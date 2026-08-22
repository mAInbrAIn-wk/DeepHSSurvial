"""
Recurrent Exam-Level Survival Analysis V2 (Keras GRU Prüfungs-Trajektorie)
===========================================================================
Erstellt ein 3D-Sequenzarray (N_studis, K_exam_max, N_features), bei dem jeder Zeitschritt
eine EINZELNE PRÜFUNG darstellt.

UPDATE V2: Fügt explizit die kumulativen Fails und den rollierenden GPA pro Prüfung hinzu,
um das Omitted Variable Bias (Confounding) aufzulösen!
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, GRU, TimeDistributed, LayerNormalization
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, plot_confusion_matrix

PADDING_VALUE = -99.0

def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PADDING_VALUE), tf.float32)
    y_true_clean = tf.maximum(y_true, 0.0)
    bce = K.binary_crossentropy(y_true_clean, y_pred)
    masked_loss = bce * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)

def build_recurrent_exam_dataset_v2(data_dir: Path, max_exams: int = 50):
    print("Lade Prüfungs- und Abschlussdaten für 3D Prüfungs-Sequenz (V2) ...")
    agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
    agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
    
    if not agg_abschluesse_path.exists():
        data_dir = Path('output_dl')
        agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
        agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
        
    df_abschluesse = pd.read_csv(agg_abschluesse_path)
    df_pruefungen = pd.read_csv(agg_pruefungen_path)
    
    df_abschluesse.columns = df_abschluesse.columns.str.strip()
    df_pruefungen.columns = df_pruefungen.columns.str.strip()
    
    df_pruefungen = df_pruefungen.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    
    # NEU IN V2: Kumulative Fehlversuche, CP-Konto und GPA berechnen
    df_pruefungen['bestanden'] = df_pruefungen['bestanden']
    df_pruefungen['is_fail'] = (~df_pruefungen['bestanden']).astype(int)
    df_pruefungen['fails_cum'] = df_pruefungen.groupby('studierenden_id')['is_fail'].cumsum()
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['cp_cum'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].cumsum()
    df_pruefungen['note_clean'] = df_pruefungen['note'].fillna(3.0)
    df_pruefungen['gpa_cum'] = df_pruefungen.groupby('studierenden_id')['note_clean'].expanding().mean().reset_index(level=0, drop=True)
    
    status_dict = df_abschluesse.set_index('studierenden_id')['status'].to_dict()
    
    studis = df_abschluesse['studierenden_id'].unique()
    num_studis = len(studis)
    
    # 12 Features pro Prüfungsschritt in V2:
    # [versuch, schwierigkeit, cp, fach_vorher, fach_glz, uebf_vorher, uebf_glz, psych_vorher, psych_glz, fails_cum, cp_cum, gpa_cum]
    n_features = 12
    
    X_seq = np.full((num_studis, max_exams, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((num_studis, max_exams, 1), PADDING_VALUE, dtype=np.float32)
    studi_events = np.zeros(num_studis, dtype=int)
    
    pr_grouped = df_pruefungen.groupby('studierenden_id')
    
    print("Erstelle 3D Sequence Array auf Prüfungs-Ebene (V2 mit Zähl-Exposition) ...")
    for i, s_id in enumerate(studis):
        if s_id not in pr_grouped.groups:
            continue
            
        studi_pr = pr_grouped.get_group(s_id)
        k_max = min(len(studi_pr), max_exams)
        
        status = str(status_dict.get(s_id, '')).strip().lower()
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        studi_events[i] = 1 if is_dropout else 0
        
        for k, row in enumerate(studi_pr.itertuples(index=False)):
            if k >= max_exams:
                break
            X_seq[i, k, :] = [
                float(row.versuch),
                float(row.schwierigkeit),
                float(row.cp),
                float(row.support_vorher_fachlich),
                float(row.support_glz_fachlich),
                float(row.support_vorher_ueberfachlich),
                float(row.support_glz_ueberfachlich),
                float(row.support_vorher_psychosozial),
                float(row.support_glz_psychosozial),
                float(row.fails_cum),  # V2 FEATURE
                float(row.cp_cum),     # V2 FEATURE
                float(row.gpa_cum)     # V2 FEATURE
            ]
            
            event_val = 1.0 if (k == len(studi_pr) - 1 and is_dropout) else 0.0
            y_seq[i, k, 0] = event_val
            
    print(f"3D Prüfungs-Tensor (V2) erfolgreich aufgebaut: X={X_seq.shape}, y={y_seq.shape}")
    return studis, X_seq, y_seq, studi_events

def train_recurrent_exam_survival_v2(data_dir: Path):
    print("\n==========================================================================")
    print("   RECURRENT EXAM-LEVEL SURVIVAL MODEL V2 (MIT GPA & FAILS_CUM)")
    print("==========================================================================")
    
    studis, X_seq, y_seq, studi_events = build_recurrent_exam_dataset_v2(data_dir)
    N, K_max, F = X_seq.shape
    
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
    y_train, y_val, y_test = y_seq[train_idx], y_seq[val_idx], y_seq[test_idx]
    
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    for X_split in [X_train, X_val, X_test]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler.transform(X_split[valid_mask])
    
    tf.random.set_seed(42)
    rec_model = Sequential([
        Masking(mask_value=PADDING_VALUE, input_shape=(K_max, F)),
        GRU(32, return_sequences=True),
        LayerNormalization(),
        Dropout(0.2),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])
    
    rec_model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=masked_binary_crossentropy)
    
    print("Trainiere Recurrent Prüfungs-GRU Survival Modell V2 ...")
    history = rec_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=256,
        verbose=1
    )
    
    y_test_pred = rec_model.predict(X_test, verbose=0)
    
    test_mask = (y_test.flatten() != PADDING_VALUE)
    y_true_flat = y_test.flatten()[test_mask]
    y_pred_flat = y_test_pred.flatten()[test_mask]
    
    auc_score = roc_auc_score(y_true_flat, y_pred_flat)
    pr_auc = average_precision_score(y_true_flat, y_pred_flat)
    brier = brier_score_loss(y_true_flat, y_pred_flat)
    
    thresh = np.percentile(y_pred_flat, 95)
    y_pred_bin = (y_pred_flat >= thresh).astype(int)
    
    prec = precision_score(y_true_flat, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_bin, zero_division=0)
    acc = accuracy_score(y_true_flat, y_pred_bin)
    rep_str = classification_report(y_true_flat, y_pred_bin, target_names=['Kein Abbruch', 'Abbruch (Top 5%)'], zero_division=0)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE RECURRENT EXAM-LEVEL V2")
    print("==========================================================================")
    print(f"  ROC-AUC (Global Ranking)         : {auc_score:.4f}")
    print(f"  PR-AUC / Average Precision (Gold): {pr_auc:.4f}")
    print("==========================================================================")
    
    metrics_dict = {
        "ROC-AUC_Seq": auc_score,
        "PR-AUC_Seq": pr_auc,
        "Brier_Score": brier,
        "Accuracy_Top5": acc,
        "Precision_Top5": prec,
        "Recall_Top5": rec,
        "F1_Top5": f1
    }
    output_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    save_metrics("recurrent_exam_survival_v2", metrics_dict, output_dir, report_str=rep_str)
    save_keras_model(rec_model, "recurrent_exam_survival_v2", output_dir)

if __name__ == '__main__':
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    train_recurrent_exam_survival_v2(data_dir)
