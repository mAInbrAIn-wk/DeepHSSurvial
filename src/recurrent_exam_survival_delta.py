"""
Recurrent Exam Survival Model Delta (GRU Exam Level)
=====================================================
Modelliert das Abbruchrisiko auf Prüfungs-Ebene unter Verwendung von
semester-aktivem Support ( active_support in dem Semester, in dem die Prüfung liegt).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Masking, GRU, LayerNormalization, TimeDistributed

from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve

def build_recurrent_exam_dataset_delta(data_dir: Path, max_exams: int = 50):
    print("Lade Prüfungs- und Abschlussdaten für 3D Prüfungs-Sequenz Delta ...")
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
    
    # Semester-lokale Support-Aktivität
    pr_sem = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg({
        'support_glz_fachlich': 'max',
        'support_glz_ueberfachlich': 'max',
        'support_glz_psychosozial': 'max'
    }).reset_index()
    
    sup_dict_fach = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_fachlich'].to_dict()
    sup_dict_uebf = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_ueberfachlich'].to_dict()
    sup_dict_psych = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_psychosozial'].to_dict()
    
    studis = df_abschluesse['studierenden_id'].unique()
    num_studis = len(studis)
    n_features = 8 # note, cp, is_fail, fach_act, uebf_act, psych_act, hzb_note, erwerb_std
    
    X_seq = np.full((num_studis, max_exams, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((num_studis, max_exams, 1), PADDING_VALUE, dtype=np.float32)
    studi_events = np.zeros(num_studis, dtype=int)
    
    df_pr_grouped = df_pruefungen.groupby('studierenden_id')
    abschluss_dict = df_abschluesse.set_index('studierenden_id').to_dict('index')
    
    for i, s_id in enumerate(studis):
        if s_id not in abschluss_dict:
            continue
        row_ab = abschluss_dict[s_id]
        status = str(row_ab['status']).strip().lower()
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        studi_events[i] = 1 if is_dropout else 0
        
        if s_id in df_pr_grouped.groups:
            studi_pr = df_pr_grouped.get_group(s_id).copy()
            num_p = min(len(studi_pr), max_exams)
            
            for k in range(num_p):
                p_row = studi_pr.iloc[k]
                sem = int(p_row['fachsemester'])
                
                note = float(p_row['note']) if not np.isnan(p_row['note']) else 3.0
                cp = float(p_row['cp']) if bool(p_row['bestanden']) else 0.0
                is_fail = 1.0 if not bool(p_row['bestanden']) else 0.0
                
                fach_act = 1.0 if sup_dict_fach.get((s_id, sem), 0) > 0 else 0.0
                uebf_act = 1.0 if sup_dict_uebf.get((s_id, sem), 0) > 0 else 0.0
                psych_act = 1.0 if sup_dict_psych.get((s_id, sem), 0) > 0 else 0.0
                
                X_seq[i, k, :] = [
                    note, cp, is_fail,
                    fach_act, uebf_act, psych_act,
                    float(row_ab['hzb_note']), float(row_ab['erwerbstaetigkeit_std'])
                ]
                
                y_seq[i, k, 0] = 1.0 if (k == num_p - 1 and is_dropout) else 0.0

    print(f"3D Prüfungs-Tensor Delta aufgebaut: X={X_seq.shape}")
    return studis, X_seq, y_seq, studi_events

def train_recurrent_exam_survival_delta(data_dir: Path):
    print("\n==========================================================================")
    print("   RECURRENT EXAM SURVIVAL MODEL DELTA (GRU EXAM LEVEL)")
    print("==========================================================================")
    
    studis, X_seq, y_seq, studi_events = build_recurrent_exam_dataset_delta(data_dir)
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
    
    model = Sequential([
        Input(shape=(K_max, F)),
        Masking(mask_value=PADDING_VALUE),
        GRU(32, return_sequences=True),
        LayerNormalization(),
        Dropout(0.2),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=masked_binary_crossentropy, metrics=['AUC'])
    
    print("Trainiere Recurrent Exam Survival Model Delta ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        verbose=0
    )
    
    preds = model.predict(X_test, verbose=0)
    mask = (y_test.flatten() != PADDING_VALUE)
    y_true_flat = y_test.flatten()[mask]
    y_pred_flat = preds.flatten()[mask]
    
    auc = roc_auc_score(y_true_flat, y_pred_flat)
    prauc = average_precision_score(y_true_flat, y_pred_flat)
    brier = brier_score_loss(y_true_flat, y_pred_flat)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE RECURRENT EXAM SURVIVAL MODEL DELTA")
    print("==========================================================================")
    print(f"    ROC-AUC                  : {auc:.4f}")
    print(f"    PR-AUC (Average Precision): {prauc:.4f}")
    print(f"    Brier Score              : {brier:.4f}")
    print("==========================================================================")
    
    metrics_dict = {
        "ROC-AUC_Exam": auc,
        "PR-AUC_Exam": prauc,
        "Brier_Score": brier
    }
    base_dir = data_dir
    save_metrics("recurrent_exam_survival_delta", metrics_dict, base_dir)
    save_keras_model(model, "recurrent_exam_survival_delta", base_dir)
    
    plot_learning_curve(history.history, "recurrent_exam_survival_delta", base_dir, metric_name='loss')
    plot_roc_curve(y_true_flat, y_pred_flat, "recurrent_exam_survival_delta", base_dir)
    plot_pr_curve(y_true_flat, y_pred_flat, "recurrent_exam_survival_delta", base_dir)
    
    print("\nTraining Recurrent Exam Survival Model Delta abgeschlossen.")

if __name__ == '__main__':
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    train_recurrent_exam_survival_delta(data_dir)
