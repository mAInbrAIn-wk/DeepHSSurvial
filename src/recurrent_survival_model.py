"""
Recurrent Survival Analysis (Keras GRU Dynamic Deep Survival)
============================================================
Kombiniert Zeitreihenanalyse (GRU/LSTM) mit Discrete-Time Survival Analysis.

Das Modell verarbeitet 3D-Tensoren (Studierende, Semester_t, Features_t) mit Masking
und lernt die bedingte Ausfallwahrscheinlichkeit h(t | X_{1..t}) für jedes Semester.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, precision_score, recall_score, f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, GRU, TimeDistributed, LayerNormalization
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report

PADDING_VALUE = -99.0

def masked_binary_crossentropy(y_true, y_pred):
    """
    Binary Cross-Entropy Loss mit Berücksichtigung von Masking (-99.0).
    """
    mask = tf.cast(tf.not_equal(y_true, PADDING_VALUE), tf.float32)
    y_true_clean = tf.maximum(y_true, 0.0)
    
    bce = K.binary_crossentropy(y_true_clean, y_pred)
    masked_loss = bce * mask
    
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)

def build_recurrent_survival_dataset(data_dir: Path, max_semesters: int = 16, blind: bool = False):
    print(f"Lade Daten für Recurrent Survival Model (blind={blind}) ...")
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
    
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)
    
    # Semesterweise Aggregation pro Student & Semester (mit Summen für Dosis-Zählung)
    sem_agg = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg(
        sem_gpa=('note', 'mean'),
        sem_cp=('cp_earned', 'sum'),
        sem_fails=('is_fail', 'sum'),
        fach_supp=('support_glz_fachlich', 'sum'),
        uebf_supp=('support_glz_ueberfachlich', 'sum'),
        psych_supp=('support_glz_psychosozial', 'sum')
    ).reset_index()
    
    # Preprocessor für 2D Feature-Matrix
    demog_cols = ['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'stg_name', 'status', 'studiendauer_semester']
    if 'migrationshintergrund' in df_abschluesse.columns:
        demog_cols.append('migrationshintergrund')
    demog_df = df_abschluesse[demog_cols].copy()
    
    print("Erstelle 3D Sequence Array (N_studis, T_max, N_features=13) ...")
    studis = demog_df['studierenden_id'].unique()
    num_studis = len(studis)
    
    # 13 Features pro Semesterschritt:
    # [gpa, cp, fails, cp_rueckstand, fach_cnt, uebf_cnt, psych_cnt, hzb, erw, erst, cum_fails_vor, delta_gpa, mig]
    n_features = 13
    
    X_seq = np.full((num_studis, max_semesters, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((num_studis, max_semesters, 1), PADDING_VALUE, dtype=np.float32)
    studi_events = np.zeros(num_studis, dtype=int)
    
    sem_lookup = sem_agg.set_index(['studierenden_id', 'fachsemester'])
    
    for i, row in enumerate(demog_df.itertuples(index=False)):
        s_id = row.studierenden_id
        max_sem = min(int(row.studiendauer_semester), max_semesters)
        status = str(row.status).strip().lower()
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        studi_events[i] = 1 if is_dropout else 0
        
        cum_cp_vorher = 0.0
        cum_fails_vorher = 0.0
        
        for sem in range(1, max_sem + 1):
            t_idx = sem - 1
            
            # Hole Semester-Features (lokale Zählungen)
            if (s_id, sem) in sem_lookup.index:
                s_data = sem_lookup.loc[(s_id, sem)]
                gpa = float(s_data['sem_gpa']) if not np.isnan(s_data['sem_gpa']) else 3.0
                cp = float(s_data['sem_cp'])
                fails = float(s_data['sem_fails'])
                fach_cnt = float(s_data['fach_supp'])
                uebf_cnt = float(s_data['uebf_supp'])
                psych_cnt = float(s_data['psych_supp'])
            else:
                gpa, cp, fails = 3.0, 0.0, 0.0
                fach_cnt, uebf_cnt, psych_cnt = 0.0, 0.0, 0.0
                
            cp_rueckstand = max(0.0, (sem - 1) * 30.0 - cum_cp_vorher)
            hzb = float(row.hzb_note)
            erw = float(row.erwerbstaetigkeit_std)
            erst = 1.0 if bool(row.erstakademiker) else 0.0
            delta_gpa = gpa - hzb
            mig = 1.0 if ('migrationshintergrund' in demog_df.columns and bool(getattr(row, 'migrationshintergrund', False))) else 0.0
            
            if blind:
                gpa = 0.0
                hzb = 0.0
                delta_gpa = 0.0
                
            X_seq[i, t_idx, :] = [
                gpa, cp, fails, cp_rueckstand,
                fach_cnt, uebf_cnt, psych_cnt,
                hzb, erw, erst, cum_fails_vorher, delta_gpa, mig
            ]
            
            cum_cp_vorher += cp
            cum_fails_vorher += fails
            
            # Target: 1 im letzten Semester des Dropouts, sonst 0
            event_val = 1.0 if (sem == max_sem and is_dropout) else 0.0
            y_seq[i, t_idx, 0] = event_val

    print(f"3D Tensor erfolgreich aufgebaut (13 Features, blind={blind}): X={X_seq.shape}, y={y_seq.shape}")
    return studis, X_seq, y_seq, studi_events

def train_recurrent_survival_model(data_dir: Path, blind: bool = False):
    suffix = "_blind" if blind else ""
    print("\n==========================================================================")
    print(f"   RECURRENT SURVIVAL MODEL (DYNAMIC DEEP SURVIVAL MIT KERAS GRU, blind={blind})")
    print("==========================================================================")
    
    studis, X_seq, y_seq, studi_events = build_recurrent_survival_dataset(data_dir, blind=blind)
    
    N, T, F = X_seq.shape
    
    # Stratifizierter 3-Wege Split (70% Train, 15% Val, 15% Test) auf Studenten-Ebene
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
    y_train, y_val, y_test = y_seq[train_idx], y_seq[val_idx], y_seq[test_idx]
    
    # Standardisiere valide (nicht-gepaddete) Features nur anhand des Train-Sets
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    for X_split in [X_train, X_val, X_test]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler.transform(X_split[valid_mask])
    
    print(f"\nSequence Split: {len(train_idx)} Train, {len(val_idx)} Val, {len(test_idx)} Test-Studierende")
    
    # Architekturentwurf
    tf.random.set_seed(42)
    rec_model = Sequential([
        Masking(mask_value=PADDING_VALUE, input_shape=(T, F)),
        GRU(32, return_sequences=True),
        LayerNormalization(),
        Dropout(0.2),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])
    
    rec_model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=masked_binary_crossentropy)
    from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, plot_confusion_matrix
    
    print("Trainiere Recurrent GRU Survival Modell ...")
    history = rec_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        verbose=1
    )
    
    y_test_pred = rec_model.predict(X_test, verbose=0)
    
    # Evaluierung auf ungepaddeten Zeitschritten
    test_mask = (y_test.flatten() != PADDING_VALUE)
    y_true_flat = y_test.flatten()[test_mask]
    y_pred_flat = y_test_pred.flatten()[test_mask]
    
    auc_score = roc_auc_score(y_true_flat, y_pred_flat)
    pr_auc = average_precision_score(y_true_flat, y_pred_flat)
    brier = brier_score_loss(y_true_flat, y_pred_flat)
    
    # Thresholding bei Top 5% Risiko-Schwellenwert
    thresh = np.percentile(y_pred_flat, 95)
    y_pred_bin = (y_pred_flat >= thresh).astype(int)
    
    prec = precision_score(y_true_flat, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_bin, zero_division=0)
    acc = accuracy_score(y_true_flat, y_pred_bin)
    rep_str = classification_report(y_true_flat, y_pred_bin, target_names=['Kein Abbruch', 'Abbruch (Top 5%)'], zero_division=0)
    
    print("\n==========================================================================")
    print(f"   ERGEBNISSE RECURRENT SURVIVAL MODELL (GRU DYNAMIC DEEP SURVIVAL, blind={blind})")
    print("==========================================================================")
    print(f"  ROC-AUC (Global Ranking)         : {auc_score:.4f}")
    print(f"  PR-AUC / Average Precision (Gold): {pr_auc:.4f}")
    print(f"  Brier Score (Kalibrierung)       : {brier:.4f}")
    print(f"  Accuracy  (Top 5% Threshold)    : {acc:.4f}")
    print(f"  Precision (Top 5% Threshold)    : {prec:.4f}")
    print(f"  Recall    (Top 5% Threshold)    : {rec:.4f}")
    print(f"  F1-Score  (Top 5% Threshold)    : {f1:.4f}")
    print("==========================================================================")
    
    # Metriken & Modell speichern
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
    model_name = f"recurrent_survival_gru{suffix}"
    save_metrics(model_name, metrics_dict, output_dir, report_str=rep_str)
    save_keras_model(rec_model, model_name, output_dir)
    plot_learning_curve(history.history, model_name, output_dir, metric_name='loss')
    plot_roc_curve(y_true_flat, y_pred_flat, model_name, output_dir)
    plot_pr_curve(y_true_flat, y_pred_flat, model_name, output_dir)
    plot_confusion_matrix(y_true_flat, y_pred_bin, model_name, output_dir)

if __name__ == '__main__':
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    train_recurrent_survival_model(data_dir)
