"""
Recurrent Survival Model Delta (GRU Semester Level)
===================================================
Trainiert ein rekurrierendes GRU-Modell auf Semester-Ebene mit semester-lokalen
Support-Interventionen und Leistungs-Deltas.
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

def build_recurrent_survival_dataset_delta(data_dir: Path, max_semesters: int = 16):
    print("Lade Daten für Recurrent Survival Model Delta (Semester Level) ...")
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
    
    sem_agg = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg(
        sem_gpa=('note', 'mean'),
        sem_cp=('cp_earned', 'sum'),
        sem_fails=('is_fail', 'sum'),
        fach_supp_active=('support_glz_fachlich', lambda x: int((x > 0).any())),
        uebf_supp_active=('support_glz_ueberfachlich', lambda x: int((x > 0).any())),
        psych_supp_active=('support_glz_psychosozial', lambda x: int((x > 0).any()))
    ).reset_index()
    
    demog_df = df_abschluesse[['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'stg_name', 'status', 'studiendauer_semester']].copy()
    
    studis = demog_df['studierenden_id'].unique()
    num_studis = len(studis)
    n_features = 9 # sem_gpa, sem_cp, sem_fails, cp_rueckstand, fach_act, uebf_act, psych_act, hzb_note, erwerb_std
    
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
        
        for sem in range(1, max_sem + 1):
            t_idx = sem - 1
            if (s_id, sem) in sem_lookup.index:
                s_data = sem_lookup.loc[(s_id, sem)]
                gpa = float(s_data['sem_gpa']) if not np.isnan(s_data['sem_gpa']) else 3.0
                cp = float(s_data['sem_cp'])
                fails = float(s_data['sem_fails'])
                fach_act = float(s_data['fach_supp_active'])
                uebf_act = float(s_data['uebf_supp_active'])
                psych_act = float(s_data['psych_supp_active'])
            else:
                gpa, cp, fails = 3.0, 0.0, 0.0
                fach_act, uebf_act, psych_act = 0.0, 0.0, 0.0
                
            cp_rueckstand = max(0.0, (sem - 1) * 30.0 - cum_cp_vorher)
            
            X_seq[i, t_idx, :] = [
                gpa, cp, fails, cp_rueckstand,
                fach_act, uebf_act, psych_act,
                float(row.hzb_note), float(row.erwerbstaetigkeit_std)
            ]
            
            cum_cp_vorher += cp
            y_seq[i, t_idx, 0] = 1.0 if (sem == max_sem and is_dropout) else 0.0

    print(f"GRU Semester Delta Tensor aufgebaut: X={X_seq.shape}")
    return studis, X_seq, y_seq, studi_events

def train_recurrent_survival_model_delta(data_dir: Path):
    print("\n==========================================================================")
    print("   RECURRENT SURVIVAL MODEL DELTA (GRU SEMESTER LEVEL)")
    print("==========================================================================")
    
    studis, X_seq, y_seq, studi_events = build_recurrent_survival_dataset_delta(data_dir)
    N, T, F = X_seq.shape
    
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
        Input(shape=(T, F)),
        Masking(mask_value=PADDING_VALUE),
        GRU(32, return_sequences=True),
        LayerNormalization(),
        Dropout(0.2),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=masked_binary_crossentropy, metrics=['AUC'])
    
    print("Trainiere Recurrent Survival Model Delta ...")
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
    print("   ERGEBNISSE RECURRENT SURVIVAL MODEL DELTA")
    print("==========================================================================")
    print(f"    ROC-AUC                  : {auc:.4f}")
    print(f"    PR-AUC (Average Precision): {prauc:.4f}")
    print(f"    Brier Score              : {brier:.4f}")
    print("==========================================================================")
    
    metrics_dict = {
        "ROC-AUC_Panel": auc,
        "PR-AUC_Panel": prauc,
        "Brier_Score": brier
    }
    base_dir = data_dir
    save_metrics("recurrent_survival_model_delta", metrics_dict, base_dir)
    save_keras_model(model, "recurrent_survival_model_delta", base_dir)
    
    plot_learning_curve(history.history, "recurrent_survival_model_delta", base_dir, metric_name='loss')
    plot_roc_curve(y_true_flat, y_pred_flat, "recurrent_survival_model_delta", base_dir)
    plot_pr_curve(y_true_flat, y_pred_flat, "recurrent_survival_model_delta", base_dir)
    
    print("\nTraining Recurrent Survival Model Delta abgeschlossen.")

if __name__ == '__main__':
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    train_recurrent_survival_model_delta(data_dir)
