"""
Dynamic DeepHit Delta Competing Risks Model (Semester Level)
============================================================
Modelliert zeitveränderliche konkurrierende Risiken (Dropout vs. Abschluss)
auf Basis von semester-lokalen Treatments (active_support) und Leistungs-Deltas.
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Masking, GRU, LayerNormalization, TimeDistributed

from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve

def build_competing_risks_dataset_delta(data_dir: Path, max_semesters: int = 16):
    print("Lade Daten für Dynamic DeepHit Delta (Semester Level) ...")
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
        fach_supp_count=('support_glz_fachlich', 'max'),
        uebf_supp_count=('support_glz_ueberfachlich', 'max'),
        psych_supp_count=('support_glz_psychosozial', 'max')
    ).reset_index()
    
    demog_df = df_abschluesse[['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'stg_name', 'status', 'studiendauer_semester']].copy()
    
    studis = demog_df['studierenden_id'].unique()
    num_studis = len(studis)
    n_features = 9 # sem_gpa, sem_cp, sem_fails, cp_rueckstand, fach_count, uebf_count, psych_count, hzb_note, erwerb_std
    
    X_seq = np.full((num_studis, max_semesters, n_features), PADDING_VALUE, dtype=np.float32)
    y_dropout = np.full((num_studis, max_semesters, 1), PADDING_VALUE, dtype=np.float32)
    y_grad = np.full((num_studis, max_semesters, 1), PADDING_VALUE, dtype=np.float32)
    studi_events = np.zeros(num_studis, dtype=int)
    
    sem_lookup = sem_agg.set_index(['studierenden_id', 'fachsemester'])
    
    for i, row in enumerate(demog_df.itertuples(index=False)):
        s_id = row.studierenden_id
        max_sem = min(int(row.studiendauer_semester), max_semesters)
        status = str(row.status).strip().lower()
        
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        is_grad = status in ['abgeschlossen']
        studi_events[i] = 1 if is_dropout else 0
        
        cum_cp_vorher = 0.0
        
        for sem in range(1, max_sem + 1):
            t_idx = sem - 1
            if (s_id, sem) in sem_lookup.index:
                s_data = sem_lookup.loc[(s_id, sem)]
                gpa = float(s_data['sem_gpa']) if not np.isnan(s_data['sem_gpa']) else 3.0
                cp = float(s_data['sem_cp'])
                fails = float(s_data['sem_fails'])
                fach_cnt = float(s_data['fach_supp_count'])
                uebf_cnt = float(s_data['uebf_supp_count'])
                psych_cnt = float(s_data['psych_supp_count'])
            else:
                gpa, cp, fails = 3.0, 0.0, 0.0
                fach_cnt, uebf_cnt, psych_cnt = 0.0, 0.0, 0.0
                
            cp_rueckstand = max(0.0, (sem - 1) * 30.0 - cum_cp_vorher)
            
            X_seq[i, t_idx, :] = [
                gpa, cp, fails, cp_rueckstand,
                fach_cnt, uebf_cnt, psych_cnt,
                float(row.hzb_note), float(row.erwerbstaetigkeit_std)
            ]
            
            cum_cp_vorher += cp
            y_dropout[i, t_idx, 0] = 1.0 if (sem == max_sem and is_dropout) else 0.0
            y_grad[i, t_idx, 0] = 1.0 if (sem == max_sem and is_grad) else 0.0

    print(f"DeepHit Delta Tensor aufgebaut (mit Zählung): X={X_seq.shape}")
    return studis, X_seq, y_dropout, y_grad, studi_events

def train_dynamic_deephit_delta_model(data_dir: Path):
    print("\n==========================================================================")
    print("   DYNAMIC DEEPHIT DELTA COMPETING RISKS MODEL (SEMESTER LEVEL)")
    print("==========================================================================")
    
    studis, X_seq, y_dropout, y_grad, studi_events = build_competing_risks_dataset_delta(data_dir)
    N, T, F = X_seq.shape
    
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
    yd_train, yd_val, yd_test = y_dropout[train_idx], y_dropout[val_idx], y_dropout[test_idx]
    yg_train, yg_val, yg_test = y_grad[train_idx], y_grad[val_idx], y_grad[test_idx]
    
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    for X_split in [X_train, X_val, X_test]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler.transform(X_split[valid_mask])
        
    tf.random.set_seed(42)
    
    inputs = Input(shape=(T, F))
    masked_in = Masking(mask_value=PADDING_VALUE)(inputs)
    
    shared_gru = GRU(32, return_sequences=True)(masked_in)
    shared_gru = LayerNormalization()(shared_gru)
    shared_gru = Dropout(0.2)(shared_gru)
    
    d_dense = TimeDistributed(Dense(16, activation='relu'))(shared_gru)
    out_dropout = TimeDistributed(Dense(1, activation='sigmoid'), name='dropout_head')(d_dense)
    
    g_dense = TimeDistributed(Dense(16, activation='relu'))(shared_gru)
    out_grad = TimeDistributed(Dense(1, activation='sigmoid'), name='graduation_head')(g_dense)
    
    deephit_delta = Model(inputs=inputs, outputs=[out_dropout, out_grad], name="Dynamic_DeepHit_Delta")
    deephit_delta.compile(
        optimizer=tf.keras.optimizers.Adam(0.005),
        loss={'dropout_head': masked_binary_crossentropy, 'graduation_head': masked_binary_crossentropy}
    )
    
    print("Trainiere Dynamic DeepHit Delta ...")
    history = deephit_delta.fit(
        X_train,
        {'dropout_head': yd_train, 'graduation_head': yg_train},
        validation_data=(X_val, {'dropout_head': yd_val, 'graduation_head': yg_val}),
        epochs=30,
        batch_size=256,
        verbose=0
    )
    
    preds = deephit_delta.predict(X_test, verbose=0)
    pred_d, pred_g = preds[0], preds[1]
    
    mask_d = (yd_test.flatten() != PADDING_VALUE)
    yd_true_flat = yd_test.flatten()[mask_d]
    yd_pred_flat = pred_d.flatten()[mask_d]
    
    mask_g = (yg_test.flatten() != PADDING_VALUE)
    yg_true_flat = yg_test.flatten()[mask_g]
    yg_pred_flat = pred_g.flatten()[mask_g]
    
    auc_d = roc_auc_score(yd_true_flat, yd_pred_flat)
    prauc_d = average_precision_score(yd_true_flat, yd_pred_flat)
    brier_d = brier_score_loss(yd_true_flat, yd_pred_flat)
    
    auc_g = roc_auc_score(yg_true_flat, yg_pred_flat)
    prauc_g = average_precision_score(yg_true_flat, yg_pred_flat)
    brier_g = brier_score_loss(yg_true_flat, yg_pred_flat)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE DYNAMIC DEEPHIT DELTA COMPETING RISKS MODELL")
    print("==========================================================================")
    print(f"  [Ursache 1: Studienabbruch]")
    print(f"    ROC-AUC                  : {auc_d:.4f}")
    print(f"    PR-AUC (Average Precision): {prauc_d:.4f}")
    print(f"    Brier Score              : {brier_d:.4f}")
    print(f"  [Ursache 2: Studienerfolg / Abschluss]")
    print(f"    ROC-AUC                  : {auc_g:.4f}")
    print(f"    PR-AUC (Average Precision): {prauc_g:.4f}")
    print(f"    Brier Score              : {brier_g:.4f}")
    print("==========================================================================")
    
    metrics_dict = {
        "ROC-AUC_Dropout": auc_d,
        "PR-AUC_Dropout": prauc_d,
        "Brier_Dropout": brier_d,
        "ROC-AUC_Graduation": auc_g,
        "PR-AUC_Graduation": prauc_g,
        "Brier_Graduation": brier_g
    }
    base_dir = data_dir
    save_metrics("dynamic_deephit_delta", metrics_dict, base_dir)
    save_keras_model(deephit_delta, "dynamic_deephit_delta", base_dir)
    
    plot_learning_curve(history.history, "dynamic_deephit_delta", base_dir, metric_name='loss')
    plot_roc_curve(yd_true_flat, yd_pred_flat, "dynamic_deephit_delta_dropout", base_dir)
    plot_pr_curve(yd_true_flat, yd_pred_flat, "dynamic_deephit_delta_dropout", base_dir)
    plot_roc_curve(yg_true_flat, yg_pred_flat, "dynamic_deephit_delta_graduation", base_dir)
    plot_pr_curve(yg_true_flat, yg_pred_flat, "dynamic_deephit_delta_graduation", base_dir)
    
    print("\nTraining DeepHit Delta abgeschlossen.")

if __name__ == '__main__':
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    train_dynamic_deephit_delta_model(data_dir)
