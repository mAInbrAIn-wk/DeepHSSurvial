"""
Dynamic DeepHit Competing Risks Model (Keras Multi-Task Edition)
================================================================
Modelliert zeitveränderliche konkurrierende Risiken (Competing Risks):
- Ursache 1: Studienabbruch / Exmatrikulation
- Ursache 2: Erfolgreicher Studienabschluss

Gemeinsames rekurrentes GRU-Backbone mit 2 ursachenspezifischen Output-Köpfen.
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
from tensorflow.keras.layers import Input, Dense, Dropout, Masking, GRU, TimeDistributed, LayerNormalization

from recurrent_survival_model import build_recurrent_survival_dataset, masked_binary_crossentropy, PADDING_VALUE

def build_competing_risks_dataset(data_dir: Path, max_semesters: int = 16):
    print("Lade Daten für Dynamic DeepHit Competing Risks ...")
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
    
    sem_agg = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg(
        sem_gpa=('note', 'mean'),
        sem_cp=('cp', lambda x: df_pruefungen.loc[x.index[df_pruefungen.loc[x.index, 'bestanden']], 'cp'].sum()),
        sem_fails=('bestanden', lambda x: (~x).sum()),
        fach_supp=('support_glz_fachlich', 'max'),
        uebf_supp=('support_glz_ueberfachlich', 'max'),
        psych_supp=('support_glz_psychosozial', 'max')
    ).reset_index()
    
    demog_df = df_abschluesse[['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'stg_name', 'status', 'studiendauer_semester']].copy()
    
    studis = demog_df['studierenden_id'].unique()
    num_studis = len(studis)
    n_features = 8
    
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
        
        cum_fach, cum_uebf, cum_psych = 0.0, 0.0, 0.0
        
        for sem in range(1, max_sem + 1):
            t_idx = sem - 1
            if (s_id, sem) in sem_lookup.index:
                s_data = sem_lookup.loc[(s_id, sem)]
                gpa = float(s_data['sem_gpa']) if not np.isnan(s_data['sem_gpa']) else 3.0
                cp = float(s_data['sem_cp'])
                fails = float(s_data['sem_fails'])
                if s_data['fach_supp'] > 0: cum_fach = 1.0
                if s_data['uebf_supp'] > 0: cum_uebf = 1.0
                if s_data['psych_supp'] > 0: cum_psych = 1.0
            else:
                gpa, cp, fails = 3.0, 0.0, 0.0
                
            X_seq[i, t_idx, :] = [
                gpa, cp, fails, cum_fach, cum_uebf, cum_psych,
                float(row.hzb_note), float(row.erwerbstaetigkeit_std)
            ]
            
            y_dropout[i, t_idx, 0] = 1.0 if (sem == max_sem and is_dropout) else 0.0
            y_grad[i, t_idx, 0] = 1.0 if (sem == max_sem and is_grad) else 0.0

    return studis, X_seq, y_dropout, y_grad, studi_events

def train_dynamic_deephit_model(data_dir: Path):
    print("\n==========================================================================")
    print("   DYNAMIC DEEPHIT COMPETING RISKS MODEL (DROPOUT VS GRADUATION)")
    print("==========================================================================")
    
    studis, X_seq, y_dropout, y_grad, studi_events = build_competing_risks_dataset(data_dir)
    
    N, T, F = X_seq.shape
    
    # 3-Wege Split (70% Train, 15% Val, 15% Test) stratifiziert nach Studenten-Event
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
    yd_train, yd_val, yd_test = y_dropout[train_idx], y_dropout[val_idx], y_dropout[test_idx]
    yg_train, yg_val, yg_test = y_grad[train_idx], y_grad[val_idx], y_grad[test_idx]
    
    # Standardisiere valide (nicht-gepaddete) Features nur anhand des Train-Sets
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    for X_split in [X_train, X_val, X_test]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler.transform(X_split[valid_mask])
    
    # Keras Multi-Task Model
    tf.random.set_seed(42)
    
    inputs = Input(shape=(T, F))
    masked_in = Masking(mask_value=PADDING_VALUE)(inputs)
    
    shared_gru = GRU(32, return_sequences=True)(masked_in)
    shared_gru = LayerNormalization()(shared_gru)
    shared_gru = Dropout(0.2)(shared_gru)
    
    # Head 1: Dropout Risk
    d_dense = TimeDistributed(Dense(16, activation='relu'))(shared_gru)
    out_dropout = TimeDistributed(Dense(1, activation='sigmoid'), name='dropout_head')(d_dense)
    
    # Head 2: Graduation Success
    g_dense = TimeDistributed(Dense(16, activation='relu'))(shared_gru)
    out_grad = TimeDistributed(Dense(1, activation='sigmoid'), name='graduation_head')(g_dense)
    
    deephit = Model(inputs=inputs, outputs=[out_dropout, out_grad], name="Dynamic_DeepHit")
    deephit.compile(
        optimizer=tf.keras.optimizers.Adam(0.005),
        loss={'dropout_head': masked_binary_crossentropy, 'graduation_head': masked_binary_crossentropy}
    )
    
    from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve
    
    print("Trainiere Dynamic DeepHit Competing Risks Modell ...")
    history = deephit.fit(
        X_train,
        {'dropout_head': yd_train, 'graduation_head': yg_train},
        validation_data=(X_val, {'dropout_head': yd_val, 'graduation_head': yg_val}),
        epochs=30,
        batch_size=256,
        verbose=1
    )
    
    preds = deephit.predict(X_test, verbose=0)
    pred_d, pred_g = preds[0], preds[1]
    
    # Evaluierung
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
    print("   ERGEBNISSE DYNAMIC DEEPHIT COMPETING RISKS MODELL")
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
    
    # Metriken & Modell speichern
    metrics_dict = {
        "ROC-AUC_Dropout": auc_d,
        "PR-AUC_Dropout": prauc_d,
        "Brier_Dropout": brier_d,
        "ROC-AUC_Graduation": auc_g,
        "PR-AUC_Graduation": prauc_g,
        "Brier_Graduation": brier_g
    }
    output_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    save_metrics("dynamic_deephit_competing", metrics_dict, output_dir)
    save_keras_model(deephit, "dynamic_deephit_competing", output_dir)
    
    plot_learning_curve(history.history, "dynamic_deephit_competing", output_dir, metric_name='loss')
    plot_roc_curve(yd_true_flat, yd_pred_flat, "dynamic_deephit_dropout", output_dir)
    plot_pr_curve(yd_true_flat, yd_pred_flat, "dynamic_deephit_dropout", output_dir)
    plot_roc_curve(yg_true_flat, yg_pred_flat, "dynamic_deephit_graduation", output_dir)
    plot_pr_curve(yg_true_flat, yg_pred_flat, "dynamic_deephit_graduation", output_dir)

if __name__ == '__main__':
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    train_dynamic_deephit_model(data_dir)
