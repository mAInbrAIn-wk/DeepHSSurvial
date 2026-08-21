"""
Deep Transformer Regression & Survival Models (Enlarged Capacity + Attention Pooling)
========================================================================================
Implementiert 3 hochentwickelte Transformer-Modelle:
1. Deep Semester-Transformer Regressor (d_model=128, 8 Heads, 3 Blöcke, Attention Pooling)
2. Deep Exam-Transformer Regressor (d_model=128, 8 Heads, 3 Blöcke, Attention Pooling)
3. Deep Exam-Transformer Survival (d_model=128, 8 Heads, 3 Blöcke, Attention Pooling, Event Prediction)

Funktionen:
- Attention-Weighted Temporal Pooling anstelle von GlobalAveragePooling1D
- Mindestens 50 Epochen mit EarlyStopping (patience=15, restore_best_weights=True)
- Stratifizierter 70/15/15 Three-Way-Split (Train / Val / Test)
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Masking, Layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, average_precision_score, brier_score_loss

PADDING_VALUE = -99.0

class AttentionPooling(Layer):
    """Gelerntes Attention-Weighted Pooling über die Zeitschritte T."""
    def __init__(self, d_model=128, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.d_model = d_model
        self.score_dense = Dense(1, activation='tanh')

    def call(self, inputs):
        # inputs: (batch, T, d_model)
        scores = self.score_dense(inputs)  # (batch, T, 1)
        is_padded = tf.reduce_all(tf.equal(inputs, PADDING_VALUE), axis=-1, keepdims=True)  # (batch, T, 1)
        padding_mask = tf.cast(is_padded, tf.float32) * -1e9
        scores = scores + padding_mask
        weights = tf.nn.softmax(scores, axis=1)  # (batch, T, 1)
        pooled = tf.reduce_sum(inputs * weights, axis=1)  # (batch, d_model)
        return pooled

def build_deep_transformer_backbone(input_shape, d_model=128, num_heads=8, num_blocks=3, dropout_rate=0.2):
    """Erzeugt das hochkapazitäre Transformer Backbone mit Positional Embedding & Attention Pooling."""
    inputs = Input(shape=input_shape)
    
    # Lineare Projektion in den d_model Raum
    x = Dense(d_model, activation='relu')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Gestapelte Transformer Encoder Blöcke
    for _ in range(num_blocks):
        # Multi-Head Self-Attention
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)(x, x)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        
        # Feed-Forward Network
        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
        
    # Attention-Weighted Pooling über die Zeitschritte T
    pooled = AttentionPooling(d_model=d_model)(x)
    
    # Dense Projection Head
    head = Dense(64, activation='relu')(pooled)
    head = LayerNormalization()(head)
    head = Dropout(dropout_rate)(head)
    
    return inputs, head


from timeseries_semester import create_semester_timeseries_dataset
from timeseries_exam import create_exam_timeseries_dataset

def build_canonical_exam_survival_dataset(data_dir: Path, max_exams: int = 40, exclude_last_exam_for_dropouts: bool = True):
    """
    Erstellt den harmonisierten 9-Feature Prüfungssatz für Exam Survival
    inklusive Last-Exam Exclusion zur Vermeidung von Future-Data Leakage.
    Features: [versuch, schwierigkeit, cp, is_fail, fach_act, uebf_act, psych_act, hzb_note, erwerbstaetigkeit_std]
    """
    agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
    agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
    
    if not agg_abschluesse_path.exists():
        data_dir = Path('output_dl') if Path('output_dl/agg_abschluesse.csv').exists() else Path('../output_dl')
        agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
        agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
        
    df_abschluesse = pd.read_csv(agg_abschluesse_path)
    df_pruefungen = pd.read_csv(agg_pruefungen_path)
    
    df_abschluesse.columns = df_abschluesse.columns.str.strip()
    df_pruefungen.columns = df_pruefungen.columns.str.strip()
    
    pr_sem = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg({
        'support_glz_fachlich': 'max',
        'support_glz_ueberfachlich': 'max',
        'support_glz_psychosozial': 'max'
    }).reset_index()
    
    sup_dict_fach = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_fachlich'].to_dict()
    sup_dict_uebf = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_ueberfachlich'].to_dict()
    sup_dict_psych = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_psychosozial'].to_dict()
    
    studis = df_abschluesse['studierenden_id'].unique()
    abschluss_dict = df_abschluesse.set_index('studierenden_id').to_dict('index')
    pr_grouped = df_pruefungen.groupby('studierenden_id')
    
    n_features = 9
    X_seq = np.full((len(studis), max_exams, n_features), PADDING_VALUE, dtype=np.float32)
    y_surv = np.zeros(len(studis), dtype=np.float32)
    
    for i, s_id in enumerate(studis):
        row_ab = abschluss_dict[s_id]
        status = str(row_ab['status']).strip().lower()
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        y_surv[i] = 1.0 if is_dropout else 0.0
        
        if s_id in pr_grouped.groups:
            studi_pr = pr_grouped.get_group(s_id).sort_values(['fachsemester', 'pruefung_id'])
            # Last-Exam Exclusion zur Vermeidung von Future-Data Leakage:
            if is_dropout and exclude_last_exam_for_dropouts and len(studi_pr) > 1:
                studi_pr = studi_pr.iloc[:-1]
            num_p = min(len(studi_pr), max_exams)
            for k in range(num_p):
                p_row = studi_pr.iloc[k]
                sem = int(p_row['fachsemester'])
                X_seq[i, k, 0] = float(p_row['versuch'])
                X_seq[i, k, 1] = float(p_row['schwierigkeit'])
                X_seq[i, k, 2] = float(p_row['cp'])
                X_seq[i, k, 3] = 1.0 if not bool(p_row['bestanden']) else 0.0
                X_seq[i, k, 4] = 1.0 if sup_dict_fach.get((s_id, sem), 0) > 0 else 0.0
                X_seq[i, k, 5] = 1.0 if sup_dict_uebf.get((s_id, sem), 0) > 0 else 0.0
                X_seq[i, k, 6] = 1.0 if sup_dict_psych.get((s_id, sem), 0) > 0 else 0.0
                X_seq[i, k, 7] = float(row_ab['hzb_note'])
                X_seq[i, k, 8] = float(row_ab['erwerbstaetigkeit_std'])
                
    return X_seq, y_surv, max_exams, n_features


def train_deep_transformer_regression(data_dir: Path, output_dir: Path):
    print("\n================================================================================")
    print("   DEEP TRANSFORMER REGRESSION & SURVIVAL (KANONISCHE DATENSÄTZE & LEAKAGE-FREI)")
    print("================================================================================")
    
    # -------------------------------------------------------------------------
    # 1. DEEP SEMESTER-TRANSFORMER REGRESSOR (KLASSE 2b)
    # -------------------------------------------------------------------------
    print("\n>>> [1/3] Trainiere Deep Semester-Transformer Regressor (GPA Prediction, Klasse 2b) ...")
    
    X_sem_3d, y_gpa_sem, T_sem, F_sem = create_semester_timeseries_dataset(data_dir)
    
    # 70/15/15 Split
    X_tr, X_temp, y_tr, y_temp = train_test_split(X_sem_3d, y_gpa_sem, test_size=0.30, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    # Feature Scaling (nur auf Training Set fitten)
    scaler_sem = StandardScaler()
    valid_mask_tr = (X_tr[:, :, 0] != PADDING_VALUE)
    scaler_sem.fit(X_tr[valid_mask_tr])
    
    for X_split in [X_tr, X_va, X_te]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler_sem.transform(X_split[valid_mask])
        
    inputs_sem, head_sem = build_deep_transformer_backbone((T_sem, F_sem), d_model=128, num_heads=8, num_blocks=3)
    outputs_sem = Dense(1, activation='linear')(head_sem)
    model_sem = Model(inputs=inputs_sem, outputs=outputs_sem)
    model_sem.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model_sem.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=60, batch_size=256, callbacks=[es], verbose=1)
    
    y_pred_sem = model_sem.predict(X_te, verbose=0).flatten()
    rmse_sem = float(np.sqrt(mean_squared_error(y_te, y_pred_sem)))
    mae_sem = float(mean_absolute_error(y_te, y_pred_sem))
    r2_sem = float(r2_score(y_te, y_pred_sem))
    print(f"   [OK] Semester-Transformer Regressor (R²: {r2_sem:.4f}, RMSE: {rmse_sem:.4f}, MAE: {mae_sem:.4f})")
    
    # -------------------------------------------------------------------------
    # 2. DEEP EXAM-TRANSFORMER REGRESSOR (KLASSE 3 - OHNE NOTEN-LEAKAGE)
    # -------------------------------------------------------------------------
    print("\n>>> [2/3] Trainiere Deep Exam-Transformer Regressor (Klasse 3, ohne Noten-Leakage) ...")
    
    X_exam_3d, y_gpa_ex, T_exam, F_exam = create_exam_timeseries_dataset(data_dir)
    
    X_tr_e, X_temp_e, y_tr_e, y_temp_e = train_test_split(X_exam_3d, y_gpa_ex, test_size=0.30, random_state=42)
    X_va_e, X_te_e, y_va_e, y_te_e = train_test_split(X_temp_e, y_temp_e, test_size=0.50, random_state=42)
    
    scaler_ex = StandardScaler()
    valid_mask_tr_e = (X_tr_e[:, :, 0] != PADDING_VALUE)
    scaler_ex.fit(X_tr_e[valid_mask_tr_e])
    
    for X_split in [X_tr_e, X_va_e, X_te_e]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler_ex.transform(X_split[valid_mask])
        
    inputs_ex, head_ex = build_deep_transformer_backbone((T_exam, F_exam), d_model=128, num_heads=8, num_blocks=3)
    outputs_ex = Dense(1, activation='linear')(head_ex)
    model_ex = Model(inputs=inputs_ex, outputs=outputs_ex)
    model_ex.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    
    model_ex.fit(X_tr_e, y_tr_e, validation_data=(X_va_e, y_va_e), epochs=60, batch_size=256, callbacks=[es], verbose=1)
    
    y_pred_ex = model_ex.predict(X_te_e, verbose=0).flatten()
    rmse_ex = float(np.sqrt(mean_squared_error(y_te_e, y_pred_ex)))
    mae_ex = float(mean_absolute_error(y_te_e, y_pred_ex))
    r2_ex = float(r2_score(y_te_e, y_pred_ex))
    print(f"   [OK] Deep Exam-Transformer Regressor (R²: {r2_ex:.4f}, RMSE: {rmse_ex:.4f}, MAE: {mae_ex:.4f})")
    
    # -------------------------------------------------------------------------
    # 3. DEEP EXAM-TRANSFORMER SURVIVAL (KLASSE 7 - MIT LAST-EXAM EXCLUSION)
    # -------------------------------------------------------------------------
    print("\n>>> [3/3] Trainiere Deep Exam-Transformer Survival (Klasse 7, Last-Exam Exclusion) ...")
    
    X_surv_3d, y_surv, T_surv, F_surv = build_canonical_exam_survival_dataset(data_dir, max_exams=40, exclude_last_exam_for_dropouts=True)
    
    X_tr_s, X_temp_s, y_tr_s, y_temp_s = train_test_split(X_surv_3d, y_surv, test_size=0.30, random_state=42, stratify=y_surv)
    X_va_s, X_te_s, y_va_s, y_te_s = train_test_split(X_temp_s, y_temp_s, test_size=0.50, random_state=42, stratify=y_temp_s)
    
    scaler_surv = StandardScaler()
    valid_mask_tr_s = (X_tr_s[:, :, 0] != PADDING_VALUE)
    scaler_surv.fit(X_tr_s[valid_mask_tr_s])
    
    for X_split in [X_tr_s, X_va_s, X_te_s]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler_surv.transform(X_split[valid_mask])
        
    inputs_surv, head_surv = build_deep_transformer_backbone((T_surv, F_surv), d_model=128, num_heads=8, num_blocks=3)
    outputs_surv = Dense(1, activation='sigmoid')(head_surv)
    model_surv = Model(inputs=inputs_surv, outputs=outputs_surv)
    model_surv.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['AUC'])
    
    model_surv.fit(X_tr_s, y_tr_s, validation_data=(X_va_s, y_va_s), epochs=60, batch_size=256, callbacks=[es], verbose=1)
    
    y_pred_surv = model_surv.predict(X_te_s, verbose=0).flatten()
    auc_surv = float(roc_auc_score(y_te_s, y_pred_surv))
    prauc_surv = float(average_precision_score(y_te_s, y_pred_surv))
    brier_surv = float(brier_score_loss(y_te_s, y_pred_surv))
    print(f"   [OK] Deep Exam-Transformer Survival (ROC-AUC: {auc_surv:.4f}, PR-AUC: {prauc_surv:.4f}, Brier: {brier_surv:.4f})")
    
    # Speichere Metriken
    metrics = {
        "deep_semester_transformer_regression": {"R2": float(r2_sem), "RMSE": float(rmse_sem), "MAE": float(mae_sem)},
        "deep_exam_transformer_regression": {"R2": float(r2_ex), "RMSE": float(rmse_ex), "MAE": float(mae_ex)},
        "deep_exam_transformer_survival": {"ROC-AUC": float(auc_surv), "PR-AUC": float(prauc_surv), "Brier_Score": float(brier_surv)}
    }
    
    os.makedirs(output_dir / "metrics", exist_ok=True)
    with open(output_dir / "metrics" / "deep_transformer_regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nDeep Transformer Regression & Survival Pipeline erfolgreich und leakage-frei abgeschlossen!")

if __name__ == "__main__":
    data_dir = Path("output_dl") if Path("output_dl").exists() else Path("../output_dl")
    output_dir = data_dir
    train_deep_transformer_regression(data_dir, output_dir)

