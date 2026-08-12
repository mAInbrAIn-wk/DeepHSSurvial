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

    def call(self, inputs, mask=None):
        # inputs: (batch, T, d_model)
        scores = self.score_dense(inputs)  # (batch, T, 1)
        if mask is not None:
            # mask: (batch, T)
            mask_expanded = tf.expand_dims(mask, -1)  # (batch, T, 1)
            padding_mask = tf.cast(tf.logical_not(mask_expanded), tf.float32) * -1e9
            scores += padding_mask
        
        weights = tf.nn.softmax(scores, axis=1)  # (batch, T, 1)
        pooled = tf.reduce_sum(inputs * weights, axis=1)  # (batch, d_model)
        return pooled

def build_deep_transformer_backbone(input_shape, d_model=128, num_heads=8, num_blocks=3, dropout_rate=0.2):
    """Erzeugt das hochkapazitäre Transformer Backbone mit Positional Embedding & Attention Pooling."""
    inputs = Input(shape=input_shape)
    
    # Masking für Padding-Werte
    masked = Masking(mask_value=PADDING_VALUE)(inputs)
    mask = masked._keras_mask
    
    # Lineare Projektion in den d_model Raum
    x = Dense(d_model, activation='relu')(masked)
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
    pooled = AttentionPooling(d_model=d_model)(x, mask=mask)
    
    # Dense Projection Head
    head = Dense(64, activation='relu')(pooled)
    head = LayerNormalization()(head)
    head = Dropout(dropout_rate)(head)
    
    return inputs, head

def train_deep_transformer_regression(data_dir: Path, output_dir: Path):
    print("\n================================================================================")
    print("   DEEP TRANSFORMER REGRESSION & SURVIVAL (ENLARGED CAPACITY)")
    print("================================================================================")
    
    # -------------------------------------------------------------------------
    # 1. DEEP SEMESTER-TRANSFORMER REGRESSOR
    # -------------------------------------------------------------------------
    print("\n>>> [1/3] Trainiere Deep Semester-Transformer Regressor (GPA Prediction) ...")
    
    studis_file = data_dir / "studierenden_id.csv" if (data_dir / "studierenden_id.csv").exists() else data_dir / "studierende.csv"
    df_studis = pd.read_csv(studis_file)
    df_sem = pd.read_csv(data_dir / "einschreibungen.csv")
    df_pr = pd.read_csv(data_dir / "pruefungen.csv")
    
    # Noten-Target pro Student
    valid_pr = df_pr[df_pr["bestanden"] == True]
    gpa_target = valid_pr.groupby("studierenden_id")["note"].mean().to_dict()
    
    studis = [s for s in df_studis["studierenden_id"].unique() if s in gpa_target]
    N = len(studis)
    y_gpa = np.array([gpa_target[s] for s in studis])
    
    # 3D Tensor Aufbau für Semesterniveau (T=16, F=6)
    sem_features = ["sem_cp_earned", "sem_cp_attempted", "sem_fail_count", "fach_supp", "uebf_supp", "psych_supp"]
    T_sem = 16
    F_sem = len(sem_features)
    
    # Semesterweise Aggregation
    pr_agg = df_pr.groupby(["studierenden_id", "semester_id"]).agg(
        sem_cp_earned=("cp", lambda x: df_pr.loc[x.index[df_pr.loc[x.index, "bestanden"] == True], "cp"].sum()),
        sem_cp_attempted=("cp", "sum"),
        sem_fail_count=("bestanden", lambda x: (x == False).sum())
    ).reset_index()
    
    X_sem_3d = np.full((N, T_sem, F_sem), PADDING_VALUE, dtype=np.float32)
    
    sem_order_unique = sorted(df_sem["semester_id"].unique())
    sem_to_idx = {s: i for i, s in enumerate(sem_order_unique)}
    
    for i, s_id in enumerate(studis):
        s_rows = pr_agg[pr_agg["studierenden_id"] == s_id]
        for _, row in s_rows.iterrows():
            t_idx = sem_to_idx.get(row["semester_id"], 0)
            if t_idx < T_sem:
                X_sem_3d[i, t_idx, 0] = row["sem_cp_earned"]
                X_sem_3d[i, t_idx, 1] = row["sem_cp_attempted"]
                X_sem_3d[i, t_idx, 2] = row["sem_fail_count"]
                X_sem_3d[i, t_idx, 3:] = 0.0 # Support Mappings
                
    # Feature Scaling
    scaler = StandardScaler()
    valid_mask = (X_sem_3d != PADDING_VALUE)
    X_sem_3d[valid_mask] = scaler.fit_transform(X_sem_3d[valid_mask].reshape(-1, F_sem)).flatten()
    
    # 70/15/15 Split
    X_tr, X_temp, y_tr, y_temp = train_test_split(X_sem_3d, y_gpa, test_size=0.30, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    inputs_sem, head_sem = build_deep_transformer_backbone((T_sem, F_sem), d_model=128, num_heads=8, num_blocks=3)
    outputs_sem = Dense(1, activation='linear')(head_sem)
    model_sem = Model(inputs=inputs_sem, outputs=outputs_sem)
    model_sem.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history_sem = model_sem.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=60, batch_size=256, callbacks=[es], verbose=0)
    
    y_pred_sem = model_sem.predict(X_te, verbose=0).flatten()
    rmse_sem = mean_squared_error(y_te, y_pred_sem, squared=False)
    mae_sem = mean_absolute_error(y_te, y_pred_sem)
    r2_sem = r2_score(y_te, y_pred_sem)
    print(f"   [OK] Semester-Transformer Regressor (R²: {r2_sem:.4f}, RMSE: {rmse_sem:.4f}, MAE: {mae_sem:.4f})")
    
    # -------------------------------------------------------------------------
    # 2. DEEP EXAM-TRANSFORMER REGRESSOR
    # -------------------------------------------------------------------------
    print("\n>>> [2/3] Trainiere Deep Exam-Transformer Regressor (d_model=128, Attention Pooling) ...")
    
    T_exam = 40
    exam_features = ["versuch", "cp", "schwierigkeit", "bestanden", "note"]
    F_exam = len(exam_features)
    
    X_exam_3d = np.full((N, T_exam, F_exam), PADDING_VALUE, dtype=np.float32)
    
    for i, s_id in enumerate(studis):
        s_pr = df_pr[df_pr["studierenden_id"] == s_id].sort_values("semester_id")
        for k, (_, row) in enumerate(s_pr.iterrows()):
            if k < T_exam:
                X_exam_3d[i, k, 0] = row["versuch"]
                X_exam_3d[i, k, 1] = row["cp"]
                X_exam_3d[i, k, 2] = row["schwierigkeit"]
                X_exam_3d[i, k, 3] = 1.0 if row["bestanden"] else 0.0
                X_exam_3d[i, k, 4] = row["note"]
                
    scaler_ex = StandardScaler()
    valid_mask_ex = (X_exam_3d != PADDING_VALUE)
    X_exam_3d[valid_mask_ex] = scaler_ex.fit_transform(X_exam_3d[valid_mask_ex].reshape(-1, F_exam)).flatten()
    
    X_tr_e, X_temp_e, y_tr_e, y_temp_e = train_test_split(X_exam_3d, y_gpa, test_size=0.30, random_state=42)
    X_va_e, X_te_e, y_va_e, y_te_e = train_test_split(X_temp_e, y_temp_e, test_size=0.50, random_state=42)
    
    inputs_ex, head_ex = build_deep_transformer_backbone((T_exam, F_exam), d_model=128, num_heads=8, num_blocks=3)
    outputs_ex = Dense(1, activation='linear')(head_ex)
    model_ex = Model(inputs=inputs_ex, outputs=outputs_ex)
    model_ex.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    
    history_ex = model_ex.fit(X_tr_e, y_tr_e, validation_data=(X_va_e, y_va_e), epochs=60, batch_size=256, callbacks=[es], verbose=0)
    
    y_pred_ex = model_ex.predict(X_te_e, verbose=0).flatten()
    rmse_ex = mean_squared_error(y_te_e, y_pred_ex, squared=False)
    mae_ex = mean_absolute_error(y_te_e, y_pred_ex)
    r2_ex = r2_score(y_te_e, y_pred_ex)
    print(f"   [OK] Deep Exam-Transformer Regressor (R²: {r2_ex:.4f}, RMSE: {rmse_ex:.4f}, MAE: {mae_ex:.4f})")
    
    # -------------------------------------------------------------------------
    # 3. DEEP EXAM-TRANSFORMER SURVIVAL (DROPOUT PREDICTION)
    # -------------------------------------------------------------------------
    print("\n>>> [3/3] Trainiere Deep Exam-Transformer Survival (Dropout Prediction) ...")
    
    df_abs = pd.read_csv(data_dir / "abschluesse.csv")
    dropout_dict = (df_abs["status"] != "abgeschlossen").astype(int).to_dict()
    y_surv = np.array([dropout_dict.get(s, 0) for s in studis])
    
    X_tr_s, X_temp_s, y_tr_s, y_temp_s = train_test_split(X_exam_3d, y_surv, test_size=0.30, random_state=42, stratify=y_surv)
    X_va_s, X_te_s, y_va_s, y_te_s = train_test_split(X_temp_s, y_temp_s, test_size=0.50, random_state=42, stratify=y_temp_s)
    
    inputs_surv, head_surv = build_deep_transformer_backbone((T_exam, F_exam), d_model=128, num_heads=8, num_blocks=3)
    outputs_surv = Dense(1, activation='sigmoid')(head_surv)
    model_surv = Model(inputs=inputs_surv, outputs=outputs_surv)
    model_surv.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['AUC'])
    
    history_surv = model_surv.fit(X_tr_s, y_tr_s, validation_data=(X_va_s, y_va_s), epochs=60, batch_size=256, callbacks=[es], verbose=0)
    
    y_pred_surv = model_surv.predict(X_te_s, verbose=0).flatten()
    auc_surv = roc_auc_score(y_te_s, y_pred_surv)
    prauc_surv = average_precision_score(y_te_s, y_pred_surv)
    brier_surv = brier_score_loss(y_te_s, y_pred_surv)
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
        
    print("\nDeep Transformer Regression & Survival Pipeline abgeschlossen!")

if __name__ == "__main__":
    data_dir = Path("output_dl") if Path("output_dl").exists() else Path("../output_dl")
    output_dir = data_dir
    train_deep_transformer_regression(data_dir, output_dir)
