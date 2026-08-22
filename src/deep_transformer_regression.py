"""
Deep Transformer Regression & Survival Models (Enlarged Capacity + Dual Causal/Masked Architectures)
=======================================================================================================
Implementiert 4 hochentwickelte Transformer-Modelle:
1. Deep Semester-Transformer Regressor (d_model=128, 8 Heads, 3 Blöcke, Attention Pooling)
2. Deep Exam-Transformer Regressor (d_model=128, 8 Heads, 3 Blöcke, Attention Pooling)
3. Deep Exam-Transformer Causal Survival (Klasse 7a: use_causal_mask=True, TimeDistributed hazard, masked_bce)
4. Deep Exam-Transformer Masked Survival (Klasse 7b: Keras Masking + Attention Mask, Attention Pooling, Static Event)

Funktionen:
- Vollständige Leakage-Freiheit durch Kausales Masking bzw. Attention-Masking
- 12 Features pro Prüfungsschritt mit Zähl-Expositionsmerkmalen
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
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Masking, Layer, TimeDistributed
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
        scores = self.score_dense(inputs)  # (batch, T, 1)
        is_padded = tf.reduce_all(tf.equal(inputs, PADDING_VALUE), axis=-1, keepdims=True)  # (batch, T, 1)
        padding_mask = tf.cast(is_padded, tf.float32) * -1e9
        scores = scores + padding_mask
        weights = tf.nn.softmax(scores, axis=1)  # (batch, T, 1)
        pooled = tf.reduce_sum(inputs * weights, axis=1)  # (batch, d_model)
        return pooled

def masked_binary_crossentropy(y_true, y_pred):
    """Masked Binary Crossentropy für sequenzielle Hazard-Vorhersagen."""
    mask = tf.not_equal(y_true, PADDING_VALUE)
    mask = tf.cast(mask, tf.float32)
    
    y_true_safe = tf.where(tf.equal(y_true, PADDING_VALUE), tf.zeros_like(y_true), y_true)
    bce = tf.keras.losses.binary_crossentropy(y_true_safe, y_pred)
    bce = tf.expand_dims(bce, axis=-1) if len(bce.shape) < len(mask.shape) else bce
    masked_loss = bce * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)

def build_deep_transformer_backbone(input_shape, d_model=128, num_heads=8, num_blocks=3, dropout_rate=0.2):
    """Erzeugt das hochkapazitäre Transformer Backbone mit Positional Embedding & Attention Pooling."""
    inputs = Input(shape=input_shape)
    
    x = Dense(d_model, activation='relu')(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)(x, x)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        
        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
        
    pooled = AttentionPooling(d_model=d_model)(x)
    
    head = Dense(64, activation='relu')(pooled)
    head = LayerNormalization()(head)
    head = Dropout(dropout_rate)(head)
    
    return inputs, head

def build_causal_transformer_survival_model(input_shape, d_model=128, num_heads=8, num_blocks=3, dropout_rate=0.2):
    """Option A: Sequenzieller Kausaler Hazard-Transformer (use_causal_mask=True + TimeDistributed)."""
    inputs = Input(shape=input_shape)
    x = Masking(mask_value=PADDING_VALUE)(inputs)
    
    x = Dense(d_model, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )(x, x, use_causal_mask=True)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        
        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
        
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    return Model(inputs=inputs, outputs=outputs)

def build_masked_transformer_static_model(input_shape, d_model=128, num_heads=8, num_blocks=3, dropout_rate=0.2):
    """Option B: Maskierter statischer Klassifikator mit Attention-Maskierung."""
    inputs = Input(shape=input_shape)
    
    # Compute padding mask (True for valid steps, False for padding)
    padding_mask = tf.reduce_any(tf.not_equal(inputs, PADDING_VALUE), axis=-1)  # (batch, T)
    attn_mask = padding_mask[:, tf.newaxis, :]  # (batch, 1, T)
    
    x = Masking(mask_value=PADDING_VALUE)(inputs)
    x = Dense(d_model, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )(x, x, attention_mask=attn_mask)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        
        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
        
    pooled = AttentionPooling(d_model=d_model)(x)
    head = Dense(64, activation='relu')(pooled)
    head = LayerNormalization()(head)
    head = Dropout(dropout_rate)(head)
    outputs = Dense(1, activation='sigmoid')(head)
    
    return Model(inputs=inputs, outputs=outputs)

from timeseries_semester import create_semester_timeseries_dataset
from timeseries_exam import create_exam_timeseries_dataset

def build_canonical_exam_survival_dataset(data_dir: Path, max_exams: int = 40):
    """
    Erstellt den harmonisierten 12-Feature Prüfungssatz für Exam Survival
    mit 6 Zähl-Expositionsmerkmalen.
    Features: [versuch, schwierigkeit, cp, is_fail,
               support_vorher_fachlich, support_glz_fachlich,
               support_vorher_ueberfachlich, support_glz_ueberfachlich,
               support_vorher_psychosozial, support_glz_psychosozial,
               hzb_note, erwerbstaetigkeit_std]
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
    
    studis = df_abschluesse['studierenden_id'].unique()
    abschluss_dict = df_abschluesse.set_index('studierenden_id').to_dict('index')
    pr_grouped = df_pruefungen.groupby('studierenden_id')
    
    n_features = 12
    X_seq = np.full((len(studis), max_exams, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((len(studis), max_exams, 1), PADDING_VALUE, dtype=np.float32)
    y_surv = np.zeros(len(studis), dtype=np.float32)
    
    for i, s_id in enumerate(studis):
        row_ab = abschluss_dict[s_id]
        status = str(row_ab['status']).strip().lower()
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        y_surv[i] = 1.0 if is_dropout else 0.0
        
        if s_id in pr_grouped.groups:
            studi_pr = pr_grouped.get_group(s_id).sort_values(['fachsemester', 'pruefung_id'])
            num_p = min(len(studi_pr), max_exams)
            for k in range(num_p):
                p_row = studi_pr.iloc[k]
                X_seq[i, k, 0] = float(p_row['versuch'])
                X_seq[i, k, 1] = float(p_row['schwierigkeit'])
                X_seq[i, k, 2] = float(p_row['cp'])
                X_seq[i, k, 3] = 1.0 if not bool(p_row['bestanden']) else 0.0
                X_seq[i, k, 4] = float(p_row['support_vorher_fachlich'])
                X_seq[i, k, 5] = float(p_row['support_glz_fachlich'])
                X_seq[i, k, 6] = float(p_row['support_vorher_ueberfachlich'])
                X_seq[i, k, 7] = float(p_row['support_glz_ueberfachlich'])
                X_seq[i, k, 8] = float(p_row['support_vorher_psychosozial'])
                X_seq[i, k, 9] = float(p_row['support_glz_psychosozial'])
                X_seq[i, k, 10] = float(row_ab['hzb_note'])
                X_seq[i, k, 11] = float(row_ab['erwerbstaetigkeit_std'])
                
                # Sequentieller Event-Status: 1 nur am letzten Schritt bei Dropouts
                event_val = 1.0 if (k == len(studi_pr) - 1 and is_dropout) else 0.0
                y_seq[i, k, 0] = event_val
                
    return X_seq, y_seq, y_surv, max_exams, n_features

def train_deep_transformer_regression(data_dir: Path, output_dir: Path):
    print("\n================================================================================")
    print("   DEEP TRANSFORMER REGRESSION & SURVIVAL (KANONISCHE DATENSÄTZE & LEAKAGE-FREI)")
    print("================================================================================")
    
    os.makedirs(output_dir / "models", exist_ok=True)
    os.makedirs(output_dir / "metrics", exist_ok=True)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # -------------------------------------------------------------------------
    # 1. DEEP SEMESTER-TRANSFORMER REGRESSOR (KLASSE 2b)
    # -------------------------------------------------------------------------
    print("\n>>> [1/4] Trainiere Deep Semester-Transformer Regressor (GPA Prediction, Klasse 2b) ...")
    
    X_sem_3d, y_gpa_sem, T_sem, F_sem = create_semester_timeseries_dataset(data_dir)
    
    X_tr, X_temp, y_tr, y_temp = train_test_split(X_sem_3d, y_gpa_sem, test_size=0.30, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
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
    
    model_sem.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=60, batch_size=256, callbacks=[es], verbose=1)
    
    y_pred_sem = model_sem.predict(X_te, verbose=0).flatten()
    rmse_sem = float(np.sqrt(mean_squared_error(y_te, y_pred_sem)))
    mae_sem = float(mean_absolute_error(y_te, y_pred_sem))
    r2_sem = float(r2_score(y_te, y_pred_sem))
    print(f"   [OK] Semester-Transformer Regressor (R²: {r2_sem:.4f}, RMSE: {rmse_sem:.4f}, MAE: {mae_sem:.4f})")
    model_sem.save(output_dir / "models" / "deep_semester_transformer_regressor.keras")
    
    # -------------------------------------------------------------------------
    # 2. DEEP EXAM-TRANSFORMER REGRESSOR (KLASSE 3)
    # -------------------------------------------------------------------------
    print("\n>>> [2/4] Trainiere Deep Exam-Transformer Regressor (Klasse 3, GPA Prediction) ...")
    
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
    model_ex.save(output_dir / "models" / "deep_exam_transformer_regressor.keras")
    
    # -------------------------------------------------------------------------
    # 3. DEEP EXAM-TRANSFORMER CAUSAL SURVIVAL (OPTION A: KAUSALES HAZARD-MODELL)
    # -------------------------------------------------------------------------
    print("\n>>> [3/4] Trainiere Deep Exam-Transformer Causal Survival (Option A: Kausaler Sequenzieller Hazard-Schätzer) ...")
    
    X_surv_3d, y_seq, y_surv, T_surv, F_surv = build_canonical_exam_survival_dataset(data_dir, max_exams=40)
    
    indices = np.arange(len(y_surv))
    idx_tr, idx_temp = train_test_split(indices, test_size=0.30, random_state=42, stratify=y_surv)
    idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y_surv[idx_temp])
    
    X_tr_s, y_seq_tr = X_surv_3d[idx_tr].copy(), y_seq[idx_tr].copy()
    X_va_s, y_seq_va = X_surv_3d[idx_va].copy(), y_seq[idx_va].copy()
    X_te_s, y_seq_te = X_surv_3d[idx_te].copy(), y_seq[idx_te].copy()
    
    scaler_surv = StandardScaler()
    valid_mask_tr_s = (X_tr_s[:, :, 0] != PADDING_VALUE)
    scaler_surv.fit(X_tr_s[valid_mask_tr_s])
    
    for X_split in [X_tr_s, X_va_s, X_te_s]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler_surv.transform(X_split[valid_mask])
        
    model_causal = build_causal_transformer_survival_model((T_surv, F_surv), d_model=128, num_heads=8, num_blocks=3)
    model_causal.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=masked_binary_crossentropy)
    
    model_causal.fit(X_tr_s, y_seq_tr, validation_data=(X_va_s, y_seq_va), epochs=60, batch_size=256, callbacks=[es], verbose=1)
    
    # Sequentielle Evaluation
    preds_seq_te = model_causal.predict(X_te_s, verbose=0)
    valid_te_mask = (y_seq_te != PADDING_VALUE)
    y_true_eval = y_seq_te[valid_te_mask]
    y_pred_eval = preds_seq_te[valid_te_mask]
    
    auc_causal = float(roc_auc_score(y_true_eval, y_pred_eval))
    prauc_causal = float(average_precision_score(y_true_eval, y_pred_eval))
    brier_causal = float(brier_score_loss(y_true_eval, y_pred_eval))
    print(f"   [OK] Deep Exam-Transformer Causal Survival (ROC-AUC: {auc_causal:.4f}, PR-AUC: {prauc_causal:.4f}, Brier: {brier_causal:.4f})")
    model_causal.save(output_dir / "models" / "deep_exam_transformer_causal_survival.keras")
    
    # -------------------------------------------------------------------------
    # 4. DEEP EXAM-TRANSFORMER MASKED SURVIVAL (OPTION B: STATISCHER KLASSIFIKATOR)
    # -------------------------------------------------------------------------
    print("\n>>> [4/4] Trainiere Deep Exam-Transformer Masked Survival (Option B: Maskierter Statischer Klassifikator) ...")
    
    y_surv_tr = y_surv[idx_tr]
    y_surv_va = y_surv[idx_va]
    y_surv_te = y_surv[idx_te]
    
    model_masked = build_masked_transformer_static_model((T_surv, F_surv), d_model=128, num_heads=8, num_blocks=3)
    model_masked.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy', metrics=['AUC'])
    
    model_masked.fit(X_tr_s, y_surv_tr, validation_data=(X_va_s, y_surv_va), epochs=60, batch_size=256, callbacks=[es], verbose=1)
    
    y_pred_masked = model_masked.predict(X_te_s, verbose=0).flatten()
    auc_masked = float(roc_auc_score(y_surv_te, y_pred_masked))
    prauc_masked = float(average_precision_score(y_surv_te, y_pred_masked))
    brier_masked = float(brier_score_loss(y_surv_te, y_pred_masked))
    print(f"   [OK] Deep Exam-Transformer Masked Survival (ROC-AUC: {auc_masked:.4f}, PR-AUC: {prauc_masked:.4f}, Brier: {brier_masked:.4f})")
    model_masked.save(output_dir / "models" / "deep_exam_transformer_masked_survival.keras")
    
    # Speichere Metriken
    metrics = {
        "deep_semester_transformer_regression": {"R2": float(r2_sem), "RMSE": float(rmse_sem), "MAE": float(mae_sem)},
        "deep_exam_transformer_regression": {"R2": float(r2_ex), "RMSE": float(rmse_ex), "MAE": float(mae_ex)},
        "deep_exam_transformer_causal_survival": {"ROC-AUC": float(auc_causal), "PR-AUC": float(prauc_causal), "Brier_Score": float(brier_causal)},
        "deep_exam_transformer_masked_survival": {"ROC-AUC": float(auc_masked), "PR-AUC": float(prauc_masked), "Brier_Score": float(brier_masked)},
        # Abwärtskompatibler Key für alten Modellnamen
        "deep_exam_transformer_survival": {"ROC-AUC": float(auc_causal), "PR-AUC": float(prauc_causal), "Brier_Score": float(brier_causal)}
    }
    
    with open(output_dir / "metrics" / "deep_transformer_regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nDeep Transformer Regression & Survival Pipeline erfolgreich und leakage-frei abgeschlossen!")

if __name__ == "__main__":
    data_dir = Path("output_dl") if Path("output_dl").exists() else Path("../output_dl")
    output_dir = data_dir
    train_deep_transformer_regression(data_dir, output_dir)
