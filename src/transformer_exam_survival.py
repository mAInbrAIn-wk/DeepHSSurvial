"""
Exam-Level Causal Transformer Survival Model (DTL Hazard auf Prüfungsebene)
=============================================================================
Transformer-Modell mit Kausalem Attention-Masking auf Ebene der einzelnen Prüfungen.
Verfolgt die Trajektorie von Prüfung zu Prüfung und berechnet die Hazard Rate h(k).
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
    precision_score, recall_score, f1_score, accuracy_score, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, TimeDistributed, Masking
import tensorflow.keras.backend as K

from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, plot_confusion_matrix
from recurrent_exam_survival import build_recurrent_exam_dataset, masked_binary_crossentropy, PADDING_VALUE

def build_exam_causal_transformer(max_exams: int, num_features: int, d_model: int = 64, num_heads: int = 4):
    inputs = Input(shape=(max_exams, num_features))
    
    # Linear projection to embedding dim
    x = Dense(d_model)(inputs)
    x = LayerNormalization()(x)
    
    # Causal Masking (Lower Triangular Matrix)
    causal_mask = tf.linalg.band_part(tf.ones((max_exams, max_exams)), -1, 0)
    causal_mask = tf.cast(causal_mask, tf.bool)
    
    # Transformer Encoder Block 1
    attn_out1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(
        x, x, attention_mask=causal_mask
    )
    x1 = LayerNormalization()(x + Dropout(0.1)(attn_out1))
    ff_out1 = Dense(d_model, activation='relu')(x1)
    x1 = LayerNormalization()(x1 + Dropout(0.1)(ff_out1))
    
    # Transformer Encoder Block 2
    attn_out2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(
        x1, x1, attention_mask=causal_mask
    )
    x2 = LayerNormalization()(x1 + Dropout(0.1)(attn_out2))
    ff_out2 = Dense(d_model, activation='relu')(x2)
    x2 = LayerNormalization()(x2 + Dropout(0.1)(ff_out2))
    
    # Output Head per Zeitschritt (Hazard Prediction h_k)
    time_dense = TimeDistributed(Dense(32, activation='relu'))(x2)
    time_dense = TimeDistributed(LayerNormalization())(time_dense)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(time_dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss=masked_binary_crossentropy)
    return model

def main():
    print("\n==========================================================================")
    print("   EXAM-LEVEL CAUSAL TRANSFORMER SURVIVAL MODEL")
    print("==========================================================================")
    
    output_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    studis, X_seq, y_seq, studi_events = build_recurrent_exam_dataset(output_dir, max_exams=50)
    N, K_max, F = X_seq.shape
    
    # 3-Wege Split (70% Train, 15% Val, 15% Test) stratifiziert nach Studenten-Event
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
    
    print(f"\nSequence Split: {len(train_idx)} Train, {len(val_idx)} Val, {len(test_idx)} Test-Studierende")
    
    tf.random.set_seed(42)
    model = build_exam_causal_transformer(K_max, F)
    model.summary()
    
    print("Trainiere Causal Exam-Transformer Survival Modell ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        verbose=1
    )
    
    y_test_pred = model.predict(X_test, verbose=0)
    
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
    print("   ERGEBNISSE EXAM-LEVEL CAUSAL TRANSFORMER SURVIVAL MODEL")
    print("==========================================================================")
    print(f"  ROC-AUC (Global Ranking)         : {auc_score:.4f}")
    print(f"  PR-AUC / Average Precision (Gold): {pr_auc:.4f}")
    print(f"  Brier Score (Kalibrierung)       : {brier:.4f}")
    print(f"  Accuracy  (Top 5% Threshold)    : {acc:.4f}")
    print(f"  Precision (Top 5% Threshold)    : {prec:.4f}")
    print(f"  Recall    (Top 5% Threshold)    : {rec:.4f}")
    print(f"  F1-Score  (Top 5% Threshold)    : {f1:.4f}")
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
    save_metrics("transformer_exam_survival", metrics_dict, output_dir, report_str=rep_str)
    save_keras_model(model, "transformer_exam_survival", output_dir)
    plot_learning_curve(history.history, "transformer_exam_survival", output_dir, metric_name='loss')
    plot_roc_curve(y_true_flat, y_pred_flat, "transformer_exam_survival", output_dir)
    plot_pr_curve(y_true_flat, y_pred_flat, "transformer_exam_survival", output_dir)
    plot_confusion_matrix(y_true_flat, y_pred_bin, "transformer_exam_survival", output_dir)

if __name__ == '__main__':
    main()
