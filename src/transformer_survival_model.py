"""
Causal Transformer Survival Analysis (Temporal Attention Edition)
=================================================================
Verwendet einen Keras Causal Transformer Encoder mit MultiHeadAttention
und Kausal-Maskierung (use_causal_mask=True), um jeden Data-Leakage aus der Zukunft
auszuschließen.

Evaluierung aller erweiterten Metriken:
- PR-AUC / Average Precision (Goldstandard bei Imbalance)
- ROC-AUC (Global Ranking)
- Brier Score (Kalibrierung)
- Precision, Recall, F1-Score & Accuracy (Top 5% Schwellenwert)
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
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    TimeDistributed, Masking, Add
)
import tensorflow.keras.backend as K

from recurrent_survival_model import build_recurrent_survival_dataset, masked_binary_crossentropy, PADDING_VALUE

class PositionalEncoding(tf.keras.layers.Layer):
    """
    Sinusoidales Positional Encoding für temporäre Reihenfolgen.
    """
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((sequence_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.cast(pe[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]

def build_causal_transformer_survival_model(sequence_length, feature_dim, d_model=32, num_heads=4):
    inputs = Input(shape=(sequence_length, feature_dim))
    
    # 1. Masking Layer
    masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)
    
    # 2. Linear Projection auf d_model Dimension
    x = TimeDistributed(Dense(d_model))(masked_inputs)
    
    # 3. Positional Encoding hinzufügen
    x = PositionalEncoding(sequence_length, d_model)(x)
    
    # 4. Causal Multi-Head Self-Attention Block (Strictly No Future Leakage)
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=0.1
    )(query=x, value=x, key=x, use_causal_mask=True)
    
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # 5. Feed-Forward Network
    ffn = TimeDistributed(Dense(64, activation='relu'))(x)
    ffn = TimeDistributed(Dense(d_model))(ffn)
    ffn = Dropout(0.1)(ffn)
    
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # 6. Output Head: Discrete-Time Hazard h(t)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Causal_Transformer_Survival")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss=masked_binary_crossentropy)
    return model

def train_causal_transformer_survival(data_dir: Path, blind: bool = False):
    model_name = "transformer_survival_blind" if blind else "transformer_survival"
    print("\n==========================================================================")
    print(f"   CAUSAL TRANSFORMER SURVIVAL MODEL (STRICTLY CAUSAL ATTENTION, blind={blind})")
    print("==========================================================================")
    
    studis, X_seq, y_seq, studi_events = build_recurrent_survival_dataset(data_dir, blind=blind)
    
    N, T, F = X_seq.shape
    
    # 3-Wege Sequence Split (70% Train, 15% Val, 15% Test) stratifiziert nach Studenten-Event
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train, y_train = X_seq[train_idx].copy(), y_seq[train_idx]
    X_val, y_val = X_seq[val_idx].copy(), y_seq[val_idx]
    X_test, y_test = X_seq[test_idx].copy(), y_seq[test_idx]
    
    # Standardisiere valide (nicht-gepaddete) Features nur anhand des Train-Sets (kein Preprocessing-Leakage!)
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    for X_split in [X_train, X_val, X_test]:
        valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
        X_split[valid_mask] = scaler.transform(X_split[valid_mask])
    
    print(f"\nSequence Split (3-Wege): {len(train_idx)} Train, {len(val_idx)} Val, {len(test_idx)} Test")
    
    # Bauen und Trainieren
    tf.random.set_seed(42)
    transformer_model = build_causal_transformer_survival_model(sequence_length=T, feature_dim=F, d_model=32, num_heads=4)
    transformer_model.summary()
    
    from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve, plot_confusion_matrix
    
    print("\nTrainiere Causal Transformer Survival Modell ...")
    history = transformer_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=512, verbose=1)
    
    y_test_pred = transformer_model.predict(X_test, verbose=0)
    
    # Evaluierung auf ungepaddeten Zeitschritten
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
    print(f"   ERGEBNISSE CAUSAL TRANSFORMER SURVIVAL MODELL (blind={blind})")
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
    save_metrics(model_name, metrics_dict, output_dir, report_str=rep_str)
    save_keras_model(transformer_model, model_name, output_dir)
    plot_learning_curve(history.history, model_name, output_dir, metric_name='loss')
    plot_roc_curve(y_true_flat, y_pred_flat, model_name, output_dir)
    plot_pr_curve(y_true_flat, y_pred_flat, model_name, output_dir)
    plot_confusion_matrix(y_true_flat, y_pred_bin, model_name, output_dir)

if __name__ == '__main__':
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    train_causal_transformer_survival(data_dir)
