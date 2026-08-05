"""
Zeitreihen-Analyse: Semester-Transformer Regressor (Abschlussnoten-Vorhersage)
=============================================================================
Transformer-Architektur auf Semester-Ebene zur Vorhersage der finalen Abschlussnote.
Verwendet Multi-Head Self-Attention, Masking und LayerNormalization.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Masking
from tensorflow.keras.callbacks import EarlyStopping

from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_parity_plot
from timeseries_semester import create_semester_timeseries_dataset, PADDING_VALUE

def build_semester_transformer(max_semesters: int, num_features: int, d_model: int = 64, num_heads: int = 4):
    inputs = Input(shape=(max_semesters, num_features))
    
    # Masking Layer
    masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)
    
    # Linear projection to embedding dim d_model
    x = Dense(d_model)(masked_inputs)
    x = LayerNormalization()(x)
    
    # Transformer Encoder Block 1
    attn_out1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x1 = LayerNormalization()(x + Dropout(0.1)(attn_out1))
    ff_out1 = Dense(d_model, activation='relu')(x1)
    x1 = LayerNormalization()(x1 + Dropout(0.1)(ff_out1))
    
    # Transformer Encoder Block 2
    attn_out2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x1, x1)
    x2 = LayerNormalization()(x1 + Dropout(0.1)(attn_out2))
    ff_out2 = Dense(d_model, activation='relu')(x2)
    x2 = LayerNormalization()(x2 + Dropout(0.1)(ff_out2))
    
    # Aggregation über die Zeitdimension
    pooled = GlobalAveragePooling1D()(x2)
    
    # Output Head
    dense_out = Dense(32, activation='relu')(pooled)
    dense_out = LayerNormalization()(dense_out)
    dense_out = Dropout(0.2)(dense_out)
    outputs = Dense(1, activation='linear')(dense_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("=" * 70)
    print("ZEITREIHEN-TRANSFORMER REGRESSOR (SEMESTER-EBENE)")
    print("=" * 70)
    
    output_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    X, y, max_sem, n_features = create_semester_timeseries_dataset(output_dir)
    
    # 3-Wege Split (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"\nDatensatz-Aufteilung:")
    print(f"  - Training Set:   {X_train.shape[0]} Sequenzen")
    print(f"  - Validation Set: {X_val.shape[0]} Sequenzen")
    print(f"  - Test Set:       {X_test.shape[0]} Sequenzen")
    
    num_seq_feats = 7
    stat_start = num_seq_feats
    
    valid_mask_train = X_train[:, :, 0] != PADDING_VALUE
    train_seq_valid = X_train[valid_mask_train][:, :num_seq_feats]
    
    scaler_seq = StandardScaler()
    scaler_seq.fit(train_seq_valid)
    
    student_mask_train = valid_mask_train.any(axis=1)
    first_valid_idx = np.argmax(valid_mask_train[student_mask_train], axis=1)
    train_stat_valid = X_train[student_mask_train, first_valid_idx, stat_start:stat_start+2]
    
    scaler_stat = StandardScaler()
    scaler_stat.fit(train_stat_valid)
    
    for X_split in [X_train, X_val, X_test]:
        valid_mask = X_split[:, :, 0] != PADDING_VALUE
        X_split[valid_mask, :num_seq_feats] = scaler_seq.transform(X_split[valid_mask, :num_seq_feats])
        X_split[valid_mask, stat_start:stat_start+2] = scaler_stat.transform(X_split[valid_mask, stat_start:stat_start+2])
    
    print("\nTrainiere Keras Semester Transformer Regressor ...")
    model = build_semester_transformer(max_sem, n_features)
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )
    
    test_preds = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    
    print("\n" + "=" * 70)
    print("ERGEBNISSE SEMESTER TRANSFORMER REGRESSOR (TEST-SET)")
    print("=" * 70)
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print("=" * 70)
    
    metrics_dict = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
    save_metrics("timeseries_semester_transformer", metrics_dict, output_dir)
    save_keras_model(model, "timeseries_semester_transformer", output_dir)
    plot_learning_curve(history.history, "timeseries_semester_transformer", output_dir, metric_name='mae')
    plot_parity_plot(y_test, test_preds, "timeseries_semester_transformer", output_dir)

if __name__ == '__main__':
    main()
