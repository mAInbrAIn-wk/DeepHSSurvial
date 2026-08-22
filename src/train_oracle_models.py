"""
Trainiert Orakel-Modelle (DeepSurv & Logistic Hazard), die Zugriff auf die
versteckten (hidden) Variablen aus dem Simulator haben, um den theoretischen
Prognose-Gewinn (ROC-AUC Lift) zu evaluieren.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from extended_cox_delta import build_delta_panel
from extended_deep_survival_delta import breslow_cox_loss

def train_oracle_models(data_dir: Path):
    print("\n==========================================================================")
    print("   ORACLE MODELS (TRAINING MIT HIDDEN VARIABLES)")
    print("==========================================================================")
    
    panel_df = build_delta_panel(data_dir)
    
    # Standard Features
    num_cols_base = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start', 'fails_prev', 'delta_cp_prev', 'cp_rueckstand']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    
    # Orakel Features
    oracle_cols = ['hidden_motivation_prev', 'hidden_soziale_integration_prev', 'hidden_erwartete_note_prev']
    
    # Feature Sets
    features_base = num_cols_base + cat_cols + treatment_cols
    features_oracle = features_base + oracle_cols
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    # Preprocessor für Oracle (Base + Oracle Cols)
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_base + oracle_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    
    X_train_oracle = preprocessor.fit_transform(train_panel[features_oracle])
    X_test_oracle = preprocessor.transform(test_panel[features_oracle])
    
    # Preprocessor für Base
    preprocessor_base = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_base),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    
    X_train_base = preprocessor_base.fit_transform(train_panel[features_base])
    X_test_base = preprocessor_base.transform(test_panel[features_base])
    
    y_train_surv = np.column_stack([train_panel['t_stop'].values, train_panel['event'].values])
    y_test_surv = np.column_stack([test_panel['t_stop'].values, test_panel['event'].values])
    y_train_event = train_panel['event'].values
    y_test_event = test_panel['event'].values
    
    def build_logistic_hazard(input_dim):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy')
        return model
        
    def build_deepsurv(input_dim):
        inputs = Input(shape=(input_dim,))
        x = Dense(32, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        x = BatchNormalization()(x)
        risk = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=risk)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=breslow_cox_loss)
        return model

    tf.random.set_seed(42)
    
    print("Trainiere Baseline Logistic Hazard...")
    lh_base = build_logistic_hazard(X_train_base.shape[1])
    lh_base.fit(X_train_base, y_train_event, epochs=30, batch_size=2048, verbose=0)
    
    print("Trainiere Oracle Logistic Hazard...")
    lh_oracle = build_logistic_hazard(X_train_oracle.shape[1])
    lh_oracle.fit(X_train_oracle, y_train_event, epochs=30, batch_size=2048, verbose=0)
    
    print("Trainiere Baseline DeepSurv...")
    ds_base = build_deepsurv(X_train_base.shape[1])
    ds_base.fit(X_train_base, y_train_surv, epochs=30, batch_size=2048, verbose=0)
    
    print("Trainiere Oracle DeepSurv...")
    ds_oracle = build_deepsurv(X_train_oracle.shape[1])
    ds_oracle.fit(X_train_oracle, y_train_surv, epochs=30, batch_size=2048, verbose=0)
    
    # Evaluierung
    pred_lh_base = lh_base.predict(X_test_base, verbose=0).flatten()
    pred_lh_oracle = lh_oracle.predict(X_test_oracle, verbose=0).flatten()
    pred_ds_base = ds_base.predict(X_test_base, verbose=0).flatten()
    pred_ds_oracle = ds_oracle.predict(X_test_oracle, verbose=0).flatten()
    
    auc_lh_base = roc_auc_score(y_test_event, pred_lh_base)
    auc_lh_oracle = roc_auc_score(y_test_event, pred_lh_oracle)
    auc_ds_base = roc_auc_score(y_test_event, pred_ds_base)
    auc_ds_oracle = roc_auc_score(y_test_event, pred_ds_oracle)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE ORAKEL-MODELLE (ROC-AUC LIFT DURCH HIDDEN VARIABLES)")
    print("==========================================================================")
    print(f"{'Modell-Typ':<30} | {'Baseline AUC':<12} | {'Oracle AUC':<12} | {'Lift':<10}")
    print("-" * 75)
    print(f"{'Logistic Hazard Delta':<30} | {auc_lh_base:<12.4f} | {auc_lh_oracle:<12.4f} | {+(auc_lh_oracle - auc_lh_base):<10.4f}")
    print(f"{'DeepSurv Delta':<30} | {auc_ds_base:<12.4f} | {auc_ds_oracle:<12.4f} | {+(auc_ds_oracle - auc_ds_base):<10.4f}")
    print("==========================================================================")
    
    # Speichere Modelle
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    lh_oracle.save(models_dir / "oracle_logistic_hazard.keras")
    ds_oracle.save(models_dir / "oracle_deepsurv.keras")
    lh_base.save(models_dir / "oracle_base_logistic_hazard.keras")
    ds_base.save(models_dir / "oracle_base_deepsurv.keras")
    print(f"Orakel-Modelle gespeichert in: {models_dir}")
    
    import json
    out_file = data_dir / "metrics" / "oracle_lift.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "Logistic_Hazard": {"base": auc_lh_base, "oracle": auc_lh_oracle, "lift": auc_lh_oracle - auc_lh_base},
            "DeepSurv": {"base": auc_ds_base, "oracle": auc_ds_oracle, "lift": auc_ds_oracle - auc_ds_base}
        }, f, indent=4)
    print(f"Orakel-Lift Metriken gespeichert in: {out_file}")
        
if __name__ == "__main__":
    possible_dirs = [Path("src/output_dl"), Path("output_dl"), Path("../output_dl")]
    target_dir = None
    for p in possible_dirs:
        if (p / "agg_abschluesse.csv").exists():
            target_dir = p
            break
    if target_dir is None:
        target_dir = Path("output_dl")
    train_oracle_models(target_dir)
