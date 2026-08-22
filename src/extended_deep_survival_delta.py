"""
Extended Neural Survival Models (Delta & Active Support Edition)
================================================================
Trainiert Extended DeepSurv Delta & Extended Discrete-Time Hazard Delta Modelle
auf dem Person-Semester Längsschnitt-Panel mit semester-lokalem Treatment
und dynamischen Leistungs-Deltas.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization
import statsmodels.formula.api as smf

from extended_cox_delta import build_delta_panel

def breslow_cox_loss(y_true, y_pred):
    """
    Keras Loss für Partial Likelihood bei intervallgezensierten Person-Semestern.
    y_true[:, 0]: t_stop
    y_true[:, 1]: event
    y_pred[:, 0]: log-risk (r_i)
    """
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]
    
    sort_idx = tf.argsort(time, direction='DESCENDING')
    risk_sorted = tf.gather(risk, sort_idx)
    event_sorted = tf.gather(event, sort_idx)
    
    exp_risk = tf.exp(risk_sorted)
    cum_exp_risk = tf.cumsum(exp_risk)
    log_risk = risk_sorted - tf.math.log(cum_exp_risk + 1e-7)
    
    uncensored_loss = -tf.reduce_sum(log_risk * event_sorted)
    num_events = tf.reduce_sum(event_sorted) + 1e-7
    return uncensored_loss / num_events

def train_extended_deep_survival_delta(data_dir: Path):
    print("\n==========================================================================")
    print("   EXTENDED NEURAL SURVIVAL MODELS (DELTA & ACTIVE SUPPORT PANEL)")
    print("==========================================================================")
    
    panel_df = build_delta_panel(data_dir)
    
    num_cols = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start', 'fails_prev', 'delta_cp_prev', 'cp_rueckstand']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    
    feature_cols = num_cols + cat_cols + treatment_cols
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    val_panel = panel_df[panel_df['studierenden_id'].isin(val_ids)].copy()
    test_panel = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    print(f"\nGroup 3-Way Split: {len(train_ids)} Train ({len(train_panel)} Zeilen), {len(val_ids)} Val ({len(val_panel)} Zeilen), {len(test_ids)} Test ({len(test_panel)} Zeilen)")
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    
    X_train = preprocessor.fit_transform(train_panel[feature_cols])
    X_val = preprocessor.transform(val_panel[feature_cols])
    X_test = preprocessor.transform(test_panel[feature_cols])
    
    y_train_surv = np.column_stack([train_panel['t_stop'].values, train_panel['event'].values])
    y_val_surv = np.column_stack([val_panel['t_stop'].values, val_panel['event'].values])
    y_test_surv = np.column_stack([test_panel['t_stop'].values, test_panel['event'].values])
    
    input_dim = X_train.shape[1]
    
    # -------------------------------------------------------------------------
    # 1. EXTENDED DEEPSURV DELTA (NEURONALES COX MODELL)
    # -------------------------------------------------------------------------
    print("\n[1/2] Trainiere Extended DeepSurv Delta ...")
    tf.random.set_seed(42)
    
    deepsurv = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.15),
        Dense(64, activation='relu'),
        LayerNormalization(),
        Dropout(0.15),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dropout(0.15),
        Dense(16, activation='relu'),
        LayerNormalization(),
        Dense(1, activation='linear', use_bias=False)
    ])
    
    deepsurv.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=breslow_cox_loss)
    es_ds = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    history_ds = deepsurv.fit(
        X_train, y_train_surv,
        validation_data=(X_val, y_val_surv),
        epochs=300, batch_size=len(X_train),
        callbacks=[es_ds], verbose=0
    )
    
    train_risk = deepsurv.predict(X_train, verbose=0).flatten()
    test_risk = deepsurv.predict(X_test, verbose=0).flatten()
    
    # -------------------------------------------------------------------------
    # 2. EXTENDED DISCRETE-TIME HAZARD DELTA (LOGISTIC HAZARD)
    # -------------------------------------------------------------------------
    print("\n[2/2] Trainiere Extended Logistic Hazard Delta ...")
    tf.random.set_seed(42)
    
    dtl_hazard = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    dtl_hazard.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['AUC'])
    es_lh = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history_lh = dtl_hazard.fit(
        X_train, train_panel['event'].values,
        validation_data=(X_val, val_panel['event'].values),
        epochs=60, batch_size=2048,
        callbacks=[es_lh], verbose=0
    )
    
    test_h_pred = dtl_hazard.predict(X_test, verbose=0).flatten()
    
    # -------------------------------------------------------------------------
    # BEWERTUNG
    # -------------------------------------------------------------------------
    auc_deepsurv = roc_auc_score(test_panel['event'], test_risk)
    auc_dtl = roc_auc_score(test_panel['event'], test_h_pred)
    brier_dtl = brier_score_loss(test_panel['event'], test_h_pred)
    pr_auc_ds = average_precision_score(test_panel['event'], test_risk)
    pr_auc_dtl = average_precision_score(test_panel['event'], test_h_pred)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE MODELLVERGLEICH DELTA-MODELLE (EVALUIERT AUF TEST-STUDIERENDEN)")
    print("==========================================================================")
    print(f"{'Modell-Typ':<35} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'Brier Score':<12}")
    print("-" * 75)
    print(f"{'Extended DeepSurv Delta':<35} | {auc_deepsurv:<10.4f} | {pr_auc_ds:<10.4f} | {'N/A':<12}")
    print(f"{'Extended Logistic Hazard Delta':<35} | {auc_dtl:<10.4f} | {pr_auc_dtl:<10.4f} | {brier_dtl:<12.4f}")
    print("-" * 75)
    
    # -------------------------------------------------------------------------
    # METRICS LOGGING & MODEL SAVING
    # -------------------------------------------------------------------------
    base_dir = data_dir
    
    # 1. Extended DeepSurv Delta
    metrics_ds = {
        "ROC-AUC_Panel": auc_deepsurv,
        "PR-AUC_Panel": pr_auc_ds
    }
    save_metrics("extended_deepsurv_delta", metrics_ds, base_dir)
    save_keras_model(deepsurv, "extended_deepsurv_delta", base_dir)
    plot_learning_curve(history_ds.history, "extended_deepsurv_delta", base_dir, metric_name='loss')
    plot_roc_curve(test_panel['event'], test_risk, "extended_deepsurv_delta", base_dir)
    plot_pr_curve(test_panel['event'], test_risk, "extended_deepsurv_delta", base_dir)
    
    # 2. Extended Logistic Hazard Delta
    metrics_dtl = {
        "ROC-AUC_Panel": auc_dtl,
        "PR-AUC_Panel": pr_auc_dtl,
        "Brier_Score": brier_dtl
    }
    save_metrics("extended_logistic_hazard_delta", metrics_dtl, base_dir)
    save_keras_model(dtl_hazard, "extended_logistic_hazard_delta", base_dir)
    plot_learning_curve(history_lh.history, "extended_logistic_hazard_delta", base_dir, metric_name='AUC')
    plot_roc_curve(test_panel['event'], test_h_pred, "extended_logistic_hazard_delta", base_dir)
    plot_pr_curve(test_panel['event'], test_h_pred, "extended_logistic_hazard_delta", base_dir)
    
    print("\nTraining und Logging für Delta-Modelle abgeschlossen.")
    print("==========================================================================")

if __name__ == '__main__':
    data_dir = Path('../output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_extended_deep_survival_delta(data_dir)
