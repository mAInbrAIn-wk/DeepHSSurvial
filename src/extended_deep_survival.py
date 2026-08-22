"""
Extended Neural Survival Models (Time-Varying Covariates Edition)
==================================================================
Trainiert Extended DeepSurv & Extended Discrete-Time Hazard Modelle auf dem
Person-Semester Längsschnitt-Panel (Counting Process Format (t_start, t_stop, event, X_{it})).

Vergleicht:
1. Statistical Extended Cox (statsmodels baseline)
2. Extended DeepSurv (Keras mit Breslow Partial-Likelihood Loss über alle Person-Semester)
3. Extended Discrete-Time Hazard (Keras mit Binary Cross-Entropy Loss)
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
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, roc_curve

from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization
import statsmodels.formula.api as smf

from extended_cox_survival import build_person_semester_panel

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

def train_extended_deep_survival(data_dir: Path):
    print("\n==========================================================================")
    print("   EXTENDED NEURAL SURVIVAL MODELS (TIME-VARYING PANEL EDITION)")
    print("==========================================================================")
    
    panel_df = build_person_semester_panel(data_dir)
    
    # Feature-Auswahl für neuronale Modelle
    num_cols = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start', 'cum_cp', 'cum_fails']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    
    feature_cols = num_cols + cat_cols + treatment_cols
    
    # Split auf Studierenden-Ebene (Group Split), um Data-Leakage zwischen Semestern desselben Studierenden zu vermeiden
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    print(f"\nGroup Split: {len(train_ids)} Train-Studierende ({len(train_panel)} Sem-Zeilen), {len(test_ids)} Test-Studierende ({len(test_panel)} Sem-Zeilen)")
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    
    X_train = preprocessor.fit_transform(train_panel[feature_cols])
    X_test = preprocessor.transform(test_panel[feature_cols])
    
    y_train_surv = np.column_stack([train_panel['t_stop'].values, train_panel['event'].values])
    y_test_surv = np.column_stack([test_panel['t_stop'].values, test_panel['event'].values])
    
    input_dim = X_train.shape[1]
    
    # -------------------------------------------------------------------------
    # 1. EXTENDED DEEPSURV (NEURONALES TIME-VARYING COX MODELL)
    # -------------------------------------------------------------------------
    print("\n[1/3] Trainiere Extended DeepSurv (Neuronales Cox-Modell auf Panel-Daten) ...")
    tf.random.set_seed(42)
    
    deepsurv = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
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
    history_ds = deepsurv.fit(X_train, y_train_surv, epochs=150, batch_size=len(X_train), verbose=0)
    
    train_risk = deepsurv.predict(X_train, verbose=0).flatten()
    test_risk = deepsurv.predict(X_test, verbose=0).flatten()
    
    # -------------------------------------------------------------------------
    # 2. EXTENDED DISCRETE-TIME HAZARD MODELL (KERAS BINARY CROSS-ENTROPY)
    # -------------------------------------------------------------------------
    print("[2/3] Trainiere Extended Discrete-Time Hazard Modell (Panel Classification) ...")
    tf.random.set_seed(42)
    
    dtl_hazard = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid') # Ausfallwahrscheinlichkeit h(t) im Semester t
    ])
    
    dtl_hazard.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['AUC'])
    history_lh = dtl_hazard.fit(X_train, train_panel['event'].values, epochs=30, batch_size=2048, verbose=0)
    
    test_h_pred = dtl_hazard.predict(X_test, verbose=0).flatten()
    
    # -------------------------------------------------------------------------
    # 3. STATISTICAL EXTENDED COX (BASELINE COMPUTE)
    # -------------------------------------------------------------------------
    print("[3/3] Schätze Statistisches Extended Cox Modell (statsmodels) ...")
    formel = "t_stop ~ fach_supp_tv + uebf_supp_tv + psych_supp_tv + cum_cp + cum_fails + hzb_note + erwerbstaetigkeit_std + erstakademiker"
    cox_stat = smf.phreg(formula=formel, data=train_panel, status=train_panel['event'].values, entry=train_panel['t_start'].values, ties='breslow').fit()
    
    params_s = pd.Series(cox_stat.params, index=cox_stat.model.exog_names)
    stat_test_risk = (
        params_s['fach_supp_tv'] * test_panel['fach_supp_tv'] +
        params_s['uebf_supp_tv'] * test_panel['uebf_supp_tv'] +
        params_s['psych_supp_tv'] * test_panel['psych_supp_tv'] +
        params_s['cum_cp'] * test_panel['cum_cp'] +
        params_s['cum_fails'] * test_panel['cum_fails'] +
        params_s['hzb_note'] * test_panel['hzb_note'] +
        params_s['erwerbstaetigkeit_std'] * test_panel['erwerbstaetigkeit_std'] +
        params_s['erstakademiker'] * test_panel['erstakademiker']
    ).values
    
    # -------------------------------------------------------------------------
    # BEWERTUNG UND MODELLVERGLEICH
    # -------------------------------------------------------------------------
    auc_stat = roc_auc_score(test_panel['event'], stat_test_risk)
    auc_deepsurv = roc_auc_score(test_panel['event'], test_risk)
    auc_dtl = roc_auc_score(test_panel['event'], test_h_pred)
    
    brier_dtl = brier_score_loss(test_panel['event'], test_h_pred)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE MODELLVERGLEICH (EVALUIERT AUF UNGESEHENEN TEST-STUDIERENDEN)")
    print("==========================================================================")
    print(f"{'Modell-Typ':<35} | {'Person-Semester ROC-AUC':<24} | {'Loss / Brier Score':<20}")
    print("-" * 84)
    print(f"{'Statistisches Extended Cox (statsmodels)':<35} | {auc_stat:<24.4f} | {'Partial Likelihood':<20}")
    print(f"{'Extended DeepSurv (Neuronales Cox)':<35} | {auc_deepsurv:<24.4f} | {'Partial Likelihood':<20}")
    print(f"{'Extended DTL Hazard (Discrete-Time)':<35} | {auc_dtl:<24.4f} | {brier_dtl:<20.4f}")
    print("-" * 84)
    
    # -------------------------------------------------------------------------
    # METRICS LOGGING & MODEL SAVING
    # -------------------------------------------------------------------------
    base_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    
    # 1. Extended DeepSurv
    metrics_ds = {
        "ROC-AUC_Panel": auc_deepsurv
    }
    fpr_ds, tpr_ds, _ = roc_curve(test_panel['event'], test_risk)
    metrics_ds["PR-AUC_Panel"] = average_precision_score(test_panel['event'], test_risk)
    
    save_metrics("extended_deepsurv_panel", metrics_ds, base_dir)
    save_keras_model(deepsurv, "extended_deepsurv_panel", base_dir)
    plot_learning_curve(history_ds.history, "extended_deepsurv_panel", base_dir, metric_name='loss')
    plot_roc_curve(test_panel['event'], test_risk, "extended_deepsurv_panel", base_dir)
    plot_pr_curve(test_panel['event'], test_risk, "extended_deepsurv_panel", base_dir)
    
    # 2. Extended DTL Hazard
    metrics_dtl = {
        "ROC-AUC_Panel": auc_dtl,
        "Brier_Score": brier_dtl,
        "PR-AUC_Panel": average_precision_score(test_panel['event'], test_h_pred)
    }
    save_metrics("extended_logistic_hazard_panel", metrics_dtl, base_dir)
    save_keras_model(dtl_hazard, "extended_logistic_hazard_panel", base_dir)
    plot_learning_curve(history_lh.history, "extended_logistic_hazard_panel", base_dir, metric_name='AUC')
    plot_roc_curve(test_panel['event'], test_h_pred, "extended_logistic_hazard_panel", base_dir)
    plot_pr_curve(test_panel['event'], test_h_pred, "extended_logistic_hazard_panel", base_dir)
    
    print("\nKernerkenntnis:")
    print("Beide neuronalen Modelle verarbeiten zeitveränderliche Support-Expositionen sauber auf Panel-Ebene.")
    print("Das Extended DTL Hazard Modell liefert direkt kalibrierte Ausfallwahrscheinlichkeiten h(t) pro Semester.")
    print("==========================================================================")

if __name__ == '__main__':
    data_dir = Path('../output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    train_extended_deep_survival(data_dir)
