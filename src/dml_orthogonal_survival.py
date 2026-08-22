"""
Double / Debiased Machine Learning (DML) Orthogonalized Survival Model
========================================================================
Implementiert Chernozhukovs DML-Orthogonalisierung zur vollständigen Elimination
des reaktiven Confounding-Bias in neuronalen Survival-Panels.

Stufe 1: Propensity Model P(A_t = 1 | W_t) -> Treatment-Residuum A_tilde_t
Stufe 2: Neuronales DML Hazard-Modell auf orthogonalisierten Residuen A_tilde_t
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from extended_cox_delta import build_delta_panel
from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_roc_curve, plot_pr_curve

def train_dml_orthogonal_survival(data_dir: Path):
    print("\n==========================================================================")
    print("   DOUBLE MACHINE LEARNING (DML) ORTHOGONALIZED SURVIVAL MODEL")
    print("==========================================================================")
    
    panel_df = build_delta_panel(data_dir)
    
    confounder_num = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start', 'fails_prev', 'delta_cp_prev', 'cp_rueckstand']
    confounder_cat = ['stg_name', 'erstakademiker']
    confounder_cols = confounder_num + confounder_cat
    
    treatment_cols = ['fach_supp_count', 'uebf_supp_count', 'psych_supp_count']
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, temp_ids = train_test_split(unique_studis, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    val_panel   = panel_df[panel_df['studierenden_id'].isin(val_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    print(f"\nGroup 3-Way Split: {len(train_ids)} Train ({len(train_panel)} Zeilen), {len(val_ids)} Val ({len(val_panel)} Zeilen), {len(test_ids)} Test ({len(test_panel)} Zeilen)")
    
    confounder_prep = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), confounder_num),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), confounder_cat)
    ])
    
    W_train = confounder_prep.fit_transform(train_panel[confounder_cols])
    W_val   = confounder_prep.transform(val_panel[confounder_cols])
    W_test  = confounder_prep.transform(test_panel[confounder_cols])
    
    # -------------------------------------------------------------------------
    # STUFE 1: TREATMENT CONDITIONAL EXPECTATION & RESIDUALS (ORTHOGONALISIERUNG)
    # -------------------------------------------------------------------------
    print("\n[Stufe 1] Schätze Treatment-Modelle E[A_t | W_t] (Ridge Regression für Zählvariablen) ...")
    
    residuals_train = []
    residuals_val = []
    residuals_test = []
    a_hat_test_list = []
    
    for supp_col in treatment_cols:
        y_treat_train = train_panel[supp_col].values.astype(float)
        y_treat_val   = val_panel[supp_col].values.astype(float)
        y_treat_test  = test_panel[supp_col].values.astype(float)
        
        reg_model = Ridge(alpha=1.0)
        reg_model.fit(W_train, y_treat_train)
        
        a_hat_train = reg_model.predict(W_train)
        a_hat_val   = reg_model.predict(W_val)
        a_hat_test  = reg_model.predict(W_test)
        
        # Orthogonalisiertes Treatment-Residuum A_tilde = A - E[A|W]
        res_train = y_treat_train - a_hat_train
        res_val   = y_treat_val - a_hat_val
        res_test  = y_treat_test - a_hat_test
        
        residuals_train.append(res_train)
        residuals_val.append(res_val)
        residuals_test.append(res_test)
        a_hat_test_list.append(a_hat_test)
        
    A_tilde_train = np.column_stack(residuals_train)
    A_tilde_val   = np.column_stack(residuals_val)
    A_tilde_test  = np.column_stack(residuals_test)
    A_hat_test    = np.column_stack(a_hat_test_list)
    
    X_train_dml = np.column_stack([W_train, A_tilde_train])
    X_val_dml   = np.column_stack([W_val, A_tilde_val])
    X_test_dml  = np.column_stack([W_test, A_tilde_test])
    
    y_train = train_panel['event'].values
    y_val   = val_panel['event'].values
    y_test  = test_panel['event'].values
    
    input_dim = X_train_dml.shape[1]
    
    # -------------------------------------------------------------------------
    # STUFE 2: NEURONALES DML HAZARD-MODELL
    # -------------------------------------------------------------------------
    print("\n[Stufe 2] Trainiere neuronales DML Hazard-Modell auf orthogonalisierten Residuen ...")
    tf.random.set_seed(42)
    
    dml_model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.20),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.10),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    dml_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    es_dml = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history_dml = dml_model.fit(
        X_train_dml, y_train,
        validation_data=(X_val_dml, y_val),
        epochs=80,
        batch_size=512,
        callbacks=[es_dml],
        verbose=1
    )
    
    # Evaluation
    test_p_pred = dml_model.predict(X_test_dml, verbose=0).flatten()
    auc_dml   = float(roc_auc_score(y_test, test_p_pred))
    prauc_dml = float(average_precision_score(y_test, test_p_pred))
    brier_dml = float(brier_score_loss(y_test, test_p_pred))
    
    print("\n==========================================================================")
    print("   DML ORTHOGONALIZED SURVIVAL PERFORMANCE")
    print("==========================================================================")
    print(f"    ROC-AUC (Panel-Ebene)    : {auc_dml:.4f}")
    print(f"    PR-AUC (Panel-Ebene)     : {prauc_dml:.4f}")
    print(f"    Brier Score              : {brier_dml:.4f}")
    print("==========================================================================")
    
    # -------------------------------------------------------------------------
    # STUFE 3: KONTRAFAKTISCHE ANALYSE (PARTIELL & ISOLIERT REALISTISCH)
    # -------------------------------------------------------------------------
    print("\n[Stufe 3] Kontrafaktische Auswertung des DML-Modells (Dual-Teststrang mit Zählung) ...")
    
    metrics_all = {
        "ROC-AUC_Panel": auc_dml,
        "PR-AUC_Panel": prauc_dml,
        "Brier_Score": brier_dml
    }
    
    for i, (supp_col, prefix, label) in enumerate([
        ('fach_supp_count', 'fach', 'Fachlicher Support'),
        ('uebf_supp_count', 'uebf', 'Überfachlicher Support'),
        ('psych_supp_count', 'psych', 'Psychosozialer Support'),
    ]):
        # 1. PARTIELL (≙ A vs C/D/E): Ziel-Support 0 vs. beobachtet, andere beobachtet
        A_tilde_c_part = A_tilde_test.copy()
        A_tilde_t_part = A_tilde_test.copy() # beobachtet
        A_tilde_c_part[:, i] = 0.0 - A_hat_test[:, i]
        
        X_c_part = np.column_stack([W_test, A_tilde_c_part])
        X_t_part = np.column_stack([W_test, A_tilde_t_part])
        
        p0_part = dml_model.predict(X_c_part, verbose=0).flatten()
        p1_part = dml_model.predict(X_t_part, verbose=0).flatten()
        rrs_part = p1_part / np.clip(p0_part, 1e-7, 1.0)
        
        mean_rr_p   = float(np.mean(rrs_part))
        median_rr_p = float(np.median(rrs_part))
        q05_p       = float(np.quantile(rrs_part, 0.05))
        q95_p       = float(np.quantile(rrs_part, 0.95))
        
        # 2. ISOLIERT REALISTISCH (≙ B vs F/G/H): Alle 0 vs. nur Ziel beobachtet, andere 0
        A_tilde_c_iso = 0.0 - A_hat_test # alle 0
        A_tilde_t_iso = 0.0 - A_hat_test # alle 0
        A_tilde_t_iso[:, i] = test_panel[supp_col].values.astype(float) - A_hat_test[:, i] # nur Ziel beobachtet
        
        X_c_iso = np.column_stack([W_test, A_tilde_c_iso])
        X_t_iso = np.column_stack([W_test, A_tilde_t_iso])
        
        p0_iso = dml_model.predict(X_c_iso, verbose=0).flatten()
        p1_iso = dml_model.predict(X_t_iso, verbose=0).flatten()
        rrs_iso = p1_iso / np.clip(p0_iso, 1e-7, 1.0)
        
        mean_rr_iso   = float(np.mean(rrs_iso))
        median_rr_iso = float(np.median(rrs_iso))
        q05_iso       = float(np.quantile(rrs_iso, 0.05))
        q95_iso       = float(np.quantile(rrs_iso, 0.95))
        
        print(f"\n--- {label} ({supp_col}) [DML Entzerrt] ---")
        print(f"  PARTIELL:           Mean RR = {mean_rr_p:.4f}, Median RR = {median_rr_p:.4f} [{q05_p:.4f}, {q95_p:.4f}]")
        print(f"  ISOLIERT (realist): Mean RR = {mean_rr_iso:.4f}, Median RR = {median_rr_iso:.4f} [{q05_iso:.4f}, {q95_iso:.4f}]")
        
        metrics_all[f"{prefix}_partial"] = {"mean_rr": mean_rr_p, "median_rr": median_rr_p, "q05": q05_p, "q95": q95_p}
        metrics_all[f"{prefix}_isolated"] = {"mean_rr": mean_rr_iso, "median_rr": median_rr_iso, "q05": q05_iso, "q95": q95_iso}
        
        # Abwärtskompatible Keys
        metrics_all[f"Mean_RR_{prefix}_DML"]   = mean_rr_p
        metrics_all[f"Median_RR_{prefix}_DML"] = median_rr_p
        metrics_all[f"Q05_RR_{prefix}_DML"]    = q05_p
        metrics_all[f"Q95_RR_{prefix}_DML"]    = q95_p

    base_dir = data_dir
    save_metrics("dml_orthogonal_survival", metrics_all, base_dir)
    save_keras_model(dml_model, "dml_orthogonal_survival", base_dir)
    
    plot_learning_curve(history_dml.history, "dml_orthogonal_survival", base_dir, metric_name='AUC')
    plot_roc_curve(test_panel['event'], test_p_pred, "dml_orthogonal_survival", base_dir)
    plot_pr_curve(test_panel['event'], test_p_pred, "dml_orthogonal_survival", base_dir)
    
    print("\nDML Orthogonalized Survival Analyse beendet.")

if __name__ == '__main__':
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    train_dml_orthogonal_survival(data_dir)
