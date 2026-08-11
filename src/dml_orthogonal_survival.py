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
from sklearn.linear_model import LogisticRegression
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
    
    treatment_cols = ['fach_supp_active', 'uebf_supp_active', 'psych_supp_active']
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    print(f"\nGroup Split: {len(train_ids)} Train-Studierende ({len(train_panel)} Sem-Zeilen), {len(test_ids)} Test-Studierende ({len(test_panel)} Sem-Zeilen)")
    
    confounder_prep = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), confounder_num),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), confounder_cat)
    ])
    
    W_train = confounder_prep.fit_transform(train_panel[confounder_cols])
    W_test  = confounder_prep.transform(test_panel[confounder_cols])
    
    # -------------------------------------------------------------------------
    # STUFE 1: PROPENSITY MODELS & RESIDUAL-CALCULATION (ORTHOGONALISIERUNG)
    # -------------------------------------------------------------------------
    print("\n[Stufe 1] Schätze Propensity-Modelle P(A_t = 1 | W_t) ...")
    
    residuals_train = []
    residuals_test = []
    
    for supp_col in treatment_cols:
        y_treat_train = train_panel[supp_col].values
        y_treat_test  = test_panel[supp_col].values
        
        prop_model = LogisticRegression(max_iter=500, C=1.0)
        prop_model.fit(W_train, y_treat_train)
        
        p_hat_train = prop_model.predict_proba(W_train)[:, 1]
        p_hat_test  = prop_model.predict_proba(W_test)[:, 1]
        
        # Orthogonalisiertes Treatment-Residuum A_tilde = A - P_hat
        res_train = y_treat_train - p_hat_train
        res_test  = y_treat_test - p_hat_test
        
        residuals_train.append(res_train)
        residuals_test.append(res_test)
        
    A_tilde_train = np.column_stack(residuals_train)
    A_tilde_test  = np.column_stack(residuals_test)
    
    X_train_dml = np.column_stack([W_train, A_tilde_train])
    X_test_dml  = np.column_stack([W_test, A_tilde_test])
    
    input_dim = X_train_dml.shape[1]
    
    # -------------------------------------------------------------------------
    # STUFE 2: NEURONALES DML HAZARD-MODELL
    # -------------------------------------------------------------------------
    print("\n[Stufe 2] Trainiere Keras DML Orthogonalized Hazard Modell ...")
    tf.random.set_seed(42)
    
    dml_model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    dml_model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='binary_crossentropy', metrics=['AUC'])
    history_dml = dml_model.fit(X_train_dml, train_panel['event'].values, epochs=30, batch_size=2048, verbose=0)
    
    test_p_pred = dml_model.predict(X_test_dml, verbose=0).flatten()
    
    auc_dml = roc_auc_score(test_panel['event'], test_p_pred)
    prauc_dml = average_precision_score(test_panel['event'], test_p_pred)
    brier_dml = brier_score_loss(test_panel['event'], test_p_pred)
    
    print("\n==========================================================================")
    print("   ERGEBNISSE DML ORTHOGONALIZED SURVIVAL MODELL")
    print("==========================================================================")
    print(f"    ROC-AUC                  : {auc_dml:.4f}")
    print(f"    PR-AUC (Average Precision): {prauc_dml:.4f}")
    print(f"    Brier Score              : {brier_dml:.4f}")
    print("==========================================================================")
    
    # -------------------------------------------------------------------------
    # STUFE 3: KONTRAFAKTISCHE ANALYSIS DES ENT ZERRTEN MODELLS
    # -------------------------------------------------------------------------
    print("\n[Stufe 3] Kontrafaktische Auswertung des DML-Modells ...")
    
    metrics_all = {
        "ROC-AUC_Panel": auc_dml,
        "PR-AUC_Panel": prauc_dml,
        "Brier_Score": brier_dml
    }
    
    for i, (supp_col, label) in enumerate([
        ('fach_supp_active',  'Fachlicher Support (aktives Semester)'),
        ('uebf_supp_active',  'Überfachlicher Support (aktives Semester)'),
        ('psych_supp_active', 'Psychosozialer Support (aktives Semester)'),
    ]):
        # Baseline Control: A_tilde when A = 0 => A_tilde = -P_hat
        # Treated: A_tilde when A = 1 => A_tilde = 1 - P_hat
        A_tilde_c = A_tilde_test.copy()
        A_tilde_t = A_tilde_test.copy()
        
        # P_hat is A_test - A_tilde_test
        p_hat_col = test_panel[supp_col].values - A_tilde_test[:, i]
        A_tilde_c[:, i] = 0.0 - p_hat_col
        A_tilde_t[:, i] = 1.0 - p_hat_col
        
        X_c_dml = np.column_stack([W_test, A_tilde_c])
        X_t_dml = np.column_stack([W_test, A_tilde_t])
        
        p0 = dml_model.predict(X_c_dml, verbose=0).flatten()
        p1 = dml_model.predict(X_t_dml, verbose=0).flatten()
        
        p0_safe = np.clip(p0, 1e-7, 1.0)
        rrs = p1 / p0_safe
        
        mean_rr   = float(np.mean(rrs))
        median_rr = float(np.median(rrs))
        q05       = float(np.quantile(rrs, 0.05))
        q95       = float(np.quantile(rrs, 0.95))
        
        print(f"\n--- {label} ({supp_col}) [DML Entzerrt] ---")
        print(f"  Mean Relative Risk (RR)  : {mean_rr:.4f}")
        print(f"  Median Relative Risk (RR): {median_rr:.4f}")
        print(f"  5%-95% KI                : [{q05:.4f}, {q95:.4f}]")
        
        prefix = supp_col.replace('_supp_active', '')
        metrics_all[f"Mean_RR_{prefix}_DML"]   = mean_rr
        metrics_all[f"Median_RR_{prefix}_DML"] = median_rr
        metrics_all[f"Q05_RR_{prefix}_DML"]    = q05
        metrics_all[f"Q95_RR_{prefix}_DML"]    = q95

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
