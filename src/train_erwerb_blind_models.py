import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

def main():
    data_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl_v2")
    if not data_dir.exists():
        data_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")

    print("================================================================================")
    print("ERWERB-BLIND MODELL-BENCHMARK: CONCOUNTER EVALUATION OHNE ERWERBSTÄTIGKEIT")
    print("================================================================================")

    # Lade 2D Panel Daten
    from extended_cox_delta import build_delta_panel
    df_panel = build_delta_panel(data_dir)

    print(f"Loaded Panel Rows: {len(df_panel)}")

    # Standard Features vs. Erwerb-Blind Features
    feature_cols_all = [c for c in df_panel.select_dtypes(include=[np.number]).columns if c not in ["studierenden_id", "t_start", "t_stop", "event", "anomalie_typ"]]
    feature_cols_standard = [c for c in feature_cols_all if c != "fach_supp_active"]
    feature_cols_blind = [c for c in feature_cols_standard if "erwerb" not in c.lower()]

    print(f"Standard Features ({len(feature_cols_standard)}): {feature_cols_standard}")
    print(f"Erwerb-Blind Features ({len(feature_cols_blind)}): {feature_cols_blind}")

    X_std = df_panel[feature_cols_standard].fillna(0).values
    X_blind = df_panel[feature_cols_blind].fillna(0).values
    A = df_panel["fach_supp_active"].values
    Y = df_panel["event"].values

    # 1. Evaluate DML with Standard Features vs Erwerb-Blind Features
    print("\n--- 1. DOUBLE MACHINE LEARNING COMPARISON ---")

    # Standard DML
    p_mod_std = LogisticRegression(max_iter=1000).fit(X_std, A)
    p_hat_std = np.clip(p_mod_std.predict_proba(X_std)[:, 1], 0.01, 0.99)
    A_res_std = A - p_hat_std

    y_mod_std = Ridge(alpha=1.0).fit(X_std, Y)
    y_hat_std = y_mod_std.predict(X_std)
    Y_res_std = Y - y_hat_std

    beta_std = Ridge(alpha=0.001).fit(A_res_std.reshape(-1, 1), Y_res_std).coef_[0]
    base_rate = np.mean(Y)
    rr_std = (base_rate + beta_std) / base_rate

    # Erwerb-Blind DML
    p_mod_blind = LogisticRegression(max_iter=1000).fit(X_blind, A)
    p_hat_blind = np.clip(p_mod_blind.predict_proba(X_blind)[:, 1], 0.01, 0.99)
    A_res_blind = A - p_hat_blind

    y_mod_blind = Ridge(alpha=1.0).fit(X_blind, Y)
    y_hat_blind = y_mod_blind.predict(X_blind)
    Y_res_blind = Y - y_hat_blind

    beta_blind = Ridge(alpha=0.001).fit(A_res_blind.reshape(-1, 1), Y_res_blind).coef_[0]
    rr_blind = (base_rate + beta_blind) / base_rate

    print(f"Ground Truth RR (Makro):                0.9972")
    print(f"Standard DML RR (mit Erwerbstätigkeit): {rr_std:.4f} (Beta: {beta_std:.6f})")
    print(f"Erwerb-Blind DML RR (ohne Erwerb):      {rr_blind:.4f} (Beta: {beta_blind:.6f})")

    # 2. Classification AUC comparison
    print("\n--- 2. DROPOUT PREDICTION ACCURACY (ROC-AUC / PR-AUC) ---")
    y_pred_std = y_hat_std
    y_pred_blind = y_hat_blind

    auc_std = roc_auc_score(Y, y_pred_std)
    pr_auc_std = average_precision_score(Y, y_pred_std)

    auc_blind = roc_auc_score(Y, y_pred_blind)
    pr_auc_blind = average_precision_score(Y, y_pred_blind)

    print(f"Standard Model  -> ROC-AUC: {auc_std:.4f} | PR-AUC: {pr_auc_std:.4f}")
    print(f"Erwerb-Blind    -> ROC-AUC: {auc_blind:.4f} | PR-AUC: {pr_auc_blind:.4f}")
    print(f"Performance Drop: ROC-AUC {auc_std - auc_blind:.4f} | PR-AUC {pr_auc_std - pr_auc_blind:.4f}")

if __name__ == "__main__":
    main()
