"""
PyTorch Marginal Structural Models (MSM) mit Inverser Propensity-Gewichtung (IPTW)
==================================================================================
Implementiert die kausale Schätzung sequentieller Fördermaßnahmen nach Robins (2000)
und Hernán et al. (2000) zur vollständigen Entflechtung zeitvariierender Confounder
mit Feedback (Time-Varying Confounding with Feedback).

Architektur:
1. Stabilisierte Gewichte (SW):
   - Zählermodell: P(A_t | A_hist, V, t) via PyTorch Logistic Net
   - Nennermodell: P(A_t | A_hist, L_t, V, t) via PyTorch Logistic Net
   - Kumulatives Produkt über Semester k=1...t mit Perzentil-Trimming (1% - 99%)
2. Gewichtete Diskrete Hazard-Regression in PyTorch:
   - Minimierung des gewichteten BCE-Verlusts: sum_it SW_it * BCE(y_it, h_it)
   - Cluster-robuste Sandwich-Kovarianzmatrix auf Studierendenebene (Huber-White)
3. Evaluierung & Causal Logging:
   - Hazard Ratios (HR) für Fachlich, Überfachlich, Psychosozial und All-Support
   - Asymptotische 95% Wald-Konfidenzintervalle & Risikoreduktion (%)
   - Protokollierung über CausalEvaluator
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import deepsupport.data_engine.feature_builder as fb
from deepsupport.evaluation.metrics_logger import CausalEvaluator, save_metrics


# =====================================================================
# PYTORCH LOGISTIC CLASSIFIER FÜR IPTW-GEWICHTE
# =====================================================================

class PyTorchWeightNet(nn.Module):
    """Kompaktes logistisches Netzwerk zur Wahrscheinlichkeitsschätzung von P(A_t=1 | Kovariaten)."""
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_prob(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)


def _train_weight_classifier(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 12,
    batch_size: int = 8192,
    lr: float = 5e-3,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """Trainiert das Propensity-Modell und liefert vorhergesagte Wahrscheinlichkeiten p in (1e-5, 1 - 1e-5)."""
    in_dim = X.shape[1]
    model = PyTorchWeightNet(in_dim=in_dim, hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t_X = torch.tensor(X, dtype=torch.float32)
    t_y = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(t_X, t_y), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for b_X, b_y in loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model.net(b_X).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, b_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), batch_size):
            chunk_X = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            p = model.forward_prob(chunk_X).cpu().numpy()
            preds.append(p)
        all_p = np.concatenate(preds)

    return np.clip(all_p, 1e-5, 1.0 - 1e-5)


# =====================================================================
# BERECHNUNG DER STABILISIERTEN GEWICHTE (SW)
# =====================================================================

def compute_torch_stabilized_weights(
    df: pd.DataFrame,
    treatment_col: str,
    time_varying_cols: List[str],
    baseline_cols: List[str],
    id_col: str = "studierenden_id",
    time_col: str = "fachsemester",
    trunc_percentiles: Tuple[float, float] = (0.01, 0.99),
    device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Berechnet stabilisierte Gewichte (SW) nach Robins & Hernán in PyTorch:
    SW_i(t) = prod_{k=1}^t [ P(A_k | A_hist, V, k) / P(A_k | A_hist, L_k, V, k) ]
    """
    df = df.copy()

    # Binäres Treatment
    A_bin = (df[treatment_col].values > 0).astype(np.float32)
    df["_A_bin"] = A_bin

    # Kumulative Historie
    df["_cum_A_prev"] = df.groupby(id_col)["_A_bin"].cumsum() - df["_A_bin"]

    # Semester One-Hot (bis Semester 16)
    sem_dummies = pd.get_dummies(df[time_col], prefix="sem", drop_first=True, dtype=float)

    # 1. Zähler-Kovariaten: [Baseline V, History A_prev, Semester-Dummies]
    X_num_df = pd.concat([df[baseline_cols], df[["_cum_A_prev"]], sem_dummies], axis=1)
    imputer_num = SimpleImputer(strategy="median")
    scaler_num = StandardScaler()
    X_num = scaler_num.fit_transform(imputer_num.fit_transform(X_num_df))

    # 2. Nenner-Kovariaten: [Baseline V, Time-varying L, History A_prev, Semester-Dummies]
    X_den_df = pd.concat([df[baseline_cols], df[time_varying_cols], df[["_cum_A_prev"]], sem_dummies], axis=1)
    imputer_den = SimpleImputer(strategy="median")
    scaler_den = StandardScaler()
    X_den = scaler_den.fit_transform(imputer_den.fit_transform(X_den_df))

    # Wahrscheinlichkeiten schätzen
    p_num = _train_weight_classifier(X_num, A_bin, epochs=12, device=device)
    p_den = _train_weight_classifier(X_den, A_bin, epochs=15, device=device)

    # Zeitschritt-Faktoren w_it
    w_t = np.where(A_bin == 1.0, p_num / p_den, (1.0 - p_num) / (1.0 - p_den))
    df["_w_t"] = w_t

    # Kumulatives Produkt pro Student: SW_i(t) = prod_{k=1}^t w_i(k)
    # Exponentieller Cumsum über log(w_t)
    df["_log_w_t"] = np.log(np.clip(w_t, 1e-4, 1e4))
    df["_cum_log_w"] = df.groupby(id_col)["_log_w_t"].cumsum()
    sw_raw = np.exp(df["_cum_log_w"].values)

    # Perzentil-Trimming
    p_low = np.percentile(sw_raw, trunc_percentiles[0] * 100)
    p_high = np.percentile(sw_raw, trunc_percentiles[1] * 100)
    sw_trimmed = np.clip(sw_raw, p_low, p_high)

    diag = {
        "mean": float(np.mean(sw_trimmed)),
        "std": float(np.std(sw_trimmed)),
        "min": float(np.min(sw_trimmed)),
        "max": float(np.max(sw_trimmed)),
        "p01": float(p_low),
        "p99": float(p_high),
    }

    return sw_trimmed, diag


# =====================================================================
# GEWICHTETE DISKRETE HAZARD-REGRESSION MIT CLUSTER-SANDWICH SE
# =====================================================================

class PyTorchMSM:
    """
    Marginal Structural Models Runner in reinem PyTorch.
    Schätzt unkonfundierte kausale Hazard Ratios für Support-Maßnahmen.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Optional[Path] = None,
        mode: str = "standard",
        temporal: str = "prev",
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        self.mode = mode
        self.temporal = temporal
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

    def fit_and_evaluate(self) -> Dict[str, Any]:
        """Führt die vollständige MSM-IPTW Analyse durch."""
        start_time = time.time()
        print("\n" + "=" * 78)
        print("   PYTORCH MARGINAL STRUCTURAL MODELS (MSM / IPTW)")
        print(f"   Modus: {self.mode} | Temporal: {self.temporal} | Device: {self.device}")
        print("=" * 78)

        # 1. Panel-Daten laden
        panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
            self.data_dir, mode=self.mode, temporal=self.temporal
        )

        # Relevante Spalten identifizieren
        baseline_candidates = [
            "hzb_note", "hzb_typ_ord", "migrationshintergrund", "erstakademiker",
            "erwerbstaetigkeit_std", "stg_BWL", "stg_Maschinenbau", "stg_Psychologie", "stg_Soziale_Arbeit"
        ]
        baseline_cols = [c for c in baseline_candidates if c in panel_df.columns]

        time_varying_candidates = [
            "fails_prev", "delta_cp_prev", "cp_rueckstand", "gpa_prev"
        ]
        time_varying_cols = [c for c in time_varying_candidates if c in panel_df.columns]

        # Treatments vorbereiten
        treatments_to_run = {}
        if "support_glz_fachlich" in panel_df.columns:
            treatments_to_run["fachlich"] = "support_glz_fachlich"
        elif "fach_supp_count" in panel_df.columns:
            treatments_to_run["fachlich"] = "fach_supp_count"

        if "support_glz_ueberfachlich" in panel_df.columns:
            treatments_to_run["ueberfachlich"] = "support_glz_ueberfachlich"
        elif "uebf_supp_count" in panel_df.columns:
            treatments_to_run["ueberfachlich"] = "uebf_supp_count"

        if "support_glz_psychosozial" in panel_df.columns:
            treatments_to_run["psychosozial"] = "support_glz_psychosozial"
        elif "psych_supp_count" in panel_df.columns:
            treatments_to_run["psychosozial"] = "psych_supp_count"

        # Gesamt-Support Spalte anlegen
        active_cols = [v for v in treatments_to_run.values() if v in panel_df.columns]
        if active_cols:
            panel_df["total_support_active"] = (panel_df[active_cols].sum(axis=1) > 0).astype(float)
            treatments_to_run["all_support"] = "total_support_active"

        y = panel_df[target_col].values.astype(np.float32)
        studis = panel_df["studierenden_id"].values
        unique_studis, studi_inverse = np.unique(studis, return_inverse=True)
        n_clusters = len(unique_studis)

        hr_estimates = {}
        hr_se = {}
        extra_info = {}

        # 2. Schleife über Treatments
        for treat_name, treat_col in treatments_to_run.items():
            print(f"\n--- Berechne MSM für Maßnahme: {treat_name.upper()} ({treat_col}) ---")

            # Stabilisierte Gewichte berechnen
            sw, diag = compute_torch_stabilized_weights(
                df=panel_df,
                treatment_col=treat_col,
                time_varying_cols=time_varying_cols,
                baseline_cols=baseline_cols,
                device=self.device
            )
            print(f"  • IPTW-Diagnostik: Mean = {diag['mean']:.4f} (Ideal=1.0) | Std = {diag['std']:.4f} | Range = [{diag['min']:.3f}, {diag['max']:.3f}]")

            # Design-Matrix: [Intercept, Treatment A_t, Baseline V]
            imputer_base = SimpleImputer(strategy="median")
            V_mat = imputer_base.fit_transform(panel_df[baseline_cols])
            A_vec = (panel_df[treat_col].values > 0).astype(np.float32).reshape(-1, 1)
            ones = np.ones((len(panel_df), 1), dtype=np.float32)

            # Design-Matrix: Intercept, Treatment, Baselines
            X_msm = np.hstack([ones, A_vec, V_mat])
            p_dim = X_msm.shape[1]

            # Weighted Logistic Regression via PyTorch (L-BFGS / AdamW)
            t_X = torch.tensor(X_msm, dtype=torch.float32).to(self.device)
            t_y = torch.tensor(y, dtype=torch.float32).to(self.device)
            t_sw = torch.tensor(sw, dtype=torch.float32).to(self.device)

            beta = torch.zeros(p_dim, requires_grad=True, device=self.device)
            opt = torch.optim.Adam([beta], lr=0.02)

            for _ in range(80):
                opt.zero_grad()
                logits = torch.matmul(t_X, beta)
                bce = F.binary_cross_entropy_with_logits(logits, t_y, reduction="none")
                loss = torch.mean(t_sw * bce)
                loss.backward()
                opt.step()

            beta_hat = beta.detach().cpu().numpy()
            beta_treat = float(beta_hat[1])

            # Vorhergesagte Wahrscheinlichkeiten
            with torch.no_grad():
                probs = torch.sigmoid(torch.matmul(t_X, beta)).cpu().numpy()

            # 3. Cluster-Robuster Huber-White Sandwich Standardfehler
            # W_diag = sw * p * (1 - p)
            v_diag = sw * probs * (1.0 - probs)
            XVX = np.dot(X_msm.T, X_msm * v_diag[:, None])
            try:
                inv_XVX = np.linalg.inv(XVX)
            except np.linalg.LinAlgError:
                inv_XVX = np.linalg.pinv(XVX)

            # Score-Residuen pro Beobachtung: r_it = sw_it * (y_it - p_it) * x_it
            r_it = (sw * (y - probs))[:, None] * X_msm

            # Aggregiere Scores auf Cluster-Ebene (Studierende)
            cluster_scores = np.zeros((n_clusters, p_dim), dtype=np.float32)
            np.add.at(cluster_scores, studi_inverse, r_it)

            # Fleisch der Sandwich-Matrix: B = sum_i s_i s_i^T
            B = np.dot(cluster_scores.T, cluster_scores)

            # Sandwich-Kovarianz: V = inv(XVX) * B * inv(XVX)
            V_robust = np.dot(inv_XVX, np.dot(B, inv_XVX))
            se_treat = float(np.sqrt(max(V_robust[1, 1], 1e-12)))

            # Causal Hazard Ratio
            hr_val = float(np.exp(beta_treat))
            ci_low = float(np.exp(beta_treat - 1.96 * se_treat))
            ci_high = float(np.exp(beta_treat + 1.96 * se_treat))
            rr_pct = float((1.0 - hr_val) * 100.0)

            hr_estimates[f"hr_{treat_name}"] = hr_val
            hr_se[f"hr_{treat_name}"] = se_treat
            extra_info[f"iptw_mean_{treat_name}"] = diag["mean"]
            extra_info[f"iptw_std_{treat_name}"] = diag["std"]

            print(f"  • Kausale Hazard Ratio: HR = {hr_val:.4f} | 95% CI: [{ci_low:.4f}, {ci_high:.4f}] | Risikoreduktion: {rr_pct:+.2f}%")

        # =====================================================================
        # PROTOKOLLIERUNG ÜBER CAUSAL EVALUATOR
        # =====================================================================
        duration = time.time() - start_time
        evaluator = CausalEvaluator(
            model_name="torch_marginal_structural_model",
            base_dir=self.output_dir,
            mode=self.mode,
            temporal=self.temporal
        )

        extra_info["training_time_s"] = duration
        extra_info["device"] = str(self.device)

        metrics_dict = evaluator.evaluate_and_log(
            hr_estimates=hr_estimates,
            hr_se=hr_se,
            extra_metrics=extra_info
        )

        return metrics_dict
