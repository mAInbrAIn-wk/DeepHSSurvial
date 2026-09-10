"""
DeepSupport PyTorch Benchmark Runner
====================================
Führt die vollständige PyTorch- und PyCox-Suite auf V4-Daten aus:
1. PyTorchLogisticHazard (Diskretes Intervall-Hazard-Modell)
2. PyTorchDeepHit (Ranking- und Likelihood-Survival)
3. PyTorchCoxPH (Extended DeepSurv mit Partial Likelihood)
4. PyTorchNextExamTransformer (Autoregressiver Noten- & Bestehens-Transformer)

Verwendet standardisierte OOP-Evaluatoren (SurvivalEvaluator, DualHeadEvaluator)
aus deepsupport.evaluation.metrics_logger und vergleicht direkt gegen Keras-Baselines.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch

from deepsupport.models.torch.data_loaders import (
    prepare_panel_dataloaders,
    prepare_sequence_dataloaders,
    prepare_exam_regressor_dataloaders,
    prepare_causal_survival_dataloaders,
)
from deepsupport.models.torch.survival import (
    PyTorchLogisticHazard,
    PyTorchDeepHit,
    PyTorchCoxPH,
    PyTorchCoxTime,
    PyTorchDeepHitCompetingRisks,
)
from deepsupport.models.torch.transformer import (
    PyTorchExamTransformerRegressor,
    PyTorchCausalExamTransformerSurvival,
    PyTorchNextExamTransformer,
)
from deepsupport.models.torch.trainer import (
    ModelTrainer,
)
from deepsupport.evaluation.metrics_logger import (
    SurvivalEvaluator,
    RegressionEvaluator,
    DualHeadEvaluator,
    save_metrics,
)


def run_survival_benchmark(
    data_dir: Path,
    output_dir: Path,
    mode: str = "standard",
    temporal: str = "prev",
    batch_size: int = 2048,
    epochs_lh: int = 25,
    epochs_dh: int = 25,
    epochs_cox: int = 25,
    epochs_ct: int = 25,
    epochs_dh_cr: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(">>> [PHASE 1 & 2] PyTorch & PyCox Survival Suite Training")
    print(f"    Daten: {data_dir} | Mode: {mode} | Temporal: {temporal}")
    print("=" * 78)

    # 1. Daten laden
    data_bundle = prepare_panel_dataloaders(
        data_dir=data_dir,
        mode=mode,
        temporal=temporal,
        batch_size=batch_size,
        seed=seed,
        panel_type="semester",
    )
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    X_train = data_bundle["X_train"]
    t_train = data_bundle["t_train"]
    e_train = data_bundle["e_train"]
    X_test = data_bundle["X_test"]
    t_test = data_bundle["t_test"]
    e_test = data_bundle["e_test"]
    ce_test = data_bundle.get("ce_test")
    test_panel = data_bundle["test_panel"]
    in_dim = data_bundle["input_dim"]
    step_test_np = np.clip(test_panel["fachsemester"].values.astype(np.int64) - 1, 0, 15)
    step_test = torch.tensor(step_test_np)

    results = {}

    # --- MODELL 1: PyTorch LogisticHazard ---
    print("\n" + "-" * 60)
    print(" 1/5: Training PyTorchLogisticHazard (Diskrete Intervall-Likelihood)")
    print("-" * 60)
    lh_model = PyTorchLogisticHazard(
        in_features=in_dim,
        num_durations=16,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
    )
    opt_lh = torch.optim.AdamW(lh_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_lh = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lh, T_max=epochs_lh)
    lh_trainer = ModelTrainer(
        model=lh_model,
        optimizer=opt_lh,
        scheduler=sched_lh,
    )
    t0 = time.time()
    hist_lh = lh_trainer.fit(train_loader, val_loader, epochs=epochs_lh, patience=6, verbose=True)
    fit_time_lh = time.time() - t0

    # Inferenz
    lh_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=lh_trainer.device)
        lh_model.to(lh_trainer.device)
        risk_lh = lh_model.predict_risk(X_test_t, step_test.to(lh_trainer.device)).cpu().numpy()

    ev_lh = SurvivalEvaluator(base_dir=output_dir, model_name="torch_logistic_hazard")
    metrics_lh = ev_lh.evaluate_and_log(
        y_true=e_test,
        y_prob=risk_lh,
        t_stop=t_test,
        mode=mode,
        temporal_type=temporal,
        fit_time_s=fit_time_lh,
    )
    results["torch_logistic_hazard"] = metrics_lh

    # --- MODELL 2: PyTorch DeepHit (Single Event) ---
    print("\n" + "-" * 60)
    print(" 2/5: Training PyTorchDeepHit (Discrete Ranking + Likelihood)")
    print("-" * 60)
    dh_model = PyTorchDeepHit(
        in_features=in_dim,
        num_durations=16,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        alpha_ranking=0.5,
    )
    opt_dh = torch.optim.AdamW(dh_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_dh = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dh, T_max=epochs_dh)
    dh_trainer = ModelTrainer(
        model=dh_model,
        optimizer=opt_dh,
        scheduler=sched_dh,
    )
    t0 = time.time()
    hist_dh = dh_trainer.fit(train_loader, val_loader, epochs=epochs_dh, patience=6, verbose=True)
    fit_time_dh = time.time() - t0

    # Inferenz
    dh_model.eval()
    with torch.no_grad():
        dh_model.to(dh_trainer.device)
        risk_dh = dh_model.predict_risk(X_test_t, step_test.to(dh_trainer.device)).cpu().numpy()

    ev_dh = SurvivalEvaluator(base_dir=output_dir, model_name="torch_deephit")
    metrics_dh = ev_dh.evaluate_and_log(
        y_true=e_test,
        y_prob=risk_dh,
        t_stop=t_test,
        mode=mode,
        temporal_type=temporal,
        fit_time_s=fit_time_dh,
    )
    results["torch_deephit"] = metrics_dh

    # --- MODELL 3: PyTorch CoxPH (Extended DeepSurv mit Breslow Baseline Hazard) ---
    print("\n" + "-" * 60)
    print(" 3/5: Training PyTorchCoxPH (Extended DeepSurv / Breslow Baseline Hazard)")
    print("-" * 60)
    cox_model = PyTorchCoxPH(
        in_features=in_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
    )
    opt_cox = torch.optim.AdamW(cox_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_cox = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cox, T_max=epochs_cox)
    cox_trainer = ModelTrainer(
        model=cox_model,
        optimizer=opt_cox,
        scheduler=sched_cox,
    )
    t0 = time.time()
    hist_cox = cox_trainer.fit(train_loader, val_loader, epochs=epochs_cox, patience=6, verbose=True)
    fit_time_cox = time.time() - t0

    # Inferenz via Breslow Baseline Hazard
    cox_model.compute_baseline_hazard(X_train, t_train, e_train)
    risk_cox = cox_model.predict_risk(X_test, step_test_np)

    ev_cox = SurvivalEvaluator(base_dir=output_dir, model_name="torch_coxph_extended")
    metrics_cox = ev_cox.evaluate_and_log(
        y_true=e_test,
        y_prob=risk_cox,
        t_stop=t_test,
        mode=mode,
        temporal_type=temporal,
        fit_time_s=fit_time_cox,
    )
    results["torch_coxph_extended"] = metrics_cox

    # --- MODELL 4: PyTorch CoxTime (Non-Proportional Hazards) ---
    print("\n" + "-" * 60)
    print(" 4/5: Training PyTorchCoxTime (Nicht-proportionale Hazard-Interaktionen g(x, t))")
    print("-" * 60)
    ct_model = PyTorchCoxTime(
        in_features=in_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        max_duration=16.0,
    )
    opt_ct = torch.optim.AdamW(ct_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_ct = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ct, T_max=epochs_ct)
    ct_trainer = ModelTrainer(
        model=ct_model,
        optimizer=opt_ct,
        scheduler=sched_ct,
    )
    t0 = time.time()
    hist_ct = ct_trainer.fit(train_loader, val_loader, epochs=epochs_ct, patience=6, verbose=True)
    fit_time_ct = time.time() - t0

    # Inferenz via zeitabhängige Baseline Hazard
    ct_model.compute_baseline_hazard(X_train, t_train, e_train)
    risk_ct = ct_model.predict_risk(X_test, step_test_np)

    ev_ct = SurvivalEvaluator(base_dir=output_dir, model_name="torch_coxtime")
    metrics_ct = ev_ct.evaluate_and_log(
        y_true=e_test,
        y_prob=risk_ct,
        t_stop=t_test,
        mode=mode,
        temporal_type=temporal,
        fit_time_s=fit_time_ct,
    )
    results["torch_coxtime"] = metrics_ct

    # --- MODELL 5: PyTorch DeepHit Competing Risks ---
    print("\n" + "-" * 60)
    print(" 5/5: Training PyTorchDeepHitCompetingRisks (Multi-Event Dropout vs. Abschluss)")
    print("-" * 60)
    dh_cr = PyTorchDeepHitCompetingRisks(
        in_features=in_dim,
        num_risks=2,
        num_durations=16,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        alpha_ranking=0.2,
        sigma_ranking=0.1,
    )
    opt_cr = torch.optim.AdamW(dh_cr.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_cr = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cr, T_max=epochs_dh_cr)
    cr_trainer = ModelTrainer(
        model=dh_cr,
        optimizer=opt_cr,
        scheduler=sched_cr,
    )
    t0 = time.time()
    hist_cr = cr_trainer.fit(train_loader, val_loader, epochs=epochs_dh_cr, patience=6, verbose=True)
    fit_time_cr = time.time() - t0

    dh_cr.eval()
    with torch.no_grad():
        dh_cr.to(cr_trainer.device)
        risk_cr_do = dh_cr.predict_risk(X_test_t, step_test.to(cr_trainer.device), risk_idx=0).cpu().numpy()
        risk_cr_gr = dh_cr.predict_risk(X_test_t, step_test.to(cr_trainer.device), risk_idx=1).cpu().numpy()

    ev_cr = SurvivalEvaluator(base_dir=output_dir, model_name="torch_deephit_competing_risks")
    extra_cr = {}
    if ce_test is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score
        e_grad_test = (ce_test == 2).astype(float)
        extra_cr["grad_roc_auc"] = float(roc_auc_score(e_grad_test, risk_cr_gr))
        extra_cr["grad_pr_auc"] = float(average_precision_score(e_grad_test, risk_cr_gr))

    metrics_cr = ev_cr.evaluate_and_log(
        y_true=e_test,
        y_prob=risk_cr_do,
        t_stop=t_test,
        mode=mode,
        temporal_type=temporal,
        fit_time_s=fit_time_cr,
        extra_metrics=extra_cr,
    )
    results["torch_deephit_competing_risks"] = metrics_cr

    # Modelle abspeichern
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lh_model.state_dict(), models_dir / "torch_logistic_hazard.pt")
    torch.save(dh_model.state_dict(), models_dir / "torch_deephit.pt")
    torch.save(cox_model.state_dict(), models_dir / "torch_coxph_extended.pt")
    torch.save(ct_model.state_dict(), models_dir / "torch_coxtime.pt")
    torch.save(dh_cr.state_dict(), models_dir / "torch_deephit_competing_risks.pt")
    print(f"\n[INFO] Alle 5 PyTorch-Survival-Gewichte in {models_dir} gespeichert.")

    return results


def run_exam_regressor_benchmark(
    data_dir: Path,
    output_dir: Path,
    mode: str = "gradeblind",
    temporal: str = "prev",
    batch_size: int = 256,
    epochs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(">>> [PHASE 3] PyTorch Exam Transformer Regressor Training (Gradeblind)")
    print(f"    Daten: {data_dir} | Mode: {mode} | Temporal: {temporal}")
    print("=" * 78)

    reg_bundle = prepare_exam_regressor_dataloaders(
        data_dir=data_dir,
        mode=mode,
        temporal=temporal,
        batch_size=batch_size,
        seed=seed,
        max_len=40,
    )
    train_loader = reg_bundle["train_loader"]
    val_loader = reg_bundle["val_loader"]
    test_loader = reg_bundle["test_loader"]
    y_test = reg_bundle["y_test"]
    feat_dim = reg_bundle["feature_dim"]

    print(f"Features: {feat_dim} | Absolventen Test: {len(y_test)}")

    reg_model = PyTorchExamTransformerRegressor(
        feat_dim=feat_dim,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        dropout=0.1,
        max_seq_len=40,
    )
    opt_reg = torch.optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_reg = torch.optim.lr_scheduler.CosineAnnealingLR(opt_reg, T_max=epochs)
    trainer = ModelTrainer(
        model=reg_model,
        optimizer=opt_reg,
        scheduler=sched_reg,
    )
    t0 = time.time()
    hist = trainer.fit(train_loader, val_loader, epochs=epochs, patience=6, verbose=True)
    fit_time = time.time() - t0

    # Inferenz
    reg_model.eval()
    preds_list = []
    with torch.no_grad():
        reg_model.to(trainer.device)
        for batch in test_loader:
            seq = batch["sequence"].to(trainer.device)
            mask = batch["mask"].to(trainer.device)
            preds = reg_model(seq, mask=mask)
            preds_list.append(preds.cpu().numpy())

    pred_notes = np.concatenate(preds_list)

    ev_reg = RegressionEvaluator(base_dir=output_dir, model_name="torch_exam_transformer_regressor")
    metrics_reg = ev_reg.evaluate_and_log(
        y_true=y_test,
        y_pred=pred_notes,
        mode=mode,
        temporal_type=temporal,
        fit_time_s=fit_time,
    )

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(reg_model.state_dict(), models_dir / "torch_exam_transformer_regressor.pt")

    return {"torch_exam_transformer_regressor": metrics_reg}


def run_causal_survival_benchmark(
    data_dir: Path,
    output_dir: Path,
    mode: str = "gradeblind",
    temporal: str = "prev",
    batch_size: int = 256,
    epochs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(">>> [PHASE 4] PyTorch Causal Exam Transformer Survival Training")
    print(f"    Daten: {data_dir} | Mode: {mode} | Temporal: {temporal}")
    print("=" * 78)

    sv_bundle = prepare_causal_survival_dataloaders(
        data_dir=data_dir,
        mode=mode,
        temporal=temporal,
        batch_size=batch_size,
        seed=seed,
        max_len=40,
    )
    train_loader = sv_bundle["train_loader"]
    val_loader = sv_bundle["val_loader"]
    test_loader = sv_bundle["test_loader"]
    y_test = sv_bundle["y_test"]
    test_mask = sv_bundle["test_mask"]
    test_student_events = sv_bundle["test_student_events"]
    feat_dim = sv_bundle["feature_dim"]

    print(f"Features: {feat_dim} | Studierende Test: {len(test_student_events)}")

    sv_model = PyTorchCausalExamTransformerSurvival(
        feat_dim=feat_dim,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        dropout=0.1,
        max_seq_len=40,
    )
    opt_sv = torch.optim.AdamW(sv_model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_sv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sv, T_max=epochs)
    trainer = ModelTrainer(
        model=sv_model,
        optimizer=opt_sv,
        scheduler=sched_sv,
    )
    t0 = time.time()
    hist = trainer.fit(train_loader, val_loader, epochs=epochs, patience=6, verbose=True)
    fit_time = time.time() - t0

    # Inferenz
    sv_model.eval()
    hazards_list = []
    with torch.no_grad():
        sv_model.to(trainer.device)
        for batch in test_loader:
            seq = batch["sequence"].to(trainer.device)
            mask = batch["mask"].to(trainer.device)
            haz = sv_model(seq, mask=mask)
            hazards_list.append(haz.cpu().numpy())

    hazards_all = np.concatenate(hazards_list, axis=0)  # (N_test, S)

    # 1. Schritt-Level Evaluierung über alle unpadded Testschritte (analog zu Keras)
    y_test_flat = y_test[test_mask].flatten()
    preds_flat = hazards_all[test_mask].flatten()

    # 2. Studierenden-Level Evaluierung via Survival-Produkt: S(K) = prod_{k=1}^K (1 - h_k)
    surv_probs = np.ones(len(y_test), dtype=np.float32)
    for i in range(len(y_test)):
        s_len = int(np.sum(test_mask[i]))
        if s_len > 0:
            h_k = np.clip(hazards_all[i, :s_len], 1e-6, 1.0 - 1e-6)
            surv_probs[i] = np.prod(1.0 - h_k)
    risk_student = 1.0 - surv_probs
    from sklearn.metrics import roc_auc_score, average_precision_score
    student_auc = float(roc_auc_score(test_student_events, risk_student))
    student_pr_auc = float(average_precision_score(test_student_events, risk_student))

    ev_sv = SurvivalEvaluator(base_dir=output_dir, model_name="torch_causal_exam_transformer_survival")
    metrics_sv = ev_sv.evaluate_and_log(
        y_true=y_test_flat,
        y_prob=preds_flat,
        mode=mode,
        temporal_type=temporal,
        fit_time_s=fit_time,
        extra_metrics={
            "student_roc_auc": student_auc,
            "student_pr_auc": student_pr_auc,
            "n_test_exam_steps": len(y_test_flat),
            "step_prevalence_pi0": float(np.mean(y_test_flat)),
        }
    )

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sv_model.state_dict(), models_dir / "torch_causal_exam_transformer_survival.pt")

    return {"torch_causal_exam_transformer_survival": metrics_sv}


def print_comparative_synopsis(results: Dict[str, Any], output_dir: Path):
    print("\n" + "=" * 98)
    print("   DEEPSUPPORT BENCHMARK SYNOPSIS: PYTORCH/PYCOX vs. KERAS/TF BASELINES")
    print("=" * 98)

    # Lade Keras Transformer Referenzwerte falls vorhanden
    keras_summary_path = output_dir / "metrics" / "deep_transformer_suite_summary_gradeblind_prev_bce_hybrid.json"
    if not keras_summary_path.exists():
        keras_summary_path = output_dir / "metrics" / "deep_transformer_suite_summary_standard_prev_bce_hybrid.json"

    keras_exam_r2 = None
    keras_exam_rmse = None
    keras_surv_auc = None
    keras_surv_prauc = None
    if keras_summary_path.exists():
        try:
            with open(keras_summary_path, "r") as f:
                k_data = json.load(f)
                k_exam = k_data.get("deep_exam_regressor", {})
                k_surv = k_data.get("deep_exam_survival", {})
                keras_exam_r2 = k_exam.get("r2_score")
                keras_exam_rmse = k_exam.get("rmse")
                keras_surv_auc = k_surv.get("roc_auc")
                keras_surv_prauc = k_surv.get("pr_auc_dropout")
        except Exception:
            pass

    header = f"{'Modell':<38} | {'Framework':<10} | {'ROC-AUC':<9} | {'PR-AUC (y=1)':<12} | {'Brier':<8} | {'C-Index / R²':<12} | {'Zeit (s)':<8}"
    print(header)
    print("-" * 105)

    for m_name, m in results.items():
        if "torch_exam_transformer_regressor" in m_name:
            r2 = f"{m.get('r2_score', 0.0):.4f}"
            rmse = f"{m.get('rmse', 0.0):.4f}"
            zeit = f"{m.get('training_time_s', 0.0):.1f}"
            print(f"{m_name:<38} | {'PyTorch':<10} | {'n/a':<9} | {'n/a':<12} | {f'RMSE={rmse}':<8} | {f'R²={r2}':<12} | {zeit:<8}")
        elif "roc_auc" in m:
            auc = f"{m.get('roc_auc', 0.0):.4f}"
            pr1 = f"{m.get('pr_auc_dropout', 0.0):.4f}"
            brier = f"{m.get('brier_score', 0.0):.4f}" if m.get("brier_score") is not None else "n/a"
            c_idx = f"{m.get('c_index', 0.0):.4f}" if m.get("c_index") is not None else "n/a"
            zeit = f"{m.get('training_time_s', 0.0):.1f}"
            print(f"{m_name:<38} | {'PyTorch':<10} | {auc:<9} | {pr1:<12} | {brier:<8} | {c_idx:<12} | {zeit:<8}")

    if keras_exam_r2 is not None:
        print("-" * 105)
        print(f"{'Keras Exam Transformer Regressor':<38} | {'TensorFlow':<10} | {'n/a':<9} | {'n/a':<12} | {f'RMSE={keras_exam_rmse:.4f}':<8} | {f'R²={keras_exam_r2:.4f}':<12} | {'Ref':<8}")
    if keras_surv_auc is not None:
        print(f"{'Keras Causal Exam Survival':<38} | {'TensorFlow':<10} | {keras_surv_auc:<9.4f} | {keras_surv_prauc:<12.4f} | {'Ref':<8} | {'n/a':<12} | {'Ref':<8}")

    print("=" * 98 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DeepSupport PyTorch & PyCox Benchmark Runner")
    parser.add_argument("--data_dir", type=str, default="data_v4_grid/S01_baseline/universe_A")
    parser.add_argument("--output_dir", type=str, default="output_v4_models/S01_baseline/universe_A")
    parser.add_argument("--mode", type=str, default=None, help="Globaler Modus-Override")
    parser.add_argument("--mode_surv", type=str, default="standard", help="Feature-Modus für Survival (Default: standard)")
    parser.add_argument("--mode_reg", type=str, default="gradeblind", help="Feature-Modus für GPA-Regression (Default: gradeblind)")
    parser.add_argument("--temporal", type=str, default="prev")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs_lh", type=int, default=20)
    parser.add_argument("--epochs_dh", type=int, default=20)
    parser.add_argument("--epochs_cox", type=int, default=20)
    parser.add_argument("--epochs_ct", type=int, default=20)
    parser.add_argument("--epochs_dh_cr", type=int, default=20)
    parser.add_argument("--epochs_trans", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_trans", action="store_true", help="Überspringe Transformer")
    parser.add_argument("--skip_surv", action="store_true", help="Überspringe Survival")

    args = parser.parse_args()

    mode_surv = args.mode if args.mode is not None else args.mode_surv
    mode_reg = args.mode if args.mode is not None else args.mode_reg

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    all_results = {}

    if not args.skip_surv:
        surv_res = run_survival_benchmark(
            data_dir=data_dir,
            output_dir=output_dir,
            mode=mode_surv,
            temporal=args.temporal,
            batch_size=args.batch_size,
            epochs_lh=args.epochs_lh,
            epochs_dh=args.epochs_dh,
            epochs_cox=args.epochs_cox,
            epochs_ct=args.epochs_ct,
            epochs_dh_cr=args.epochs_dh_cr,
            seed=args.seed,
        )
        all_results.update(surv_res)

    if not args.skip_trans:
        reg_res = run_exam_regressor_benchmark(
            data_dir=data_dir,
            output_dir=output_dir,
            mode=mode_reg,
            temporal=args.temporal,
            batch_size=256,
            epochs=args.epochs_trans,
            seed=args.seed,
        )
        all_results.update(reg_res)

        causal_res = run_causal_survival_benchmark(
            data_dir=data_dir,
            output_dir=output_dir,
            mode=mode_reg,
            temporal=args.temporal,
            batch_size=256,
            epochs=args.epochs_trans,
            seed=args.seed,
        )
        all_results.update(causal_res)

    # Speichern des Gesamt-JSON
    summary_path = output_dir / "metrics" / f"pytorch_suite_summary_{mode_surv}_{mode_reg}_{args.temporal}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"[INFO] Zusammenfassung der PyTorch-Suite gespeichert unter: {summary_path}")

    # Synopse ausgeben
    print_comparative_synopsis(all_results, output_dir)


if __name__ == "__main__":
    main()
