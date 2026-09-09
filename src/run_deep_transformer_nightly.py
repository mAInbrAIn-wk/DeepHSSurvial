"""
Nightly Runner: Modern Deep Transformer Suite (9-Run Benchmark)
===============================================================
Orchestriert den vollständigen Nachtlauf für die modernisierte
Deep Transformer Suite auf der V4.1 Baseline-Kohorte (N=50.000, Uni A).

Umfasst exakt 9 Durchläufe:
- 5 Feature-Modi (gradeblind, standard, blind, oracle, realistic) mit Hybrid-Reg.
- 4 Regularisierungs-Modi (l2, dropout, elasticnet, none) im Default-Modus gradeblind.

Kein Focal Loss (strikte Nutzung von BCE für saubere Kalibrierung und Survival-AUC).
25 Epochen mit EarlyStopping (patience=15) und ReduceLROnPlateau (patience=5).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
from pathlib import Path

# Ensure src is in pythonpath
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_transformer_regression import train_deep_transformer_models


def run_nightly_suite():
    data_dir = PROJECT_ROOT / "data_v4_grid" / "S01_baseline" / "universe_A"
    output_dir = PROJECT_ROOT / "output_v4_models" / "S01_baseline" / "universe_A"
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("   START DES DEEP TRANSFORMER NACHTLAUFS (9 BENCHMARK-DURCHLÄUFE)")
    print(f"   Input-Daten : {data_dir}")
    print(f"   Output-Ziel : {output_dir}")
    print("   Epochen     : 25 (EarlyStopping Patience=15, ReduceLROnPlateau Patience=5)")
    print("   Batch-Size  : 128")
    print("   Loss-Type   : BCE (Focal Loss deaktiviert)")
    print("=" * 80)

    total_start = time.time()
    results = {}

    # -------------------------------------------------------------
    # Teil 1: 5 Feature-Modi (Hybrid Regularisierung)
    # -------------------------------------------------------------
    feature_modes = ["gradeblind", "standard", "blind", "oracle", "realistic"]
    print("\n>>> TEIL 1: 5 FEATURE-MODI (Reg-Type = hybrid) <<<")

    for i, mode in enumerate(feature_modes, 1):
        run_name = f"feature_mode_{mode}"
        print(f"\n[{i}/9] Starte Durchlauf '{mode}' (Hybrid Regularisierung)...")
        t0 = time.time()
        try:
            m = train_deep_transformer_models(
                data_dir=data_dir,
                output_dir=output_dir,
                temporal="prev",
                mode=mode,
                d_model=64,
                num_heads=4,
                num_blocks=2,
                reg_type="hybrid",
                l2_val=1e-4,
                loss_type="bce",
                epochs=25,
                batch_size=128
            )
            elapsed = time.time() - t0
            results[run_name] = {
                "status": "success",
                "mode": mode,
                "reg_type": "hybrid",
                "elapsed_s": elapsed,
                "metrics": m
            }
            sem = m.get('deep_semester_regressor', {})
            ex = m.get('deep_exam_regressor', {})
            sv = m.get('deep_exam_survival_bce', {})
            print(f"--> Fertig in {elapsed:.1f}s | Sem R2: {sem.get('r2_score', sem.get('r2', 0.0)):.4f} | Exam R2: {ex.get('r2_score', ex.get('r2', 0.0)):.4f} | Surv AUC: {sv.get('roc_auc', 0.0):.4f}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"--> FEHLER in {mode}: {e}")
            results[run_name] = {
                "status": "error",
                "error": str(e),
                "elapsed_s": elapsed
            }

    # -------------------------------------------------------------
    # Teil 2: 4 Regularisierungs-Modi (Default: gradeblind)
    # -------------------------------------------------------------
    reg_modes = ["l2", "dropout", "elasticnet", "none"]
    print("\n>>> TEIL 2: 4 REGULARISIERUNGS-MODI (Mode = gradeblind) <<<")

    for i, reg in enumerate(reg_modes, 6):
        run_name = f"reg_mode_{reg}"
        print(f"\n[{i}/9] Starte Durchlauf 'gradeblind' mit Reg-Type '{reg}'...")
        t0 = time.time()
        try:
            m = train_deep_transformer_models(
                data_dir=data_dir,
                output_dir=output_dir,
                temporal="prev",
                mode="gradeblind",
                d_model=64,
                num_heads=4,
                num_blocks=2,
                reg_type=reg,
                l2_val=1e-4,
                loss_type="bce",
                epochs=25,
                batch_size=128
            )
            elapsed = time.time() - t0
            results[run_name] = {
                "status": "success",
                "mode": "gradeblind",
                "reg_type": reg,
                "elapsed_s": elapsed,
                "metrics": m
            }
            sem = m.get('deep_semester_regressor', {})
            ex = m.get('deep_exam_regressor', {})
            sv = m.get('deep_exam_survival_bce', {})
            print(f"--> Fertig in {elapsed:.1f}s | Sem R2: {sem.get('r2_score', sem.get('r2', 0.0)):.4f} | Exam R2: {ex.get('r2_score', ex.get('r2', 0.0)):.4f} | Surv AUC: {sv.get('roc_auc', 0.0):.4f}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"--> FEHLER in {reg}: {e}")
            results[run_name] = {
                "status": "error",
                "error": str(e),
                "elapsed_s": elapsed
            }

    total_duration = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"   [FERTIG] NACHTLAUF BEENDET NACH {total_duration / 3600:.2f} STUNDEN ({total_duration:.1f}s)")
    print("=" * 80)

    # Master-Summary JSON speichern
    summary_path = metrics_dir / "deep_transformer_nightly_9runs_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_duration_s": total_duration,
            "total_duration_hours": total_duration / 3600,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": results
        }, f, indent=4, ensure_ascii=False)

    # Markdown Summary
    md_path = metrics_dir / "deep_transformer_nightly_9runs_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Modern Deep Transformer Suite: 9-Run Benchmark Summary\n\n")
        f.write(f"- **Datum / Uhrzeit**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Gesamtlaufzeit**: {total_duration / 3600:.2f} Stunden ({total_duration:.1f}s)\n")
        f.write("- **Datensatz**: S01 Baseline (universe_A, N=50.000)\n")
        f.write("- **Hyperparameter**: $d=64$, 4 Heads, 2 Blocks, Epochs=25, Batch=128\n\n")
        f.write("## 1. Feature-Modi Vergleich (Hybrid-Regularisierung)\n\n")
        f.write("| Modus | Semester R² | Semester RMSE | Exam R² | Exam RMSE | Survival ROC-AUC | Survival PR-AUC | Dauer (s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for mode in feature_modes:
            res = results.get(f"feature_mode_{mode}", {}).get("metrics", {})
            sem = res.get("deep_semester_regressor", {})
            ex = res.get("deep_exam_regressor", {})
            sv = res.get("deep_exam_survival_bce", {})
            el = results.get(f"feature_mode_{mode}", {}).get("elapsed_s", 0.0)
            f.write(f"| `{mode}` | {sem.get('r2_score', sem.get('r2', 0.0)):.4f} | {sem.get('rmse', 0.0):.4f} | {ex.get('r2_score', ex.get('r2', 0.0)):.4f} | {ex.get('rmse', 0.0):.4f} | {sv.get('roc_auc', 0.0):.4f} | {sv.get('pr_auc_dropout', 0.0):.4f} | {el:.1f} |\n")

        f.write("\n## 2. Regularisierungs-Vergleich (Default: `gradeblind`)\n\n")
        f.write("| Regularisierung | Semester R² | Semester RMSE | Exam R² | Exam RMSE | Survival ROC-AUC | Survival PR-AUC | Dauer (s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for reg in ["hybrid"] + reg_modes:
            if reg == "hybrid":
                run_key = "feature_mode_gradeblind"
            else:
                run_key = f"reg_mode_{reg}"
            res = results.get(run_key, {}).get("metrics", {})
            sem = res.get("deep_semester_regressor", {})
            ex = res.get("deep_exam_regressor", {})
            sv = res.get("deep_exam_survival_bce", {})
            el = results.get(run_key, {}).get("elapsed_s", 0.0)
            f.write(f"| `{reg}` | {sem.get('r2_score', sem.get('r2', 0.0)):.4f} | {sem.get('rmse', 0.0):.4f} | {ex.get('r2_score', ex.get('r2', 0.0)):.4f} | {ex.get('rmse', 0.0):.4f} | {sv.get('roc_auc', 0.0):.4f} | {sv.get('pr_auc_dropout', 0.0):.4f} | {el:.1f} |\n")

    print(f"\n[OK] Zusammenfassung gespeichert unter:\n  {summary_path}\n  {md_path}")


if __name__ == "__main__":
    run_nightly_suite()
