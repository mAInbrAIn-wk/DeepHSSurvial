"""
Fast Core Suite Runner (V4.1)
=============================
Fuehrt alle leichten und mittelschweren Modellklassen sowie saemtliche
Kausale Inferenz- und Kontrafaktik-Skripte schnell und automatisiert aus.
Laufzeit pro Szenario: ca. 15-25 Minuten.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, List
import psutil

# Pfade sauber auflösen (3 Ebenen über runners: repo_root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.data_engine.aggregate import aggregiere_daten
from deepsupport.models.extended_cox import train_extended_cox_model
from deepsupport.models.extended_deepsurv import train_extended_deep_survival
from deepsupport.models.semester_gru import train_recurrent_survival_model
from deepsupport.models.dynamic_deephit import train_dynamic_deephit_model
from deepsupport.models.semester_transformer import train_transformer_survival
from deepsupport.models.exam_gru import train_recurrent_exam_survival_model
from deepsupport.models.exam_transformer import train_transformer_exam_survival
from deepsupport.models.baseline_classifiers import run_baseline_training
from deepsupport.models.baseline_regressors import run_regression_training
from deepsupport.models.dml_orthogonal import train_dml_orthogonal_survival
from deepsupport.models.dml_transformer import train_transformer_dml

from deepsupport.models.train_oracle_models import train_oracle_models
from deepsupport.models.train_erwerb_blind_models import train_erwerb_blind_models
from deepsupport.models.deep_survival import train_deep_survival
from deepsupport.evaluation.mediation_analysis import run_structural_mediation_analysis
from deepsupport.evaluation.causal.counterfactual_hr_analyzer import analyze_counterfactual_hr
from deepsupport.evaluation.causal.counterfactual_deephit_fixed import main as run_counterfactual_deephit
from deepsupport.evaluation.causal.counterfactual_grade_transformer import analyze_counterfactual_grade_transformer
from deepsupport.evaluation.causal.counterfactual_oracle_logistic_hazard import analyze_counterfactual_oracle_logistic_hazard
import plot_calibration_curves

class PipelineBenchmarkTracker:
    def __init__(self):
        self.steps = []
        self.process = psutil.Process(os.getpid())

    def run_step(self, step_name: str, func, *args, **kwargs):
        print("\n" + "=" * 80)
        print(f"   START: {step_name}")
        print(f"   Zeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        mem_start = self.process.memory_info().rss / (1024 * 1024)
        t0 = time.time()

        result = None
        status = "PASSED"
        err_msg = None

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - t0
            mem_end = self.process.memory_info().rss / (1024 * 1024)
            print(f"\n[OK] {step_name} ERFOLGREICH BEENDET ({elapsed/60:.2f} Min. / {elapsed:.1f}s | RAM: {mem_end:.1f}MB)")
        except Exception as e:
            elapsed = time.time() - t0
            mem_end = self.process.memory_info().rss / (1024 * 1024)
            status = "FAILED"
            err_msg = str(e)
            print(f"\n[FEHLER] {step_name} FEHLGESCHLAGEN nach {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

        self.steps.append({
            "step_name": step_name,
            "status": status,
            "duration_s": round(elapsed, 2),
            "ram_start_mb": round(mem_start, 1),
            "ram_end_mb": round(mem_end, 1),
            "ram_delta_mb": round(mem_end - mem_start, 1),
            "error": err_msg
        })
        return result

    def export_report(self, output_dir: Path):
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True, parents=True)

        json_path = diag_dir / "fast_suite_benchmark_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_duration_s": sum(s["duration_s"] for s in self.steps),
                "steps": self.steps
            }, f, indent=2)

        md_path = diag_dir / "fast_suite_benchmark_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Fast Core Suite Benchmark Report\n\n")
            f.write(f"**Generiert am:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Gesamtlaufzeit:** {sum(s['duration_s'] for s in self.steps)/60:.2f} Minuten\n\n")
            f.write("| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for s in self.steps:
                f.write(f"| {s['step_name']} | {s['status']} | {s['duration_s']} | {s['ram_start_mb']} | {s['ram_end_mb']} | {s['ram_delta_mb']:+} |\n")

        print(f"\n[REPORT] Fast Suite Benchmark gespeichert unter: {md_path}")


def run_fast_suite(data_dir: Path, output_dir: Optional[Path] = None, temporal: str = 'prev', modes: list = None, population_seed: int = 42):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir = PROJECT_ROOT / data_dir

    if modes is None:
        modes = ['standard', 'gradeblind']
        
    target_out = Path(output_dir) if output_dir else data_dir
    target_out.mkdir(parents=True, exist_ok=True)
    os.environ['DATA_DIR'] = str(data_dir)
    os.environ['OUTPUT_DIR'] = str(target_out)
    tracker = PipelineBenchmarkTracker()
    total_t0 = time.time()

    print("*" * 80)
    print("   FAST CORE SUITE RUNNER (V4.2 Modular)")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')} | Temporal: {temporal} | Seed: {population_seed}")
    print(f"   Data Dir:   {data_dir.resolve()}")
    print(f"   Output Dir: {target_out.resolve()}")
    print(f"   Aktive Modi: {modes}")
    print("*" * 80)

    # 0. Sicherstellen, dass aggregierte Daten da sind (DuckDB)
    if not (data_dir / 'agg_abschluesse.csv').exists():
        print(f"\n[INFO] Aggregierte Daten fehlen in {data_dir}. Starte DuckDB Aggregation...")
        aggregiere_daten(data_dir, backend='duckdb')

    # 1. MODUS-ABHAENGIGE SCHNELLE MODELLE
    for mode in modes:
        print(f"\n{'='*80}\n   FAST SUITE -> MODUS: {mode.upper()}\n{'='*80}")

        tracker.run_step(f"Extended Cox [{mode}]", lambda m=mode: train_extended_cox_model(data_dir=data_dir, temporal=temporal, mode=m))
        tracker.run_step(f"Extended DeepSurv [{mode}]", lambda m=mode: train_extended_deep_survival(data_dir=data_dir, temporal=temporal, mode=m))
        tracker.run_step(f"Recurrent Survival GRU [{mode}]", lambda m=mode: train_recurrent_survival_model(data_dir=data_dir, max_semesters=16, temporal=temporal, mode=m))
        tracker.run_step(f"Dynamic DeepHit Competing Risks [{mode}]", lambda m=mode: train_dynamic_deephit_model(data_dir=data_dir, max_semesters=16, temporal=temporal, mode=m))
        tracker.run_step(f"Transformer Survival [{mode}]", lambda m=mode: train_transformer_survival(data_dir=data_dir, max_semesters=16, temporal=temporal, mode=m))
        tracker.run_step(f"Recurrent Exam Survival GRU [{mode}]", lambda m=mode: train_recurrent_exam_survival_model(data_dir=data_dir, max_exams=40, temporal=temporal, mode=m))
        tracker.run_step(f"Transformer Exam Survival [{mode}]", lambda m=mode: train_transformer_exam_survival(data_dir=data_dir, max_exams=40, temporal=temporal, mode=m))
        tracker.run_step(f"Landmark Baseline Classifiers [{mode}]", lambda m=mode: run_baseline_training(data_dir=data_dir, mode=m))
        tracker.run_step(f"Landmark Regression [{mode}]", lambda m=mode: run_regression_training(data_dir=data_dir, mode=m))
        tracker.run_step(f"DML Orthogonal Survival [{mode}]", lambda m=mode: train_dml_orthogonal_survival(data_dir=data_dir, temporal=temporal, mode=m))
        tracker.run_step(f"Transformer DML [{mode}]", lambda m=mode: train_transformer_dml(data_dir=data_dir, temporal=temporal, mode=m))

    # 2. SCHNELLE SPEZIAL- & DIAGNOSE-MODELLE
    print(f"\n{'='*80}\n   FAST SUITE -> SPEZIAL- UND STATISTISCHE DIAGNOSEMODELLE\n{'='*80}")
    tracker.run_step("Oracle Models (Lift Analysis)", lambda: train_oracle_models(data_dir=data_dir, temporal=temporal))
    tracker.run_step("DSGVO Realistic Models", lambda: train_erwerb_blind_models(data_dir=data_dir, temporal=temporal))
    tracker.run_step("Deep Survival Landmark (LH & DS)", lambda: train_deep_survival(data_dir=data_dir))
    tracker.run_step("Strukturelle Mediationsanalyse", lambda: run_structural_mediation_analysis(data_dir=data_dir))
    tracker.run_step("Plot Calibration Curves", lambda: plot_calibration_curves.main(data_dir=data_dir))

    # 3. KONTRAFAKTISCHE INFERENZ-SUITE (VOLLSTAENDIG & AUTARK)
    print(f"\n{'='*80}\n   FAST SUITE -> KONTRAFAKTISCHE INFERENZ-SUITE\n{'='*80}")
    tracker.run_step("Counterfactual HR Analyzer (Extended Cox/Panel)", lambda: analyze_counterfactual_hr(data_dir=data_dir))
    tracker.run_step("Counterfactual DeepHit Competing Risks", lambda: run_counterfactual_deephit(data_dir=data_dir))
    tracker.run_step("Counterfactual Grade Transformer", lambda: analyze_counterfactual_grade_transformer(data_dir=data_dir))
    tracker.run_step("Counterfactual Oracle Logistic Hazard", lambda: analyze_counterfactual_oracle_logistic_hazard(data_dir=data_dir))

    tracker.export_report(target_out)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 80)
    print(f"   FAST CORE SUITE ERFOLGREICH BEENDET ({total_elapsed/60:.2f} Minuten)")
    print("=" * 80)
    return tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fast Core Suite Runner V4.2")
    parser.add_argument('--data_dir', type=str, default="data_v4_grid/S01_baseline/universe_A")
    parser.add_argument('--output_dir', type=str, default=None, help="Optionales separates Ausgabe-Verzeichnis")
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--modes', type=str, default='standard,gradeblind', help="Kommagetrennte Liste der Modi")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    mode_list = [m.strip() for m in args.modes.split(',') if m.strip()]
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_fast_suite(
        data_dir=Path(args.data_dir),
        output_dir=out_dir,
        temporal=args.temporal,
        modes=mode_list,
        population_seed=args.seed
    )
