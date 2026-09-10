r"""
CLI Runner für Marginal Structural Models (MSM) mit IPTW
=========================================================
Fuehrt die vollständige MSM-Kausalanalyse auf dem Person-Semester-Panel durch.

Verwendung:
    $env:PYTHONPATH = "src"
    C:\GitHub_public\.venv\Scripts\python.exe src/run_msm_analysis.py --data_dir data_v4_grid/S01_baseline/universe_A
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0].parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deepsupport.evaluation.causal.marginal_structural_model import run_full_msm_suite


def main():
    parser = argparse.ArgumentParser(description="Run Marginal Structural Model (MSM) Analysis with IPTW")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_v4_grid/S01_baseline/universe_A",
        help="Pfad zum Szenario- und Universums-Ordner"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Zielordner fuer Metriken und Diagnoseberichte (Standard: data_dir)"
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_path = Path(args.output_dir) if args.output_dir else Path("output_v4_models/S01_baseline/universe_A")

    run_full_msm_suite(data_dir=data_path, output_dir=out_path)


if __name__ == "__main__":
    main()
