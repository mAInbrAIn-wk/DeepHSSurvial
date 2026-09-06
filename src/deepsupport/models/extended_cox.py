"""
Extended Cox Proportional Hazards Model (Time-Varying Covariates Edition)
========================================================================
Modelliert den Einfluss von Support-Maßnahmen als ZEITVERÄNDERLICHE KOVARIATE
(Time-Varying Treatment X_i(t)) über die gesamte Studiendauer.

Unterstützt über feature_builder.py alle Modi (standard, gradeblind, blind, oracle, realistic)
sowie temporale Varianten (temporal='prev' für Delta/Vorwerte, temporal='cum' für Gesamthistorie).
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import metrics_logger and feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from deepsupport.evaluation.metrics_logger import save_metrics, SurvivalEvaluator
import deepsupport.data_engine.feature_builder as fb


def fit_extended_cox_model(panel_df: pd.DataFrame,
                           feature_cols: list,
                           target_col: str = 'event',
                           temporal: str = 'prev',
                           mode: str = 'standard',
                           output_dir: Path = None):
    print("\n" + "=" * 74)
    print(f"   EXTENDED COX PROPORTIONAL HAZARDS MODEL (PHREG, temporal={temporal}, mode={mode})")
    print("=" * 74)

    # Bereinigung fehlender Werte
    valid_cols = [c for c in feature_cols if c in panel_df.columns]
    model_df = panel_df.dropna(subset=valid_cols + ['t_start', 't_stop', target_col]).copy()

    print(f"Schätze Modell mit {len(model_df)} Person-Semester Zeilen und {len(valid_cols)} Features...")

    formel = f"t_stop ~ {' + '.join(valid_cols)}"

    try:
        cox_mod = smf.phreg(
            formula=formel,
            data=model_df,
            status=model_df[target_col].values,
            entry=model_df['t_start'].values,
            ties='breslow'
        )
        results = cox_mod.fit()
    except Exception as e:
        print(f"Fehler beim Fitten des Modells: {e}")
        return None

    print("\n--- ESTIMATION RESULTS ---")
    print(results.summary())

    params_s = pd.Series(results.params, index=results.model.exog_names)
    hr = np.exp(params_s)

    print("\nExtrahierte Hazard Ratios:")
    for var_name, value in hr.items():
        print(f"  • {var_name:<30}: HR = {value:.4f}")

    if output_dir:
        hr_f = float(hr.get('fach_supp_count', hr.get('support_glz_fachlich', 1.0)))
        hr_u = float(hr.get('uebf_supp_count', hr.get('support_glz_ueberfachlich', 1.0)))
        hr_p = float(hr.get('psych_supp_count', hr.get('support_glz_psychosozial', 1.0)))

        model_key = "extended_cox_delta" if temporal == 'prev' else "extended_cox_panel"
        evaluator = SurvivalEvaluator(model_name=model_key, output_dir=output_dir, temporal=temporal, mode=mode)
        hr_estimates = {
            "hr_fachlich": hr_f,
            "hr_ueberfachlich": hr_u,
            "hr_psychosozial": hr_p
        }
        
        evaluator.evaluate_and_log(hr_estimates=hr_estimates)
        
        if temporal == 'prev':
            evaluator_panel = SurvivalEvaluator(model_name="extended_cox_panel", output_dir=output_dir, temporal=temporal, mode=mode)
            evaluator_panel.evaluate_and_log(hr_estimates=hr_estimates)

    print("=" * 74)
    return results


def train_extended_cox_model(data_dir: Path = Path('src/output_dl'),
                             temporal: str = 'prev',
                             mode: str = 'standard'):
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode=mode, temporal=temporal
    )
    return fit_extended_cox_model(panel_df, feature_cols, target_col, temporal=temporal, mode=mode, output_dir=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extended Cox Survival Model")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    args = parser.parse_args()

    train_extended_cox_model(Path(args.data_dir), temporal=args.temporal, mode=args.mode)
