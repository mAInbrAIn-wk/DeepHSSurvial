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
from deepsupport.evaluation.metrics_logger import save_metrics, SurvivalEvaluator, CausalEvaluator
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
        def _get_val(s, *keys, default=1.0):
            if hasattr(s, 'get'):
                for k in keys:
                    v = s.get(k)
                    if v is not None and pd.notna(v):
                        return float(v)
            elif hasattr(s, '__getitem__'):
                for k in keys:
                    if k in s and pd.notna(s[k]):
                        return float(s[k])
            return default

        hr_f = _get_val(hr, 'fach_supp_count', 'support_glz_fachlich', default=1.0)
        hr_u = _get_val(hr, 'uebf_supp_count', 'support_glz_ueberfachlich', default=1.0)
        hr_p = _get_val(hr, 'psych_supp_count', 'support_glz_psychosozial', default=1.0)

        metrics_dict = {
            "model_type": f"extended_cox_{temporal}_{mode}",
            "temporal": temporal,
            "mode": mode,
            "Support_HR_Fach_count": hr_f,
            "Support_HR_Uebf_count": hr_u,
            "Support_HR_Psych_count": hr_p,
            "fach_partial": {"mean_hr": hr_f, "median_hr": hr_f},
            "fach_isolated": {"mean_hr": hr_f, "median_hr": hr_f},
            "uebf_partial": {"mean_hr": hr_u, "median_hr": hr_u},
            "uebf_isolated": {"mean_hr": hr_u, "median_hr": hr_u},
            "psych_partial": {"mean_hr": hr_p, "median_hr": hr_p},
            "psych_isolated": {"mean_hr": hr_p, "median_hr": hr_p},
            "Support_HR_Fach_tv": hr_f,
            "Support_HR_Uebf_tv": hr_u,
            "Support_HR_Psych_tv": hr_p
        }
        if 'cum_fails' in hr:
            metrics_dict['HR_cum_fails'] = float(hr['cum_fails'])
        if 'cum_cp' in hr:
            metrics_dict['HR_cum_cp'] = float(hr['cum_cp'])
        if 'fails_prev' in hr:
            metrics_dict['HR_fails_prev'] = float(hr['fails_prev'])
        if 'delta_cp_prev' in hr:
            metrics_dict['HR_delta_cp_prev'] = float(hr['delta_cp_prev'])

        model_key = "extended_cox_delta" if temporal == 'prev' else "extended_cox_panel"
        hr_estimates = {
            "hr_fachlich": hr_f,
            "hr_ueberfachlich": hr_u,
            "hr_psychosozial": hr_p
        }
        bse_series = pd.Series(results.bse, index=results.model.exog_names) if hasattr(results, 'bse') and results.bse is not None else pd.Series(dtype=float)
        hr_se = {
            "hr_fachlich": _get_val(bse_series, 'fach_supp_count', 'support_glz_fachlich', default=0.05),
            "hr_ueberfachlich": _get_val(bse_series, 'uebf_supp_count', 'support_glz_ueberfachlich', default=0.05),
            "hr_psychosozial": _get_val(bse_series, 'psych_supp_count', 'support_glz_psychosozial', default=0.05)
        }

        evaluator = CausalEvaluator(base_dir=output_dir, model_name=model_key, temporal=temporal, mode=mode)
        evaluator.evaluate_and_log(hr_estimates=hr_estimates, hr_se=hr_se, extra_metrics=metrics_dict, mode=mode, temporal_type=temporal)
        save_metrics(model_key, metrics_dict, output_dir)
        
        if temporal == 'prev':
            evaluator_panel = CausalEvaluator(base_dir=output_dir, model_name="extended_cox_panel", temporal=temporal, mode=mode)
            evaluator_panel.evaluate_and_log(hr_estimates=hr_estimates, hr_se=hr_se, extra_metrics=metrics_dict, mode=mode, temporal_type=temporal)
            save_metrics("extended_cox_panel", metrics_dict, output_dir)

    print("=" * 74)
    return results


def train_extended_cox_model(data_dir: Path = Path('src/output_dl'),
                             temporal: str = 'prev',
                             mode: str = 'standard',
                             output_dir: Path = None):
    if output_dir is None:
        out_env = os.environ.get("OUTPUT_DIR")
        output_dir = Path(out_env) if out_env else Path(data_dir)
    panel_df, feature_cols, target_col, _ = fb.build_semester_panel_df(
        data_dir, mode=mode, temporal=temporal
    )
    return fit_extended_cox_model(panel_df, feature_cols, target_col, temporal=temporal, mode=mode, output_dir=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extended Cox Survival Model")
    parser.add_argument('--data_dir', type=str, default='src/output_dl')
    parser.add_argument('--temporal', type=str, default='prev', choices=['prev', 'cum'])
    parser.add_argument('--mode', type=str, default='standard')
    args = parser.parse_args()

    train_extended_cox_model(Path(args.data_dir), temporal=args.temporal, mode=args.mode)
