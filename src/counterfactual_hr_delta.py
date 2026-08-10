"""
Counterfactual Hazard Ratio Analysis (Extended DeepSurv Delta Edition)
========================================================================
Berechnet empirische kontrafaktische Hazard Ratios für die semester-lokalen
Support-Interventionen (fachlich, überfachlich, psychosozial) basierend auf
dem Extended DeepSurv Delta Modell.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import tensorflow as tf

from extended_cox_delta import build_delta_panel
from extended_deep_survival_delta import breslow_cox_loss
from metrics_logger import save_metrics

def analyze_counterfactual_hr_delta(data_dir: Path):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL HAZARD RATIO ANALYSIS (EXTENDED DEEPSURV DELTA)")
    print("==========================================================================")
    
    panel_df = build_delta_panel(data_dir)
    
    num_cols = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start', 'fails_prev', 'delta_cp_prev', 'cp_rueckstand']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_active', 'uebf_supp_active', 'psych_supp_active']
    
    feature_cols = num_cols + cat_cols + treatment_cols
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel  = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    preprocessor.fit(train_panel[feature_cols])
    
    model_path = data_dir / "models" / "extended_deepsurv_delta.keras"
    if not model_path.exists():
        print(f"Modell {model_path} nicht gefunden!")
        return
        
    model = tf.keras.models.load_model(model_path, custom_objects={'breslow_cox_loss': breslow_cox_loss})
    print(f"Modell geladen: {model_path.name}  |  Input-Dim: {model.input_shape}")

    metrics_all = {}

    for supp_col, label in [
        ('fach_supp_active',  'Fachlicher Support (aktives Semester)'),
        ('uebf_supp_active',  'Überfachlicher Support (aktives Semester)'),
        ('psych_supp_active', 'Psychosozialer Support (aktives Semester)'),
    ]:
        control = test_panel.copy()
        treated = test_panel.copy()
        control[supp_col] = 0.0
        treated[supp_col] = 1.0

        X_c = preprocessor.transform(control[feature_cols])
        X_t = preprocessor.transform(treated[feature_cols])

        h0 = model.predict(X_c, verbose=0).flatten()
        h1 = model.predict(X_t, verbose=0).flatten()

        hrs = np.exp(h1 - h0)
        mean_hr   = float(np.mean(hrs))
        median_hr = float(np.median(hrs))
        min_hr    = float(np.min(hrs))
        max_hr    = float(np.max(hrs))
        q05       = float(np.quantile(hrs, 0.05))
        q95       = float(np.quantile(hrs, 0.95))

        print(f"\n--- {label} ({supp_col}) ---")
        print(f"  Mean HR:     {mean_hr:.4f}")
        print(f"  Median HR:   {median_hr:.4f}")
        print(f"  Min / Max:   {min_hr:.4f} / {max_hr:.4f}")
        print(f"  5%–95% CI:   [{q05:.4f}, {q95:.4f}]")
        direction = "senkt" if median_hr < 1.0 else "ERHÖHT"
        print(f"  -> Median HR {direction} das Risiko {'um ' + f'{(1-median_hr)*100:.1f}%' if median_hr < 1.0 else 'um ' + f'{(median_hr-1)*100:.1f}%'}.")

        prefix = supp_col.replace('_supp_active', '')
        metrics_all[f"Mean_HR_{prefix}"]   = mean_hr
        metrics_all[f"Median_HR_{prefix}"] = median_hr
        metrics_all[f"Min_HR_{prefix}"]    = min_hr
        metrics_all[f"Max_HR_{prefix}"]    = max_hr
        metrics_all[f"Q05_HR_{prefix}"]    = q05
        metrics_all[f"Q95_HR_{prefix}"]    = q95

    print("\n" + "=" * 74)
    save_metrics("counterfactual_hr_delta", metrics_all, data_dir)
    print("Counterfactual HR Delta Analyse abgeschlossen.")

if __name__ == '__main__':
    data_dir = Path('../output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    analyze_counterfactual_hr_delta(data_dir)
