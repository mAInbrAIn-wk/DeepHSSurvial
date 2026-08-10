"""
Counterfactual Relative Risk Analysis für Extended Logistic Hazard Delta
==========================================================================
Berechnet das Relative Risiko (RR) für jeden der drei semester-lokalen Support-Typen
auf dem leistungsstarken Extended Logistic Hazard Delta Modell (ROC-AUC = 0.7992).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from extended_cox_delta import build_delta_panel
from metrics_logger import save_metrics

def analyze_counterfactual_rr_logistic_hazard_delta(data_dir: Path):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL RELATIVE RISK ANALYSIS (EXTENDED LOGISTIC HAZARD DELTA)")
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
    
    model_path = data_dir / "models" / "extended_logistic_hazard_delta.keras"
    if not model_path.exists():
        print(f"Modell {model_path} nicht gefunden!")
        return
        
    model = tf.keras.models.load_model(model_path)
    print(f"Modell geladen: {model_path.name}")

    metrics_all = {}

    for supp_col, label in [
        ('fach_supp_active',  'Fachlicher Support (aktives Semester)'),
        ('uebf_supp_active',  'Überfachlicher Support (aktives Semester)'),
        ('psych_supp_active', 'Psychosozialer Support (aktives Semester)'),
    ]:
        control = test_panel.copy()
        treated = test_panel.copy()
        control['fach_supp_active'] = 0.0
        control['uebf_supp_active'] = 0.0
        control['psych_supp_active'] = 0.0
        
        treated['fach_supp_active'] = 0.0
        treated['uebf_supp_active'] = 0.0
        treated['psych_supp_active'] = 0.0
        treated[supp_col] = 1.0

        X_c = preprocessor.transform(control[feature_cols])
        X_t = preprocessor.transform(treated[feature_cols])

        p0 = model.predict(X_c, verbose=0).flatten()
        p1 = model.predict(X_t, verbose=0).flatten()

        p0_safe = np.clip(p0, 1e-7, 1.0)
        rrs = p1 / p0_safe
        
        mean_rr   = float(np.mean(rrs))
        median_rr = float(np.median(rrs))
        min_rr    = float(np.min(rrs))
        max_rr    = float(np.max(rrs))
        q05       = float(np.quantile(rrs, 0.05))
        q95       = float(np.quantile(rrs, 0.95))

        print(f"\n--- {label} ({supp_col}) ---")
        print(f"  Mean RR:     {mean_rr:.4f}")
        print(f"  Median RR:   {median_rr:.4f}")
        print(f"  Min / Max:   {min_rr:.4f} / {max_rr:.4f}")
        print(f"  5%–95% CI:   [{q05:.4f}, {q95:.4f}]")
        direction = "senkt" if median_rr < 1.0 else "ERHÖHT"
        print(f"  -> Median RR {direction} das Abbruchrisiko {'um ' + f'{(1-median_rr)*100:.1f}%' if median_rr < 1.0 else 'um ' + f'{(median_rr-1)*100:.1f}%'}.")

        prefix = supp_col.replace('_supp_active', '')
        metrics_all[f"Mean_RR_{prefix}"]   = mean_rr
        metrics_all[f"Median_RR_{prefix}"] = median_rr
        metrics_all[f"Min_RR_{prefix}"]    = min_rr
        metrics_all[f"Max_RR_{prefix}"]    = max_rr
        metrics_all[f"Q05_RR_{prefix}"]    = q05
        metrics_all[f"Q95_RR_{prefix}"]    = q95

    print("\n" + "=" * 74)
    save_metrics("counterfactual_rr_logistic_hazard_delta", metrics_all, data_dir)
    print("Counterfactual Relative Risk Analysis für Logistic Hazard Delta abgeschlossen.")

if __name__ == '__main__':
    data_dir = Path('../output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    analyze_counterfactual_rr_logistic_hazard_delta(data_dir)
