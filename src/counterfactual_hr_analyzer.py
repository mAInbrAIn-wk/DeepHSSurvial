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

from extended_cox_survival import build_person_semester_panel
from extended_deep_survival import breslow_cox_loss

def analyze_counterfactual_hr(data_dir: Path):
    print("\n==========================================================================")
    print("   COUNTERFACTUAL HAZARD RATIO ANALYSIS (EXTENDED DEEPSURV)")
    print("==========================================================================")
    
    panel_df = build_person_semester_panel(data_dir)
    
    num_cols = ['hzb_note', 'erwerbstaetigkeit_std', 't_stop', 't_start']
    cat_cols = ['stg_name', 'erstakademiker']
    treatment_cols = ['fach_supp_tv', 'uebf_supp_tv', 'psych_supp_tv']
    
    feature_cols = num_cols + cat_cols + treatment_cols
    
    unique_studis = np.array(panel_df['studierenden_id'].unique().tolist())
    train_ids, test_ids = train_test_split(unique_studis, test_size=0.20, random_state=42)
    
    train_panel = panel_df[panel_df['studierenden_id'].isin(train_ids)].copy()
    test_panel = panel_df[panel_df['studierenden_id'].isin(test_ids)].copy()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), cat_cols),
        ('treatments', 'passthrough', treatment_cols)
    ])
    
    preprocessor.fit(train_panel[feature_cols])
    
    model_path = data_dir / "models" / "extended_deepsurv_panel.keras"
    if not model_path.exists():
        print(f"Modell {model_path} nicht gefunden!")
        return
        
    model = tf.keras.models.load_model(model_path, custom_objects={'breslow_cox_loss': breslow_cox_loss})
    
    test_panel_control = test_panel.copy()
    test_panel_treated = test_panel.copy()
    
    test_panel_control['fach_supp_tv'] = 0.0
    test_panel_treated['fach_supp_tv'] = 1.0
    
    X_control = preprocessor.transform(test_panel_control[feature_cols])
    X_treated = preprocessor.transform(test_panel_treated[feature_cols])
    
    h0 = model.predict(X_control, verbose=0).flatten()
    h1 = model.predict(X_treated, verbose=0).flatten()
    
    hrs = np.exp(h1 - h0)
    
    print("\nVerteilung der empirischen kontrafaktischen Hazard Ratios (fach_supp_tv):")
    print("-" * 60)
    print(f"Mean HR:    {np.mean(hrs):.4f}")
    print(f"Median HR:  {np.median(hrs):.4f}")
    print(f"Min HR:     {np.min(hrs):.4f}")
    print(f"Max HR:     {np.max(hrs):.4f}")
    print(f"5% Quantil: {np.quantile(hrs, 0.05):.4f}")
    print(f"95% Quantil:{np.quantile(hrs, 0.95):.4f}")
    print("-" * 60)
    print("Ein Median < 1.0 bedeutet, dass der Support risikosenkend wirkt (Kausale Effektschätzung).")

if __name__ == '__main__':
    data_dir = Path('../output_dl')
    if not data_dir.exists():
        data_dir = Path('output_dl')
    analyze_counterfactual_hr(data_dir)
