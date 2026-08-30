"""
Kalibrierungsanalyse & Reliability Diagrams
============================================
Erstellt Kalibrierungskurven (Reliability Diagrams) und Brier Score Analysen
für die probabilistischen Survival-Modelle unter Klassen-Imbalance.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import tensorflow as tf

from extended_cox_delta import build_delta_panel
from metrics_logger import save_metrics
import tensorflow.keras.backend as K

PADDING_VALUE = -99.0

@tf.keras.utils.register_keras_serializable()
def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PADDING_VALUE), tf.float32)
    y_true_clean = tf.maximum(y_true, 0.0)
    bce = K.binary_crossentropy(y_true_clean, y_pred)
    return tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-7)


def main(data_dir=None):
    print("\n==========================================================================")
    print("   KALIBRIERUNGSANALYSE & RELIABILITY DIAGRAMME")
    print("==========================================================================")
    
    if data_dir is not None:
        data_dir = Path(data_dir)
    elif os.environ.get('DATA_DIR'):
        data_dir = Path(os.environ['DATA_DIR'])
    else:
        data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    plots_dir = data_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
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
    X_test = preprocessor.transform(test_panel[feature_cols])
    y_test = test_panel['event'].values
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfekt kalibriert")
    
    # 1. Extended Logistic Hazard Delta
    lh_model_path = data_dir / "models" / "extended_logistic_hazard_delta.keras"
    if lh_model_path.exists():
        lh_model = tf.keras.models.load_model(lh_model_path)
        y_prob_lh = lh_model.predict(X_test, verbose=0).flatten()
        prob_true_lh, prob_pred_lh = calibration_curve(y_test, y_prob_lh, n_bins=10)
        brier_lh = brier_score_loss(y_test, y_prob_lh)
        plt.plot(prob_pred_lh, prob_true_lh, "s-", label=f"Logistic Hazard Delta (Brier={brier_lh:.4f})")
        print(f"Logistic Hazard Delta Brier Score: {brier_lh:.4f}")

    # 2. Dynamic DeepHit Delta (Dropout / Event 1)
    dh_model_path = data_dir / "models" / "dynamic_deephit_delta.keras"
    if dh_model_path.exists():
        try:
            dh_model = tf.keras.models.load_model(dh_model_path)
            dh_preds = dh_model.predict(X_test, verbose=0)
            if isinstance(dh_preds, list):
                y_prob_dh = dh_preds[0].flatten()
            else:
                y_prob_dh = dh_preds[:, 0]
                
            prob_true_dh, prob_pred_dh = calibration_curve(y_test, y_prob_dh, n_bins=10)
            brier_dh = brier_score_loss(y_test, y_prob_dh)
            plt.plot(prob_pred_dh, prob_true_dh, "o-", label=f"DeepHit Delta (Brier={brier_dh:.4f})")
            print(f"DeepHit Delta (Dropout) Brier Score: {brier_dh:.4f}")
        except Exception as e:
            print(f"DeepHit Kalibrierungskurve übersprungen: {e}")

    plt.xlabel("Vorhergesagte Abbruchwahrscheinlichkeit")
    plt.ylabel("Tatsächlicher Anteil Abbrüche")
    plt.title("Reliability Diagram (Kalibrierungskurve auf Testdaten)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    out_plot = plots_dir / "calibration_curves.png"
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Kalibrierungsdiagramm erfolgreich gespeichert unter: {out_plot}")
    
    metrics = {}
    if 'brier_lh' in locals(): metrics["Brier_Logistic_Hazard_Delta"] = float(brier_lh)
    if 'brier_dh' in locals(): metrics["Brier_DeepHit_Delta"] = float(brier_dh)
    if 'brier_rnn' in locals(): metrics["Brier_Recurrent_Delta"] = float(brier_rnn)
    save_metrics("calibration_analysis", metrics, data_dir)

if __name__ == '__main__':
    main()
