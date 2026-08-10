"""
Counterfactual Relative Risk Analysis für Dynamic DeepHit Delta
================================================================
Berechnet das Relative Risiko (RR) für jeden der drei semester-lokalen Support-Typen
auf dem neu trainierten Dynamic DeepHit Delta Competing Risks Modell.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dynamic_deephit_delta_model import build_competing_risks_dataset_delta
from recurrent_survival_model import masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics

def main():
    print("\n==========================================================================")
    print("   COUNTERFACTUAL RELATIVE RISK ANALYSIS (DYNAMIC DEEPHIT DELTA)")
    print("==========================================================================")
    
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    model_path = data_dir / 'models' / 'dynamic_deephit_delta.keras'
    
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        return
        
    print("Lade Datensatz & DeepHit Delta Modell...")
    studis, X_seq, y_dropout, y_grad, studi_events = build_competing_risks_dataset_delta(data_dir, max_semesters=16)
    N, K_max, F = X_seq.shape
    
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train = X_seq[train_idx].copy()
    X_test = X_seq[test_idx].copy()
    
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    model = tf.keras.models.load_model(model_path, custom_objects={'masked_binary_crossentropy': masked_binary_crossentropy})
    valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
    
    # Control: Support = 0 für alle 3 Typen
    # Features: 0:gpa, 1:cp, 2:fails, 3:cp_rueckstand, 4:fach_act, 5:uebf_act, 6:psych_act, 7:hzb, 8:erwerb
    X_control = X_test.copy()
    X_control[valid_mask_test, 4] = 0.0 # fach
    X_control[valid_mask_test, 5] = 0.0 # uebf
    X_control[valid_mask_test, 6] = 0.0 # psych
    X_control[valid_mask_test] = scaler.transform(X_control[valid_mask_test])
    
    preds_control_dropout = model.predict(X_control, verbose=0)[0].flatten()
    preds_control_dropout = np.clip(preds_control_dropout[valid_mask_test.flatten()], 1e-7, 1.0)
    
    metrics_all = {}
    
    for feature_idx, supp_name in [(4, 'fach'), (5, 'uebf'), (6, 'psych')]:
        X_treated = X_test.copy()
        X_treated[valid_mask_test, 4] = 0.0
        X_treated[valid_mask_test, 5] = 0.0
        X_treated[valid_mask_test, 6] = 0.0
        X_treated[valid_mask_test, feature_idx] = 1.0
        X_treated[valid_mask_test] = scaler.transform(X_treated[valid_mask_test])
        
        preds_treated_dropout = model.predict(X_treated, verbose=0)[0].flatten()
        preds_treated_dropout = preds_treated_dropout[valid_mask_test.flatten()]
        
        rrs = preds_treated_dropout / preds_control_dropout
        
        mean_rr = float(np.mean(rrs))
        median_rr = float(np.median(rrs))
        q05 = float(np.quantile(rrs, 0.05))
        q95 = float(np.quantile(rrs, 0.95))
        
        print(f"\n--- Support-Typ: {supp_name.upper()} ---")
        print(f"  Mean Relative Risk (RR)  : {mean_rr:.4f}")
        print(f"  Median Relative Risk (RR): {median_rr:.4f}")
        print(f"  5%-95% KI                : [{q05:.4f}, {q95:.4f}]")
        
        metrics_all[f"Mean_RR_{supp_name}"] = mean_rr
        metrics_all[f"Median_RR_{supp_name}"] = median_rr
        metrics_all[f"Q05_RR_{supp_name}"] = q05
        metrics_all[f"Q95_RR_{supp_name}"] = q95

    save_metrics("counterfactual_rr_deephit_delta", metrics_all, data_dir)
    print("\nCounterfactual Analysis für DeepHit Delta abgeschlossen.")

if __name__ == '__main__':
    main()
