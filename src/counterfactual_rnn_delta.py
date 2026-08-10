"""
Counterfactual Relative Risk Analysis für Recurrent GRU v2 (Exam Level)
========================================================================
Berechnet das Relative Risiko (RR) für jeden der drei Support-Typen separat
auf dem bestehenden Prüfungs-Sequenzmodell (recurrent_exam_survival_v2.keras).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from recurrent_exam_survival_v2 import build_recurrent_exam_dataset_v2, masked_binary_crossentropy, PADDING_VALUE
from metrics_logger import save_metrics

def main():
    print("\n==========================================================================")
    print("   COUNTERFACTUAL INFERENCE: RECURRENT GRU V2 (EXAM LEVEL)")
    print("==========================================================================")
    
    data_dir = Path('../output_dl') if Path('../output_dl').exists() else Path('output_dl')
    model_path = data_dir / 'models' / 'recurrent_exam_survival_v2.keras'
    if not model_path.exists():
        model_path = data_dir / 'recurrent_exam_survival_v2.keras'
        
    if not model_path.exists():
        print(f"Modell nicht gefunden: {model_path}")
        return
        
    print("Lade Datensatz & Modell...")
    studis, X_seq, y_seq, studi_events = build_recurrent_exam_dataset_v2(data_dir)
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
    
    # Baseline Prediction (unbehandelt für alle 3 Support-Typen)
    X_control_base = X_test.copy()
    X_control_base[valid_mask_test, 3] = 0.0 # fach
    X_control_base[valid_mask_test, 4] = 0.0 # uebf
    X_control_base[valid_mask_test, 5] = 0.0 # psych
    X_control_base[valid_mask_test] = scaler.transform(X_control_base[valid_mask_test])
    
    preds_control = model.predict(X_control_base, verbose=0)
    
    # Extrahieren des Risikos im letzten gültigen Zeitschritt für jeden Test-Studenten
    risk_control_last = []
    last_step_indices = []
    for i in range(len(test_idx)):
        steps = np.where(X_test[i, :, 0] != PADDING_VALUE)[0]
        if len(steps) > 0:
            last_step_indices.append(steps[-1])
            risk_control_last.append(preds_control[i, steps[-1], 0])
        else:
            last_step_indices.append(-1)
            risk_control_last.append(0.0)
            
    risk_control_last = np.clip(np.array(risk_control_last), 1e-7, 1.0)
    
    metrics_all = {}
    
    for feature_idx, supp_name in [(3, 'fach'), (4, 'uebf'), (5, 'psych')]:
        X_treated = X_test.copy()
        X_treated[valid_mask_test, 3] = 0.0
        X_treated[valid_mask_test, 4] = 0.0
        X_treated[valid_mask_test, 5] = 0.0
        X_treated[valid_mask_test, feature_idx] = 1.0 # Einzelne Intervention
        X_treated[valid_mask_test] = scaler.transform(X_treated[valid_mask_test])
        
        preds_treated = model.predict(X_treated, verbose=0)
        risk_treated_last = []
        for i, last_step in enumerate(last_step_indices):
            if last_step >= 0:
                risk_treated_last.append(preds_treated[i, last_step, 0])
            else:
                risk_treated_last.append(0.0)
                
        risk_treated_last = np.array(risk_treated_last)
        rrs = risk_treated_last / risk_control_last
        
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

    save_metrics("counterfactual_rnn_delta", metrics_all, data_dir)
    print("\nCounterfactual Analysis für RNN v2 abgeschlossen.")

if __name__ == '__main__':
    main()
