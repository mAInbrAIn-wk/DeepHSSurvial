"""
Counterfactual Inference Wrapper für Semester-Transformer Survival
==================================================================
Lädt das Semester-Level Transformer Survival Modell (transformer_survival.keras)
und berechnet die Counterfactual Hazard Ratio für fachlichen Support.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from recurrent_survival_model import build_recurrent_survival_dataset, masked_binary_crossentropy, PADDING_VALUE
from transformer_survival_model import PositionalEncoding

def run_counterfactual_transformer():
    print("Starte Counterfactual Inference für Transformer (Semester-Ebene)...")
    
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'output_dl'
    models_dir = data_dir / 'models'
    
    model_path = models_dir / 'transformer_survival.keras'
    
    custom_obj = {
        'masked_binary_crossentropy': masked_binary_crossentropy,
        'PositionalEncoding': PositionalEncoding
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_obj)
    
    # 8 Features: [sem_gpa, sem_cp, sem_fails, fach_supp_cum, uebf_supp_cum, psych_supp_cum, hzb_note, erwerb_std]
    studis, X_seq, y_seq, studi_events = build_recurrent_survival_dataset(data_dir, max_semesters=16)
    N, K_max, F = X_seq.shape
    
    train_idx, temp_idx, _, y_temp_event = train_test_split(
        np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp_event, test_size=0.50, random_state=42, stratify=y_temp_event
    )
    
    X_train = X_seq[train_idx].copy()
    X_test = X_seq[test_idx].copy()
    
    FACH_SUPP_IDX = 3  # gpa=0, cp=1, fails=2, fach_supp_cum=3
    
    valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
    
    X_test_control = X_test.copy()
    X_test_treated = X_test.copy()
    
    X_test_control[valid_mask_test, FACH_SUPP_IDX] = 0.0
    X_test_treated[valid_mask_test, FACH_SUPP_IDX] = 1.0
    
    scaler = StandardScaler()
    valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
    scaler.fit(X_train[valid_mask_train])
    
    X_test_control[valid_mask_test] = scaler.transform(X_test_control[valid_mask_test])
    X_test_treated[valid_mask_test] = scaler.transform(X_test_treated[valid_mask_test])
    
    preds_control = model.predict(X_test_control, verbose=0)
    preds_treated = model.predict(X_test_treated, verbose=0)
    
    preds_c_flat = preds_control.flatten()[valid_mask_test.flatten()]
    preds_t_flat = preds_treated.flatten()[valid_mask_test.flatten()]
    
    preds_c_flat = np.clip(preds_c_flat, 1e-7, 1.0)
    
    mean_c = np.mean(preds_c_flat)
    mean_t = np.mean(preds_t_flat)
    mean_hr = np.mean(preds_t_flat / preds_c_flat)
    median_hr = np.median(preds_t_flat / preds_c_flat)
    global_hr = mean_t / mean_c
    
    print("\n==========================================================================")
    print("   COUNTERFACTUAL INFERENCE (TRANSFORMER - SEMESTER LEVEL)")
    print("==========================================================================")
    print(f"  Ø Dropout-Risiko (Ohne Support)  : {mean_c:.4f}")
    print(f"  Ø Dropout-Risiko (Mit Support)   : {mean_t:.4f}")
    print(f"  Kausale HR (Mean HR)             : {mean_hr:.4f}")
    print(f"  Kausale HR (Median HR)           : {median_hr:.4f}")
    print(f"  Kausale HR (Global Ratio)        : {global_hr:.4f}")
    print("==========================================================================")

if __name__ == "__main__":
    run_counterfactual_transformer()
