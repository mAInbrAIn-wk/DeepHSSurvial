"""
Feature Grid Master Evaluation & Benchmark Pipeline
===================================================
Orchestriert das Cross-Szenario Training (S01-S15).
Strikte Trennung von Input-Daten und Output-Modellen/Metriken.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import deepsupport.data_engine.feature_builder as fb
from deepsupport.evaluation.metrics_logger import save_metrics
from deepsupport.models.semester_gru import build_gru_model, masked_binary_crossentropy
from deepsupport.models.semester_transformer import build_causal_transformer_survival_model

PADDING_VALUE = -999.0
MODES = ["standard", "gradeblind", "blind", "oracle", "realistic"]

def run_causal_evaluation(model, X_test, valid_mask_test, f_indices):
    cf_results = {}
    supp_configs = [
        ('fach', 'fach_supp', 'Fachlicher Support'),
        ('uebf', 'uebf_supp', 'Überfachlicher Support'),
        ('psych', 'psych_supp', 'Psychosozialer Support')
    ]
    
    for prefix, key, s_name in supp_configs:
        feat_idx = f_indices.get(key)
        if feat_idx is None:
            cf_results[f"{prefix}_partial"] = {"mean_rr": None, "median_rr": None}
            cf_results[f"{prefix}_isolated"] = {"mean_rr": None, "median_rr": None}
            continue
            
        X_c_part = X_test.copy()
        X_t_part = X_test.copy()
        X_c_part[valid_mask_test, feat_idx] = 0.0
        
        p0_p = model.predict(X_c_part, verbose=0).flatten()[valid_mask_test.flatten()]
        p1_p = model.predict(X_t_part, verbose=0).flatten()[valid_mask_test.flatten()]
        rrs_p = p1_p / np.clip(p0_p, 1e-7, 1.0)
        
        X_c_iso = X_test.copy()
        X_t_iso = X_test.copy()
        for _, other_key, _ in supp_configs:
            other_idx = f_indices.get(other_key)
            if other_idx is not None:
                X_c_iso[valid_mask_test, other_idx] = 0.0
                X_t_iso[valid_mask_test, other_idx] = 0.0
        X_t_iso[valid_mask_test, feat_idx] = X_test[valid_mask_test, feat_idx]
        
        p0_i = model.predict(X_c_iso, verbose=0).flatten()[valid_mask_test.flatten()]
        p1_i = model.predict(X_t_iso, verbose=0).flatten()[valid_mask_test.flatten()]
        rrs_i = p1_i / np.clip(p0_i, 1e-7, 1.0)
        
        cf_results[f"{prefix}_partial"] = {"mean_rr": float(np.mean(rrs_p)), "median_rr": float(np.median(rrs_p))}
        cf_results[f"{prefix}_isolated"] = {"mean_rr": float(np.mean(rrs_i)), "median_rr": float(np.median(rrs_i))}
        
    return cf_results

def run_grid_evaluation(architecture_name: str, build_model_fn, data_dir: Path, output_dir: Path, max_timesteps: int, level: str = 'semester'):
    results = {}
    
    for mode in MODES:
        print(f"\n>>> Trainiere {architecture_name} im Modus: [{mode.upper()}] ...")
        
        if level == 'semester':
            studis, X_seq, y_seq, studi_events, f_names, f_indices = fb.build_semester_sequence_tensor(
                data_dir, max_semesters=max_timesteps, mode=mode
            )
        else:
            studis, X_seq, y_seq, studi_events, f_names, f_indices = fb.build_exam_sequence_tensor(
                data_dir, max_exams=max_timesteps, mode=mode
            )
            
        N, T, F = X_seq.shape
        
        train_idx, temp_idx, _, y_temp = train_test_split(
            np.arange(N), studi_events, test_size=0.30, random_state=42, stratify=studi_events
        )
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
        
        X_train, X_val, X_test = X_seq[train_idx].copy(), X_seq[val_idx].copy(), X_seq[test_idx].copy()
        y_train, y_val, y_test = y_seq[train_idx], y_seq[val_idx], y_seq[test_idx]
        
        scaler = StandardScaler()
        valid_mask_train = (X_train[:, :, 0] != PADDING_VALUE)
        scaler.fit(X_train[valid_mask_train])
        
        for X_split in [X_train, X_val, X_test]:
            valid_mask = (X_split[:, :, 0] != PADDING_VALUE)
            X_split[valid_mask] = scaler.transform(X_split[valid_mask])
            
        tf.random.set_seed(42)
        model = build_model_fn(sequence_length=T, feature_dim=F)
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss=masked_binary_crossentropy)
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=128,
            callbacks=[es],
            verbose=0
        )
        
        valid_mask_test = (X_test[:, :, 0] != PADDING_VALUE)
        y_pred = model.predict(X_test, verbose=0)
        
        y_true_flat = y_test[valid_mask_test]
        y_pred_flat = y_pred[valid_mask_test]
        
        auc = roc_auc_score(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else None
        prauc = average_precision_score(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else None
        brier = brier_score_loss(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else None
        
        print(f"   --> PR-AUC: {prauc:.4f} | ROC-AUC: {auc:.4f} | Brier: {brier:.4f}")
        
        cf_results = run_causal_evaluation(model, X_test, valid_mask_test, f_indices)
        
        results[mode] = {
            "n_features": F,
            "roc_auc": auc,
            "pr_auc": prauc,
            "brier_score": brier,
            "counterfactual": cf_results
        }
        
        # Save dynamically using the new metrics_logger to the OUTPUT DIR
        save_metrics(architecture_name, results[mode], output_dir, mode=mode, temporal_type='flat')
        
    return results

def main(data_root: Optional[Union[str, Path]] = None, output_root: Optional[Union[str, Path]] = None):
    data_root = Path(data_root) if data_root else Path('data_v4_grid')
    output_root = Path(output_root) if output_root else Path('output_v4_models')
    
    if not data_root.exists():
        raise FileNotFoundError(f"Datenquelle {data_root} nicht gefunden!")
        
    print(f"Starte Master Grid Orchestrierung...")
    print(f"Lese Rohdaten von: {data_root}")
    print(f"Schreibe Modelle/Metriken nach: {output_root}")
    
    # 1. Finde alle Szenarien (S01_baseline, S02_supp_half, etc.)
    scenarios = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('S')])
    if not scenarios:
        # Fallback: Wenn data_root direkt auf ein Szenario zeigt
        scenarios = [data_root]
        
    for scenario_dir in scenarios:
        scenario_name = scenario_dir.name
        print("\n" + "="*80)
        print(f"   SZENARIO: {scenario_name}")
        print("="*80)
        
        scenario_out = output_root / scenario_name
        scenario_out.mkdir(parents=True, exist_ok=True)
        
        # Core Models trainieren und isoliert in scenario_out speichern
        run_grid_evaluation('grid_semester_gru', build_gru_model, scenario_dir, scenario_out, max_timesteps=16, level='semester')
        run_grid_evaluation('grid_semester_transformer', build_causal_transformer_survival_model, scenario_dir, scenario_out, max_timesteps=16, level='semester')
        run_grid_evaluation('grid_exam_gru', build_gru_model, scenario_dir, scenario_out, max_timesteps=40, level='exam')

if __name__ == "__main__":
    main()
