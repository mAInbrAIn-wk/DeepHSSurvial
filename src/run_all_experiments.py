"""
Master Runner Script: Alle Modell-Trainings & Experimente nacheinander ausführen
================================================================================
Führt alle 20+ Modellvarianten nacheinander aus, speichert Metriken (.json & .md),
Keras-Modelle (.keras) und Plots (.png).

Erweitert um: Extended-Modelle (Cox, DeepSurv, LogHaz), DML, Delta-Varianten,
Counterfactual-Analysen, Oracle-Modelle und Kalibrierungskurven.
"""

import os
import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def run_all():
    print("=" * 80)
    print("   STARTE COMPLETE MODELL-SUITE (STANDARD, EXTENDED & KAUSAL-VARIANTEN)")
    print("=" * 80)
    
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    
    # =========================================================================
    # STUFE 1: Baseline MLP & ML Classifiers
    # =========================================================================
    print("\n>>> [1/20] Trainere MLP Baseline Classification (Standard & Blind) ...")
    from train_mlp_baseline import main as run_mlp_baseline
    run_mlp_baseline(blind=False)
    run_mlp_baseline(blind=True)
    
    print("\n>>> [2/20] Trainiere MLP Regression ...")
    from train_mlp_regression import main as run_mlp_regression
    run_mlp_regression()
    
    # =========================================================================
    # STUFE 2: Zeitreihen-Regressoren
    # =========================================================================
    print("\n>>> [3/20] Trainiere Semester-LSTM Regressor ...")
    from timeseries_semester import main as run_ts_semester
    run_ts_semester()
    
    print("\n>>> [4/20] Trainiere Semester-Transformer Regressor ...")
    from timeseries_semester_transformer import main as run_ts_semester_tf
    run_ts_semester_tf()
    
    print("\n>>> [5/20] Trainiere Exam-GRU Regressor ...")
    from timeseries_exam import main as run_ts_exam
    run_ts_exam()
    
    print("\n>>> [6/20] Trainiere Exam-Transformer Regressor ...")
    from timeseries_exam_transformer import main as run_ts_exam_tf
    run_ts_exam_tf()
    
    # =========================================================================
    # STUFE 3: Survival-Modelle (Recurrent & Transformer)
    # =========================================================================
    print("\n>>> [7/20] Trainiere Recurrent Survival GRU (Standard & Blind) ...")
    from recurrent_survival_model import train_recurrent_survival_model
    train_recurrent_survival_model(data_dir, blind=False)
    train_recurrent_survival_model(data_dir, blind=True)
    
    print("\n>>> [8/20] Trainiere Recurrent Exam Survival GRU ...")
    from recurrent_exam_survival import train_recurrent_exam_survival
    train_recurrent_exam_survival(data_dir)
    
    print("\n>>> [9/20] Trainiere Causal Transformer Survival ...")
    from transformer_survival_model import train_causal_transformer_survival
    train_causal_transformer_survival(data_dir)
    
    print("\n>>> [10/20] Trainiere Causal Exam-Transformer Survival ...")
    from transformer_exam_survival import main as run_tf_exam_surv
    run_tf_exam_surv()
    
    # =========================================================================
    # STUFE 4: Dynamic DeepHit & Competing Risks
    # =========================================================================
    print("\n>>> [11/20] Trainiere Dynamic DeepHit Competing Risks ...")
    from dynamic_deephit_model import train_dynamic_deephit_model
    train_dynamic_deephit_model(data_dir)
    
    print("\n>>> [12/20] Trainiere Dynamic DeepHit Delta ...")
    from dynamic_deephit_delta_model import train_dynamic_deephit_delta_model
    train_dynamic_deephit_delta_model(data_dir)
    
    # =========================================================================
    # STUFE 5: DeepSurv & Logistic Hazard (Landmark + Extended)
    # =========================================================================
    print("\n>>> [13/20] Trainiere DeepSurv & Logistic Hazard Landmark ...")
    from deep_survival import main as run_deep_surv
    run_deep_surv()
    
    print("\n>>> [14/20] Trainiere Extended DeepSurv & Logistic Hazard (Panel) ...")
    from extended_deep_survival import train_extended_deep_survival
    train_extended_deep_survival(data_dir)
    
    print("\n>>> [15/20] Trainiere Extended DeepSurv & Logistic Hazard (Delta) ...")
    from extended_deep_survival_delta import train_extended_deep_survival_delta
    train_extended_deep_survival_delta(data_dir)
    
    print("\n>>> [16/20] Trainiere Extended Exam Survival ...")
    from extended_exam_survival import train_extended_exam_survival
    train_extended_exam_survival(data_dir)
    
    # =========================================================================
    # STUFE 6: DML (Double Machine Learning)
    # =========================================================================
    print("\n>>> [17/20] Trainiere DML Orthogonal Survival ...")
    from dml_orthogonal_survival import train_dml_orthogonal_survival
    train_dml_orthogonal_survival(data_dir)
    
    # =========================================================================
    # STUFE 7: Delta-Varianten (Recurrent)
    # =========================================================================
    print("\n>>> [18/20] Trainiere Recurrent Survival Model Delta ...")
    from recurrent_survival_model_delta import train_recurrent_survival_model_delta
    train_recurrent_survival_model_delta(data_dir)
    
    print("\n>>> [19/20] Trainiere Recurrent Exam Survival Delta ...")
    from recurrent_exam_survival_delta import train_recurrent_exam_survival_delta
    train_recurrent_exam_survival_delta(data_dir)
    
    # =========================================================================
    # STUFE 8: Deep Transformer Regression & Survival (Enlarged Capacity)
    # =========================================================================
    print("\n>>> [20/21] Trainiere Deep Transformer Regression & Survival (d_model=128, Attention Pooling) ...")
    from deep_transformer_regression import train_deep_transformer_regression
    train_deep_transformer_regression(data_dir, data_dir)
    
    # =========================================================================
    # STUFE 9: Oracle-Modelle & Kalibrierung
    # =========================================================================
    print("\n>>> [21/21] Trainiere Oracle-Modelle & Kalibrierungskurven ...")
    from train_oracle_models import train_oracle_models
    train_oracle_models(data_dir)
    
    from plot_calibration_curves import main as plot_calibration_curves
    plot_calibration_curves()
    
    print("\n" + "=" * 80)
    print("   ALLE MODELL-TRAININGS ERFOLGREICH BEENDET!")
    print("=" * 80)

if __name__ == '__main__':
    run_all()
