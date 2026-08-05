"""
Master Runner Script: Alle Modell-Trainings & Experimente nacheinander ausführen
================================================================================
Führt alle 15+ Modellvarianten (inkl. Notenblinden Klassifikationen & Regressoren)
nacheinander aus, speichert Metriken (.json & .md), Keras-Modelle (.keras) und Plots (.png).
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
    print("   STARTE COMPLETE MODELL-SUITE (STANDARD & NOTENBLINDE VARIANTEN)")
    print("=" * 80)
    
    data_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    
    # 1. Baseline MLP & ML Classifiers (Standard & Blind)
    print("\n>>> [1/12] Trainere MLP Baseline Classification (Standard & Blind) ...")
    from train_mlp_baseline import main as run_mlp_baseline
    run_mlp_baseline(blind=False)
    run_mlp_baseline(blind=True)
    
    # 2. MLP Regression
    print("\n>>> [2/12] Trainiere MLP Regression ...")
    from train_mlp_regression import main as run_mlp_regression
    run_mlp_regression()
    
    # 3. Semester LSTM Regressor
    print("\n>>> [3/12] Trainiere Semester-LSTM Regressor ...")
    from timeseries_semester import main as run_ts_semester
    run_ts_semester()
    
    # 4. Semester Transformer Regressor
    print("\n>>> [4/12] Trainiere Semester-Transformer Regressor ...")
    from timeseries_semester_transformer import main as run_ts_semester_tf
    run_ts_semester_tf()
    
    # 5. Exam GRU Regressor
    print("\n>>> [5/12] Trainiere Exam-GRU Regressor ...")
    from timeseries_exam import main as run_ts_exam
    run_ts_exam()
    
    # 6. Exam Transformer Regressor
    print("\n>>> [6/12] Trainiere Exam-Transformer Regressor ...")
    from timeseries_exam_transformer import main as run_ts_exam_tf
    run_ts_exam_tf()
    
    # 7. Recurrent Survival GRU (Standard & Blind)
    print("\n>>> [7/12] Trainiere Recurrent Survival GRU (Standard & Blind) ...")
    from recurrent_survival_model import train_recurrent_survival_model
    train_recurrent_survival_model(data_dir, blind=False)
    train_recurrent_survival_model(data_dir, blind=True)
    
    # 8. Recurrent Exam Survival GRU
    print("\n>>> [8/12] Trainiere Recurrent Exam Survival GRU ...")
    from recurrent_exam_survival import train_recurrent_exam_survival
    train_recurrent_exam_survival(data_dir)
    
    # 9. Causal Transformer Survival (Standard & Blind)
    print("\n>>> [9/12] Trainiere Causal Transformer Survival (Standard) ...")
    from transformer_survival_model import train_causal_transformer_survival
    train_causal_transformer_survival(data_dir)
    
    # 10. Exam Causal Transformer Survival
    print("\n>>> [10/12] Trainiere Causal Exam-Transformer Survival ...")
    from transformer_exam_survival import main as run_tf_exam_surv
    run_tf_exam_surv()
    
    # 11. Dynamic DeepHit Competing Risks
    print("\n>>> [11/12] Trainiere Dynamic DeepHit Competing Risks ...")
    from dynamic_deephit_model import train_dynamic_deephit_model
    train_dynamic_deephit_model(data_dir)
    
    # 12. DeepSurv Landmark Survival
    print("\n>>> [12/12] Trainiere DeepSurv & Logistic Hazard Landmark ...")
    from deep_survival import main as run_deep_surv
    run_deep_surv()
    
    print("\n" + "=" * 80)
    print("   ALLE MODELL-TRAININGS ERFOLGREICH BEENDET!")
    print("=" * 80)

if __name__ == '__main__':
    run_all()
