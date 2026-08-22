"""
Master Retraining & Counterfactual Analysis Pipeline (V3.3 Count-Feature Edition)
==================================================================================
Führt das vollständige Retraining aller 13+ Überlebens-, Regressions- und Kausalmodelle
auf den neuen Zähl-Expositionsmerkmalen durch und führt anschließend die gesamte
kontrafaktische Dual-Strang-Evaluationssuite (Partiell vs. Isoliert Realistisch) aus.
"""

import os
import sys
import time
from pathlib import Path

# Projekt-Pfade einbinden
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(SRC_DIR)
data_dir = Path("output_dl") if Path("output_dl").exists() else Path("../output_dl")

def run_step(step_name, func, *args, **kwargs):
    print("\n" + "=" * 80)
    print(f"   START: {step_name}")
    print(f"   Zeit: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    t0 = time.time()
    try:
        res = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n[OK] {step_name} ERFOLGREICH BEENDET ({elapsed/60:.2f} Min. / {elapsed:.1f} Sek.)")
        return res
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[FEHLER] {step_name} FEHLGESCHLAGEN nach {elapsed:.1f} Sek.: {e}")
        import traceback
        traceback.print_exc()

def main():
    total_start = time.time()
    print("*" * 80)
    print("   MASTER NACHTLAUF: MODELL-TRAINING & KONTRAFAKTISCHE ANALYSEN (V3.3)")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("*" * 80)
    
    # -------------------------------------------------------------------------
    # 1. TRADITIONELLE SURVIVAL- UND HAZARD-MODELLE
    # -------------------------------------------------------------------------
    from extended_cox_survival import train_extended_cox_model
    run_step("1. Extended Cox Proportional Hazards (Panel)", train_extended_cox_model, data_dir)
    
    from extended_deep_survival import train_extended_deep_survival
    run_step("2. Extended DeepSurv (Panel, Breslow Loss)", train_extended_deep_survival, data_dir)
    
    from extended_cox_delta import fit_extended_cox_delta
    run_step("3. Extended Cox Delta (Semester-Lokal)", fit_extended_cox_delta, data_dir)
    
    from extended_deep_survival_delta import train_extended_deep_survival_delta
    run_step("4. Extended DeepSurv Delta (Semester-Lokal, Breslow Loss)", train_extended_deep_survival_delta, data_dir)
    
    # -------------------------------------------------------------------------
    # 2. REKURRIERENDE SURVIVAL- UND COMPETING-RISKS-MODELLE
    # -------------------------------------------------------------------------
    from recurrent_survival_model import train_recurrent_survival_model
    run_step("5. Recurrent Survival GRU (Semester-Ebene)", train_recurrent_survival_model, data_dir)
    
    from recurrent_survival_model_delta import train_recurrent_survival_model_delta
    run_step("6. Recurrent Survival GRU Delta (Semester-Ebene)", train_recurrent_survival_model_delta, data_dir)
    
    from transformer_survival_model import train_causal_transformer_survival
    run_step("7. Transformer Survival (Semester-Ebene)", train_causal_transformer_survival, data_dir)
    
    from dynamic_deephit_delta_model import train_dynamic_deephit_delta_model
    run_step("8. Dynamic DeepHit Delta (Competing Risks)", train_dynamic_deephit_delta_model, data_dir)
    
    from recurrent_exam_survival import train_recurrent_exam_survival
    run_step("9. Recurrent Exam Survival GRU Base (9 Features)", train_recurrent_exam_survival, data_dir)
    
    from recurrent_exam_survival_v2 import train_recurrent_exam_survival_v2
    run_step("10. Recurrent Exam Survival GRU V2 (12 Features)", train_recurrent_exam_survival_v2, data_dir)
    
    from recurrent_exam_survival_delta import train_recurrent_exam_survival_delta
    run_step("11. Recurrent Exam Survival GRU Delta (12 Features)", train_recurrent_exam_survival_delta, data_dir)
    
    # -------------------------------------------------------------------------
    # 3. DEEP TRANSFORMER REGRESSION & DUAL SURVIVAL (OPTION A + B)
    # -------------------------------------------------------------------------
    from deep_transformer_regression import train_deep_transformer_regression
    run_step("12. Deep Transformer Suite (Semester Regressor, Exam Regressor, Causal Hazard Option A, Masked Static Option B)", train_deep_transformer_regression, data_dir, data_dir)
    
    # -------------------------------------------------------------------------
    # 4. ORACLE- UND KAUSALE DML-BENCHMARKS
    # -------------------------------------------------------------------------
    from train_oracle_models import train_oracle_models
    run_step("13. Oracle Baseline Models (Vollständige Latente Variablen)", train_oracle_models, data_dir)
    
    from dml_orthogonal_survival import train_dml_orthogonal_survival
    run_step("14. Double Machine Learning (Orthogonal Causal Ridge Pipeline)", train_dml_orthogonal_survival, data_dir)
    
    # -------------------------------------------------------------------------
    # 5. KONTRAFAKTISCHE INFERENZ SUITE (DUAL-TESTSTRANG: PARTIELL + ISOLIERT)
    # -------------------------------------------------------------------------
    from counterfactual_hr_analyzer import analyze_counterfactual_hr
    run_step("15. Counterfactual HR Extended DeepSurv Panel", analyze_counterfactual_hr, data_dir)
    
    from counterfactual_hr_delta import analyze_counterfactual_hr_delta
    run_step("16. Counterfactual HR Extended DeepSurv Delta", analyze_counterfactual_hr_delta, data_dir)
    
    from counterfactual_rr_logistic_hazard_delta import analyze_counterfactual_rr_logistic_hazard_delta
    run_step("17. Counterfactual RR Logistic Hazard Delta", analyze_counterfactual_rr_logistic_hazard_delta, data_dir)
    
    from counterfactual_rr_deephit_delta import main as run_cf_deephit
    run_step("18. Counterfactual RR Dynamic DeepHit Delta", run_cf_deephit)
    
    from counterfactual_inference_semester_transformer import run_counterfactual_transformer
    run_step("19. Counterfactual HR Semester Transformer", run_counterfactual_transformer)
    
    from counterfactual_rnn_delta import main as run_cf_rnn_v2
    run_step("20. Counterfactual RR Exam GRU V2 Delta", run_cf_rnn_v2)
    
    from counterfactual_rr_exam_rnn_delta import main as run_cf_exam_rnn_delta
    run_step("21. Counterfactual RR Exam GRU Delta", run_cf_exam_rnn_delta)
    
    from counterfactual_rnn_semester_delta import main as run_cf_sem_delta
    run_step("22. Counterfactual RR Semester GRU Delta (13 Features)", run_cf_sem_delta)
    
    # -------------------------------------------------------------------------
    # 6. ORACLE COUNTERFACTUAL INFERENZ & NOTEN-/BESTEHENSEFFEKTE
    # -------------------------------------------------------------------------
    from counterfactual_oracle_logistic_hazard import analyze_counterfactual_oracle_logistic_hazard
    run_step("23. Counterfactual RR Oracle Logistic Hazard (Latente Variablen)", analyze_counterfactual_oracle_logistic_hazard, data_dir)
    
    from counterfactual_oracle_deepsurv import analyze_counterfactual_oracle_deepsurv
    run_step("24. Counterfactual HR Oracle DeepSurv (Latente Variablen)", analyze_counterfactual_oracle_deepsurv, data_dir)
    
    from grade_effect_linear import analyze_grade_effects_linear
    run_step("25. Lineare Noteneffekt-Analyse (OLS auf Prüfungsebene)", analyze_grade_effects_linear, data_dir)
    
    from pass_rate_analysis import analyze_pass_rates
    run_step("26. Bestehensquoten-Analyse (Logit auf Prüfungsebene)", analyze_pass_rates, data_dir)
    
    from counterfactual_grade_transformer import analyze_counterfactual_grade_transformer
    run_step("27. Kontrafaktische Notenanlyse (Deep Exam Transformer Regressor)", analyze_counterfactual_grade_transformer, data_dir)
    
    total_elapsed = time.time() - total_start
    print("\n" + "*" * 80)
    print("   MASTER NACHTLAUF VOLLSTÄNDIG ABGESCHLOSSEN (27 SCHRITTE)!")
    print(f"   Gesamtdauer: {total_elapsed/60:.2f} Minuten ({total_elapsed/3600:.2f} Stunden)")
    print(f"   Ende: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("*" * 80)

if __name__ == "__main__":
    main()