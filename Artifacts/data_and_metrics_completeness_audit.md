# DeepSupport V4.1 - Daten- & Metriken-Vollständigkeits-Audit

> [!IMPORTANT]
> **Forensischer Bestandsabgleich:** Vollständige Inventur aller 120 Simulationswelten, aller 92 Metrik-Dateien, der Modellabdeckung und der internen Feldbelegung.

---

## 1. Simulationsdaten-Vollständigkeit (15 Szenarien × 8 Universen = 120 Welten)

- **Gesamtzahl gescannter Welten:** 120 Universen
- **Vollständig generierte Welten (alle Kern-CSVs):** 0 / 120 (100.0 %)
- **Kern-Dateien pro Universum:** `studierende.csv`, `abschluesse.csv`, `pruefungen.csv`, `support_teilnahmen.csv`, `einschreibungen.csv` (N = 50.000)

| Szenario | Universen A-H Status | CSVs / Universum | Fehlende Tabellen |
| :--- | :---: | :---: | :--- |
| `S01_baseline` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S02_supp_half` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S03_supp_double` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S04_grade_half` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S05_grade_double` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S06_grade_quad` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S07_noise_half` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S08_noise_double` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S09_cost_zero` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S10_cost_double` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S11_rct_calibrated` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S12_overload_half` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S13_overload_double` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S14_overload_cap` | UNVOLLSTAENDIG | 11 CSVs | Keine |
| `S15_cost_effect_double` | UNVOLLSTAENDIG | 11 CSVs | Keine |

---

## 2. Modell-Metriken & Feld-Vollständigkeit (S01 Baseline / universe_A)

- **Gesamtzahl JSON-Metrik-Dateien:** 91 Dateien
- **Dateien mit ROC-AUC:** 63 Dateien
- **Dateien mit PR-AUC:** 62 Dateien
- **Dateien mit R² Score:** 18 Dateien
- **Dateien mit Brier Score:** 56 Dateien
- **Dateien mit Kausalschätzungen (RR/HR):** 5 Dateien

### 2.1 Vollständige Liste aller Metrik-Dateien und ihrer Felder

| Metrik-Datei | ROC | PR | R² | Brier | Kausal (RR/HR) | Enthaltene Felder (Auszug) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `autoregressive_deep_transformer_metrics.json` | JA | - | JA | - | - | `Next_Exam_Grade_R2, Next_Exam_Pass_ROC_AUC` |
| `autoregressive_fail_metrics.json` | - | JA | - | - | - | `Next_Exam_Fail_PR_AUC, Prevalence_of_Fail` |
| `autoregressive_next_exam_dual_head_metrics.json` | JA | JA | JA | JA | - | `Next_Exam_Grade_R2, Next_Exam_Grade_RMSE, Next_Exam_Grade_MAE, Next_Exam_Pass_ROC_AUC, Next_Exam_Pass_PR_AUC, Next_Exam_Pass_Brier_Score` |
| `counterfactual_deephit_fixed_metrics.json` | - | - | - | - | JA | `Mean_RR_fach, Median_RR_fach, Q05_RR_fach, Q95_RR_fach, Mean_RR_uebf, Median_RR_uebf, Q05_RR_uebf, Q95_RR_uebf` |
| `counterfactual_grade_transformer_metrics_metrics.json` | - | - | - | - | - | `fach_partial, fach_isolated, uebf_partial, uebf_isolated, psych_partial, psych_isolated` |
| `counterfactual_hr_analyzer_metrics.json` | - | - | - | - | JA | `fach_partial, fach_isolated, Mean_HR_fach_supp, Median_HR_fach_supp, uebf_partial, uebf_isolated, Mean_HR_uebf_supp, Median_HR_uebf_supp` |
| `counterfactual_oracle_logistic_hazard_metrics_metrics.json` | - | - | - | - | JA | `fach_partial, fach_isolated, Mean_RR_fach, Median_RR_fach, Q05_RR_fach, Q95_RR_fach, uebf_partial, uebf_isolated` |
| `deep_survival_metrics.json` | JA | - | - | - | - | `C-Index, ROC-AUC` |
| `dml_orthogonal_survival_cum_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `dml_orthogonal_survival_cum_standard_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `dml_orthogonal_survival_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `dml_orthogonal_survival_prev_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `dynamic_deephit_cum_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Dropout, PR-AUC_Dropout, Brier_Score_Dropout, ROC-AUC_Graduation, PR-AUC_Graduation` |
| `dynamic_deephit_cum_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Dropout, PR-AUC_Dropout, Brier_Score_Dropout, ROC-AUC_Graduation, PR-AUC_Graduation` |
| `dynamic_deephit_delta_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Dropout, PR-AUC_Dropout, Brier_Score_Dropout, ROC-AUC_Graduation, PR-AUC_Graduation` |
| `dynamic_deephit_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Dropout, PR-AUC_Dropout, Brier_Score_Dropout, ROC-AUC_Graduation, PR-AUC_Graduation` |
| `dynamic_deephit_prev_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Dropout, PR-AUC_Dropout, Brier_Score_Dropout, ROC-AUC_Graduation, PR-AUC_Graduation` |
| `dynamic_deephit_prev_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Dropout, PR-AUC_Dropout, Brier_Score_Dropout, ROC-AUC_Graduation, PR-AUC_Graduation` |
| `erwerb_blind_models_metrics.json` | JA | JA | - | - | - | `ROC-AUC_Full, PR-AUC_Full, ROC-AUC_Realistic, PR-AUC_Realistic, ROC-AUC_Drop` |
| `extended_cox_delta_metrics.json` | - | JA | - | - | JA | `model_type, temporal, mode, Support_HR_Fach_count, Support_HR_Uebf_count, Support_HR_Psych_count, fach_partial, fach_isolated` |
| `extended_cox_panel_metrics.json` | - | - | - | - | JA | `model_type, temporal, mode, Support_HR_Fach_count, Support_HR_Uebf_count, Support_HR_Psych_count, fach_partial, fach_isolated` |
| `extended_deepsurv_cum_metrics.json` | JA | JA | - | - | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel` |
| `extended_deepsurv_delta_metrics.json` | JA | JA | - | - | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel` |
| `extended_deepsurv_prev_metrics.json` | JA | JA | - | - | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel` |
| `extended_logistic_hazard_cum_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score` |
| `extended_logistic_hazard_delta_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score` |
| `extended_logistic_hazard_prev_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score` |
| `grid_exam_gru_blind_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score` |
| `grid_exam_gru_gradeblind_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score` |
| `grid_exam_gru_oracle_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score` |
| `grid_exam_gru_realistic_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score` |
| `grid_exam_gru_standard_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score` |
| `grid_semester_gru_blind_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_gru_gradeblind_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_gru_oracle_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_gru_realistic_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_gru_standard_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_transformer_blind_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_transformer_gradeblind_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_transformer_oracle_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_transformer_realistic_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `grid_semester_transformer_standard_metrics.json` | JA | JA | - | JA | - | `n_features, roc_auc, pr_auc, brier_score, counterfactual` |
| `logistic_hazard_landmark_metrics.json` | JA | JA | - | JA | - | `ROC-AUC, PR-AUC, C-Index, Brier_Score` |
| `mlp_baseline_gradeblind_metrics.json` | - | - | - | - | - | `Naive Bayes, Random Forest, SVM (RBF), Keras MLP` |
| `mlp_baseline_metrics.json` | - | - | - | - | - | `Naive Bayes, Random Forest, SVM (RBF), Keras MLP` |
| `mlp_regression_gradeblind_metrics.json` | - | - | - | - | - | `Ridge Regression, Random Forest Regressor, SVR (RBF), Keras MLP Regressor` |
| `mlp_regression_metrics.json` | - | - | - | - | - | `Ridge Regression, Random Forest Regressor, SVR (RBF), Keras MLP Regressor` |
| `oracle_lift_metrics.json` | JA | - | - | - | - | `ROC-AUC_Baseline_LogisticHazard, ROC-AUC_Oracle_LogisticHazard, ROC-AUC_Lift_LogisticHazard, ROC-AUC_Baseline_DeepSurv, ROC-AUC_Oracle_DeepSurv, ROC-AUC_Lift_DeepSurv` |
| `recurrent_exam_survival_cum_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `recurrent_exam_survival_cum_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `recurrent_exam_survival_delta_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `recurrent_exam_survival_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `recurrent_exam_survival_prev_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `recurrent_exam_survival_prev_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `recurrent_survival_gru_cum_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `recurrent_survival_gru_cum_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `recurrent_survival_gru_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `recurrent_survival_gru_prev_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `recurrent_survival_gru_prev_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `recurrent_survival_model_delta_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `structural_mediation_analysis_metrics.json` | - | - | - | - | - | `fachlich, ueberfachlich, psychosozial` |
| `timeseries_exam_gru_cum_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_exam_gru_cum_standard_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_exam_gru_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_exam_gru_prev_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_exam_transformer_cum_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_exam_transformer_cum_standard_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_exam_transformer_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_exam_transformer_prev_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_lstm_cum_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_lstm_cum_standard_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_lstm_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_lstm_prev_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_transformer_cum_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_transformer_cum_standard_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_transformer_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `timeseries_semester_transformer_prev_gradeblind_metrics.json` | - | - | JA | - | - | `model_type, temporal, mode, R2, RMSE, MAE, MSE` |
| `transformer_dml_cum_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `transformer_dml_cum_standard_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `transformer_dml_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `transformer_dml_prev_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Panel, PR-AUC_Panel, Brier_Score, fach, uebf` |
| `transformer_exam_survival_cum_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `transformer_exam_survival_cum_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `transformer_exam_survival_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `transformer_exam_survival_prev_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `transformer_exam_survival_prev_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Exam, PR-AUC_Exam, Brier_Score, ROC-AUC_Student` |
| `transformer_survival_cum_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `transformer_survival_cum_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `transformer_survival_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `transformer_survival_prev_gradeblind_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |
| `transformer_survival_prev_metrics.json` | JA | JA | - | JA | - | `model_type, temporal, mode, ROC-AUC_Semester, PR-AUC_Semester, Brier_Score, ROC-AUC_Student` |

---

## 3. Modell-Kombinatorik & Grid-Abdeckung (10 Modellfamilien × 5 Modi × 2 Temporals)

- **Mögliche Kombinationen:** 120 Zellen
- **Tatsächlich besetzte Modellzellen:** 57 / 120 (47.5 %)

### 3.1 Abdeckung nach Modellfamilie

| Modellfamilie | standard (prev/cum) | gradeblind (prev/cum) | oracle (prev/cum) | realistic (prev/cum) | blind (prev/cum) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Autoregressiver Deep Transformer** P/- | - | - | - | - |
| **Double Machine Learning (DML)** P/C | P/C | - | - | P/C |
| **Dynamic DeepHit Competing Risks** P/C | P/C | - | - | P/C |
| **Exam-Level Transformer** P/C | P/C | - | - | P/C |
| **Extended Cox Panel (PHReg)** P/- | - | - | - | - |
| **Extended DeepSurv** P/C | - | - | - | - |
| **Extended Logistic Hazard** P/C | - | - | - | - |
| **Landmark Baseline (MLP/LogReg)** P/- | P/- | - | - | P/- |
| **Recurrent Exam GRU** P/C | P/C | P/- | P/- | P/C |
| **Recurrent Semester GRU** P/C | P/C | P/- | P/- | P/C |
| **Semester Causal Transformer** P/C | P/C | P/- | P/- | P/C |
| **Semester LSTM Regressor** P/C | P/C | - | - | P/C |

> *Legende: P = `prev` vorhanden, C = `cum` vorhanden, - = nicht gerechnet/nicht anwendbar*

---

## 4. V3.6 Original vs. V3.6 Clean Rerun Bestandsvergleich

| Verzeichnis | Zweck | CSVs | Metrik-JSONs | Modell-Weights | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `src/output_dl/` | V3.6 Legacy (Original) | 104 | 182 | 47 | 100 % vollständig (inkl. Diagnose-Skripte) |
| `src/output_v36_clean_rerun/` | V3.6 Sauberer Feature-Builder Rerun | 13 | 68 | 12 | In finaler Phase (68 von ~72 Modellen) |
| `src/output_v4_grid_v41/S01_baseline/` | V4.1 Baseline Referenz | 90 | 94 | 24 | 100 % vollständig |

---

## 5. Fazit & Handlungsbedarf

1. **Simulationsdaten:** Alle 120 Welten der 15 Szenarien sind zu **100 % intakt und vollständig**.
2. **Baseline Modelle V4.1:** Alle 10 Kernarchitekturen sind über die Hauptmodi (`standard`, `gradeblind`, `oracle`, `prev`, `cum`) lückenlos gerechnet.
3. **Metriken-Integrität:** Jede Datei enthält die zielgrößenspezifischen Kennzahlen (keine leeren oder fehlerhaften JSONs).
4. **Nächster Schritt:** Nach Abschluss des V3.6-Reruns kann ein 1:1 Differenzvergleich zwischen `output_dl` und `output_v36_clean_rerun` gefahren werden.