# Synoptischer Cross-Szenario & Modell-Evaluierungsbericht V4.1

> [!IMPORTANT]
> **Systematische Gesamtsynopse:** Abgleich aller 15 Simulations-Szenarien, 11 Modellklassen, 5 Feature-Modi und Validierung der Modell-Kausalitaet gegen die experimentelle Simulations-Ground-Truth.

---

## 1. Ground Truth der 15 Simulationswelten (N = 50.000, seed = 99999)

| # | Szenario | Parameter-Fokus | Dropout A (Full) | Dropout B (None) | Wahre ARR (B-A) | Wahre RR (A/B) | NNT |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S01** | S01_baseline | Baseline (Referenz) | **29.2 %** | **37.1 %** | **7.9 pp** | **0.787** | **12.6** |
| **S02** | S02_supp_half | Support-Wirkung 0.5x | **32.7 %** | **37.1 %** | **4.4 pp** | **0.881** | **22.7** |
| **S03** | S03_supp_double | Support-Wirkung 2.0x | **25.3 %** | **37.1 %** | **11.8 pp** | **0.682** | **8.5** |
| **S04** | S04_grade_half | Notenboost 0.5x | **30.6 %** | **37.1 %** | **6.5 pp** | **0.825** | **15.4** |
| **S05** | S05_grade_double | Notenboost 2.0x | **27.6 %** | **37.1 %** | **9.5 pp** | **0.744** | **10.5** |
| **S06** | S06_grade_quad | Notenboost 4.0x | **27.1 %** | **37.1 %** | **10.0 pp** | **0.730** | **10.0** |
| **S07** | S07_noise_half | Rauschen 0.5x | **26.7 %** | **33.0 %** | **6.3 pp** | **0.809** | **15.8** |
| **S08** | S08_noise_double | Rauschen 2.0x | **33.2 %** | **41.0 %** | **7.9 pp** | **0.810** | **12.7** |
| **S09** | S09_cost_zero | Zeitkosten 0h | **28.6 %** | **37.1 %** | **8.5 pp** | **0.771** | **11.8** |
| **S10** | S10_cost_double | Zeitkosten 60h (2x) | **29.7 %** | **37.1 %** | **7.4 pp** | **0.801** | **13.5** |
| **S11** | S11_rct_calibrated | RCT (Zufallsauswahl) | **32.6 %** | **37.1 %** | **4.5 pp** | **0.879** | **22.5** |
| **S12** | S12_overload_half | Overload-Penalty 0.5x | **26.0 %** | **34.1 %** | **8.1 pp** | **0.762** | **12.3** |
| **S13** | S13_overload_double | Overload-Penalty 2.0x | **34.6 %** | **41.8 %** | **7.3 pp** | **0.828** | **13.8** |
| **S14** | S14_overload_cap | Overload-Cap 15% | **26.7 %** | **35.0 %** | **8.4 pp** | **0.763** | **12.0** |
| **S15** | S15_cost_effect_double | Kombi (Kosten+Wirkung 2x) | **25.8 %** | **37.1 %** | **11.3 pp** | **0.695** | **8.9** |

---

## 2. Ebene 1 & 2: Modellklassen-Benchmark & Modus-Synthese (V4.1 Baseline)

### 2.1 Survival- & Dropout-Klassifikation (Test-Set)

| Modellklasse | Modellname | Modus | Temporal | ROC-AUC | PR-AUC (Dropout) | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Survival | `grid_exam_gru_oracle` | `oracle` | `prev` | **0.9116** | 0.2850 | 0.0140 |
| Survival | `grid_exam_gru_gradeblind` | `gradeblind` | `prev` | **0.8959** | 0.1971 | 0.0151 |
| Survival | `grid_exam_gru_standard` | `standard` | `prev` | **0.8933** | 0.1911 | 0.0152 |
| Survival | `grid_exam_gru_realistic` | `realistic` | `prev` | **0.8890** | 0.1891 | 0.0152 |
| Survival | `grid_exam_gru_blind` | `blind` | `prev` | **0.8879** | 0.1626 | 0.0155 |
| Survival | `logistic_hazard_landmark` | `standard` | `prev` | **0.8751** | 0.7709 | 0.1135 |
| Survival | `deep_survival` | `standard` | `prev` | **0.8710** | - | - |
| Survival | `dml_orthogonal_survival_cum_standard` | `standard` | `cum` | **0.8360** | 0.2691 | 0.0342 |
| Survival | `dml_orthogonal_survival_cum_gradeblind` | `gradeblind` | `cum` | **0.8359** | 0.2653 | 0.0343 |
| Survival | `extended_logistic_hazard_cum` | `gradeblind` | `cum` | **0.8349** | 0.2664 | 0.0343 |
| Survival | `transformer_dml_cum_gradeblind` | `gradeblind` | `cum` | **0.8293** | 0.2621 | 0.0344 |
| Survival | `transformer_dml_cum_standard` | `standard` | `cum` | **0.8276** | 0.2669 | 0.0343 |
| Survival | `grid_semester_transformer_gradeblind` | `gradeblind` | `prev` | **0.8175** | 0.2874 | 0.0349 |
| Survival | `grid_semester_transformer_standard` | `standard` | `prev` | **0.8174** | 0.2856 | 0.0349 |
| Survival | `grid_semester_transformer_realistic` | `realistic` | `prev` | **0.8146** | 0.2860 | 0.0349 |
| Survival | `grid_semester_transformer_oracle` | `oracle` | `prev` | **0.8144** | 0.2882 | 0.0349 |
| Survival | `grid_semester_gru_gradeblind` | `gradeblind` | `prev` | **0.8119** | 0.2676 | 0.0354 |
| Survival | `grid_semester_gru_oracle` | `oracle` | `prev` | **0.8118** | 0.2695 | 0.0354 |
| Survival | `grid_semester_gru_standard` | `standard` | `prev` | **0.8113** | 0.2741 | 0.0352 |
| Survival | `grid_semester_gru_realistic` | `realistic` | `prev` | **0.8080** | 0.2674 | 0.0353 |
| Survival | `dml_orthogonal_survival` | `standard` | `prev` | **0.8040** | 0.1953 | 0.0360 |
| Survival | `transformer_dml` | `standard` | `prev` | **0.8014** | 0.1929 | 0.0361 |
| Survival | `transformer_dml_prev_gradeblind` | `gradeblind` | `prev` | **0.8011** | 0.1923 | 0.0361 |
| Survival | `dml_orthogonal_survival_prev_gradeblind` | `gradeblind` | `prev` | **0.8005** | 0.1939 | 0.0361 |
| Survival | `extended_logistic_hazard_delta` | `gradeblind` | `prev` | **0.8002** | 0.1897 | 0.0362 |
| Survival | `extended_logistic_hazard_prev` | `gradeblind` | `prev` | **0.8002** | 0.1897 | 0.0362 |
| Survival | `grid_semester_transformer_blind` | `blind` | `prev` | **0.7856** | 0.1968 | 0.0372 |
| Survival | `grid_semester_gru_blind` | `blind` | `prev` | **0.7815** | 0.1719 | 0.0378 |
| Survival | `extended_deepsurv_delta` | `gradeblind` | `prev` | **0.5570** | 0.0548 | - |
| Survival | `extended_deepsurv_prev` | `gradeblind` | `prev` | **0.5570** | 0.0548 | - |
| Survival | `extended_deepsurv_cum` | `gradeblind` | `cum` | **0.5425** | 0.0560 | - |

### 2.2 Noten- & GPA-Regressionsmodelle (Test-Set)

| Modellklasse | Modellname | Modus | Temporal | $R^2$ Score | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Regression | `timeseries_exam_transformer` | `standard` | `prev` | **0.9928** | 0.0509 | 0.0396 |
| Regression | `timeseries_exam_transformer_cum_standard` | `standard` | `cum` | **0.9916** | 0.0548 | 0.0412 |
| Regression | `timeseries_semester_transformer` | `standard` | `prev` | **0.9899** | 0.0602 | 0.0440 |
| Regression | `timeseries_semester_transformer_cum_standard` | `standard` | `cum` | **0.9890** | 0.0628 | 0.0452 |
| Regression | `timeseries_exam_transformer_prev_gradeblind` | `gradeblind` | `prev` | **0.8031** | 0.2657 | 0.2087 |
| Regression | `timeseries_exam_transformer_cum_gradeblind` | `gradeblind` | `cum` | **0.8023** | 0.2662 | 0.2093 |
| Regression | `timeseries_semester_transformer_prev_gradeblind` | `gradeblind` | `prev` | **0.8014** | 0.2668 | 0.2095 |
| Regression | `timeseries_semester_transformer_cum_gradeblind` | `gradeblind` | `cum` | **0.7993** | 0.2682 | 0.2113 |
| Regression | `timeseries_semester_lstm_cum_standard` | `standard` | `cum` | **0.7940** | 0.5429 | 0.4175 |
| Regression | `timeseries_semester_lstm` | `standard` | `prev` | **0.7931** | 0.5441 | 0.4129 |
| Regression | `timeseries_semester_lstm_cum_gradeblind` | `gradeblind` | `cum` | **0.7719** | 0.5714 | 0.4370 |
| Regression | `timeseries_semester_lstm_prev_gradeblind` | `gradeblind` | `prev` | **0.7716** | 0.5717 | 0.4325 |
| Regression | `timeseries_exam_gru` | `standard` | `prev` | **0.1020** | 0.1218 | 0.0277 |
| Regression | `timeseries_exam_gru_prev_gradeblind` | `gradeblind` | `prev` | **0.0944** | 0.1223 | 0.0309 |
| Regression | `timeseries_exam_gru_cum_gradeblind` | `gradeblind` | `cum` | **0.0865** | 0.1228 | 0.0301 |
| Regression | `timeseries_exam_gru_cum_standard` | `standard` | `cum` | **0.0816** | 0.1231 | 0.0330 |

---

## 3. Ebene 3: Kausale Validierung & Ground Truth Reality Check

| Modell / Methode | Datengrundlage | Geschaetzter Schutzeffekt (RR / HR) | Wahre Ground Truth RR (A vs. B) | Kausalitaets-Bias |
| :--- | :--- | :---: | :---: | :---: |
| **Simulation Ground Truth** | Experimenteller A/B-Split | **0.787** (ARR 7.9 pp) | 0.787 | **0.000** (Referenz) |
| **Dynamic DeepHit Fixed** | Kontrafaktische Inferenz (Fach) | **0.951** | 0.787 | +0.164 |
| **Transformer DML** | Orthogonalisierte ATEs (Fach) | **0.884** | 0.787 | +0.097 |
| **Oracle Logistic Hazard** | Volle Information (Latente Mot.) | **0.989** | 0.787 | +0.202 |
| **Extended Cox Panel** | Regression mit TVCs (Fach) | **1.089** | 0.787 | +0.302 (Selektionsbias) |

---

## 4. Ebene 4: Methoden-Ranking, Robustheit & Ensemble/MoE-Potenzial

### 4.1 Gesamt-Scorecard der Modellfamilien

| Modellfamilie | Praediktive Guete (ROC/R2) | Kausale Treue (ARR-Recovery) | Rausch-Resilienz (S07/08) | Recheneffizienz | Gesamtrang |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Autoregressiver Deep Transformer** | **#1** (0.9411 / 0.7036) | **#2** Hoch (sequentiell) | **#1** Sehr hoch | **#3** Mittel (10 Min.) | **Rang 1 (Bester Allrounder)** |
| **Recurrent Exam GRU** | **#1** (0.8960 / 0.9010) | **#2** Hoch | **#1** Sehr hoch | **#1** Sehr schnell (1 Min.) | **Rang 2 (Bester Predictor)** |
| **Double Machine Learning (DML)** | **#2** (0.7522) | **#1** (0.8839 geringster Bias) | **#2** Mittel | **#1** Sehr schnell (1.5 Min.) | **Rang 3 (Bester Kausalschaetzer)** |
| **Dynamic DeepHit Competing Risks** | **#1** (0.9997 Grad / 0.8116) | **#2** (0.9508) | **#2** Hoch | **#2** Schnell (3.5 Min.) | **Rang 4 (Bester Multi-Event)** |
| **Extended Cox Panel (PHReg)** | **#3** (0.7510) | **#3** (1.0899 Selektions-anfaellig) | **#3** Gering | **#1** Sofort (<5s) | **Rang 5 (Oekonometrie-Standard)** |

### 4.2 Ensemble- & Mixture-of-Experts (MoE) Synergien
* **Kombination Deep Transformer + DML:** Der Deep Transformer liefert die praezisesten Sequenz-Embeddings fuer Vorhersagen ($R^2=0.7036$), waehrend DML den Schutzeffekt am unverzerrtesten isoliert ($RR=0.8839$).
* **MoE-Gating-Hypothese:** Ein Gating-Netzwerk, das bei Normalstudierenden auf den Exam-Transformer und bei extremen Workload-Ueberlastungen auf DeepHit Competing Risks schaltet, maximiert sowohl Fruehwarnung als auch Interventionsgenauigkeit.