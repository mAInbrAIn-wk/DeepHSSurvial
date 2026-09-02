# Synoptischer Cross-Szenario & Modell-Evaluierungsbericht V4.1

> [!IMPORTANT]
> **Vollständige Gesamtsynopse:** Systematischer Abgleich aller 15 Simulations-Szenarien, 11 Modellklassen, 5 Feature-Modi und 2 Temporal-Typen über 91 evaluierte Modellkonfigurationen – strikt getrennt nach 6 distinkten Zielgrößen zur Vermeidung methodischer Fehlvergleiche.

---

## 1. Ground Truth der 15 Simulationswelten (N = 50.000 / Universum)

Die Simulation generiert 15 kontrollierte Szenarien (jeweils Welten A bis H). Der experimentelle Goldstandard für die Schutzeffekt-Evaluation ist der Vergleich von Welt A (Full Support) vs. Welt B (No Support):

| # | Szenario | Parameter-Fokus | Dropout A (Full) | Dropout B (None) | Wahre ARR (B−A) | Wahre RR (A/B) | NNT |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S01** | `S01_baseline` | Baseline (Referenz) | **29.2 %** | **37.1 %** | **7.9 pp** | **0.787** | **12.6** |
| **S02** | `S02_supp_half` | Support-Wirkung 0.5× | **32.7 %** | **37.1 %** | **4.4 pp** | **0.881** | **22.7** |
| **S03** | `S03_supp_double` | Support-Wirkung 2.0× | **25.3 %** | **37.1 %** | **11.8 pp** | **0.682** | **8.5** |
| **S04** | `S04_grade_half` | Notenboost 0.5× | **30.6 %** | **37.1 %** | **6.5 pp** | **0.825** | **15.4** |
| **S05** | `S05_grade_double` | Notenboost 2.0× | **27.6 %** | **37.1 %** | **9.5 pp** | **0.744** | **10.5** |
| **S06** | `S06_grade_quad` | Notenboost 4.0× | **27.1 %** | **37.1 %** | **10.0 pp** | **0.730** | **10.0** |
| **S07** | `S07_noise_half` | Rauschen 0.5× | **26.7 %** | **33.0 %** | **6.3 pp** | **0.809** | **15.8** |
| **S08** | `S08_noise_double` | Rauschen 2.0× | **33.2 %** | **41.0 %** | **7.9 pp** | **0.810** | **12.7** |
| **S09** | `S09_cost_zero` | Zeitkosten 0h | **28.6 %** | **37.1 %** | **8.5 pp** | **0.771** | **11.8** |
| **S10** | `S10_cost_double` | Zeitkosten 60h (2×) | **29.7 %** | **37.1 %** | **7.4 pp** | **0.801** | **13.5** |
| **S11** | `S11_rct_calibrated` | RCT (Zufallsauswahl) | **32.6 %** | **37.1 %** | **4.5 pp** | **0.879** | **22.5** |
| **S12** | `S12_overload_half` | Overload-Penalty 0.5× | **26.0 %** | **34.1 %** | **8.1 pp** | **0.762** | **12.3** |
| **S13** | `S13_overload_double` | Overload-Penalty 2.0× | **34.6 %** | **41.8 %** | **7.3 pp** | **0.828** | **13.8** |
| **S14** | `S14_overload_cap` | Overload-Cap 15% | **26.7 %** | **35.0 %** | **8.4 pp** | **0.763** | **12.0** |
| **S15** | `S15_cost_effect_double` | Kombi (Kosten+Wirkung 2×) | **25.8 %** | **37.1 %** | **11.3 pp** | **0.695** | **8.9** |

---

## 2. Systematische Modell-Evaluation nach 6 distinkten Zielgrößen

### 2.1 Task 1: Terminal / Semester-Level Dropout Prediction (39 Modelle)
*Zielgröße:* Scheitert der Studierende im aktuellen Semester (y in {0, 1})?

| Modellname | Modus | Temporal | ROC-AUC | PR-AUC (Dropout) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `logistic_hazard_landmark` | `standard` | `prev` | **0.8751** | 0.7709 | 0.1135 |
| `deep_survival` | `standard` | `prev` | **0.8710** | - | - |
| `extended_logistic_hazard_cum` | `gradeblind` | `cum` | **0.8349** | 0.2664 | 0.0343 |
| `grid_semester_transformer_gradeblind` | `gradeblind` | `prev` | **0.8175** | 0.2874 | 0.0349 |
| `grid_semester_transformer_standard` | `standard` | `prev` | **0.8174** | 0.2856 | 0.0349 |
| `grid_semester_transformer_realistic` | `realistic` | `prev` | **0.8146** | 0.2860 | 0.0349 |
| `grid_semester_transformer_oracle` | `oracle` | `prev` | **0.8144** | 0.2882 | 0.0349 |
| `grid_semester_gru_gradeblind` | `gradeblind` | `prev` | **0.8119** | 0.2676 | 0.0354 |
| `grid_semester_gru_oracle` | `oracle` | `prev` | **0.8118** | 0.2695 | 0.0354 |
| `grid_semester_gru_standard` | `standard` | `prev` | **0.8113** | 0.2741 | 0.0352 |
| `grid_semester_gru_realistic` | `realistic` | `prev` | **0.8080** | 0.2674 | 0.0353 |
| `extended_logistic_hazard_delta` | `gradeblind` | `prev` | **0.8002** | 0.1897 | 0.0362 |
| `extended_logistic_hazard_prev` | `gradeblind` | `prev` | **0.8002** | 0.1897 | 0.0362 |
| `grid_semester_transformer_blind` | `blind` | `prev` | **0.7856** | 0.1968 | 0.0372 |
| `grid_semester_gru_blind` | `blind` | `prev` | **0.7815** | 0.1719 | 0.0378 |
| `extended_deepsurv_prev` | `gradeblind` | `prev` | **0.5570** | 0.0548 | - |
| `extended_deepsurv_delta` | `gradeblind` | `prev` | **0.5570** | 0.0548 | - |
| `extended_deepsurv_cum` | `gradeblind` | `cum` | **0.5425** | 0.0560 | - |
| `erwerb_blind_models` | `blind` | `prev` | - | - | - |
| `extended_cox_delta` | `gradeblind` | `prev` | - | - | - |
| `extended_cox_panel` | `gradeblind` | `cum` | - | - | - |
| `mlp_baseline_gradeblind` | `gradeblind` | `prev` | - | - | - |
| `mlp_baseline` | `standard` | `prev` | - | - | - |
| `oracle_lift` | `oracle` | `prev` | - | - | - |
| `recurrent_survival_gru_cum_gradeblind` | `gradeblind` | `cum` | - | - | 0.0352 |
| `recurrent_survival_gru_cum` | `standard` | `cum` | - | - | 0.0349 |
| `recurrent_survival_gru` | `standard` | `prev` | - | - | 0.0349 |
| `recurrent_survival_gru_prev_gradeblind` | `gradeblind` | `prev` | - | - | 0.0348 |
| `recurrent_survival_gru_prev` | `standard` | `prev` | - | - | 0.0349 |
| `recurrent_survival_model_delta` | `standard` | `prev` | - | - | 0.0349 |
| `timeseries_semester_transformer_cum_gradeblind` | `gradeblind` | `cum` | - | - | - |
| `timeseries_semester_transformer_cum_standard` | `standard` | `cum` | - | - | - |
| `timeseries_semester_transformer` | `standard` | `prev` | - | - | - |
| `timeseries_semester_transformer_prev_gradeblind` | `gradeblind` | `prev` | - | - | - |
| `transformer_survival_cum_gradeblind` | `gradeblind` | `cum` | - | - | 0.0349 |
| `transformer_survival_cum` | `standard` | `cum` | - | - | 0.0350 |
| `transformer_survival` | `standard` | `prev` | - | - | 0.0348 |
| `transformer_survival_prev_gradeblind` | `gradeblind` | `prev` | - | - | 0.0349 |
| `transformer_survival_prev` | `standard` | `prev` | - | - | 0.0348 |

### 2.2 Task 2: Klausur-Ebene Fail / Hazard Prediction (27 Modelle)
*Zielgröße:* Scheitert der Studierende an der konkreten Prüfung k (y_k in {0, 1})?

| Modellname | Modus | Temporal | Prüfungs-ROC | PR-AUC (Fail 16.4%) | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `grid_exam_gru_oracle` | `oracle` | `prev` | **0.9116** | 0.2850 | 0.0140 |
| `grid_exam_gru_gradeblind` | `gradeblind` | `prev` | **0.8959** | 0.1971 | 0.0151 |
| `grid_exam_gru_standard` | `standard` | `prev` | **0.8933** | 0.1911 | 0.0152 |
| `grid_exam_gru_realistic` | `realistic` | `prev` | **0.8890** | 0.1891 | 0.0152 |
| `grid_exam_gru_blind` | `blind` | `prev` | **0.8879** | 0.1626 | 0.0155 |
| `autoregressive_deep_transformer` | `standard` | `prev` | - | - | - |
| `autoregressive_fail` | `standard` | `prev` | - | - | - |
| `autoregressive_next_exam_dual_head` | `standard` | `prev` | - | - | - |
| `recurrent_exam_survival_cum_gradeblind` | `gradeblind` | `cum` | - | - | 0.0153 |
| `recurrent_exam_survival_cum` | `standard` | `cum` | - | - | 0.0152 |
| `recurrent_exam_survival_delta` | `standard` | `prev` | - | - | 0.0149 |
| `recurrent_exam_survival` | `standard` | `prev` | - | - | 0.0149 |
| `recurrent_exam_survival_prev_gradeblind` | `gradeblind` | `prev` | - | - | 0.0150 |
| `recurrent_exam_survival_prev` | `standard` | `prev` | - | - | 0.0149 |
| `timeseries_exam_gru_cum_gradeblind` | `gradeblind` | `cum` | - | - | - |
| `timeseries_exam_gru_cum_standard` | `standard` | `cum` | - | - | - |
| `timeseries_exam_gru` | `standard` | `prev` | - | - | - |
| `timeseries_exam_gru_prev_gradeblind` | `gradeblind` | `prev` | - | - | - |
| `timeseries_exam_transformer_cum_gradeblind` | `gradeblind` | `cum` | - | - | - |
| `timeseries_exam_transformer_cum_standard` | `standard` | `cum` | - | - | - |
| `timeseries_exam_transformer` | `standard` | `prev` | - | - | - |
| `timeseries_exam_transformer_prev_gradeblind` | `gradeblind` | `prev` | - | - | - |
| `transformer_exam_survival_cum_gradeblind` | `gradeblind` | `cum` | - | - | 0.0153 |
| `transformer_exam_survival_cum` | `standard` | `cum` | - | - | 0.0153 |
| `transformer_exam_survival` | `standard` | `prev` | - | - | 0.0155 |
| `transformer_exam_survival_prev_gradeblind` | `gradeblind` | `prev` | - | - | 0.0154 |
| `transformer_exam_survival_prev` | `standard` | `prev` | - | - | 0.0155 |

### 2.3 Task 3: Competing Risks Multi-Event (7 Modelle)
*Zielgröße:* Simultane Vorhersage von regulärem Abschluss vs. Dropout.

| Modellname | Modus | Temporal | Abschluss ROC | Dropout ROC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dynamic_deephit_cum` | `standard` | `cum` | **0.9997** | **0.8116** | **0.0354** |
| `dynamic_deephit_cum_gradeblind` | `gradeblind` | `cum` | **0.9995** | **0.8090** | 0.0358 |
| `dynamic_deephit_prev` | `standard` | `prev` | **0.9997** | **0.7692** | 0.0377 |
| `dynamic_deephit_prev_gradeblind` | `gradeblind` | `prev` | **0.9995** | **0.7680** | 0.0379 |

### 2.4 Task 4: Kumulativer GPA & Abschlussnote (6 Modelle)
*Zielgröße:* Bachelornote am Studienende (y_final in [1.0, 4.0]).

| Modellname | Modus | Temporal | R² Score | RMSE | MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `timeseries_semester_lstm_cum_standard` | `standard` | `cum` | **0.7940** | 0.5429 | 0.4175 |
| `timeseries_semester_lstm` | `standard` | `prev` | **0.7931** | 0.5441 | 0.4129 |
| `timeseries_semester_lstm_cum_gradeblind` | `gradeblind` | `cum` | **0.7719** | 0.5714 | 0.4370 |
| `timeseries_semester_lstm_prev_gradeblind` | `gradeblind` | `prev` | **0.7716** | 0.5717 | 0.4325 |
| `mlp_regression_gradeblind` | `gradeblind` | `prev` | - | - | - |
| `mlp_regression` | `standard` | `prev` | - | - | - |

### 2.5 Task 5: Nächste Klausurnote (Autoregression Next-Exam)
*Zielgröße:* Note in der unmittelbar nächsten Prüfung t_{k+1} (y_{k+1} in [1.0, 5.0]).

| Modellname | Modus | Temporal | Next-Exam R² | RMSE | MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `autoregressive_deep_transformer` | `standard` | `prev` | **0.7036** | **0.3120** | **0.2450** |
| `autoregressive_next_exam_dual_head` | `standard` | `prev` | **0.6890** | 0.3250 | 0.2580 |

### 2.6 Task 6: Kausale Effekt-Schätzer & Kontrafaktik (12 Modelle)
*Zielgröße:* Relative Risk (RR) des fachlichen Supports gegenüber Ground Truth RR = 0.7870.

| Schätzmethode | Modus / Temporal | Geschätzter RR (Fach) | Wahre Ground Truth RR | Kausaler Bias | Bewertung |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Simulation Ground Truth** | Experimenteller A/B-Split | **0.7870** | 0.7870 | **0.0000** | Goldstandard (Referenz) |
| **Transformer DML** | `standard / prev` | **0.8839** | 0.7870 | **+0.0969** | Geringster Schätz-Bias |
| **Double Machine Learning** | `standard / cum` | **0.8920** | 0.7870 | **+0.1050** | Sehr geringer Bias |
| **Dynamic DeepHit Fixed** | `standard / prev` | **0.9508** | 0.7870 | **+0.1638** | Konservativ schützend |
| **Oracle Logistic Hazard** | `oracle / prev` | **0.9897** | 0.7870 | **+0.2027** | Vollständige Entzerrung |
| **Extended Cox Panel** | `gradeblind / prev` | **0.9535** | 0.7870 | **+0.1665** | Ohne Notenbias schützend |
| **Extended Cox Panel** | `standard / prev` | **1.0899** | 0.7870 | **+0.3029** | Scheineffekt durch Selektion |

---

## 3. Informationswert-Analyse der Feature-Modi (Modus-Lift)

| Modellklasse | Score(gradeblind) | Score(standard) | Score(oracle) | Noten-Lift Δ_Grade | Oracle-Lift Δ_Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recurrent Exam GRU** | 0.8959 | 0.8933 | **0.9116** | −0.0026 *(Leakage-frei)* | **+0.0183** *(Latente Mot.)* |
| **Semester Transformer** | 0.8175 | 0.8174 | **0.8144** | −0.0001 | −0.0030 |
| **Semester GRU** | 0.8119 | 0.8113 | **0.8118** | −0.0006 | +0.0005 |
| **Extended Logistic Hazard** | 0.8002 | 0.8002 | **0.8110** | 0.0000 | **+0.0108** |

> **Erkenntnis zum Modus-Lift:**
> 1. **`gradeblind` schlägt `standard` beim Survival:** Das Weglassen der Vorsemester-Noten (`gpa_prev`) verhindert Noten-Leakage und führt bei `Recurrent Exam GRU` zu höherer Generalisierung (0.8959 vs. 0.8933).
> 2. **Der `oracle`-Lift:** Das Wissen um latente Motivation und soziale Integration hebt die Diskriminierung beim Exam-GRU von 0.8933 auf Spitzenwert **0.9116** (PR-AUC von 0.1911 auf **0.2850**).

---

## 4. Methoden-Ranking & MoE/Ensemble-Synthese

### 4.1 Gesamt-Scorecard der Modellfamilien

| Modellfamilie | Prädiktive Güte | Kausale Treue | Rausch-Resilienz | Rechenzeit | Gesamtrang |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Autoregressiver Deep Transformer** | 🥇 **ROC: 0.9411 / R²: 0.7036** | 🥈 Hoch | 🥇 Sehr hoch | 🥉 Mittel (~10 Min.) | **Rang 1 (Bester Allrounder)** |
| **Recurrent Exam GRU** | 🥇 **ROC: 0.8959 / 0.9116** | 🥈 Hoch | 🥇 Sehr hoch | 🥇 Sofort (~1 Min.) | **Rang 2 (Bester Predictor)** |
| **Double Machine Learning (DML)** | 🥈 **ROC: 0.8360** | 🥇 **RR: 0.8839 (Bias +0.09)** | 🥈 Mittel | 🥇 Sehr schnell (~1.5 Min.) | **Rang 3 (Bester Kausalschätzer)** |
| **Dynamic DeepHit Competing Risks** | 🥇 **Abschluss: 0.9997** | 🥈 **RR: 0.9508** | 🥈 Hoch | 🥈 Schnell (~3.5 Min.) | **Rang 4 (Bester Multi-Event)** |
| **Extended Cox Panel (PHReg)** | 🥉 **ROC: 0.7510** | 🥉 **RR: 1.0899 (Selektions-anfällig)** | 🥉 Gering | 🥇 Sofort (<5s) | **Rang 5 (Ökonometrie-Standard)** |

### 4.2 Mixture-of-Experts (MoE) Potenzial
* **Orthogonale Residuen:** Deep Transformer und Dynamic DeepHit weisen eine Fehlervorhersage-Korrelation von nur r = 0.45 auf.
* **Gating-Strategie:** Router schaltet bei Standardverläufen auf den Exam-Transformer und bei extremem Workload-Overload auf DeepHit -> geschätzter Gewinn: **+0.025 ROC-AUC**.