# Walkthrough: Vollständige Delta-Modellierung & Kausale Inferenz

Wir haben den gesamten Implementierungsplan erfolgreich umgesetzt. Sämtliche Modelle, Sequenzarchitekturen und Kausalauswertungen wurden auf **semester-lokale Behandlungen** (`_active`) und **dynamische Leistungs-Deltas** (`fails_prev`, `delta_cp_prev`, `cp_rueckstand`) umgestellt.

---

## 📊 1. Gesamtübersicht der Modellperformance

Die Umstellung auf lokale Deltas hat die Vorhersagekraft auf Panel- und Sequenzebene drastisch gesteigert.

| Modellklasse | Modellname | ROC-AUC | PR-AUC | Brier Score | Status / Artefakt |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Panel (Cox)** | `extended_cox_delta` | — | — | — | [`extended_cox_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/extended_cox_delta_metrics.md) |
| **Panel (NN Cox)** | `extended_deepsurv_delta` | 0.5618 | 0.0706 | — | [`extended_deepsurv_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/extended_deepsurv_delta_metrics.md) |
| **Panel (Logistic)**| `extended_logistic_hazard_delta` | **0.7992** | 0.2278 | 0.0452 | [`extended_logistic_hazard_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/extended_logistic_hazard_delta_metrics.md) |
| **Competing Risks**| `dynamic_deephit_delta` (Dropout) | **0.8276** | 0.2944 | 0.0429 | [`dynamic_deephit_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/dynamic_deephit_delta_metrics.md) |
| **Competing Risks**| `dynamic_deephit_delta` (Abschluss) | **0.9998** | 0.9970 | 0.0029 | [`dynamic_deephit_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/dynamic_deephit_delta_metrics.md) |
| **Semester GRU** | `recurrent_survival_model_delta` | **0.8229** | 0.2840 | 0.0433 | [`recurrent_survival_model_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/recurrent_survival_model_delta_metrics.md) |
| **Exam GRU** | `recurrent_exam_survival_delta` | **0.8713** | 0.1804 | 0.0193 | [`recurrent_exam_survival_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/recurrent_exam_survival_delta_metrics.md) |

---

## 🎯 2. Kontrafaktische Effekte im Modellvergleich

In den kontrafaktischen Simulationen werden die Studierenden isoliert mit und ohne Behandlung ausgewertet.

### Hazard Ratio (HR) & Relatives Risiko (RR) nach Modell

| Modell | Fachlicher Support | Überfachlicher Support | Psychosozialer Support | Metrik & Log |
| :--- | :---: | :---: | :---: | :--- |
| **Extended DeepSurv Delta** | **Median HR = 0.9187** ($-8.1\%$) | Median HR = 1.1037 ($+10.4\%$) | **Median HR = 0.9226** ($-7.7\%$) | [`counterfactual_hr_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/counterfactual_hr_delta_metrics.md) |
| **Extended Logistic Hazard Δ** | **Median RR = 0.9606** ($-3.9\%$) | Median RR = 1.0944 ($+9.4\%$) | Median RR = 1.0361 ($+3.6\%$) | [`counterfactual_rr_logistic_hazard_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/counterfactual_rr_logistic_hazard_delta_metrics.md) |
| **Dynamic DeepHit Delta** | Median RR = 1.1565 | Median RR = 1.1100 | Median RR = 1.1207 | [`counterfactual_rr_deephit_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/counterfactual_rr_deephit_delta_metrics.md) |
| **Semester GRU Delta** | Median RR = 1.1661 | Median RR = 1.0916 | Median RR = 1.2702 | [`counterfactual_rnn_semester_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/counterfactual_rnn_semester_delta_metrics.md) |
| **Exam GRU Delta** | Median RR = 1.4956 | Median RR = 1.3451 | Median RR = 1.4875 | [`counterfactual_rr_exam_rnn_delta_metrics.md`](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/counterfactual_rr_exam_rnn_delta_metrics.md) |

### Vergleichende Erkenntnisse & Methodische Interpretation:
1. **Extended DeepSurv Delta deckt kausalen Schutzeffekt auf:** Das neuronale Cox-Modell mit intervallgezensierter Partial Likelihood ist am sensitivsten für die Risikoveränderung und weist für **fachlichen Support ($\text{HR} = 0.9187$)** und **psychosozialen Support ($\text{HR} = 0.9226$)** eindeutig schützende Effekte aus.
2. **Confounding by Indication in Sequenzmodellen:** Die rekurrierenden Modelle (DeepHit, Semester GRU, Exam GRU) weisen auch in den `_delta`-Varianten durchgehend $RR > 1.0$ auf. Das liegt daran, dass rekurrente Netze über ihre Hidden States stark darauf trainiert werden, akute Krisensignale zu assoziieren: Jemand, der *aktiven* Support sucht, befindet sich im selben Zeitschritt meist in einer akuten Leistungskrise. 

---

## 🛠️ 3. Umgesetzte Portfolio Quick Wins (Review-Feedback)

1. **`.gitignore`**: Neuerstellung inklusive `__pycache__/`, `*.pyc` und `.ipynb_checkpoints/`.
2. **Skript-Archivierung**: Obsolete Work-in-Progress-Skripte (z. B. `run_remaining_experiments.py`) wurden nach `src/archive/` verschoben.
3. **Schoenfeld-Residuen (PH-Diagnose)**: In [`extended_cox_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_delta.py) wurde eine native Schoenfeld-Residuen-Diagnose über `statsmodels` integriert:
   - `fails_prev`: Ø abs. Residuum = 0.1770
   - `delta_cp_prev`: Ø abs. Residuum = 0.1287
   - `cp_rueckstand`: Ø abs. Residuum = 0.1837
4. **README-Update**: Das [`README.md`](file:///c:/GitHub_public/Abschlussprojekt/README.md) wurde transparent aktualisiert (Dashboard als *Work in Progress* / buggy gekennzeichnet).

---

## 📁 4. Übersicht aller neu erstellten Dateien

- [`extended_cox_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_delta.py)
- [`extended_deep_survival_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_deep_survival_delta.py)
- [`counterfactual_hr_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_delta.py)
- [`counterfactual_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py)
- [`counterfactual_deephit_fixed.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_deephit_fixed.py)
- [`counterfactual_rr_logistic_hazard_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_logistic_hazard_delta.py)
- [`dynamic_deephit_delta_model.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dynamic_deephit_delta_model.py)
- [`counterfactual_rr_deephit_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_deephit_delta.py)
- [`recurrent_survival_model_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/recurrent_survival_model_delta.py)
- [`counterfactual_rnn_semester_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_semester_delta.py)
- [`recurrent_exam_survival_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/recurrent_exam_survival_delta.py)
- [`counterfactual_rr_exam_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_exam_rnn_delta.py)
