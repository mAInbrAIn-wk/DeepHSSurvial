# Vollständiges Modell-Inventar, Modellklassen & Implementierungsplan

**Datum:** 20. August 2026 (erweitert: 21. August 2026)  
**Projekt:** DeepSupport – Wirksamkeitsanalyse von Hochschulsupport (V3.3)

---

## Vorbemerkung: Antworten auf alle Detailfragen & Kommentare

### 1. Analyse der Skript-Diskrepanz: `deep_transformer_regression.py` vs. `timeseries_semester.py` (Kategorie 2b)
> *Frage/Kommentar:* "Warum ist das hier so anders gelöst, als in den anderen Fällen? Bitte analysiere das Skript und vergleiche es mit mindestens einem anderen derselben Kategorie."

**Vergleichsanalyse:**

| Kriterium | `timeseries_semester.py` (Referenz Klasse 2b) | `deep_transformer_regression.py` (Neu / Fehlerhaft) |
| :--- | :--- | :--- |
| **Datenquellen** | Liest alle 8 Primärtabellen (`studierende.csv`, `module.csv`, `einschreibungen.csv`, `pruefungen.csv`, `support_angebote.csv`, `support_teilnahmen.csv`, etc.) | Liest nur 4 Tabellen (`studierende.csv`, `einschreibungen.csv`, `pruefungen.csv`, `module.csv`) – Support-Tabellen werden gar nicht erst geladen! |
| **Support-Features** | 4 differenzierte Merkmale: `sem_support_fachlich_relevant`, `sem_support_fachlich_sonst`, `sem_support_ueberfachlich`, `sem_support_psychosozial` (präzise gejoint) | Deklariert 3 Spalten (`fach_supp`, `uebf_supp`, `psych_supp`), setzt diese aber in Zeile 131 pauschal auf `0.0` als unvollständiger Platzhalter. |
| **Demographische Features** | Vollständig eingebunden: `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, plus One-Hot-Encoding für `stg_name` und `hzb_typ` | Komplett weggelassen (0 demographische Merkmale). |
| **Zielvariable (Target)** | `sem_avg_note.mean()` über alle Fachsemester (Gesamt-GPA) | `df_pr[df_pr["bestanden"] == True].groupby()["note"].mean()` (nur bestandene Prüfungen – unüblich und inkonsistent) |
| **Folge für Metriken** | $R^2 = 0{,}9144$, $\text{RMSE} = 0{,}3108$ | $R^2 = 0{,}5046$, $\text{RMSE} = 0{,}5135$ (Modell leidet unter Feature-Entzug) |

**Ursache der Diskrepanz:**  
In `deep_transformer_regression.py` wurde ein improvisierter, isolierter Aggregations-Loop geschrieben, anstatt die etablierte Pipeline `create_semester_timeseries_dataset` aus `timeseries_semester.py` wiederzuverwenden.

**Lösung:**  
Nicht noch ein eigener Aggregations-Code, sondern `deep_transformer_regression.py` refactoren, sodass es direkt die standardisierte Funktion `create_semester_timeseries_dataset(data_dir)` nutzt. Dadurch wird vollständige Feature- und Target-Gleichheit innerhalb der Klasse 2b hergestellt.

---

### 2. Vollständige Auflösung aller 13 Counterfactual-Skripte
> *Frage/Kommentar:* "Warum stehen da Fragezeichen? Kannst Du bitte in den Skripten nachsehen, welches Modell sie erwarten? Welches wäre denn sinnvoll?"

Alle 13 Counterfactual-Skripte wurden im Quelltext analysiert und mit dem Dateisystem in `output_dl/models/` abgeglichen:

| # | Counterfactual-Skript | Erwartetes Modell (`.keras`) | Modell auf Disk vorhanden? | Berechnete Kausal-Metrik | Status |
| :---: | :--- | :--- | :---: | :--- | :---: |
| 1 | [`counterfactual_deepsurv.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_deepsurv.py) | `deepsurv_landmark.keras` | ✅ (225 KB) | Bootstrap Pseudo-HR (Landmark) | Bereit zur Ausführung |
| 2 | [`counterfactual_hr_analyzer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_analyzer.py) | `extended_deepsurv_panel.keras` | ✅ (97 KB) | Kausale HR per Person-Semester Panel | Bereit zur Ausführung |
| 3 | [`counterfactual_hr_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_delta.py) | `extended_deepsurv_delta.keras` | ✅ (225 KB) | Kausale HR für DeepSurv Delta | Bereit zur Ausführung |
| 4 | [`counterfactual_rr_logistic_hazard_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_logistic_hazard_delta.py) | `extended_logistic_hazard_delta.keras` | ✅ (55 KB) | Kausales RR für DTL Hazard Delta | Bereit zur Ausführung |
| 5 | [`counterfactual_rr_deephit_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_deephit_delta.py) | `dynamic_deephit_delta.keras` | ✅ (126 KB) | Kausales RR für DeepHit Delta Competing Risks | Bereit zur Ausführung |
| 6 | [`counterfactual_inference_deephit.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_deephit.py) | `dynamic_deephit_competing.keras` | ✅ (125 KB) | Kausale HR für DeepHit Base | Bereit zur Ausführung |
| 7 | [`counterfactual_deephit_fixed.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_deephit_fixed.py) | `dynamic_deephit_competing.keras` | ✅ (125 KB) | Kausales RR für DeepHit Base (korrigiert) | Bereit zur Ausführung |
| 8 | [`counterfactual_inference_semester_transformer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_semester_transformer.py) | `transformer_survival.keras` | ✅ (198 KB) | Kausale HR für Semester-Transformer Survival | Bereit zur Ausführung |
| 9 | [`counterfactual_rnn_semester_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_semester_delta.py) | `recurrent_survival_model_delta.keras` | ✅ (100 KB) | Kausales RR für Semester-GRU Delta | Bereit zur Ausführung |
| 10 | [`counterfactual_inference.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference.py) | `recurrent_exam_survival.keras` | ✅ (97 KB) | Kausale HR für Exam-GRU Base | Bereit zur Ausführung |
| 11 | [`counterfactual_rr_exam_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_exam_rnn_delta.py) | `recurrent_exam_survival_delta.keras` | ✅ (99 KB) | Kausales RR für Exam-GRU Delta | Bereit zur Ausführung |
| 12 | [`counterfactual_rnn.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn.py) | `recurrent_exam_survival_v2.keras` | ❌ (fehlt) | Kausales RR für Exam-GRU V2 | Benötigt vorheriges Training von `recurrent_exam_survival_v2.py` |
| 13 | [`counterfactual_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py) | `recurrent_exam_survival_v2.keras` | ❌ (fehlt) | Kausales RR für Exam-GRU V2 | Benötigt vorheriges Training von `recurrent_exam_survival_v2.py` |

**Ergebnis:** 11 von 13 Counterfactual-Skripten können **sofort** auf den vorhandenen trainierten Modellen ausgeführt werden! Nur die Skripte 12 und 13 warten auf das Modell `recurrent_exam_survival_v2.keras`.

---

### 3. Weitere Antworten auf die vorherigen Anmerkungen
- **Klasse 6 (`cp_rueckstand`):** Wird vereinheitlicht. Da `cp_rueckstand` eine deterministische Funktion der kumulierten CP ist, wird es aus den Delta-Varianten entfernt. Klasse 6 erhält einheitlich 8 Features.
- **Klasse 2 Aufspaltung:** Sauber getrennt in **2a** (Statische Landmark-Regression, S1-S2 Aggregat) und **2b** (Semester-Sequenz-Regression).
- **Extended Cox Delta HRs:** Werden durch Aufnahme von `extended_cox_delta.py` in die Runner-Pipeline erstmals sauber berechnet und als `extended_cox_delta_metrics.json` gespeichert.

---

## Teil 1: Vollständiges Modell-Inventar (Stand V3.3)

### A. Klassifikations-Baselines (Statisches Outcome: Abbruch ja/nein)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 1 | `train_mlp_baseline.py` | Naive Bayes | `agg_abschluesse.csv` | `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`, `hzb_typ`, `AVG_note_sem1-2`, `AVG_cp_sem1-2`, `fehlversuche_sem12`, `Fach/Uebf/Psych_supp_sem12`, `any_support_sem12` | `status` (multiclass) | Acc=0,731, F1-Macro=0,505 |
| 2 | `train_mlp_baseline.py` | Random Forest | idem | idem | idem | Acc=0,793, F1=0,393 |
| 3 | `train_mlp_baseline.py` | SVM | idem | idem | idem | Acc=0,723, F1=0,313 |
| 4 | `train_mlp_baseline.py` | Keras MLP | idem | idem | idem | Acc=0,793, ROC-AUC=0,913, PR-AUC=0,563 |
| 5 | `train_mlp_baseline.py` [blind] | Naive Bayes (blind) | idem | ohne `AVG_note`, `hzb_note` | idem | Acc=0,750 |
| 6 | `train_mlp_baseline.py` [blind] | Random Forest (blind) | idem | idem | idem | Acc=0,787 |
| 7 | `train_mlp_baseline.py` [blind] | SVM (blind) | idem | idem | idem | Acc=0,648 |
| 8 | `train_mlp_baseline.py` [blind] | Keras MLP (blind) | idem | idem | idem | Acc=0,792, ROC-AUC=0,908 |

---

### B. Noten-Regression (Target: Abschlussnote / GPA)

#### B1. Statische Regression (Klasse 2a – Landmark S1-S2)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 9 | `train_mlp_regression.py` | Linear Ridge | `agg_abschluesse.csv` | `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`, `hzb_typ`, `AVG_note_sem1-2`, `AVG_cp_sem1-2`, `fehlversuche_sem12`, `support_sem12`-Felder | `abschlussnote` | R²=0,846, RMSE=0,247 |
| 10 | `train_mlp_regression.py` | SVR | idem | idem | idem | R²=0,867, RMSE=0,229 |
| 11 | `train_mlp_regression.py` | Random Forest Reg. | idem | idem | idem | R²=0,848, RMSE=0,245 |
| 12 | `train_mlp_regression.py` | Keras MLP Regressor | idem | idem | idem | R²=0,869, RMSE=0,227 |

#### B2. Semester-Sequenz Regression (Klasse 2b – Semester-Zeitreihe)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 13 | `timeseries_semester.py` | Semester-LSTM Regressor | 8 Primärtabellen | Sequenziell (CP, Fails, 4x Support) + Statisch (Demographie, Stg, HZB) | `sem_avg_note` (Gesamt-GPA) | R²=0,914, RMSE=0,311 |
| 14 | `timeseries_semester_transformer.py` | Semester-Transformer Reg. | idem | idem | idem | R²=0,908, RMSE=0,322 |
| 17 | `deep_transformer_regression.py` [M1] | Deep Semester-Transformer Reg. (d=128) | ad-hoc CSVs | Unvollständig: Support=0, keine Demographie (**Bug**) | nur bestandene Noten | R²=0,505 ⚠️ (wird refactort) |

#### B3. Exam-Sequenz Regression (Klasse 3 – Prüfungs-Zeitreihe)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 15 | `timeseries_exam.py` | Exam-GRU Regressor | `agg_pruefungen.csv` + Stammdaten | Sequenziell (Versuch, CP, Schw., Support vor/glz, Lag-Fails) + Statisch | `note` (mean) | R²=0,903, RMSE=0,329 |
| 16 | `timeseries_exam_transformer.py` | Exam-Transformer Reg. | idem | idem | idem | R²=0,905, RMSE=0,325 |
| 18 | `deep_transformer_regression.py` [M2] | Deep Exam-Transformer Reg. (d=128) | `pruefungen.csv` | `versuch`, `cp`, `schwierigkeit`, `bestanden`, **`note`** (**Leakage**) | `note` (mean) | R²=0,999 ❌ (wird refactort) |

---

### C. Landmark Survival (Statisch, nach S1–S2)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 19 | `deep_survival.py` | DeepSurv Landmark (Neural Cox) | `agg_abschluesse.csv` | `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`, `hzb_typ`, `AVG_note_sem1-2`, `AVG_cp_sem1-2`, `Fach/Uebf/Psych_supp` | Survival: `(t_stop, event)` | C-Index=0,741; HR_fach=1,092 ⚠️, HR_uebf=1,053 ⚠️, HR_psych=0,932 |
| 20 | `deep_survival.py` | DTL Hazard Landmark | idem | idem | Discrete Hazard über 14 Semester | C-Index=0,735, ROC-AUC=0,860, PR-AUC=0,715 |

---

### D. Semester-Level Panel-Survival (Zeitvariierende Features, Counting Process)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 21 | `extended_cox_delta.py` | Extended Cox | Panel via `build_delta_panel` | `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `t_start`, `t_stop`, `fails_prev`, `delta_cp_prev`, `cp_rueckstand`, `fach/uebf/psych_supp_active` | `event` (Dropout bei t_stop) | ⏳ Wird in Runner aufgenommen |
| 21b | `extended_deep_survival.py` | Extended DeepSurv Panel | Panel via `build_person_semester_panel` | Zeitvariierende Panel-Features | idem | ROC-AUC=0,563, PR-AUC=0,052 |
| 21c | `extended_deep_survival.py` | Extended DTL Hazard Panel | idem | idem | idem | ROC-AUC=0,769, PR-AUC=0,208 |
| 22 | `extended_deep_survival_delta.py` | Extended DeepSurv Delta | Panel via `build_delta_panel` | Panel-Features mit Deltas | idem | ROC-AUC=0,535, PR-AUC=0,050 |
| 23 | `extended_deep_survival_delta.py` | Extended DTL Hazard Delta | idem | idem | idem | ROC-AUC=0,761, PR-AUC=0,201, Brier=0,037 |
| 24 | `dml_orthogonal_survival.py` | DML Orthogonalized Survival | idem | idem + Propensity-Residualen | idem | ROC-AUC=0,765; RR_fach=0,799 ⚠️, RR_uebf=1,098 ⚠️, RR_psych=0,908 |
| 25 | `train_oracle_models.py` | Oracle DTL Hazard Delta | idem | idem + `hidden_*_prev` (Motivation, SozInt, ErwNote) | idem | C-Index=0,776 (+0,9 % Lift) |
| 34 | `extended_exam_survival.py` | Extended DTL Hazard Exam | `agg_pruefungen.csv` (Exam-Panel) | Exam-Level Panel-Features | Exam-Level Dropout | ROC-AUC=0,864, PR-AUC=0,176 |

---

### E. Semester-Level Sequenz-Survival (GRU / Transformer, TimeDistributed)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 26 | `recurrent_survival_model.py` | Recurrent Survival GRU | `agg_abschluesse.csv` + `agg_pruefungen.csv` | `sem_gpa`, `sem_cp`, `sem_fails`, `fach/uebf/psych_supp_cum`, `hzb_note`, `erwerbstaetigkeit_std` (8F) | Dropout am letzten Semester | ROC-AUC=0,790, PR-AUC=0,223 |
| 27 | `recurrent_survival_model.py` [blind] | Recurrent Survival GRU (blind) | idem | wie oben, `sem_gpa=0`, `hzb_note=0` | idem | ROC-AUC=0,791, PR-AUC=0,229 |
| 28 | `recurrent_survival_model_delta.py` | Recurrent Survival GRU Delta | idem | 8F wie oben + `cp_rueckstand` (9F) | idem | ROC-AUC=0,789, PR-AUC=0,226 |
| 29 | `transformer_survival_model.py` | Transformer Survival (Semester) | idem | 8F wie GRU (26) | idem | ROC-AUC=0,791, PR-AUC=0,228 |
| 30 | `dynamic_deephit_delta_model.py` | Dynamic DeepHit Delta (Competing Risks) | idem | 9F wie Delta-Modell | 2 Köpfe: Dropout + Abschluss | ROC-AUC_Dropout=0,794, PR-AUC=0,230 |
| 30b | `dynamic_deephit_model.py` | Dynamic DeepHit Base (Competing Risks) | idem | 8F wie GRU (26) | idem | ROC-AUC_Dropout=0,792, PR-AUC=0,216 |

---

### F. Exam-Level Sequenz-Survival (GRU / Transformer, TimeDistributed)

| # | Skript | Modellname | Datenquelle | Features | Target | Metriken |
| :---: | --- | --- | --- | --- | --- | --- |
| 31 | `recurrent_exam_survival.py` | Recurrent Exam GRU (Base) | `agg_abschluesse.csv` + `agg_pruefungen.csv` | `versuch`, `schwierigkeit`, `cp`, `fach/uebf/psych_supp_cum` (6F) | Dropout am letzten Exam | ROC-AUC=0,845, PR-AUC=0,142 |
| 31b | `recurrent_exam_survival_v2.py` | Recurrent Exam GRU V2 | idem | 6F + `fails_cum`, `cp_cum`, `gpa_cum` (9F) | idem | ⏳ Wird in Runner aufgenommen |
| 32 | `recurrent_exam_survival_delta.py` | Recurrent Exam GRU Delta | idem | `note`, `cp`, `is_fail`, `fach/uebf/psych_act`, `hzb_note`, `erwerbstaetigkeit_std` (8F) | idem | ROC-AUC=0,850, PR-AUC=0,139 |
| 33 | `transformer_exam_survival.py` | Exam Causal Transformer Survival | idem | 6F wie Base (31) | idem | ROC-AUC=0,832, PR-AUC=0,127 |
| 38 | `deep_transformer_regression.py` [M3] | Deep Exam-Transformer Survival | `pruefungen.csv` | `versuch`, `cp`, `schwierigkeit`, `bestanden`, `note` (**Leakage**) | `status != abgeschlossen` | ROC-AUC=0,9999 ❌ (wird refactort) |

---

### G. Kausal-Inferenz & Causal ML

| # | Modell / Skript | Methode | Treatment-Definition | Output / Metriken | Status |
| :---: | --- | --- | --- | --- | :---: |
| GT | `calculate_true_effect.py` | 5 Universen Simulation (V3.3) | Kontrafaktische Welten A–E | RR: Fach=**0,958**, Übf=**0,939**, Psych=**0,951** | ✅ Benchmark |
| 19 | `deep_survival.py` | DeepSurv Landmark HR | Statischer Indikator S1-S2 | HR: Fach=1,092 ❌, Übf=1,053 ❌, Psych=0,932 ✅ | Gelaufen |
| 21 | `extended_cox_delta.py` | Extended Cox PH | Semester-aktiver Support | Exakte Cox-HRs per Support-Typ | ⏳ Im Runner ausführen |
| 24 | `dml_orthogonal_survival.py` | DML Orthogonal Hazard | Orthogonalisierte Residuen | RR: Fach=0,799 ❌, Übf=1,098 ❌, Psych=0,908 ✅ | Gelaufen |
| 36 | `train_transformer_dml.py` | Deep Transformer-DML | Transformer-Repräsentation + DML | RR: Fach=1,017 ⚠️, Übf=0,996 ⚠️, Psych=0,957 ✅ | Gelaufen |
| CF-1 | `counterfactual_hr_analyzer.py` | DeepSurv Panel CF | $X_{\text{treat}}=1$ vs $X_{\text{ctrl}}=0$ | Kausale HR Verteilung (Mean, Median, CI) | ⏳ Ausführen |
| CF-2 | `counterfactual_hr_delta.py` | DeepSurv Delta CF | $X_{\text{treat}}=1$ vs $X_{\text{ctrl}}=0$ | Kausale HR Verteilung (Mean, Median, CI) | ⏳ Ausführen |
| CF-3 | `counterfactual_rr_logistic_hazard_delta.py` | DTL Hazard Delta CF | $X_{\text{treat}}=1$ vs $X_{\text{ctrl}}=0$ | Kausales RR Verteilung (Mean, Median, CI) | ⏳ Ausführen |
| CF-4 | `counterfactual_rr_deephit_delta.py` | DeepHit Delta CF | $X_{\text{treat}}=1$ vs $X_{\text{ctrl}}=0$ | Kausales RR Verteilung (Mean, Median, CI) | ⏳ Ausführen |
| CF-5 | `counterfactual_inference_semester_transformer.py` | Semester-Transformer CF | $X_{\text{treat}}=1$ vs $X_{\text{ctrl}}=0$ | Kausale HR (Mean, Median, Global) | ⏳ Ausführen |
| CF-6 | `counterfactual_rr_exam_rnn_delta.py` | Exam-GRU Delta CF | $X_{\text{treat}}=1$ vs $X_{\text{ctrl}}=0$ | Kausales RR Verteilung (Mean, Median, CI) | ⏳ Ausführen |

---

## Teil 2: Modellklassen-Gruppierung (Klar & Diskriminierungsfrei)

```
KLASSE 1: Statische Klassifikation (Abbruch / Abschluss / Exmatrikulation)
  → Mitglieder: Naive Bayes, Random Forest, SVM, Keras MLP (+ jeweilige Blind-Versionen)
  → Features: HZB, Erwerbstätigkeit, Erstakademiker, Studiengang, HZB-Typ + S1-S2-Performance (11 Features)
  → Target: Absolventenstatus (multiclass / binary)

KLASSE 2a: Statische Noten-Regression (Landmark)
  → Mitglieder: Linear Ridge, SVR, Random Forest Regressor, Keras MLP Regressor
  → Features: Identisch mit Klasse 1 (Demographie + S1-S2 Aggregat)
  → Target: Abschlussnote (Imputiert / Absolventen)

KLASSE 2b: Semester-Sequenz Noten-Regression
  → Mitglieder: Semester-LSTM, Semester-Transformer, Deep Semester-Transformer (nach Fix)
  → Features: Zeitreihe der Semesterleistungen (CP, Fails, 4x Support) + Statische Demographie
  → Target: Gesamt-Notendurchschnitt (GPA)

KLASSE 3: Exam-Sequenz Noten-Regression
  → Mitglieder: Exam-GRU Regressor, Exam-Transformer Regressor
  → Features: Zeitreihe pro Prüfung (Versuch, CP, Schwierigkeit, Support, Lag-Fails) + Demographie
  → Target: Durchschnittliche Prüfungsnote

KLASSE 4: Landmark Survival (Statisch, Zeithorizont ab S3)
  → Mitglieder: DeepSurv Landmark, Discrete-Time Logistic Hazard Landmark
  → Features: Demographie + S1-S2 Leistungs- und Support-Merkmale
  → Target: Überlebenszeit (t_stop) & Event (Dropout)

KLASSE 5: Semester-Panel Survival (Counting Process Längsschnitt)
  → Mitglieder: Extended Cox Delta, Extended DeepSurv Panel/Delta, Extended DTL Hazard Panel/Delta, DML Orthogonal, Oracle DTL
  → Features: Person-Semester Panel (t_start, t_stop, fails_prev, delta_cp_prev, cp_rueckstand, Support-Aktivität, Demographie)
  → Target: Event (Dropout im Semesterintervall)

KLASSE 6: Semester-Sequenz Survival (TimeDistributed Hazard)
  → Mitglieder: Recurrent Survival GRU (Base/Delta/Blind), Transformer Survival, Dynamic DeepHit (Base/Delta)
  → Features: 8-Feature Sequenztensor (sem_gpa, sem_cp, sem_fails, 3x Support, hzb_note, erwerbstaetigkeit_std)
  → Target: Event-Signal am finalen Semesterschritt

KLASSE 7: Exam-Sequenz Survival (Prüfungs-Trajektorie)
  → Mitglieder: Recurrent Exam GRU (Base/V2/Delta), Exam Causal Transformer, Deep Exam-Transformer Survival (nach Refactoring)
  → Features: Harmonisierter 9-Feature Prüfungstensor (siehe Abschnitt 4.2)
  → Target: Event-Signal am finalen Prüfungsschritt (mit Last-Exam Exclusion vor dem Dropout-Moment)

KLASSE 8: Kausal-Inferenz & Causal ML
  → Mitglieder: Ground Truth (5 Universen), Extended Cox Delta, DML Orthogonal, Deep Transformer-DML, plus 6x Counterfactual Inference Skripte
  → Ziel: Identifikation des unvoreingenommenen Kausaleffekts (RR / HR) pro Support-Typ
```

---

## Teil 3: Vollständiger HR/RR-Vergleich (Aktueller Stand & Ausblick)

| Modellklasse & Modell | Methode | RR / HR fachlich | RR / HR überfachlich | RR / HR psychosozial | Daten-Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ground Truth (Simulation V3.3)** | 5 Universen Simulation | **0,9579** | **0,9387** | **0,9514** | ✅ Ground Truth |
| **Klasse 4:** DeepSurv Landmark | Cox Hazard Ratio | 1,0921 ❌ | 1,0534 ❌ | 0,9323 ✅ | Vorhanden (Landmark Confounding) |
| **Klasse 5:** Extended Cox Delta | Semi-parametrische Cox PH | *(Wird berechnet)* | *(Wird berechnet)* | *(Wird berechnet)* | ⏳ Im nächsten Runner-Lauf |
| **Klasse 5:** DML Orthogonal Survival | Double ML Residual Hazard | 0,7994 ❌ | 1,0980 ❌ | 0,9078 ✅ | Vorhanden (Residuales Confounding) |
| **Klasse 8:** Deep Transformer-DML | Causal Transformer + DML | 1,0172 ⚠️ | 0,9957 ⚠️ | 0,9569 ✅ | Vorhanden (Überdämpfung) |
| **Klasse 5:** CF DeepSurv Delta | Kontrafaktische Simulation | *(Wird berechnet)* | *(Wird berechnet)* | *(Wird berechnet)* | ⏳ Skript bereit |
| **Klasse 5:** CF DTL Hazard Delta | Kontrafaktische Simulation | *(Wird berechnet)* | *(Wird berechnet)* | *(Wird berechnet)* | ⏳ Skript bereit |
| **Klasse 6:** CF DeepHit Delta | Kontrafaktische Simulation | *(Wird berechnet)* | *(Wird berechnet)* | *(Wird berechnet)* | ⏳ Skript bereit |
| **Klasse 6:** CF Semester-Transformer | Kontrafaktische Simulation | *(Wird berechnet)* | *(Wird berechnet)* | *(Wird berechnet)* | ⏳ Skript bereit |
| **Klasse 7:** CF Exam-GRU Delta | Kontrafaktische Simulation | *(Wird berechnet)* | *(Wird berechnet)* | *(Wird berechnet)* | ⏳ Skript bereit |

---

## Teil 4: Implementierungsplan & Maßnahmenkatalog

### 4.1 Refactoring von `deep_transformer_regression.py`

1. **Semester-Regressor (Klasse 2b):**
   - Entfernen des unvollständigen Ad-hoc Datenladers.
   - Einbindung der Standardfunktion `create_semester_timeseries_dataset(data_dir)` aus `timeseries_semester.py`.
   - Identischer 70/15/15 Split, identisches GPA-Target $\rightarrow$ Faire Vergleichbarkeit mit Semester-LSTM und Semester-Transformer.

2. **Exam-Regressor (Klasse 3):**
   - Entfernen von `note` aus den Input-Features (Beseitigung des trivialen Selbstvorhersage-Leakages).
   - Einbindung der Prüfungsserie inklusive Support-Features und HZB/Erwerbstätigkeit (analog `timeseries_exam.py`).

3. **Exam-Survival (Klasse 7):**
   - Umstellung auf den harmonisierten 9-Feature Prüfungstensor (siehe 4.2).
   - **Beseitigung des Future-Data Leakages:** Anwendung der *Last-Exam Exclusion* für Abbrecher bzw. Prognose aus dem Verlauf vor Eintritt des finalen Exmatrikulationsereignisses.

---

### 4.2 Feature-Harmonisierung & Standard-Builder

#### A. Klasse 6 (Semester-Sequenz) $\rightarrow$ Kanonischer 8-Feature Satz
Vereinheitlichung aller Modelle (`recurrent_survival_model`, `recurrent_survival_model_delta`, `transformer_survival_model`, `dynamic_deephit_model`, `dynamic_deephit_delta_model`):
- `sem_gpa`, `sem_cp`, `sem_fails`, `fach_act`, `uebf_act`, `psych_act`, `hzb_note`, `erwerbstaetigkeit_std`.
- `cp_rueckstand` wird gedroppt, um absolute Äquivalenz herzustellen.

#### B. Klasse 7 (Exam-Sequenz) $\rightarrow$ Kanonischer 9-Feature Satz
Implementierung einer zentralen Funktion `build_canonical_exam_dataset(data_dir, max_exams=50, blind_type=None)`:
1. `versuch` (1, 2, 3)
2. `schwierigkeit` (Modulschwierigkeit)
3. `cp` (Modul-CP)
4. `is_fail` (Fehlversuchsindikator)
5. `fach_supp_act` (aktiver fachlicher Support)
6. `uebf_supp_act` (aktiver überfachlicher Support)
7. `psych_supp_act` (aktiver psychosozialer Support)
8. `hzb_note` (Abiturnote / HZB)
9. `erwerbstaetigkeit_std` (Arbeitsbelastung in Std/Woche)

---

### 4.3 Systematische Blind-Varianten

Durch die Vereinheitlichung der Dataset-Builder lassen sich Blind-Varianten konsistent über alle Klassen hinweg evaluieren:

| Variante | Maskierte Features (auf 0 gesetzt) | Getestete Hypothese | Anwendbare Modellklassen |
| :--- | :--- | :--- | :--- |
| **Standard (Voll)** | Keine (alle 8 bzw. 9 Features aktiv) | Maximale Vorhersagekraft mit allen verfügbaren Signalen | Alle Klassen |
| **Noten-Blind** | `sem_gpa = 0` bzw. `is_fail = 0` und `hzb_note = 0` | Prognosegüte ohne Berücksichtigung historischer oder aktueller Notenleistungen | Klassen 1, 2a, 6, 7 |
| **Erwerbs-Blind** | `erwerbstaetigkeit_std = 0` | Einfluss der Erwerbstätigkeit auf Verzerrung und Diskriminierungsfreiheit | Klassen 1, 5, 6, 7 |
| **Voll-Blind** | Noten- und Erwerbsmerkmale maskiert | Reine Verhaltens- und Strukturprognose (nur CP & Support-Nutzung) | Klassen 6, 7 |

---

### 4.4 Ergänzung der Master-Orchestrierung (`run_all_experiments.py`)

Folgende Aufrufe werden in `run_all_experiments.py` und `run_overnight.py` integriert:

1. **Extended Cox Delta:**
   ```python
   from extended_cox_delta import build_delta_panel, fit_extended_cox_delta
   panel = build_delta_panel(data_dir)
   fit_extended_cox_delta(panel, base_dir=data_dir)
   ```
2. **Recurrent Exam Survival V2:**
   ```python
   from recurrent_exam_survival_v2 import train_recurrent_exam_survival_v2
   train_recurrent_exam_survival_v2(data_dir)
   ```
3. **Automatisierter Counterfactual-Block (Schritt 10 im Runner):**
   - Ausführung aller 6 zentralen Counterfactual-Analysen (`counterfactual_hr_analyzer`, `counterfactual_hr_delta`, `counterfactual_rr_logistic_hazard_delta`, `counterfactual_rr_deephit_delta`, `counterfactual_inference_semester_transformer`, `counterfactual_rr_exam_rnn_delta`).
   - Speicherung aller generierten HR/RR-Werte in einer zusammenfassenden JSON-Tabelle `causal_counterfactual_comparison.json`.

---

## 5. Arbeitsplan für die Umsetzung

1. **Schritt 1: Refactoring von `deep_transformer_regression.py`** (Behebung von Bug & Leakage, Angleichung an Klasse 2b und Klasse 7).
2. **Schritt 2: Feature-Harmonisierung der Klassen 6 und 7** (Standardisierter 8- und 9-Feature Builder).
3. **Schritt 3: Implementierung der Blind-Pipelines** für Klasse 6 und Klasse 7.
4. **Schritt 4: Runner-Aktualisierung & Ausführung** aller Modelle inklusive Extended Cox Delta, Exam V2 und aller Counterfactual-Skripte.
5. **Schritt 5: Gesamt-Evaluation & Update der Dokumentation** (ReadMe, Hypothesis Evolution, Walkthrough).
