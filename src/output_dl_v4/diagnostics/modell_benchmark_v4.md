# Master-Nachtlauf Benchmark-Report: Modell-Suite auf V4-Daten

**Datengrundlage:** Simulation V4 (50.000 Studierende, 8 Paralleluniversen, varianzkorrigierte Beta-Verteilungen mit $\kappa_{\text{HZB}}=6.5$, $\kappa_{\text{Alter}}=12.8$, $\kappa_{\text{Walk}}=95.0$)  
**Orchestrierung:** `run_overnight.py` mit `--data_dir output_dl_v4 --skip_sim`  
**Gesamtlaufzeit:** 195,36 Minuten (~3,25 Stunden) – **20 von 20 Pipeline-Schritten erfolgreich (100% PASSED)**  

---

## 1. Executive Summary & Modell-Synopse

| Modell-Klasse | Modell | Zielgröße | ROC-AUC | PR-AUC | $R^2$ / Acc | Brier |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Klasse 1 (Landmark)** | Keras MLP Classifier | Dropout (Sem 2) | **0.8716** | **0.7484** | 84,61 % | 0.1109 |
| | SVM (RBF) | Dropout (Sem 2) | 0.8219 | 0.7198 | **84,63 %** | 0.1195 |
| | Random Forest | Dropout (Sem 2) | 0.8507 | 0.7154 | 83,49 % | 0.1199 |
| | Naive Bayes | Dropout (Sem 2) | 0.8581 | 0.7271 | 83,39 % | 0.1419 |
| | Keras MLP Regressor | Abschlussnote | — | — | **0.8844** (MAE 0.16) | — |
| | SVR (RBF) | Abschlussnote | — | — | 0.8828 (MAE 0.16) | — |
| | Random Forest Regressor | Abschlussnote | — | — | 0.8680 (MAE 0.17) | — |
| | Ridge Regression | Abschlussnote | — | — | 0.8674 (MAE 0.18) | — |
| **Klasse 2 (Orakel / DSGVO)** | Logistic Hazard (Full) | Dropout (Panel) | 0.7955 | 0.1552 | — | — |
| | Logistic Hazard (Realistic/DSGVO) | Dropout (Panel) | 0.7809 | 0.1511 | — (Verlust: -0.0147) | — |
| | Logistic Hazard (Oracle) | Dropout (Panel) | **0.8029** | — | — (+0.0086 Lift) | — |
| **Klasse 3 (DML Causal)** | DML Orthogonal Survival | Hazard Panel | 0.7966 | 0.1668 | — | 0.0341 |
| | Deep Transformer-DML | Hazard Panel | 0.7948 | 0.1555 | — | 0.0343 |
| **Klasse 5 (Trad. Survival)** | Extended Cox (PHReg) | Hazard Ratio | (Stat. Modell) | — | — | — |
| | Extended Logistic Hazard | Hazard Panel | **0.7986** | 0.1694 | — | **0.0340** |
| | Extended DeepSurv | Hazard Panel | 0.5695 | 0.0512 | — | — |
| **Klasse 6 (Semester Seq.)** | Dynamic DeepHit Competing | Dropout / Abschluss | **0.8109** / **0.9995** | 0.1917 / 0.9951 | — | 0.0346 |
| | Recurrent Survival GRU | Dropout (Semester) | 0.7972 | 0.1880 | — | 0.0341 |
| | Causal Semester Transformer | Dropout (Semester) | 0.7941 | 0.1887 | — | 0.0341 |
| | Semester Timeseries Transformer | Abschlussnote | — | — | **0.9915** (RMSE 0.057) | — |
| | Semester Timeseries LSTM | GPA Semester | — | — | 0.7807 (RMSE 0.544) | — |
| **Klasse 7 (Exam Seq.)** | Recurrent Exam Survival GRU | Dropout (Exam) | **0.9096** | **0.2404** | — | **0.0137** |
| | Causal Exam Transformer | Dropout (Exam) | 0.8927 | 0.1730 | — | 0.0144 |
| | Exam Timeseries Transformer | Klausurnote | — | — | **0.9942** (RMSE 0.047) | — |
| | Exam Timeseries GRU | Klausurnote | — | — | 0.0941 | — |
| **Klasse 8 (Deep Suite)** | Deep Semester Transformer Regr. | Abschlussnote | — | — | **0.9869** (RMSE 0.071) | — |
| | Deep Exam Transformer Regr. | Klausurnote | — | — | **0.9918** (RMSE 0.056) | — |
| | Deep Exam Transformer Survival | Dropout (Exam) | 0.8938 | 0.1507 | — | — |
| **Klasse 8B (Autoregressor)** | Next-Exam Dual-Head (Note $k+1$) | Klausurnote | — | — | $R^2 = 0.4216$ (MAE 0.75) | — |
| | Next-Exam Dual-Head (Pass $k+1$) | Bestehen Klausur | **0.9277** | **0.9868** | — | 0.0750 |

---

## 2. Detaillierte Befunde nach Modell-Klassen

### 2.1 Landmark Baselines (Klasse 1)
- **Klassifikation (Dropout-Risiko nach Semester 2):**
  - Keras MLP führt mit **ROC-AUC = 0.8716** und **PR-AUC = 0.7484** bei einer Genauigkeit von **84,61 %**.
  - Random Forest (85,07 % AUC) und Naive Bayes (85,81 % AUC) folgen dicht dahinter.
- **Abschlussnoten-Regression:**
  - Keras MLP Regressor erreicht **$R^2 = 0.8844$** (MAE = 0.1619 Notenpunkte).
  - SVR (0.8828), Random Forest (0.8680) und Ridge (0.8674) zeigen eine konsistente Vorhersagekraft nach dem Grundstudium.

### 2.2 Sequenzmodelle auf Prüfungs- vs. Semesterebene (Klasse 6 & 7)
- **Prüfungsebene (Exam-Level):**
  - Das **Recurrent Exam Survival GRU** erzielt mit **ROC-AUC = 0.9096** und **PR-AUC = 0.2404** (Brier = 0.0137) den höchsten Diskriminationswert für den genauen Dropout-Zeitpunkt.
  - Der **Exam Timeseries Transformer** prognostiziert Einzelnoten mit **$R^2 = 0.9942$** (RMSE = 0.047).
- **Semesterebene (Semester-Level):**
  - **Dynamic DeepHit** dominiert bei Competing Risks: **ROC-AUC = 0.8109** für Dropout und **ROC-AUC = 0.9995** für erfolgreichen Studienabschluss.
  - **Semester Timeseries Transformer** erreicht für die Abschlussnote **$R^2 = 0.9915$**.

### 2.3 Kausalschätzungen & Hazard Ratios (Klasse 5 & 3)
- **Extended Cox Modell (Statsmodels PHReg):**
  - `fach_supp_count`: **$\text{HR} = 0.8902$** ($p < 0.0001$) $\rightarrow$ Signifikant protektiv (~11 % Hazard-Reduktion pro fachlicher Teilnahme).
  - `uebf_supp_count`: **$\text{HR} = 1.0081$** ($p = 0.369$).
  - `psych_supp_count`: **$\text{HR} = 0.9850$** ($p = 0.257$).
  - `fails_prev`: **$\text{HR} = 1.1982$** ($p < 0.0001$) $\rightarrow$ Jeder Fehlversuch im Vorsemester erhöht das Dropout-Risiko um ~20 %.
  - `delta_cp_prev`: **$\text{HR} = 0.9500$** ($p < 0.0001$) $\rightarrow$ Jeder erworbene CP senkt das Risiko um 5 %.
- **Double Machine Learning (DML):**
  - Behält mit ROC-AUC = 0.7966 und Brier = 0.0341 die Vorhersagequalität bei voller Residual-Orthogonalisierung zur Ausschaltung des Confounding-by-Indication.

### 2.4 Orakel- und DSGVO-Realitätsanalyse (Klasse 2)
- **Orakel-Lift ($\Delta$ AUC):**
  - Zugriff auf unbeobachtbare mentale Zustände (`hidden_motivation`, `hidden_soziale_integration`, `hidden_erwartete_note`) bringt einen Lift von **+0.0086** im Logistic Hazard Modell (0.7942 $\rightarrow$ 0.8029).
- **DSGVO / Feature-Blindness:**
  - Entfernung sensibler demographischer Merkmale (Migration, Erstakademiker, Erwerbsstunden) senkt die Vorhersagekraft moderat von **0.7955 auf 0.7809** ($\Delta = -0.0147$). Das System bleibt auch unter strengen Datenschutzauflagen einsatzfähig.

### 2.5 Autoregressor & Strukturelle Mediation (Klasse 8B & AP8)
- **Next-Exam Autoregressor:**
  - Bestehen der nächsten Klausur ($k+1$): **ROC-AUC = 0.9277**, **PR-AUC = 0.9868**, Brier = 0.0750.
  - Note der nächsten Klausur: **$R^2 = 0.4216$**, MAE = 0.7486 Notenpunkte.
- **Strukturelle Mediationsanalyse (Imai / Pearl Framework):**
  - **Fachlicher Support:** Gesamteffekt $\text{OR} = 0.9347$, davon 4,8 % mediiert über die Klausurnote ($\text{ACME OR} = 0.9968$), 95,2 % direkter Entlastungseffekt.
  - **Überfachlicher Support:** $\text{OR} = 1.0409$, Anteil über Noten mediiert: 82,8 %.
  - **Psychosozialer Support:** $\text{OR} = 0.9804$, direkter schützender Effekt ($\text{ADE OR} = 0.9660$).

---

## 3. Laufzeit- und Ressourcenprofil

| Phase / Schritt | Dauer | RAM Peak |
| :--- | :---: | :---: |
| 1. Extended Cox PHReg | 11,3 s | 453 MB |
| 2. Extended DeepSurv / Logistic Hazard | 3,7 Min. | 656 MB |
| 3–5. Semester Sequence Suite (GRU, DeepHit, Transformer) | 6,9 Min. | 1.090 MB |
| 6–7. Exam Sequence Suite (GRU, Transformer) | 19,2 Min. | 1.430 MB |
| 8–9. Landmark Baselines & Regressionen | 4,2 Min. | 1.614 MB |
| 10–11. DML & Transformer-DML | 8,4 Min. | 1.691 MB |
| 12–15. Timeseries Regressionen (LSTM, GRU, Transformers) | 20,3 Min. | 2.002 MB |
| 16–17. Oracle & DSGVO Benchmarks | 2,0 Min. | 2.028 MB |
| 18. Deep Transformer Suite (Enlarged Capacity) | 97,8 Min. | 2.599 MB |
| 19. Autoregressive Next-Exam Prediction | 32,5 Min. | 2.672 MB |
| 20. Strukturelle Mediationsanalyse | 7,7 s | 2.616 MB |
| **Gesamte Master-Pipeline** | **195,4 Min. (3,25 h)** | **2.672 MB** |
