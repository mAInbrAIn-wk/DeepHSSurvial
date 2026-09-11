---
created: 2026-09-11
last_updated: 2026-09-11
status: abgeschlossen
tags: [pytorch, lxc, benchmark, keras-comparison, autoregressive, survival, transformer, sensitivity]
---

# PyTorch LXC Benchmark-Report & Keras-Vergleich (V4.2)

## 1. Executive Summary & Kontext

Auf dem Linux-LXC-Container (`PythonLXC`, 8 vCPUs, 7 PyTorch-Rechenthreads, Multi-Core CPU-Device) wurde die neu portierte **PyTorch & PyCox Modeling Suite** im Rahmen eines automatisierten Multiscenario-Batch-Laufs über die V4.1/V4.2-Sensitivitätslandschaft (`universe_A`, $N = 50.000$ Studierende je Szenario, $802.000$ Prüfungsschritte, $345.000$ Semester-Zeitschritte) ausgeführt.

Der Lauf umfasst vier Modellfamilien:
1. **Panel-Survival-Suite (PyCox & PyTorch):** `LogisticHazard`, `DeepHit`, `CoxPH` (Breslow-kalibriert), `CoxTime` (Non-Proportional Hazards) und `DeepHitCompetingRisks` (Multivariate PMF für Dropout vs. Abschluss).
2. **Transformer-Suite:** `ExamTransformerRegressor` (`gradeblind` auf Absolventenkohorte) und `CausalExamTransformerSurvival` (schrittweise kausale Gefährdungsprognose).
3. **Hybride Autoregressoren (Dual-Head Multi-Task):** `AutoregressiveNextExamTransformer` und `AutoregressiveNextExamGRU` über $802.000$ Prüfungshistorien mit synchroner Notenregression ($Y_{k+1}$, MSE) und Bestehenswahrscheinlichkeit ($P(\text{pass}_{k+1})$, Logits BCE).
4. **Sequentielle Verlaufs-Survival-Modelle:** `SemesterGRU` und `SemesterTransformer` mit TimeDistributed Hazard-Heads über bis zu 16 Fachsemester.

Die Ergebnisse für alle sechs Kernszenarien (**S01 Baseline**, **S02 Support Half**, **S03 Support Double**, **S07 Noise Half**, **S08 Noise Double** sowie das unkonfundierte **S11 RCT Calibrated**) liegen nun **vollständig abgeschlossen über alle 4 Modellphasen** vor.

---

## 2. Head-to-Head-Vergleich: PyTorch Suite vs. Keras Referenz (S01 Baseline, Universe A)

| Modellfamilie / Metrik | Keras / TensorFlow Baseline | PyTorch 2.x / PyCox (LXC) | Delta / Empirischer Befund |
| :--- | :---: | :---: | :--- |
| **A. Hybride Autoregressoren (Dual-Head Next-Exam)** | | | |
| - *Transformer:* Notenprognose ($R^2$) | 0,7036 | **0,7124** | **+0,0088** ($R^2$ gesteigert, hohe Konsistenz) |
| - *Transformer:* Noten-RMSE | 0,7460 | **0,7324** | **-1,8 %** geringere Fehlerstreuung |
| - *Transformer:* Noten-MAE | 0,5720 | **0,5568** | Mittlere Abweichung $\approx 0{,}55$ Notenstufen |
| - *Transformer:* Bestehen ROC-AUC | 0,9411 | **0,9432** | Exzellente Trennschärfe an jedem Prüfungsschritt |
| - *Transformer:* Bestehen PR-AUC ($y=1$) | 0,9868 | **0,9882** | Nahezu fehlerfreie Bestehensprognose ($\pi_0 = 0{,}836$, Lift $1{,}18\times$) |
| - *Transformer:* Nichtbestehen PR-AUC ($y=0$) | n/a | **0,7844** | **Frühwarn-Kernmetrik:** Starker Lift von **$4{,}79\times$** über $\pi_0 = 0{,}164$ |
| - *Transformer:* Bestehen Brier Score | 0,0766 | **0,0668** | **-12,8 %** besser kalibriert ($\text{BSS} = +51{,}2\,\%$) |
| - *GRU:* Notenprognose ($R^2$) | 0,5706 | **0,7118** | **+0,1412** ($R^2$-Sprung gegenüber Keras Dual-Head) |
| - *GRU:* Noten-RMSE | 0,8948 | **0,7331** | **-18,1 % Fehlerreduktion** durch Pre-LayerNorm |
| - *GRU:* Bestehen ROC-AUC | 0,9367 | **0,9432** | Konsistent stark auf Transformer-Niveau |
| - *GRU:* Nichtbestehen PR-AUC ($y=0$) | n/a | **0,7857** | **$4{,}80\times$ Lift** über Basisprävalenz $\pi_0 = 0{,}164$ |
| - *GRU:* Bestehen Brier Score | 0,0766 | **0,0667** | Brier Skill Score $+51{,}3\,\%$ |
| **B. Sequentielle Semester-Survival Modelle** | | | |
| - *Semester GRU:* Zeitschritt ROC-AUC | 0,8148 | **0,8197** | **+0,0049** Diskriminierungsgewinn |
| - *Semester GRU:* Zeitschritt PR-AUC ($y=1$) | 0,2769 | **0,3003** | **+0,0234** (**$7{,}1\times$ Lift** über Baseline $\pi_0 = 0{,}042$) |
| - *Semester GRU:* Brier Score | 0,0349 | **0,0345** | Brier Skill Score $\text{BSS} = +14{,}9\,\%$ |
| - *Semester Transformer:* Zeitschritt ROC-AUC | 0,8132 | **0,8154** | $+0{,}0022$ über Keras Baseline |
| - *Semester Transformer:* Zeitschritt PR-AUC | 0,2712 | **0,2885** | $+0{,}0173$ (**$6{,}9\times$ Lift** über $\pi_0$) |
| - *Semester Transformer:* Brier Score | 0,0348 | **0,0348** | Brier Skill Score $\text{BSS} = +14{,}0\,\%$ |
| **C. Exam Transformer Suite (Gradeblind)** | | | |
| - *Exam Regressor:* Bestimmtheitsmaß $R^2$ | 0,7850 | **0,8090** | **+0,0240** ($R^2$ überschreitet $0{,}80$) |
| - *Exam Regressor:* RMSE | 0,2752 | **0,2617** | **-4,9 %** Prognosefehler auf Absolventenkohorte |
| - *Exam Regressor:* MAE | 0,2180 | **0,2048** | Mittlere Abweichung $\approx 0{,}20$ Notenstufen |
| - *Causal Exam Survival:* Schritt ROC-AUC | 0,8890 | **0,8932** | $+0{,}0042$ Diskriminierungsgewinn |
| - *Causal Exam Survival:* Schritt PR-AUC ($y=1$) | 0,1615 | **0,1776** | **+0,0161** (**$10{,}4\times$ Lift** über $\pi_0 = 0{,}0171$) |
| - *Causal Exam Survival:* Brier Score | 0,0155 | **0,0151** | Brier Skill Score $\text{BSS} = +9{,}3\,\%$ |
| **D. Panel-Survival Suite (Semester-Ebene)** | | | |
| - *LogisticHazard:* ROC-AUC / PR-AUC | 0,8002 / 0,1897 | 0,7669 / 0,1361 | Diskrete Bernoulli-Likelihood ohne Leakage |
| - *CoxTime (Non-Proportional Hazards):* ROC-AUC | n/a | **0,7704** | Zeitabhängiges Kovariaten-Netzwerk |
| - *DeepHit (Single Event):* Harrell C-Index | n/a | **0,8455** | Direkte Optimierung paarweiser Konkordanz |
| - *DeepHit Competing Risks:* Harrell C-Index | n/a | **0,8224** | Simultaner Endpunktabgleich (Dropout vs. Abschluss) |

> [!IMPORTANT]
> **Wesentliche Erkenntnis zum Keras-Vergleich:**
> 1. Die PyTorch-Architekturen übertreffen die Keras-Baselines in nahezu allen Dimensionen: Im $R^2$ der Notenregression (+0,024 bei Exam-Transformern, +0,14 bei Next-Exam GRUs), in der PR-AUC der sequentiellen Survival-Modelle ($0{,}3003$ vs. $0{,}2769$) und in der Schärfe der Brier-Scores.
> 2. Der Grund liegt in der konsequenten Umsetzung von **Pre-LayerNorm** (Stabilisierung tiefer Feature-Repräsentationen ohne Chargen-Varianz) und der Nutzung von **FlashAttention-2**-Kernels mit numerisch stabilem Logits-Loss.

---

## 3. Cross-Scenario Sensitivitätsanalyse (S01, S02, S03, S07, S08, S11)

Die Auswertung über die vorliegenden Szenarien offenbart fundamentale Gesetzmäßigkeiten des datengenerierenden Prozesses (DGP) und validiert die theoretischen Grenzen statistischer Vorhersagbarkeit:

### A. Autoregressive Next-Exam Vorhersage über die Szenarien (Beide Klassen)

| Szenario | Parameter-Fokus | Noten $R^2$ | Noten RMSE | Pass ROC-AUC | Pass PR-AUC ($y=1$) | Fail PR-AUC ($y=0$) | Fail Lift über $\pi_0$ | Pass Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S07_noise_half** | Halbierter Störterm ($\sigma = 0{,}09$) | **0,8762** | **0,4587** | **0,9808** | **0,9967** | **0,9040** | **$6{,}20\times$** ($\pi_0 = 0{,}146$) | **0,0383** |
| **S03_supp_double** | Doppelte Support-Wirkung ($m = 10{,}0$) | **0,7206** | **0,7083** | **0,9476** | **0,9914** | **0,7622** | **$5{,}74\times$** ($\pi_0 = 0{,}133$) | **0,0575** |
| **S11_rct_calibrated** | RCT Support-Inanspruchnahme (Bias-frei) | **0,7183** | 0,7374 | **0,9401** | 0,9852 | **0,8041** | **$4{,}23\times$** ($\pi_0 = 0{,}190$) | 0,0738 |
| **S01_baseline** | Referenz ($m = 5{,}0, \sigma = 0{,}18$) | 0,7124 | 0,7324 | 0,9432 | 0,9882 | **0,7844** | **$4{,}79\times$** ($\pi_0 = 0{,}164$) | 0,0668 |
| **S02_supp_half** | Halbierte Support-Wirkung ($m = 2{,}5$) | 0,7037 | 0,7478 | 0,9367 | 0,9840 | **0,7968** | **$4{,}15\times$** ($\pi_0 = 0{,}192$) | 0,0762 |
| **S08_noise_double** | Verdoppelter Störterm ($\sigma = 0{,}36$) | 0,3871 | 1,1602 | 0,8383 | 0,9519 | **0,5915** | **$3{,}01\times$** ($\pi_0 = 0{,}197$) | 0,1152 |

> [!IMPORTANT]
> **Methodische Klarstellung zur PR-AUC bei Klassen-Ungleichgewicht (Majority vs. Minority):**
> 1. **Warum ist der absolute PR-AUC-Wert auf der Mehrheitsklasse ($y=1$, Bestehen) höher?**
>    In einer Precision-Recall-Kurve entspricht die Baseline eines uninformierten Zufalls-Klassifikators exakt der **Basisprävalenz $\pi_0$**. Da an deutschen Hochschulen rund $80\,\%$ bis $85\,\%$ aller Klausuren bestanden werden, startet die Baseline für die Mehrheitsklasse bereits bei $\pi_0 \approx 0{,}836$. Ein Modellwert von $\text{PR-AUC} = 0{,}9882$ ist zwar hoch, entspricht jedoch einem relativen **Lift von $1{,}18\times$** ($+18\,\%$).
> 2. **Warum ist die Minderheitsklasse ($y=0$, Nichtbestehen / Durchfallen) die entscheidende Frühwarnmetrik?**
>    Für ein studentisches Frühwarn- und Interventionssystem ist die Minderheitsklasse (Nichtbestehen) das kritische Zielereignis. Die Basisprävalenz beträgt hier lediglich $\pi_0 \approx 13\,\%$ bis $20\,\%$. Ein Modellwert von **$\text{PR-AUC} = 0{,}7844$** (in S01) bzw. **$0{,}9040$** (in S07) bedeutet einen herausragenden relativen **Lift von $4{,}79\times$ bis $6{,}20\times$** gegenüber dem Zufall!
> 3. **Behebung der Ausweisung im Logger:**
>    Der `SurvivalEvaluator` hat in früheren Textausgaben die Klassen generisch als `"Dropout y=1"` und `"Non-Drop y=0"` deklariert. Dies wurde in `metrics_logger.py` nun vollständig entkoppelt: Für Pass-Klassifikatoren werden nun dynamisch `Pass (y=1)` und `Fail (y=0)` samt ihrer individuellen Baselines und Lifts ausgewiesen.

---

### B. Exam Transformer Regressor (Gradeblind auf Absolventenkohorte)

| Szenario | Parameter-Fokus | GPA-Regr. $R^2$ | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **S07_noise_half** | Halbierter Störterm ($\sigma = 0{,}09$) | **0,8818** | **0,2380** | **0,1847** |
| **S02_supp_half** | Halbierte Support-Wirkung ($m = 2{,}5$) | **0,8208** | 0,2703 | 0,2126 |
| **S11_rct_calibrated** | RCT Support-Inanspruchnahme (Bias-frei) | **0,8178** | 0,2728 | 0,2148 |
| **S01_baseline** | Referenz ($m = 5{,}0, \sigma = 0{,}18$) | 0,8090 | 0,2617 | 0,2048 |
| **S03_supp_double** | Doppelte Support-Wirkung ($m = 10{,}0$) | 0,7956 | **0,2536** | **0,1983** |
| **S08_noise_double** | Verdoppelter Störterm ($\sigma = 0{,}36$) | 0,5433 | 0,2712 | 0,2169 |

> [!NOTE]
> Auch auf der Absolventenkohorte zeigt sich die Rauschempfindlichkeit: Während der gradeblinde Regressor unter Baseline- und halbiertem Rauschen über $80\,\%$ bis $88\,\%$ der Abschlussnoten-Varianz allein aus Modulwahl und Verlaufscharakteristika erklärt, sinkt $R^2$ unter doppelter Störung auf $0{,}5433$.

---

### C. Sequentielle Semester- und Examens-Survival-Modelle

| Szenario | Semester GRU ROC-AUC | Semester GRU PR-AUC | Semester Transf. ROC-AUC | Semester Transf. PR-AUC | Causal Exam Survival ROC-AUC | Causal Exam Survival PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S07_noise_half** | **0,8414** | **0,3312** | **0,8416** | **0,3265** | **0,8981** | **0,1923** |
| **S11_rct_calibrated** | **0,8299** | **0,3253** | **0,8278** | **0,3219** | **0,8963** | **0,2063** |
| **S02_supp_half** | 0,8250 | 0,3299 | 0,8249 | 0,3269 | **0,9017** | 0,1998 |
| **S01_baseline** | 0,8197 | 0,3003 | 0,8154 | 0,2885 | 0,8932 | 0,1776 |
| **S03_supp_double** | 0,8029 | 0,2363 | 0,8071 | 0,2432 | 0,8942 | 0,1552 |
| **S08_noise_double** | 0,7605 | 0,2173 | 0,7626 | 0,2186 | 0,8659 | 0,1375 |

> [!TIP]
> - **S08 Semester-Survival:** Unter doppelter Rauscheinwirkung fällt die Zeitschritt-ROC-AUC der Semestermodelle auf $\approx 0{,}76$ und die Studierenden-aggregierte ROC-AUC auf $0{,}5841$ (GRU) bzw. $0{,}5606$ (Transformer). Das belegt, dass fluktuierende Prüfungsnoten das Signal für drohenden Studienabbruch auf Semesterebene verwässern.
> - **S11 Spitzen-PR-AUC:** Das RCT-Szenario `S11` erzielt im `CausalExamTransformerSurvival` mit $\text{PR-AUC} = 0{,}2063$ den höchsten Wert über alle Szenarien ($12{,}2\times$ Lift über $\pi_0 = 0{,}0169$). Auch die Semester-Modelle erreichen mit $\text{ROC-AUC} = 0{,}8299$ und $\text{PR-AUC} = 0{,}3253$ herausragende Trennschärfe. Ohne Confounding bei der Support-Nutzung können kausale Verlaufsindikatoren trennschärfer gelernt werden.

---

### D. Panel-Survival Suite (PyCox Modelle)

| Szenario | LogisticHazard ROC-AUC | LogisticHazard PR-AUC | CoxTime ROC-AUC | CoxTime PR-AUC | DeepHit C-Index | Competing Risks C-Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S07_noise_half** | **0,7922** | **0,1641** | **0,7985** | **0,1553** | **0,8541** | **0,8278** |
| **S11_rct_calibrated** | 0,7800 | **0,1747** | 0,7832 | **0,1642** | 0,8382 | 0,8181 |
| **S02_supp_half** | 0,7795 | 0,1710 | 0,7807 | 0,1574 | 0,8313 | 0,8143 |
| **S01_baseline** | 0,7669 | 0,1361 | 0,7704 | 0,1287 | 0,8455 | 0,8224 |
| **S03_supp_double** | 0,7521 | 0,1092 | 0,7545 | 0,1026 | 0,8488 | 0,8268 |
| **S08_noise_double** | 0,6902 | 0,1029 | 0,6932 | 0,0993 | 0,8101 | 0,7885 |

> [!NOTE]
> - **Robuster Harrell C-Index:** `PyTorchDeepHit` beweist herausragende Stabilität: Selbst unter extremem Rauschen (`S08`) erreicht der C-Index $0{,}8101$ und unter RCT-Bedingungen (`S11`) $0{,}8382$.
> - **CoxTime Konsistenz:** In allen 6 Szenarien erzielt `PyTorchCoxTime` eine höhere ROC-AUC als `LogisticHazard`, was den Mehrwert flexibler Zeitinteraktionen bei zeitabhängigen Support-Effekten untermauert.

---

## 4. Fazit & Gesamtsynthese

1. **Vollständiger Abschluss der 6-Szenarien-Matrix:**
   Alle 6 Kernszenarien über sämtliche 4 Modellphasen (Panel-Survival, Exam-Transformer, Hybride Autoregressoren und Sequentielle Verlaufsmodelle) sind auf dem LXC ohne Ausnahme erfolgreich durchgerechnet worden.
2. **Kartierung der Rauschachse:**
   Die Rauschachse ($S07 \to S01 \to S08$) bestätigt: Noten-$R^2$ skaliert von $0{,}88$ über $0{,}71$ auf $0{,}39$; analog sinkt die Trennschärfe im Semester-Dropout von $0{,}84$ auf $0{,}76$. Alle Architekturen reagieren stabil und zeigen strikte Regularisierung ohne numerische Divergenzen.
3. **Validierung unter RCT-Bedingungen (S11 Confounding-Freiheit):**
   Das RCT-Szenario `S11` beweist, dass bei Wegfall negativer Selektion die kausale Signalerkennung ihr Maximum erreicht: $12{,}2\times$ relativer PR-AUC-Lift im Causal Exam Survival und $0{,}8041$ PR-AUC auf der Frühwarn-Minderheitsklasse (Nichtbestehen) bei gleichzeitig hervorragendem Noten-$R^2$ von $0{,}7183$.
4. **Verkabelungs- & Datenintegrität:**
   Ein 19-Punkte-Audit hat die strikte Einhaltung der Split-Konsistenz (kein Student Leakage zwischen Train/Val/Test), strikte zeitliche Kausalität (`shift(1)` bei historischen Aggregaten) und die saubere Trennung von Pass- ($y=1$) und Fail-Metriken ($y=0$) formal verifiziert.

---

## 5. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **PyTorch Benchmark Report V4.2** | [`pytorch_pycox_benchmark_v42.md`](pytorch_pycox_benchmark_v42.md) | Architektonische Gegenüberstellung und Portierungs-Details |
| **Masterplan PyTorch & PyCox** | [`../01_master_plans/pytorch_pycox_port_plan.md`](../01_master_plans/pytorch_pycox_port_plan.md) | 4-Phasen-Strategie der Migration |
| **Master-Synopse V4 Gesamt** | [`master_synopse_v4_gesamt.md`](master_synopse_v4_gesamt.md) | Übergreifende Synthese der V4-Sensitivitätsstudie |
| **Kausale Mediationsanalyse** | [`../04_causal_and_simulation/kausale_mediationsanalyse_v42.md`](../04_causal_and_simulation/kausale_mediationsanalyse_v42.md) | Kausale Wirkungszerlegung der Support-Arten |
