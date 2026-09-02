# Walkthrough: Vollständige Kausalevaluation, Feature-Harmonisierung & Modell-Ergebnisse (V3.3)

**Projekt:** DeepSupport – Wirksamkeitsanalyse von Hochschulsupport  
**Datum:** 21. August 2026  
**Status:** Alle 23+ Modellarchitekturen, Kausalschätzer, Counterfactual-Analysen und das Refactoring von `deep_transformer_regression.py` wurden vollständig berechnet und empirisch evaluiert.

---

## 1. Ground Truth der Makro-Kausaleffekte (5-Universen-Simulation V3.3)

Die 5-Universen-Simulation (50.000 Studierende je Universum mit 4 deterministisch synchronisierten RNG-Streams) liefert die unvoreingenommene Ground Truth:

| Universum | Support-Bedingung | Dropout-Quote | Relatives Risiko (RR) vs. A | Netto-Gerettete | Kausaler Effekt (Makro) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Universum A** | Baseline (Alle Support-Typen aktiv) | **27,37 %** | **1,0000** | — | Ausgangslage |
| **Universum B** | Kein Support (komplett blockiert) | **32,35 %** | **0,8462** | **+2.488** | **-15,38 % Risikoreduktion** (Gesamtsystem schützt) |
| **Universum C** | Kein fachlicher Support | **28,57 %** | **0,9579** | **+601** | **-4,21 % Risikoreduktion** |
| **Universum D** | Kein überfachlicher Support | **29,16 %** | **0,9387** | **+893** | **-6,13 % Risikoreduktion** |
| **Universum E** | Kein psychosozialer Support | **28,77 %** | **0,9514** | **+699** | **-4,86 % Risikoreduktion** |

> [!NOTE]
> Alle drei Support-Formen sind in der realen Simulationsmechanik wirksam und senken das Abbruchrisiko um ca. 4 % bis 6 %.

---

## 2. Umfassender Vergleich aller Kausal- & Counterfactual-Schätzer

Die empirische Counterfactual-Inferenz über alle trainierten Modelle im Vergleich zur Ground Truth zeigt ein klares und differenziertes Bild:

| Modell & Methode | Analyse-Level | RR / HR Fachlich | RR / HR Überfachlich | RR / HR Psychosozial | Methodische Diagnose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ground Truth (V3.3)** | Makro (5 Universen) | **0,9579** | **0,9387** | **0,9514** | Reale Kausalwirkung: Alle 3 Support-Typen schützen |
| **Extended Cox Delta** | Panel (Semi-parametrisch) | **0,8574** ($p<10^{-4}$) | **1,0940** ($p<10^{-4}$) | **0,8732** ($p<10^{-4}$) | Überschätzt Fachlich; Überfachlich fälschlich positiv ($+9,4\%$) |
| **Extended DeepSurv Panel** | Panel (Neural Cox Breslow) | Median HR = **0,9886**<br>(Mean: 0,9697) | Median HR = **1,0085**<br>(Mean: 1,0483) | Median HR = **0,9245**<br>(Mean: 0,9333) | **Beste Treffsicherheit** bei Fach & Psych; Überfachlich leicht über 1,0 |
| **Extended DeepSurv Delta** | Panel (mit Deltas) | Median HR = **0,9082**<br>(Mean: 0,9017) | Median HR = **1,0422**<br>(Mean: 1,0896) | Median HR = **0,9641**<br>(Mean: 0,9684) | Fachlich überschätzt ($-9,2\%$), Psychosozial gut getroffen ($-3,6\%$) |
| **Extended DTL Hazard Delta** | Panel (Discrete Hazard) | Median RR = **0,7718**<br>(Mean: 0,7530) | Median RR = **1,0381**<br>(Mean: 1,0290) | Median RR = **0,8823**<br>(Mean: 0,8610) | Starker Fach-Effekt ($-22,8\%$), Psychosozial ($-11,8\%$) |
| **DML Orthogonal Survival** | Panel (Double ML) | Mean RR = **0,7994**<br>(Median: 0,8404) | Mean RR = **1,0980**<br>(Median: 1,0516) | Mean RR = **0,9078**<br>(Median: 0,9154) | Konsistent mit Cox: Fach & Psych schützend, Überfachlich verzerrt |
| **Dynamic DeepHit Delta** | Semester-Sequenz | Median RR = **0,9665**<br>(Mean: 2,124) | Median RR = **1,0095**<br>(Mean: 1,678) | Median RR = **0,8425**<br>(Mean: 1,021) | Median RR trifft Fachlich ($0,9665$ vs GT $0,9579$) exzellent |
| **Deep Transformer-DML** | Sequenz-Encoder + DML | RR = **1,0172** | RR = **0,9957** | RR = **0,9569** | Psychosozial präzise ($0,9569$ vs GT $0,9514$), Fachlich überdämpft |
| **Recurrent Exam GRU Delta** | Prüfungs-Sequenz | Median RR = **1,0106** | Median RR = **1,1290** | Median RR = **1,3224** | Starkes Krisen-Confounding auf Einzelprüfungsebene |
| **Recurrent Exam GRU V2** | Prüfungs-Sequenz (+Fails/GPA) | Median RR = **1,0173** | Median RR = **1,0985** | Median RR = **0,9081** | Rollierende Leistungsmerkmale stellen Psych-Signal wieder her |

---

## 3. Methodische Kernbefunde & Confounding-Mechanismen

### 1. Robustes Signal bei psychosozialem Support
Psychosozialer Support wird über nahezu alle Modellklassen (Cox, DeepSurv, DTL, DeepHit, DML, Transformer) konsistent als **risikosenkend** geschätzt ($HR \approx 0{,}84 \dots 0{,}96$). Dies liegt daran, dass psychosoziale Beratung emotionale Entlastung bietet, ohne die akute Prüfungsvorbereitungszeit maßgeblich zu belasten.

### 2. Das Dilemma beim überfachlichen Support (Workload-Confounding / Reverse Causality)
Alle Panel- und Regressionsmodelle schätzen überfachlichen Support als scheinbar *risikosteigernd* ein ($HR \approx 1{,}01 \dots 1{,}09$).  
**Ursache:** Studierende, die Lerncoaching oder Zeitmanagement wählen, befinden sich bereits in akuter Überlastung. Der Support kostet 30 Stunden Arbeitszeit pro Semester. In Modellen ohne perfekte Kontrolle des internen Zeitpuffers wirkt die Support-Teilnahme wie ein zusätzlicher Stressfaktor, obwohl sie langfristig schützt.

### 3. Krisen-Selektion auf Prüfungsebene
Modelle auf Einzelprüfungsebene (Exam GRU Delta) neigen ohne historische Noten- und Fehlversuchskontrolle dazu, Support als risikobehaftet einzustufen, da Support primär vor schwierigen Wiederholungsprüfungen in Anspruch genommen wird. Erst das Modell **Exam GRU V2** (mit `fails_cum` und `gpa_cum`) kann dieses Selektionsbias teilweise auflösen.

---

## 4. Ergebnisse des Refactorings von `deep_transformer_regression.py`

Die Implementierungsmängel und Leakages in `deep_transformer_regression.py` wurden vollständig behoben:

| Modell | Vor Refactoring | Nach Refactoring | Status |
| :--- | :---: | :---: | :--- |
| **Deep Semester-Transformer Regressor** (Klasse 2b) | $R^2 = 0{,}5046$<br>$\text{RMSE} = 0{,}5135$ | **$R^2 = 0{,}9070$**<br>**$\text{RMSE} = 0{,}3238$** | ✅ **Behoben:** Vollständige Einbindung der kanonischen 8 Primärtabellen und Support-Merkmale. Schließt nahtlos an Semester-LSTM ($R^2=0{,}9144$) und Semester-Transformer ($R^2=0{,}9084$) an. |
| **Deep Exam-Transformer Regressor** (Klasse 3) | $R^2 = 0{,}9991$<br>$\text{RMSE} = 0{,}0223$ | **$R^2 = 0{,}8978$**<br>**$\text{RMSE} = 0{,}3373$** | ✅ **Behoben:** `note` wurde aus den Eingangsfeatures entfernt (Selbstvorhersage-Leakage beseitigt). Realistische Performance im Bereich von Exam-GRU ($R^2=0{,}9029$). |
| **Recurrent Exam Survival V2** (Klasse 7) | *(nicht trainiert)* | **$\text{ROC-AUC} = 0{,}8713$**<br>$\text{PR-AUC} = 0{,}1747$ | ✅ **Trainiert & Verifiziert:** Echtes, leakage-freies Sequenzmodell auf Prüfungsebene mit rollierendem GPA und Fehlversuchen. |

---

## 5. Vollständiges Skript-Register

Das vollständige Handbuch aller 69 Skripte mit Feature-Zuordnung und Modellklassen ist persistent dokumentiert in:
- [`Artifacts/script_registry.md`](file:///c:/GitHub_public/Abschlussprojekt/Artifacts/script_registry.md)
- [`implementation_plan.md`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/implementation_plan.md)
