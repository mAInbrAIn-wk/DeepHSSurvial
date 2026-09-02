# DeepSupport Master-Topologie & Roadmap-Konsolidierung

Dieses Dokument klärt endgültig die numerischen Inkonsistenzen (Skripte, Dateien, Modelle), dokumentiert die aktuelle Modell-Kombinatorik und bringt die bestehenden Refactoring- und Erweiterungspläne in eine strategische Reihenfolge.

---

## 1. Entwirrung der Zahlen: Skripte, Dateien & Modelle

### A. Die 104 Python-Skripte (Warum nicht 45 oder 69?)
Eine Live-Zählung im Verzeichnis `src/` ergibt aktuell exakt **104 `.py`-Dateien**. 
* **Woher kamen die anderen Zahlen?** Das alte Registry-Dokument vom 21. August zählte 69 Dateien. Die Angabe "45" in der letzten Nachricht war eine mentale Filterung auf die reinen Kernmodelle und Daten-Pipelines.
* **Wie setzen sich die 104 Skripte zusammen?**
  * **10** Kern-Modellarchitekturen (`timeseries_exam_transformer.py`, `dynamic_deephit_model.py`, etc.)
  * **19** `counterfactual_*.py` Skripte für isolierte kausale Inferenz
  * **16** `analyze_*.py` Skripte für empirische Teilauswertungen
  * **14** `run_*.py` Skripte für Overnight-Läufe und Batch-Jobs
  * **~45** Legacy-Simulationen (`v1`-`v3`), Alt-Trainer und Hilfs-Module.
* **Fazit:** Die Codebase ist durch Ad-hoc-Analysen massiv gewuchert. Dies bestätigt die Notwendigkeit des Refactorings (siehe Phase 3).

### B. 91 Modelle vs. 92 Dateien im Metrik-Ordner
Im Ordner `S01_baseline/universe_A/metrics/` liegen exakt **92 JSON-Dateien**.
* **91 Dateien** sind echte individuelle Modell-Ergebnisse (z.B. `grid_exam_gru_standard_metrics.json`).
* **Die 92. Datei** ist `feature_grid_master_benchmark.json`. Sie ist kein Modell, sondern ein Aggregations-Log, das die Ergebnisse anderer Modelle zusammenfasst.

### C. Die 120 Simulationswelten
* **15 Data-Generating Processes (Szenarien S01 bis S15)** (z.B. Baseline, RCT, Überlastung).
* **8 Universen pro Szenario (A bis H)**: Dies ist **kein** stochastisches Rauschen, sondern es sind **kontrafaktisch simulierte Alternativwelten**. Bei exakt gleichen Studierenden-Eigenschaften (identischer Seed) wird lediglich die Zuweisung zum Support-Programm variiert. Universum A ist die Trainingswelt, B-H dienen der exakten kausalen Isolierung (Berechnung der Ground-Truth Treatment-Effekte auf Studierenden-Ebene).
* $15 \times 8 = 120$ Welten mit je $N=50.000$ Studierenden.

---

## 2. Die Modell-Architektur-Topologie (Die 91 Modelle)

Die 91 Modelle generieren sich kombinatorisch aus **10 Kernfamilien**, **5 Feature-Modi** (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`) und **2 Temporals** (`prev` [Panel/Flat] vs. `cum` [Zeitreihe]).

| Kern-Architektur | Kurzbeschreibung & Status | Kombinatorische Varianten (JSON-Output) |
| :--- | :--- | :--- |
| **1. Landmark MLP / Regression** | Querschnitts-Baseline (Ende Sem 1/2) für Status & GPA. | 4 Dateien (`mlp_baseline`, `mlp_regression` je std/gradeblind) |
| **2. Extended Cox Panel** | Ökonometrisches Proportional-Hazards-Modell mit Time-Varying Covariates. | 2 Dateien (`extended_cox_panel`, `_delta`) |
| **3. DeepSurv (Breslow)** | Neuronales Survival nach Katzman et al. (2018). | 4 Dateien (`deep_survival`, `extended_deepsurv_*`) |
| **4. Neural Logistic Hazard** | Discrete-Time Survival Model, Cross-Entropy optimiert. | 4 Dateien (`logistic_hazard_landmark`, `extended_logistic_*`) |
| **5. Semester Recurrent GRU** | Sequenzmodell auf Semesterebene ($t \in [1, 10]$). | 5 Dateien (`grid_semester_gru_*` [alle 5 Modi]) |
| **6. Semester Transformer** | Causal Masked Attention auf Semesterebene (Klasse 8). | 5 Dateien (`grid_semester_transformer_*` [alle 5 Modi]) |
| **7. Exam Recurrent GRU** | Sequenzmodell auf Prüfungsebene ($t \in [1, 40]$). **Top Predictor**. | 9 Dateien (`recurrent_exam_survival_*`, `grid_exam_gru_*`) |
| **8. Exam Transformer** | Multi-Head Attention auf Prüfungsebene. | 2 Dateien (`transformer_exam_survival`, `timeseries_exam_transformer`) |
| **9. Dynamic DeepHit** | Competing Risks (Dropout vs. Abschluss). | 8 Dateien (`dynamic_deephit_cum/prev_*` + Delta) |
| **10. Causal ML (DML)** | Double Machine Learning (Neyman Orthogonalisierung). | 6 Dateien (`dml_orthogonal_survival_*`, `transformer_dml`) |
| **11. Next-Exam AutoReg** | Dual-Head Transformer (Vorhersage der Folgeklausur). | 2 Dateien (`autoregressive_deep_transformer_*`) |
| **+ Legacy/Diagnostik-Aliase** | Counterfactuals, Lifts und Erwerbs-Splits. | 40 Dateien (Aliase & Kontrafaktische Logs) |
| **GESAMT** | | **91 Modell-Dateien** |

---

## 3. Konsolidierung der Erweiterungs- & Refactoring-Pläne (Roadmap V2)

Basierend auf der strategischen Ausrichtung "Konsolidierung vor Erweiterung" ergibt sich folgende Reihung:

### 🥇 Phase 1: Clean Repo Refactoring & Metrik-Standardisierung (Das Aufräumen)
* **Kritik:** 104 Skripte sind unübersichtlich. Zudem zeigte der Audit, dass abzüglich der 40 Aliase/Diagnose-Logs noch **51 echte Modell-JSONs** übrig bleiben, die asymmetrisch über die Familien verteilt sind (einige haben alle 5 Modi, andere nur 2). Die Output-Strukturen der JSONs sind ebenfalls historisch gewachsen und uneinheitlich.
* **Entscheidung:** Wir strukturieren das Projekt hart in Packages (`src/models/`, `src/simulation/`, `src/evaluation/`) und verschieben Ad-hoc-Skripte nach `archive/`. Dabei führen wir eine **einheitliche Metrik-API** ein: Jedes Modell generiert fortan zwingend die exakt gleichen JSON-Keys (ROC, PR, Brier, R2, etc.), sodass wir künftig eine saubere $10 \times 5 \times 2$ Matrix (100 Dateien) erhalten.

### 🥈 Phase 2: Dashboard-Erweiterung (Tab 2-5)
* **Kritik:** Die Auswertungen zu Kausalität, Bias und Stresstests (S02-S15) sind aktuell nur im Markdown. Dashboards sind technisch fehleranfällig (wie bei Plotly/CSP gesehen).
* **Entscheidung:** Umsetzung der bereits bestehenden detaillierten Pläne zur Überführung der Text-Reports in interaktive, offline-fähige SVGs. Eine saubere Backend-Datenbank (durch Phase 1) ist hierfür zwingend erforderlich.

### 🥉 Phase 3: Multi-Szenario-Robustheit (S02-S15, Universum A)
* **Kritik:** Modelle laufen momentan primär auf `S01_baseline/universe_A`. 
* **Entscheidung:** Wir werten die **Top-3 Modelle** auf den abweichenden Szenarien S02 bis S15 (stets nur auf `universe_A`!) aus, um Robustheit unter Stress (z. B. Überlastung, RCT) zu evaluieren.

### 🔬 Phase 4: MoE / Stacking mit kontrafaktischem Ground-Truth (Erweiterung)
* **Kritik:** MoE ist spannend, aber eine inhaltliche Erweiterung und kommt daher *nach* der Konsolidierung.
* **Der Clou:** Da die Universen B-H exakte kontrafaktische Zwillinge auf Studentenebene sind, wissen wir für jeden individuellen Studierenden, ob eine Support-Maßnahme *wirklich* geholfen hätte. Wir können evaluieren, ob z.B. DeepHit oder Exam-GRU bestimmte *Typen* von Studierenden besser vorhersagt. Das Router-Netzwerk trainiert dann direkt auf den individuellen Fehlerprofilen aus den Kontrafaktik-Welten. Ein extremer Mehrwert für den Abschlussbericht!

### 🔮 Phase 5: PyTorch / PyCox Portierung (Future Work)
* **Kritik:** Aktuelle Custom-Losses in Keras sind stabil. PyTorch Portierung (für nativen DeepSurv/PC-Hazard) wird als Ausblick für Folgearbeiten dokumentiert.
