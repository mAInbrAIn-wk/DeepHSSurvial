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
* **8 Universen pro Szenario (A bis H)** (Stochastic Noise).
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

## 3. Konsolidierung der Erweiterungs- & Refactoring-Pläne (Roadmap)

Wir haben historisch ca. 5 Masterpläne generiert. Hier ist die strategische Reihung inklusive kritischer Würdigung:

### 🥇 Phase 1: Ergebnissynthese & Ensembling (Höchster ROI, Jetzt)
1. **Dashboard-Erweiterung (Tab 2-5):**
   * *Kritik:* Die Auswertungen zu Kausalität, Bias, Stresstests (S02-S15) und MoE sind aktuell nur im Markdown. Eine Überführung der Tabs ins HTML-Dashboard macht die Ergebnisse greifbar und präsentationsfertig.
2. **Mixture of Experts (MoE) Experiment:**
   * *Kritik:* Der Exam GRU ($0{,}893$) und Dynamic DeepHit ($0{,}811$) haben sehr unterschiedliche Stärken. Ein einfaches Router-Skript, das Vorhersagen kombiniert, ist ein starkes "Cutting-Edge"-Feature für den Abschlussbericht.

### 🥈 Phase 2: Robustheit & Multi-Szenario (Wissenschaftlicher Kern)
1. **Multi-Szenario-Grid-Run (S02-S15):**
   * *Kritik:* Alle 120 Welten wurden per Simulation generiert, aber die DL-Modelle liefen bisher vor allem auf `S01_baseline`. Es wäre Verschwendung (Compute und Zeit), *alle 91 Modelle* auf allen 120 Welten zu trainieren.
   * *Entscheidung:* Wir wählen nur die **Top-3 Modelle** (z.B. Exam GRU, DeepHit, DML) und evaluieren sie auf S02-S15, um Stresstests (z.B. RCT vs. Bias) zu validieren.

### 🥉 Phase 3: Clean Repo Refactoring (Das Aufräumen)
1. **Codebase Restrukturierung (vgl. *Codebase Registry Plan*):**
   * *Kritik:* 104 Skripte sind für Dritte unwartbar.
   * *Entscheidung:* Sobald die inhaltliche Arbeit abgeschlossen ist, strukturieren wir das Projekt hart in Packages (`src/models/`, `src/simulation/`, `src/evaluation/`) und verschieben die ca. 70 überflüssigen `analyze_` und `counterfactual_` Skripte in einen `archive/` Ordner.

### 🔬 Phase 4: PyTorch / PyCox Portierung (Optionaler Ausblick)
1. **TensorFlow/Keras $\rightarrow$ PyTorch Port:**
   * *Kritik:* PyCox bietet native und stabilere Implementierungen für DeepSurv, Logistic Hazard und PC-Hazard.
   * *Entscheidung:* Aktuell funktionieren unsere Keras-Custom-Losses hervorragend. Der PyTorch-Port ist ein perfektes Thema für eine **"Future Work"** Sektion oder einen späteren Architektur-Sprint, sollte aber das aktuelle Einreichen/Konsolidieren der Arbeit nicht blockieren.
