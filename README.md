# DeepSupport: Wirksamkeitsanalyse von Hochschulsupport via Deep Learning & Causal Machine Learning

**Autor:** Wilfried Keller  
**Kontext:** Abschlussprojekt im Kurs *Deep Learning* (Dr. Bernd Ebenhoch)  
**Datum:** August 2026  

---

## KI-Transparenz & Methodischer Stack

Alle Inhalte dieses Projekts (Code-Architektur, Datengenerierungs-Engine, Modellierung, Kausalanalyse, Audits und Dokumentation) wurden in intensiver, transparenter Auseinandersetzung mit KI-Systemen entwickelt, überprüft, reviewed, korrigiert und erweitert.

- **Entwicklungsumgebung & Orchestrierung:** Antigravity IDE / Antigravity Agent
- **Integrierte LLM-Modelle (Pair Programming & Code Generation):** Gemini 3.1 Pro, Gemini 3.6 Flash, Claude Opus 4.6
- **Weitere KI-Tools & Exploration (via Mammouth.ai):** Claude Opus/Sonnet 5, ChatGPT 5.6, ChatGPT Sol, Kimi K2.5 / K3
- **Dokumentations-Artefakte:** Sämtliche Berichte, Reviews und Walkthroughs im Ordner `Artifacts/` sowie im System-Kontext sind direkte, transparente KI-generierte Audit-Protokolle.

---

## Projektübersicht & Kausale Herausforderung

Dieses Projekt analysiert datengetrieben die Wirksamkeit von Unterstützungsangeboten (z. B. fachliche Tutorien, überfachliche Workshops, psychosoziale Beratung) an Hochschulen.

Die Kernherausforderung liegt in der Auflösung des **Selektionsbias**, des **Time Availability Confoundings** und des **Immortal-Time-Bias**:
Da leistungsschwächere Studierende oder Studierende mit viel Erwerbstätigkeit (20h/Woche) an Supportmaßnahmen teilnehmen, kommen naive Machine-Learning-Modelle oft zu dem fehlerhaften Schluss, dass Support das Studienabbruch-Risiko erhöht (*Dropout-Paradoxon*).

### Methodischer Ansatz:
1. **5-Universen Counterfactual Simulator (V3.3):** Stochastische Simulation von 50.000 Studierenden, deren identische Klone in 5 parallelen Universen (A: Alle Angebote, B: Kein Support, C: Kein fachlicher Support, D: Kein überfachlicher Support, E: Kein psychosozialer Support) simuliert werden. In Version V3.3 wurde die Stochastik über **dedizierte RNG-Streams und index-basiertes Prüfungsrauschen** entkoppelt, um eine exakte kontrafaktische Vergleichbarkeit der Universen zu gewährleisten.
2. **Kausale Survival-Analyse (Longitudinal Panels):** Überführung der Studienverläufe in Person-Semester-Panels (Counting Process Format) mit zeitvariablen Vorsemester-Deltas (`fails_prev`, `delta_cp_prev`, `cp_rueckstand`).
3. **Double Machine Learning (DML) & Causal Transformer Benchmark:** Systematischer Vergleich von Standard-DML (Tabular Cox) und Causal Masked Transformer-DML zur Schätzung der kausalen Treatment-Effekte.

---

## Ergebnisse der Simulation & Modell-Evaluierung (V3.3)

### 1. Makroskopische Ground Truth (5 Universen × 50.000 Studierende)
| Universum | Konfiguration | Dropout-Rate | Relatives Risiko (RR) vs. A | Netto-Gerettete vs. A | Kausale Wirkung auf Makro-Ebene |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Universum A** | Baseline (Alle Support-Typen) | **27,37 %** | **1,0000** | — | Ausgangslage |
| **Universum E** | Kein psychosozialer Support | **28,77 %** | **0,9514** | **+699** | **-4,86 % Risikoreduktion** (Psychosozialer Support schützt) |

---

### 2. Kausalschätzer Benchmark vs. Realität (Umfassende Evaluation)

Die Evaluierung der Kausalschätzer und Counterfactual-Analysen zeigt die Grenzen und Stärken der verschiedenen Modellklassen:

| Modell & Methode | Analyse-Level | RR / HR Fachlich | RR / HR Überfachlich | RR / HR Psychosozial | Methodische Bewertung |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ground Truth (V3.3)** | Makro (5 Universen) | **0,9579** | **0,9387** | **0,9514** | Reale Kausalwirkung: Alle 3 Support-Typen schützen |
| **Extended Cox Delta** | Panel (Semi-parametrisch) | **0,8574** ($p<10^{-4}$) | **1,0940** ($p<10^{-4}$) | **0,8732** ($p<10^{-4}$) | Überschätzt Fachlich; Überfachlich fälschlich positiv ($+9,4\%$) |
| **Extended DeepSurv Panel** | Panel (Neural Cox Breslow) | Median HR = **0,9886** | Median HR = **1,0085** | Median HR = **0,9245** | **Beste Treffsicherheit** bei Fachlich & Psychosozial |
| **Extended DeepSurv Delta** | Panel (mit Deltas) | Median HR = **0,9082** | Median HR = **1,0422** | Median HR = **0,9641** | Fachlich überschätzt ($-9,2\%$), Psychosozial gut getroffen |
| **Extended DTL Hazard Delta** | Panel (Discrete Hazard) | Median RR = **0,7718** | Median RR = **1,0381** | Median RR = **0,8823** | Starker Fach-Effekt ($-22,8\%$), Psychosozial ($-11,8\%$) |
| **DML Orthogonal Survival** | Panel (Double ML) | Mean RR = **0,7994** | Mean RR = **1,0980** | Mean RR = **0,9078** | Fachlich & Psychosozial protektiv, Überfachlich verzerrt |
| **Dynamic DeepHit Delta** | Semester-Sequenz | Median RR = **0,9665** | Median RR = **1,0095** | Median RR = **0,8425** | Median RR trifft Fachlich exzellent ($0,9665$ vs GT $0,9579$) |
| **Deep Transformer-DML** | Sequenz-Encoder + DML | RR = **1,0172** | RR = **0,9957** | RR = **0,9569** | Psychosozial präzise ($0,9569$ vs GT $0,9514$), Fachlich überdämpft |
| **Recurrent Exam GRU V2** | Prüfungs-Sequenz (+Fails/GPA) | Median RR = **1,0173** | Median RR = **1,0985** | Median RR = **0,9081** | Rollierende Leistungsmerkmale stellen Psych-Signal wieder her |

> [!NOTE]
> **Methodischer Kernbefund:**  
> - **Psychosozialer Support** wird über nahezu alle Panel- und Sequenz-Modelle robust als risikosenkend erkannt ($HR \approx 0,84 \dots 0,96$).
> - **Überfachlicher Support (Lerncoaching/Zeitmanagement)** leidet unter dem **Workload-Confounding / Reverse Causality**: Studierende in akuter Zeitnot wählen den Support, wodurch die 30 Stunden Support-Kosten kurzfristig das Risiko erhöhen, wenn die Entlastung nicht sofort greift.

---

## Modell-Portfolio Performance (V3.3 Datensatz)

### Abbruch- & Survival-Vorhersage
| Modell | Level / Typ | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: |
| **Recurrent Exam Survival V2** | Exam Sequence (mit roll. GPA/Fails) | **0,8713** | 0,1747 | 0,0168 |
| Extended Logistic Hazard Exam Delta | Exam Level Panel | 0,8636 | 0,1757 | 0,0169 |
| Logistic Hazard Landmark | Static Landmark | 0,8597 | 0,7146 | — |
| Recurrent Exam Survival GRU Delta | Exam Sequence | 0,8504 | 0,1389 | 0,0175 |
| Recurrent Exam Survival GRU (Base) | Exam Sequence | 0,8453 | 0,1420 | 0,0174 |
| Transformer Survival (Semester) | Semester Sequence | 0,7909 | 0,2284 | 0,0365 |
| Dynamic DeepHit Delta (Dropout) | Multi-Task Competing | 0,7942 | 0,2301 | 0,0366 |
| Recurrent Survival GRU (Semester) | Semester Sequence | 0,7898 | 0,2234 | 0,0368 |
| Extended Logistic Hazard Delta | Semester Panel | 0,7694 | 0,2081 | 0,0370 |
| DML Orthogonal Survival | Causal Panel | 0,7694 | 0,2081 | 0,0370 |

### Noten- & GPA-Regression
| Modell | Typ | $R^2$ Score | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **Semester-LSTM Regressor** | Semester Sequence (T=16) | **0,9144** | 0,3108 | 0,2352 |
| **Semester-Transformer Regressor** | Semester Sequence (T=16) | **0,9084** | 0,3215 | 0,2448 |
| **Deep Semester-Transformer Regressor** | Semester Sequence ($d=128$, Attn) | **0,9070** | 0,3238 | 0,2472 |
| **Exam-GRU Regressor** | Exam Sequence (T=40) | **0,9029** | 0,3289 | 0,2480 |
| **Deep Exam-Transformer Regressor** | Exam Sequence ($d=128$, ohne Noten-Leakage) | **0,8978** | 0,3373 | 0,2559 |
| Keras MLP Regression | Static Tabular | 0,8694 | 0,2272 | 0,1731 |
| SVR (Support Vector Regression) | Static Tabular | 0,8668 | 0,2294 | 0,1752 |
| Random Forest Regression | Static Tabular | 0,8484 | 0,2448 | 0,1857 |
| Linear Ridge Regression | Static Linear | 0,8461 | 0,2466 | 0,1914 |

---

## Vollständiges Skript-Register & Architektur-Handbuch

👉 **[`Artifacts/script_registry.md`](Artifacts/script_registry.md)**

---

## Methodik & Modell-Stufen

Das Projekt implementiert und vergleicht über **21 Modellarchitekturen**:

### Stufe 0: Statische Baselines & Noten-Regression
- **Klassifikation:** Naive Bayes, Random Forest (79,25 % Acc), SVM (72,29 % Acc), Keras MLP Classifier (79,29 % Acc).
- **Noten-Regression:** Linear Ridge ($R^2=0.8461$), SVR ($R^2=0.8668$), Keras MLP Regressor ($R^2=0.8694$, MAE=0.1731).

### Stufe 1: Landmark Survival (Statisch)
- **Modelle:** DeepSurv (Keras Cox-Partial-Likelihood) und Discrete-Time Logistic (DTL) Hazard.

### Stufe 2: Extended Panel Survival (Zeitveränderlich & Delta-Features)
- **Modelle:** Extended Cox (statsmodels), Extended DeepSurv Delta, Extended DTL Hazard Delta.
- **Ansatz:** Aufspaltung in Person-Semester-Panels mit Vorsemester-Deltas (`fails_prev`, `delta_cp_prev`, `cp_rueckstand`).

### Stufe 3: Sequenz-Survival & Competing Risks
- **Modelle:** Semester- & Exam-Level LSTMs, Causal Masked Transformer, Dynamic DeepHit Delta (Multi-Task Competing Risks).

### Stufe 4: Double Machine Learning (DML) & Causal Transformer
- **Modelle:** DML Orthogonalized Survival, Erwerb-Blind DML, Deep Causal Transformer-DML.

---

## Projektarchitektur & Orchestrierung

```mermaid
flowchart TD
    subgraph Data Pipeline
        A["run_overnight.py\n(Master-Nachtlauf V3.3)"] --> B["simulation_v3.py\n(5-Universen Stochastik)"]
        B --> C["export.py\n(CSV-Export & hidden Logging)"]
        C --> D["aggregate.py\n(Delta Feature Engineering)"]
        D --> E["validate.py\n(Prüfung & Doku-Generierung)"]
    end

    subgraph Causal ML Pipeline
        E --> F["calculate_true_effect.py\n(Ground Truth ATT)"]
        F --> G["run_all_experiments.py\n(21 Modelle Training)"]
        G --> H["train_transformer_dml.py\n(Deep Transformer-DML)"]
        G --> I["deep_transformer_regression.py\n(Deep Exam-Transformer)"]
    end
```

### Die wichtigsten Skripte im `src/` Ordner:
- `simulation_v3.py`: Core-Engine des 5-Universen-Simulators mit dekoppelten RNG-Streams und index-basiertem Prüfungsrauschen.
- `calculate_true_effect.py`: Berechnet die empirische Makro Ground Truth über alle 5 Universen.
- `train_transformer_dml.py`: Deep Causal Transformer-DML für kausale Effektschätzung über alle 3 Support-Typen.
- `deep_transformer_regression.py`: Hochkapazitäre Transformer mit Attention-Weighted Pooling ($ROC\text{-}AUC=0.9999$).
- `analyze_grade_effects.py`: Detaillierte Noteneffekt-Analyse nach Prüfungsversuchen.
