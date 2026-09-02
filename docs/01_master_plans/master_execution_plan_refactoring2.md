# Master Execution Plan: Codebase Refactoring & Evaluierung

Dieses Dokument vereint sämtliche Analysen, Deep Dives und Architektur-Entscheidungen zu einem finalen, detaillierten Ausführungsplan. Es dient als exaktes Drehbuch für das Refactoring aller 104 Skripte.

## 1. Die Leitprinzipien (Paradigmen)

1. **Keine 0-Imputation:** Der `metrics_logger.py` wird ein einheitliches Schema forcieren, aber fehlende Metriken strikt als `null` (JSON) oder `NaN` protokollieren. Eine fehlende Metrik darf niemals durch `0.0` verschleiert werden. Der Logger reicht Parameter (`architecture`, `mode`, `dataset`) sauber durch.
2. **One Model = One Script (Modularisierung):** Die Modelldateien enthalten ausschließlich die Keras/Scikit-Learn Architektur (`build_model`) und den reinen Trainingsaufruf. Die MLPs und Scikit-Learn-Modelle werden (da sie schnell und basal sind) als Ausnahme in kompakteren "Classifier/Regressor"-Blöcken gebündelt, aber ebenfalls modular vom Runner getrennt.
3. **Runner als reine Orchestratoren:** Der Grid-Runner kopiert keine Keras-Layer mehr in sich selbst. Er iteriert über Daten/Modi, importiert das Modell und leitet die Resultate an den Metrics-Logger weiter.
4. **Causal Inference Sicherheit:** Die 16 Kontrafaktik-Skripte werden erhalten und ihre hartcodierten `.keras`-Ladebefehle werden präzise auf die neuen Dateinamen überschrieben.

---

## 2. Ziel-Topologie

```text
Abschlussprojekt/
├── src/deepsupport/
│   ├── data_engine/     # Pipeline, Aggregation, Feature Builder
│   ├── simulation/      # Data Generating Process (S01-S15)
│   ├── models/          # Architekturen (Keras, Scikit-Learn)
│   ├── evaluation/      # Metrik-Logger, Dashboards
│   │   └── causal/      # Die 16 Kontrafaktik-Skripte
│   └── runners/         # Fast Suite, Heavy Suite, Grid Runner
├── archive/             # Veraltete Skripte & Legacy-Analysen
└── Artifacts/           # Dokumentation & Registry
```

---

## 3. Exaktes Datei-Mapping (Verschiebungen & Umbenennungen)

### A. Core Models -> `src/deepsupport/models/`
Die Monolithen werden aufgespalten, Namenskonventionen vereinheitlicht:
* `train_mlp_baseline.py` -> `baseline_classifiers.py` (RF, SVC, MLP)
* `train_mlp_regression.py` -> `baseline_regressors.py` (RF, SVR, MLP)
* `extended_cox_survival.py` -> `extended_cox.py`
* `deep_survival.py` & `extended_deep_survival.py` -> `deep_survival.py`
* `logistic_hazard_landmark.py` & `extended_logistic_hazard.py` -> `neural_hazard.py`
* `recurrent_survival_model.py` -> `semester_gru.py`
* `transformer_survival_model.py` -> `semester_transformer.py`
* `recurrent_exam_survival.py` -> `exam_gru.py`
* `transformer_exam_survival.py` -> `exam_transformer.py`
* `dynamic_deephit_model.py` -> `dynamic_deephit.py`
* `dml_orthogonal_survival.py` & `train_transformer_dml.py` -> `dml_causal_nets.py`
* `autoregressive_next_exam.py` -> `autoregressive_gru.py`
* `autoregressive_deep_transformer.py` -> `autoregressive_transformer.py`
* `train_oracle_models.py` -> Wird aufgelöst (Oracle-Modus wird direkt vom Grid-Runner übergeben).

### B. Evaluation & Causal -> `src/deepsupport/evaluation/`
* `metrics_logger.py` -> `metrics_logger.py`
* `analyze_cross_scenario_models.py` -> `cross_scenario_engine.py`
* `structural_mediation_analysis.py` -> `mediation_analysis.py`
* **Die 16 Kontrafaktik-Skripte** (z. B. `counterfactual_hr_analyzer.py`, `counterfactual_grade_transformer.py`) wandern nach `evaluation/causal/`. In allen Dateien werden die Ladebefehle (`load_model(...)`) per Regex an die neuen Namen aus Kategorie A angepasst.

### C. Data Engine & Simulation -> `src/deepsupport/...`
* **Data Engine:** `feature_builder.py`, `aggregate.py`, `config.py`
* **Simulation:** `simulation_v4.py` -> `engine.py`, `run_v4_simulation_grid.py` -> `sensitivity_grid.py`, `calculate_true_effect.py` -> `ground_truth.py`

### D. Runners -> `src/deepsupport/runners/`
* `run_feature_grid_experiments.py` -> `grid_runner.py` (Gestrippt! Keine Keras-Layer mehr).
* `run_fast_suite.py` -> `fast_suite.py` (Ruft nur noch `grid_runner.py` auf).
* `run_heavy_suite.py` -> `heavy_suite.py`

---

## 4. Das Archiv (`archive/`)
**Folgende Dateien (und alle weiteren `analyze_*.py` ohne Core-Relevanz) wandern ins Archiv:**
* `recurrent_exam_survival_v2.py`, `recurrent_exam_survival_delta.py`, `extended_cox_delta.py`, `dynamic_deephit_delta_model.py`
* Alle `timeseries_*.py` Aliase (`timeseries_exam.py`, `timeseries_semester.py`, etc.)
* Alle Legacy Simulationen (`simulation.py`, `simulation_v2.py`, `simulation_v3.py`)
* Einmal-Analysen wie `analyze_amortization_timeline.py`, `analyze_module_drops.py`
* Alte Runner: `run_all_experiments.py`, `run_retrain_all.py`

---

## 5. Ausführungs-Phasen

**Phase 1: Bereinigung & Topologie** (Startet jetzt)
- `mkdir` für die neue Package-Struktur.
- Verschieben der Archiv-Dateien nach `archive/`.

**Phase 2: Code-Umstrukturierung**
- Verschieben und Umbenennen der Core-Modelle und Causal-Skripte.
- Suchen und Ersetzen der hartcodierten Dateipfade in den Causal-Skripten.
- Anpassen der `import`-Statements in allen Skripten (da sie nun in Packages liegen, z. B. `from deepsupport.models import ...`).

**Phase 3: Runner & Logger Refactoring**
- Umschreiben des `metrics_logger.py` (Null-Werte zulassen, Schema definieren, Argumente durchreichen).
- Strip des `run_feature_grid_experiments.py` (Auslagerung der Layer in die Modelle).
