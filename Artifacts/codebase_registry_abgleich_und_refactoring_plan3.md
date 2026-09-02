# Codebase Script-Register Abgleich & Zukünftiger Refactoring-Plan

> [!IMPORTANT]
> **Forensische Bestandsaufnahme & Architekturplan:**
> 1. Detail-Abgleich aller 91 Metrik-Dateien gegen das Skript-Register (`Artifacts/script_registry.md`).
> 2. Vollständige Aufklärung der ROC- vs. PR-AUC-Diskrepanz (63 vs. 62 Dateien).
> 3. Entwurf eines sauberen, modularen Refactoring- und Archivierungsplans für die spätere Konsolidierung.

---

## 1. Aufklärung der ROC- vs. PR-AUC-Diskrepanz (63 vs. 62 Dateien)

Bei der automatisierten Inventur der 91 Metrik-Dateien auf `S01_baseline/universe_A/metrics/` ergab sich ein minimaler Versatz zwischen Dateien mit ROC-AUC (63) und PR-AUC (62). 

Der Einzel-Audit aller JSON-Strukturen zeigt die exakten Ursachen:

### A. Dateien mit ROC-AUC, aber OHNE PR-AUC (3 Dateien):
1. **`autoregressive_deep_transformer_metrics.json`:**
   * *Ursache:* Das ursprüngliche Single-Head Transformer-Skript hat nur `Next_Exam_Grade_R2` (Regression) und `Next_Exam_Pass_ROC_AUC` (Bestehens-Wahrscheinlichkeit) geloggt. Der PR-AUC-Wert für Nichtbestehen wurde in einem separaten Schritt berechnet.
   * *Auflösung:* Im nachfolgenden Dual-Head-Skript (`autoregressive_next_exam_dual_head_metrics.json`) wurden bereits beide Maße gemeinsam erfasst.
2. **`deep_survival_metrics.json`:**
   * *Ursache:* Klassisches PySurv/Breslow-Baseline-Skript aus V3.0, das nach Standard-Survival-Konvention nur Harrell's `C-Index` und zeitabhängigen `ROC-AUC` (Uno/Heagerty) geloggt hat.
3. **`oracle_lift_metrics.json`:**
   * *Ursache:* Dies ist kein Einzelmodell, sondern ein diagnostisches Delta-JSON, das den relativen ROC-Lift zwischen Standard- und Oracle-Hazard-Modellen vergleicht (`ROC-AUC_Lift_LogisticHazard`, `ROC-AUC_Lift_DeepSurv`).

### B. Dateien mit PR-AUC, aber OHNE ROC-AUC (2 Dateien):
1. **`autoregressive_fail_metrics.json`:**
   * *Ursache:* Ergänzungs-Evaluator zu Punkt A1, der spezifisch die seltene Minderheitsklasse Klausur-Durchfall (Prävalenz 16.4 %) isoliert analysiert und nur `Next_Exam_Fail_PR_AUC` ausgegeben hat.
2. **`extended_cox_delta_metrics.json`:**
   * *Ursache:* Das Skript ist ein rein ökonometrischer Hazard-Ratio-Schätzer (PHReg). Der String-Filter hat auf das Suffix `_prev` in den Spaltennamen (`HR_delta_cp_prev`) reagiert. Das Modell schätzt Hazard Ratios $\text{HR}$, berechnet jedoch keine Klassifikations-ROC/PR-Kurven.

---

## 2. Abgleich mit dem Skript-Register (`src/*.py`)

Im Projektverzeichnis `src/` existieren **45 Python-Skripte** mit Modell-, Simulations- oder Evaluierungsbezug. Der Abgleich mit dem Skript-Register (`Artifacts/script_registry.md`) zeigt eine klare Zuordnung:

```
src/ (Aktueller Bestand)
├── 1. Daten- & Feature-Engine (Zentraler Backbone)
│   ├── feature_builder.py                [NEU in V4.1: Einheitliche Tensor- & Panel-Pipeline]
│   ├── aggregate.py / aggregate_semester_data_v3.py
│   └── metrics_logger.py                 [Zentrales Logging für alle Runs]
│
├── 2. Simulation & Sensitivitäts-Grid
│   ├── simulation_v4.py                  [Aktueller V4.1 Kern mit 15 Parametern]
│   ├── run_v4_simulation_grid.py         [Grid-Runner für 120 Welten]
│   ├── simulation_v3.py / v2.py / v1.py  [Legacy-Stände V1-V3 -> Archiv-Kandidaten]
│   └── calculate_true_effect.py          [Ground Truth ATE/ARR Berechnung]
│
├── 3. Modellfamilien (Die 10 Kernarchitekturen)
│   ├── Landmark:       train_mlp_baseline.py, train_mlp_regression.py, landmark_prediction.py
│   ├── Ökonometrie:    extended_cox_survival.py, extended_cox_delta.py
│   ├── Deep Survival:  deep_survival.py, extended_deep_survival.py, extended_deep_survival_delta.py
│   ├── Hazard (NN):    extended_logistic_hazard_prev/cum/delta
│   ├── Semester Rec.:  recurrent_survival_model.py, recurrent_survival_model_delta.py
│   ├── Semester Trans: transformer_survival_model.py, timeseries_semester_transformer.py
│   ├── Exam Recurrent: recurrent_exam_survival.py, recurrent_exam_survival_v2/delta.py
│   ├── Exam Trans.:    transformer_exam_survival.py, timeseries_exam_transformer.py
│   ├── Competing Risk: dynamic_deephit_model.py, dynamic_deephit_delta_model.py
│   ├── Causal ML:      dml_orthogonal_survival.py, train_transformer_dml.py
│   └── Autoregression: autoregressive_deep_transformer.py, autoregressive_next_exam.py
│
└── 4. Evaluierung & Diagnostik
    ├── analyze_cross_scenario_models.py  [NEU: Synoptische Evaluierungs-Engine]
    ├── audit_data_completeness.py        [NEU: Forensischer Vollständigkeits-Auditor]
    ├── structural_mediation_analysis.py  [Imai/Pearl Mediationsanalyse]
    └── counterfactual_*.py (8 Skripte)   [Kontrafaktische Schutz-Inferenz]
```

---

## 3. Zukünftiger Refactoring- & Archivierungsplan (Clean Repo Design)

> [!TIP]
> **Ziel des Refactorings (für spätere Phase):**
> Überführung der historisch gewachsenen Einzelskripte in ein sauberes, modulares Python-Package mit sprechender Namenskonvention, klaren Unterordnern und Verschiebung aller Legacy-Zwischenschritte in ein `archive/`-Verzeichnis.

### 3.1 Geplante Ziel-Verzeichnisstruktur

```
Abschlussprojekt/
├── src/deepsupport/                      # Modulares Package
│   ├── __init__.py
│   │
│   ├── simulation/                       # Simulations-Engine
│   │   ├── __init__.py
│   │   ├── engine.py                     # ehemals simulation_v4.py
│   │   ├── student_generator.py          # Populations-Generierung (N=50k)
│   │   ├── sensitivity_grid.py           # ehemals run_v4_simulation_grid.py
│   │   └── ground_truth.py               # ehemals calculate_true_effect.py
│   │
│   ├── data_engine/                      # Daten-Pipeline & Feature-Builder
│   │   ├── __init__.py
│   │   ├── feature_builder.py            # Einheitliche Tensor-/Panel-Schnittstelle
│   │   ├── datacube.py                   # DuckDB/NumPy Backend-Aggregation
│   │   └── splits.py                     # Leckage-freie Student-Level Splits
│   │
│   ├── models/                           # Die 10 Kernarchitekturen (sauber typisiert)
│   │   ├── __init__.py
│   │   ├── landmark_mlp.py               # Baseline MLP & Regressoren
│   │   ├── extended_cox.py               # Ökonometrisches PHReg mit TVCs
│   │   ├── deep_survival.py              # DeepSurv (Breslow Likelihood)
│   │   ├── neural_hazard.py              # Discrete-Time Logistic Hazard
│   │   ├── recurrent_semester_gru.py     # Semester-Level GRU
│   │   ├── semester_transformer.py       # Causal Masked Semester Transformer
│   │   ├── recurrent_exam_gru.py         # Exam-Level GRU (Top Predictor)
│   │   ├── exam_transformer.py           # Exam-Level Multi-Head Attention
│   │   ├── dynamic_deephit.py            # Competing Risks (Dropout vs. Abschluss)
│   │   ├── dml_causal.py                 # Double Machine Learning (Neyman Orthogonal)
│   │   └── autoregressive_transformer.py # Dual-Head Next-Exam Noten- & Pass-Modell
│   │
│   ├── evaluation/                       # Auswertung, Metriken & Kausalität
│   │   ├── __init__.py
│   │   ├── cross_scenario_engine.py      # ehemals analyze_cross_scenario_models.py
│   │   ├── metrics_logger.py             # Standardisiertes JSON- & Curve-Logging
│   │   ├── causal_inference.py           # Kontrafaktische RR/HR-Schützer
│   │   ├── mediation_analysis.py         # Strukturelle Mediation (Pearl)
│   │   └── completeness_auditor.py       # ehemals audit_data_completeness.py
│   │
│   └── visualization/                    # Dashboards & Visualisierung
│       ├── __init__.py
│       └── dashboard_builder.py          # Generierung des Standalone-HTML-Dashboards
│
├── archive/                              # Reines Archiv historischer Zwischenstände
│   ├── legacy_simulation/                # simulation_v1.py, v2.py, v3.py
│   ├── legacy_runners/                   # run_retrain_all.py, run_all_experiments.py
│   ├── redundant_delta_scripts/          # extended_cox_delta.py, recurrent_survival_delta.py
│   └── unmodular_trainers/               # V2 Einzelskripte vor feature_builder
│
├── Artifacts/                            # Berichte, Master-CSVs, Dashboards
│   ├── dashboard_cross_scenario.html
│   ├── v41_cross_scenario_gesamtreview.md
│   ├── data_and_metrics_completeness_audit.md
│   └── v41_all_92_models_by_target.csv
│
└── config/                               # Konfigurationsdateien & Presets
    └── default_config.py
```

---

## 4. Geplante Archivierungs-Kandidaten (Verschiebung nach `archive/`)

Sobald die finale Konsolidierungsphase abgeschlossen ist, können folgende **18 redundante Skripte** gefahrlos ins Archiv verschoben werden:

1. `src/simulation_v1.py`, `src/simulation_v2.py`, `src/simulation_v3.py` *(abgelöst durch `simulation_v4.py`)*.
2. `src/recurrent_survival_model_delta.py`, `src/recurrent_exam_survival_delta.py`, `src/recurrent_exam_survival_v2.py` *(vollständig integriert in die regulären Skripte via `temporal='prev'/'cum'`)*.
3. `src/extended_cox_delta.py`, `src/extended_deep_survival_delta.py` *(abgelöst durch `extended_cox_survival.py`)*.
4. `src/dynamic_deephit_delta_model.py` *(abgelöst durch `dynamic_deephit_model.py`)*.
5. `src/counterfactual_rr_deephit_delta.py`, `src/counterfactual_rr_logistic_hazard_delta.py` *(abgelöst durch `counterfactual_deephit_fixed.py`)*.
6. `src/run_retrain_all.py`, `src/run_all_experiments.py` *(abgelöst durch `run_fast_suite.py` und `feature_grid_runner`)*.
7. `src/test_deepsurv_scaling.py`, `src/analyze_v3_deep.py`, `src/analyze_mechanics_deepdive.py` *(historische Einmal-Tests)*.

---

## 5. Zusammenfassung

1. **Kein Datenverlust:** Alle 91 Metrik-Dateien und alle 120 Simulationswelten sind vollständig erfasst und in den Berichten verankert.
2. **Klare Aufklärung:** Die minimale Differenz zwischen ROC (63) und PR-AUC (62) rührt von 3 reinen Regressions-/Survival-Spezialfällen her und ist methodisch plausibel.
3. **Refactoring-Plan:** Der Plan steht bereit und kann nach Abschluss der inhaltlichen Auswertungen strukturiert und risikofrei umgesetzt werden.
