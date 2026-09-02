# Historische Skript-Registry (Vollstaendiger Audit)

Dieser Bericht listet ALLE Python-Skripte im Projekt auf (Stand vor dem Refactoring), inklusive Funktion, benoetigter Daten und Abhaengigkeiten.

### `aggregate.py`
- **Funktion:** Datenaggregation für HSDS Datensatz (3-Way Backend: DuckDB / NumPy / Pandas)
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_amortization_timeline.py`
- **Funktion:** Berechnet den Amortisationszeitpunkt von Support-Massnahmen (Break-Even Analyse).
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_cross_scenario_models.py`
- **Funktion:** Hierarchische Cross-Szenario & Modell-Evaluierungs-Engine V4.1
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** metrics JSONs

### `analyze_exmat_remaining.py`
- **Funktion:** Analysiert die verbleibende Studiendauer bis zur Exmatrikulation nach einem Fail.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_g1_exmatrikulation_and_workload.py`
- **Funktion:** Untersucht den Zusammenhang zwischen Erstsemester-Workload (G1) und Abbruchwahrscheinlichkeit.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_grade_effects.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_mechanics_deepdive.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_module_drops.py`
- **Funktion:** Analysiert das Abwurfverhalten von Modulen (Pruefungsabmeldungen) unter Stress.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_overload_victims.py`
- **Funktion:** Identifiziert und analysiert Studierende, die an systematischem Workload-Overload scheitern.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_support_effects.py`
- **Funktion:** Umfassende Analyse der kontrafaktischen Support-Effekte.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv), metrics JSONs

### `analyze_time_amortization.py`
- **Funktion:** Berechnet den Amortisationszeitpunkt von Support-Massnahmen (Break-Even Analyse).
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv), metrics JSONs

### `analyze_v3_deep.py`
- **Funktion:** Gründliche Evaluation der Simulation V3 Ergebnisse
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv), metrics JSONs

### `analyze_v3_followup.py`
- **Funktion:** Vertiefte V3-Analyse: Antworten auf alle Nutzer-Rückfragen
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_v4_counterfactual.py`
- **Funktion:** Vollständige kontrafaktische Supportanalyse & Migrationsanalyse
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `analyze_v4_grid_sensitivity.py`
- **Funktion:** Analyse & Visualisierung des V4 Simulations-Sensitivitäts-Grids
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** metrics JSONs

### `analyze_workload_delay.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `run_remaining_experiments.py`
- **Funktion:** Runner Script: Führt verbleibende Experimente ab Schritt 7 aus
- **Abhaengigkeiten:** deep_survival, dynamic_deephit_model, transformer_exam_survival, transformer_survival_model
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `audit_data_completeness.py`
- **Funktion:** DeepSupport V4.1 - Daten- & Metriken-Vollstaendigkeits-Auditor
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** metrics JSONs

### `autoregressive_deep_transformer.py`
- **Funktion:** Stellt Funktionen bereit: build_deep_transformer_dual_head, train_autoregressive_deep_transformer...
- **Abhaengigkeiten:** autoregressive_next_exam
- **Benoetigte Daten:** DATA_DIR, metrics JSONs, Keras Models

### `autoregressive_next_exam.py`
- **Funktion:** Autoregressive Next-Exam Prediction Model (Dual-Head Multi-Task Edition)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `benchmark_backbone_sanity_check.py`
- **Funktion:** 3-Way Backbone Sanity-Check & Performance Benchmark (Pandas vs. DuckDB vs. NumPy)
- **Abhaengigkeiten:** aggregate, feature_builder
- **Benoetigte Daten:** feature_builder API

### `calculate_true_effect.py`
- **Funktion:** Berechnung des zugrundeliegenden Support-Effekts
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `config.py`
- **Funktion:** Konfiguration und Stammdaten für das Deep Learning Absolventenprojekt.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `counterfactual_deephit_fixed.py`
- **Funktion:** Counterfactual Relative Risk Analysis für Dynamic DeepHit (Semester Level - Existing Model)
- **Abhaengigkeiten:** feature_builder, metrics_logger, recurrent_survival_model
- **Benoetigte Daten:** feature_builder API, Keras Models

### `counterfactual_deepsurv.py`
- **Funktion:** Kontrafaktische Inferenz-Berechnung (RR/HR) spezifisch fuer DeepSurv.
- **Abhaengigkeiten:** deep_survival
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `counterfactual_grade_transformer.py`
- **Funktion:** Kontrafaktische Noteninferenz für Deep Exam Transformer Regressor
- **Abhaengigkeiten:** deep_transformer_regression, feature_builder, metrics_logger, transformer_survival_model
- **Benoetigte Daten:** feature_builder API, metrics JSONs, Keras Models

### `counterfactual_hr_analyzer.py`
- **Funktion:** Stellt Funktionen bereit: analyze_counterfactual_hr...
- **Abhaengigkeiten:** extended_deep_survival, feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `counterfactual_hr_delta.py`
- **Funktion:** Counterfactual Hazard Ratio Analysis (Extended DeepSurv Delta Edition)
- **Abhaengigkeiten:** extended_cox_delta, extended_deep_survival_delta, metrics_logger
- **Benoetigte Daten:** Keras Models

### `counterfactual_inference.py`
- **Funktion:** Counterfactual Inference Wrapper für Deep Survival Modelle
- **Abhaengigkeiten:** recurrent_exam_survival
- **Benoetigte Daten:** metrics JSONs

### `counterfactual_inference_deephit.py`
- **Funktion:** Counterfactual Inference Wrapper für Dynamic DeepHit
- **Abhaengigkeiten:** dynamic_deephit_model, recurrent_survival_model
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `counterfactual_inference_semester_transformer.py`
- **Funktion:** Counterfactual Inference Wrapper für Semester-Transformer Survival
- **Abhaengigkeiten:** metrics_logger, recurrent_survival_model, transformer_survival_model
- **Benoetigte Daten:** DATA_DIR, Keras Models

### `counterfactual_oracle_deepsurv.py`
- **Funktion:** Counterfactual Hazard Ratio Analysis für Oracle DeepSurv
- **Abhaengigkeiten:** extended_cox_delta, extended_deep_survival_delta, metrics_logger
- **Benoetigte Daten:** Keras Models

### `counterfactual_oracle_logistic_hazard.py`
- **Funktion:** Counterfactual Relative Risk Analysis für Oracle Logistic Hazard
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `counterfactual_rnn.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** recurrent_exam_survival_v2
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `counterfactual_rnn_delta.py`
- **Funktion:** Counterfactual Relative Risk Analysis für Recurrent GRU v2 (Exam Level)
- **Abhaengigkeiten:** metrics_logger, recurrent_exam_survival_v2
- **Benoetigte Daten:** Keras Models

### `counterfactual_rnn_semester_delta.py`
- **Funktion:** Counterfactual Relative Risk Analysis für Recurrent Survival Model Delta (Semester Level)
- **Abhaengigkeiten:** metrics_logger, recurrent_survival_model, recurrent_survival_model_delta
- **Benoetigte Daten:** Keras Models

### `counterfactual_rr_deephit_delta.py`
- **Funktion:** Counterfactual Relative Risk Analysis für Dynamic DeepHit Delta
- **Abhaengigkeiten:** dynamic_deephit_delta_model, metrics_logger, recurrent_survival_model
- **Benoetigte Daten:** DATA_DIR, Keras Models

### `counterfactual_rr_exam_rnn_delta.py`
- **Funktion:** Counterfactual Relative Risk Analysis für Recurrent Exam Survival Delta
- **Abhaengigkeiten:** metrics_logger, recurrent_exam_survival_delta, recurrent_survival_model
- **Benoetigte Daten:** DATA_DIR, Keras Models

### `counterfactual_rr_logistic_hazard_delta.py`
- **Funktion:** Counterfactual Relative Risk Analysis für Extended Logistic Hazard Delta
- **Abhaengigkeiten:** extended_cox_delta, metrics_logger
- **Benoetigte Daten:** Keras Models

### `dashboard_educational.py`
- **Funktion:** Educational Survival Dashboard: Kausalität vs. Data Leakage
- **Abhaengigkeiten:** extended_cox_survival
- **Benoetigte Daten:** CSV (read_csv)

### `dashboard_survival_dl.py`
- **Funktion:** Unified Deep & Classic Survival Analysis Web Dashboard (Clean Landmark Edition)
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `deep_survival.py`
- **Funktion:** Deep Survival Analysis (Landmark DeepSurv & Logistic Hazard)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `deep_transformer_regression.py`
- **Funktion:** Deep Transformer Regression & Survival Models (Enlarged Capacity + Dual Causal/Masked Architectures)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, metrics JSONs, Keras Models

### `dml_orthogonal_survival.py`
- **Funktion:** Double / Debiased Machine Learning (DML) Orthogonalized Survival Model
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `dynamic_deephit_delta_model.py`
- **Funktion:** DEPRECATED / CONSOLIDATED WRAPPER: dynamic_deephit_delta_model.py
- **Abhaengigkeiten:** dynamic_deephit_model, feature_builder
- **Benoetigte Daten:** feature_builder API

### `dynamic_deephit_model.py`
- **Funktion:** Dynamic DeepHit Competing Risks Model (Semester Level)
- **Abhaengigkeiten:** feature_builder, metrics_logger, recurrent_survival_model
- **Benoetigte Daten:** feature_builder API, Keras Models

### `eval_autoregressive_fail.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** autoregressive_next_exam
- **Benoetigte Daten:** DATA_DIR, metrics JSONs

### `export.py`
- **Funktion:** Hilfsskript zum Exportieren von Daten (z.B. nach CSV/JSON).
- **Abhaengigkeiten:** config, models
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `extended_cox_delta.py`
- **Funktion:** DEPRECATED / CONSOLIDATED WRAPPER: extended_cox_delta.py
- **Abhaengigkeiten:** extended_cox_survival, feature_builder
- **Benoetigte Daten:** feature_builder API

### `extended_cox_survival.py`
- **Funktion:** Extended Cox Proportional Hazards Model (Time-Varying Covariates Edition)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API

### `extended_deep_survival.py`
- **Funktion:** Extended Neural Survival Models (Time-Varying & Delta Panel Edition)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `extended_deep_survival_delta.py`
- **Funktion:** DEPRECATED / CONSOLIDATED WRAPPER: extended_deep_survival_delta.py
- **Abhaengigkeiten:** extended_deep_survival
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `extended_exam_survival.py`
- **Funktion:** Extended Neural Survival Analysis (Prüfungs-basierte Panel Edition)
- **Abhaengigkeiten:** metrics_logger
- **Benoetigte Daten:** CSV (read_csv), Keras Models

### `extract_config.py`
- **Funktion:** Stellt Funktionen bereit: extract_configs...
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `feature_builder.py`
- **Funktion:** Feature Builder & Harmonization Module (Feature Factory)
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `grade_effect_linear.py`
- **Funktion:** Lineare Noteneffekt-Analyse (OLS / Ridge auf Prüfungsebene)
- **Abhaengigkeiten:** metrics_logger
- **Benoetigte Daten:** CSV (read_csv), metrics JSONs

### `landmark_prediction.py`
- **Funktion:** Stellt Funktionen bereit: build_landmark_dataset, main...
- **Abhaengigkeiten:** autoregressive_deep_transformer, autoregressive_next_exam, feature_builder
- **Benoetigte Daten:** feature_builder API, DATA_DIR

### `main.py`
- **Funktion:** Zentraler historischer Einstiegspunkt fuer das Gesamtprojekt (Legacy).
- **Abhaengigkeiten:** aggregate, config, export, simulation, validate
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `metrics_logger.py`
- **Funktion:** Stellt Funktionen bereit: ensure_dir, get_output_dirs, save_metrics...
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** metrics JSONs, Keras Models

### `models.py`
- **Funktion:** Historische Sammlung von Modellklassen/Definitionen vor der Modularisierung.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `oracle_mediation_analysis.py`
- **Funktion:** Oracle Mediationsanalyse (Imai / Pearl Framework)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, metrics JSONs

### `pass_rate_analysis.py`
- **Funktion:** Bestehensquoten-Analyse (Logistische Regression auf Prüfungsebene)
- **Abhaengigkeiten:** metrics_logger
- **Benoetigte Daten:** CSV (read_csv), metrics JSONs

### `plot_breakeven.py`
- **Funktion:** Visualisiert den Break-Even-Point von Zeitinvestitionen vs. Notenertrag.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `plot_calibration_curves.py`
- **Funktion:** Kalibrierungsanalyse & Reliability Diagrams
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, DATA_DIR, Keras Models

### `recurrent_exam_survival.py`
- **Funktion:** Recurrent Exam-Level Survival Analysis (Keras GRU Sequenz auf Prüfungsebene)
- **Abhaengigkeiten:** feature_builder, metrics_logger, recurrent_survival_model
- **Benoetigte Daten:** feature_builder API, Keras Models

### `recurrent_exam_survival_delta.py`
- **Funktion:** DEPRECATED / CONSOLIDATED WRAPPER: recurrent_exam_survival_delta.py
- **Abhaengigkeiten:** feature_builder, recurrent_exam_survival
- **Benoetigte Daten:** feature_builder API

### `recurrent_exam_survival_v2.py`
- **Funktion:** DEPRECATED / CONSOLIDATED WRAPPER: recurrent_exam_survival_v2.py
- **Abhaengigkeiten:** recurrent_exam_survival
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `recurrent_survival_model.py`
- **Funktion:** Recurrent Survival Analysis (Keras GRU Dynamic Deep Survival)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `recurrent_survival_model_delta.py`
- **Funktion:** DEPRECATED / CONSOLIDATED WRAPPER: recurrent_survival_model_delta.py
- **Abhaengigkeiten:** feature_builder, recurrent_survival_model
- **Benoetigte Daten:** feature_builder API

### `run_all_experiments.py`
- **Funktion:** Master Runner Script: Alle Modell-Trainings & Experimente nacheinander ausführen
- **Abhaengigkeiten:** counterfactual_hr_delta, counterfactual_inference_semester_transformer, counterfactual_rr_deephit_delta, counterfactual_rr_exam_rnn_delta, counterfactual_rr_logistic_hazard_delta, deep_survival, deep_transformer_regression, dml_orthogonal_survival, dynamic_deephit_delta_model, dynamic_deephit_model, extended_cox_delta, extended_deep_survival, extended_deep_survival_delta, extended_exam_survival, plot_calibration_curves, recurrent_exam_survival, recurrent_exam_survival_delta, recurrent_exam_survival_v2, recurrent_survival_model, recurrent_survival_model_delta, timeseries_exam, timeseries_exam_transformer, timeseries_semester, timeseries_semester_transformer, train_mlp_baseline, train_mlp_regression, train_oracle_models, transformer_exam_survival, transformer_survival_model
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `run_fast_suite.py`
- **Funktion:** Fast Core Suite Runner (V4.1)
- **Abhaengigkeiten:** counterfactual_deephit_fixed, counterfactual_grade_transformer, counterfactual_hr_analyzer, counterfactual_oracle_logistic_hazard, deep_survival, dml_orthogonal_survival, dynamic_deephit_model, extended_cox_survival, extended_deep_survival, plot_calibration_curves, recurrent_exam_survival, recurrent_survival_model, run_feature_grid_experiments, structural_mediation_analysis, timeseries_exam, timeseries_exam_transformer, timeseries_semester, timeseries_semester_transformer, train_erwerb_blind_models, train_mlp_baseline, train_mlp_regression, train_oracle_models, train_transformer_dml, transformer_exam_survival, transformer_survival_model
- **Benoetigte Daten:** DATA_DIR

### `run_feature_grid_experiments.py`
- **Funktion:** Feature Grid Master Evaluation & Benchmark Pipeline
- **Abhaengigkeiten:** feature_builder, metrics_logger, recurrent_survival_model, transformer_survival_model
- **Benoetigte Daten:** feature_builder API, DATA_DIR, metrics JSONs, Keras Models

### `run_gradeblind_background.py`
- **Funktion:** Hilfsskript zur Datenverarbeitung oder Analyse.
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `run_heavy_suite.py`
- **Funktion:** Heavy Deep Suite Runner (V4.1)
- **Abhaengigkeiten:** autoregressive_deep_transformer, autoregressive_next_exam, deep_transformer_regression, eval_autoregressive_fail, landmark_prediction
- **Benoetigte Daten:** DATA_DIR

### `run_master_suite.py`
- **Fehler beim Parsen:** invalid non-printable character U+FEFF (<unknown>, line 1)

### `run_overnight.py`
- **Funktion:** Master Orchestration: Vollständiger Nachtlauf Pipeline (V3.6)
- **Abhaengigkeiten:** autoregressive_next_exam, deep_transformer_regression, dml_orthogonal_survival, dynamic_deephit_model, extended_cox_survival, extended_deep_survival, feature_builder, metrics_logger, recurrent_exam_survival, recurrent_survival_model, simulation_v3, structural_mediation_analysis, timeseries_exam, timeseries_exam_transformer, timeseries_semester, timeseries_semester_transformer, train_erwerb_blind_models, train_mlp_baseline, train_mlp_regression, train_oracle_models, train_transformer_dml, transformer_exam_survival, transformer_survival_model
- **Benoetigte Daten:** feature_builder API, metrics JSONs

### `run_overnight_v41.py`
- **Funktion:** Master Orchestration: Extended Overnight Runner Pipeline (V4.1)
- **Abhaengigkeiten:** autoregressive_deep_transformer, autoregressive_next_exam, counterfactual_hr_delta, counterfactual_inference_semester_transformer, counterfactual_rr_deephit_delta, counterfactual_rr_exam_rnn_delta, counterfactual_rr_logistic_hazard_delta, deep_survival, deep_transformer_regression, dml_orthogonal_survival, dynamic_deephit_delta_model, dynamic_deephit_model, extended_cox_delta, extended_cox_survival, extended_deep_survival, extended_deep_survival_delta, extended_exam_survival, landmark_prediction, plot_calibration_curves, recurrent_exam_survival, recurrent_exam_survival_delta, recurrent_exam_survival_v2, recurrent_survival_model, recurrent_survival_model_delta, run_feature_grid_experiments, structural_mediation_analysis, timeseries_exam, timeseries_exam_transformer, timeseries_semester, timeseries_semester_transformer, train_erwerb_blind_models, train_mlp_baseline, train_mlp_regression, train_oracle_models, train_transformer_dml, transformer_exam_survival, transformer_survival_model
- **Benoetigte Daten:** DATA_DIR

### `run_retrain_all.py`
- **Funktion:** Master Retraining & Counterfactual Analysis Pipeline (V3.6)
- **Abhaengigkeiten:** run_overnight
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `run_suite_chained.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** run_overnight_v41
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `run_transfer_learning.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** autoregressive_next_exam
- **Benoetigte Daten:** metrics JSONs, Keras Models

### `run_v4_simulation_grid.py`
- **Funktion:** V4 Simulation Sensitivity Grid Search Runner (Multiprocessing)
- **Abhaengigkeiten:** config, export, simulation_v4
- **Benoetigte Daten:** metrics JSONs

### `run_v4_test.py`
- **Funktion:** Stellt Funktionen bereit: main...
- **Abhaengigkeiten:** aggregate, config, export, simulation_v4, validate
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `run_v4_universes.py`
- **Funktion:** Stellt Funktionen bereit: run_v4_universes...
- **Abhaengigkeiten:** config, export, simulation_v4
- **Benoetigte Daten:** metrics JSONs

### `simulate_universes_fgh.py`
- **Funktion:** Partielle Simulation der Universen F, G, H (Ground-Truth Isolations-Welten)
- **Abhaengigkeiten:** aggregate, config, export, simulation_v3
- **Benoetigte Daten:** metrics JSONs

### `simulation.py`
- **Funktion:** Stellt Funktionen bereit: get_exam_noise, generiere_studierende_v3, simuliere_verlaeufe_v3...
- **Abhaengigkeiten:** aggregate, config, export, models, simulation_v2
- **Benoetigte Daten:** metrics JSONs

### `simulation_v2.py`
- **Funktion:** Stellt Funktionen bereit: _erzeuge_semester_liste, generiere_stammdaten, generiere_studierende...
- **Abhaengigkeiten:** aggregate, config, export, models
- **Benoetigte Daten:** metrics JSONs

### `simulation_v3.py`
- **Funktion:** Stellt Funktionen bereit: get_exam_noise, generiere_studierende_v3, simuliere_verlaeufe_v3...
- **Abhaengigkeiten:** aggregate, config, export, models, simulation_v2
- **Benoetigte Daten:** metrics JSONs

### `simulation_v4.py`
- **Funktion:** Stellt Funktionen bereit: get_exam_noise, _erzeuge_semester_liste, generiere_stammdaten...
- **Abhaengigkeiten:** config, models
- **Benoetigte Daten:** Kein direkter Daten-I/O

### `structural_mediation_analysis.py`
- **Funktion:** Strukturelle Mediationsanalyse (Imai / Pearl Causal Mediation Framework)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, metrics JSONs

### `test_deepsurv_scaling.py`
- **Fehler beim Parsen:** invalid non-printable character U+FEFF (<unknown>, line 1)

### `timeseries_exam.py`
- **Funktion:** Zeitreihen-Analyse: Variante 2 (Prüfungs-basierte Zeitreihe GRU)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `timeseries_exam_transformer.py`
- **Funktion:** Zeitreihen-Analyse: Prüfungs-Transformer Regressor (Abschlussnoten-Vorhersage)
- **Abhaengigkeiten:** feature_builder, metrics_logger, timeseries_exam
- **Benoetigte Daten:** feature_builder API, Keras Models

### `timeseries_semester.py`
- **Funktion:** Zeitreihen-Analyse: Variante 1 (Semester-basierte Zeitreihe)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `timeseries_semester_transformer.py`
- **Funktion:** Zeitreihen-Analyse: Semester-Transformer Regressor (Abschlussnoten-Vorhersage)
- **Abhaengigkeiten:** feature_builder, metrics_logger, timeseries_semester
- **Benoetigte Daten:** feature_builder API, Keras Models

### `train_erwerb_blind_models.py`
- **Funktion:** Erwerb-Blind / DSGVO Realistic Model Training
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `train_mlp_baseline.py`
- **Funktion:** Training Script: Status-Vorhersage mit 3-Wege-Split (Train/Val/Test) & Lernkurven
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `train_mlp_regression.py`
- **Funktion:** Training Script: Abschlussnoten-Vorhersage (Regression)
- **Abhaengigkeiten:** feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `train_oracle_models.py`
- **Funktion:** Oracle Models (Theoretical Predictability Upper Bound)
- **Abhaengigkeiten:** extended_deep_survival, feature_builder, metrics_logger
- **Benoetigte Daten:** feature_builder API, Keras Models

### `train_transformer_dml.py`
- **Funktion:** Deep Causal Transformer-DML Pipeline
- **Abhaengigkeiten:** feature_builder, metrics_logger, recurrent_survival_model, transformer_survival_model
- **Benoetigte Daten:** feature_builder API, metrics JSONs, Keras Models

### `train_v3_multi_task.py`
- **Funktion:** Stellt Funktionen bereit: build_v37_multi_task_model, prepare_multi_task_dataset, main...
- **Abhaengigkeiten:** autoregressive_deep_transformer, autoregressive_next_exam, feature_builder
- **Benoetigte Daten:** feature_builder API

### `transformer_exam_survival.py`
- **Funktion:** Exam-Level Causal Transformer Survival Model (DTL Hazard auf Prüfungsebene)
- **Abhaengigkeiten:** feature_builder, metrics_logger, recurrent_survival_model
- **Benoetigte Daten:** feature_builder API, Keras Models

### `transformer_survival_model.py`
- **Funktion:** Causal Transformer Survival Analysis (Semester Attention Edition)
- **Abhaengigkeiten:** feature_builder, metrics_logger, recurrent_survival_model
- **Benoetigte Daten:** feature_builder API, Keras Models

### `validate.py`
- **Funktion:** Validierung & Dokumentation für HSDS Datensatz (DL-Edition)
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** CSV (read_csv)

### `verify_feature_migration.py`
- **Funktion:** Feature Builder Migration & Model Verification Test Suite
- **Abhaengigkeiten:** Keine internen Abhaengigkeiten
- **Benoetigte Daten:** feature_builder API, DATA_DIR, metrics JSONs