# Änderungsbericht: Codebase Refactoring & Archivierung (Phase 1)

Dieser Bericht definiert die exakten Pfade und Zuweisungen für das anstehende Refactoring aller 104 Skripte im `src/`-Verzeichnis. Das Ziel ist eine saubere, modulare Package-Struktur, in der Legacy-Skripte archiviert werden und Runner-Skripte ihrer Namensgebung gerecht werden.

> [!TIP]
> **Das neue Paradigma: "One Model = One Script + Wrapper"**
> Auf Deinen Vorschlag hin etablieren wir ein strenges Design-Pattern: Die Skripte in `src/deepsupport/models/` enthalten **nur noch die reine Architektur und einen `train_model(X, y)`-Einstiegspunkt**. Die gesamte Logik rund um Modus-Schleifen (`standard`, `oracle` etc.), das Laden via Feature-Builder und das Logging wird **strikt** in die Runner bzw. Grid-Wrapper ausgelagert. Dadurch werden die Modellskripte universell einsetzbare Bausteine.

## 1. Die neue Verzeichnisstruktur

```text
Abschlussprojekt/
├── src/deepsupport/
│   ├── data_engine/     (Datenaggregation & Feature Engineering)
│   ├── simulation/      (Data Generating Process)
│   ├── models/          (Die reinen Modellarchitekturen)
│   ├── evaluation/      (Metriken & Kausal-Inferenz)
│   └── runners/         (Wrapper, Fast Suite & Heavy Suite)
├── archive/             (Eingefrorene Legacy- & Einmal-Analysen)
└── Artifacts/           (Berichte, Dashboards & Registry)
```

## 2. Zuordnung der aktiven Skripte (Verschiebung in Packages)

### A. `src/deepsupport/data_engine/`
* `feature_builder.py` -> `feature_builder.py`
* `aggregate.py` -> `aggregate.py`
* `config.py` -> `config.py`

### B. `src/deepsupport/simulation/`
* `simulation_v4.py` -> `engine.py`
* `run_v4_simulation_grid.py` -> `sensitivity_grid.py`
* `simulate_universes_fgh.py` -> `simulate_universes_fgh.py`
* `calculate_true_effect.py` -> `ground_truth.py`

### C. `src/deepsupport/models/`
Hier verbleiben **nur die Kernarchitekturen** (und wir trennen sie trennscharf nach Architektur, z.B. GRU vs. Transformer).
* `train_mlp_baseline.py` & `train_mlp_regression.py` -> `landmark_mlp.py`
* `extended_cox_survival.py` -> `extended_cox.py`
* `deep_survival.py` & `extended_deep_survival.py` -> `deep_survival.py`
* `logistic_hazard_landmark.py` & `extended_logistic_hazard.py` -> `neural_hazard.py`
* `recurrent_survival_model.py` -> `semester_gru.py`
* `transformer_survival_model.py` -> `semester_transformer.py`
* `recurrent_exam_survival.py` -> `exam_gru.py`
* `transformer_exam_survival.py` -> `exam_transformer.py`
* `dynamic_deephit_model.py` -> `dynamic_deephit.py`
* `dml_orthogonal_survival.py` & `train_transformer_dml.py` -> `dml_causal.py`
* **Trennung der Autoregressiven Modelle:**
  * `autoregressive_next_exam.py` (RNN/Linear-basiert) -> `autoregressive_gru.py`
  * `autoregressive_deep_transformer.py` (Attention-basiert) -> `autoregressive_transformer.py`

### D. `src/deepsupport/evaluation/`
* `analyze_cross_scenario_models.py` -> `cross_scenario_engine.py`
* `metrics_logger.py` -> `metrics_logger.py`
* `audit_data_completeness.py` -> `completeness_auditor.py`
* `counterfactual_deephit_fixed.py` -> `causal_inference_deephit.py`
* `counterfactual_hr_analyzer.py` -> `causal_inference_hr.py`
* `structural_mediation_analysis.py` -> `mediation_analysis.py`

### E. `src/deepsupport/runners/`
Hier wird der Inhalt massiv bereinigt, um das Paradigma (Wrapper übernimmt die Arbeit) zu erfüllen:
* `run_feature_grid_experiments.py`: Der eigentliche Grid-Runner (läuft über alle Modi).
* `run_fast_suite.py`: Wird **gestrippt**, sodass es **nur noch** den Grid-Runner aufruft! Keine Standalone-Modelle mehr.
* `run_heavy_suite.py`: Behält die Aufrufe für die Deep Transformer und Autoregressiven Modelle (die zu teuer für den Grid-Run sind).

---

## 3. Die Archivierungs-Liste (`archive/`)
**Über 60 Skripte** sind historisch gewachsen, veraltet oder waren Einmal-Analysen (die als `analyze_` oder `counterfactual_` benannt waren, aber inzwischen abgelöst wurden). Diese werden in `archive/` eingefroren.

* **Legacy Simulationen:** `simulation.py`, `simulation_v2.py`, `simulation_v3.py`
* **Redundante Delta-Skripte:** `extended_cox_delta.py`, `extended_deep_survival_delta.py`, `recurrent_exam_survival_delta.py`
* **Einmal-Analysen:** Alle `analyze_*.py` Skripte (wie `analyze_amortization_timeline.py`, `analyze_overload_victims.py`), die gute Insights lieferten, aber nicht zur iterativen Core-Pipeline gehören. (Ihre Funktion ist im `historical_script_registry_full.md` dokumentiert).
* **Veraltete Runner:** `run_all_experiments.py`, `run_retrain_all.py`, `run_v3_multi_task.py`

---

## 4. Nächste Umsetzungsschritte

1. **Freigabe:** Abnahme dieses Plans durch den Nutzer.
2. **Archivierung:** Verschiebung der identifizierten Legacy-Skripte nach `archive/`.
3. **Restrukturierung:** Neuanlage der Ordner in `src/deepsupport/` und Verschieben/Umbenennen der Kern-Skripte gemäß Paradigma.
4. **Runner-Fix:** Umschreiben der `run_fast_suite.py`.
