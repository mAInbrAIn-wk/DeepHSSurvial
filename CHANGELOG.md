# Changelog

Alle nennenswerten Änderungen an diesem Projekt werden in dieser Datei dokumentiert.
Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/).

## [3.6.0] - 2026-08-24

### Hinzugefügt / Erledigt
- **AP0 (3-Way-Backbone & Feature-Factory):** `src/aggregate.py` unterstützt 3 austauschbare Backends (`duckdb`, `numpy`, `pandas`), `cp_attempted`-Spalte und optimierten Multi-Column Student-Join. `src/feature_builder.py` mit Vektorisierung in NumPy (34x Speedup), temporalem Switch (`temporal='prev'|'cum'`), `build_exam_panel_df`, Competing-Risks Dual-Target und flexiblen Landmark-Targets implementiert.
- **AP1 (Feature-Builder-Migration & Skript-Konsolidierung):** Alle 8 Modellklassen auf `src/feature_builder.py` umgestellt. Redundante Skripte (`*_delta.py`, `*_v2.py`) durch transparente, abwärtskompatible Wrapper ersetzt. Automatisierte Smoke-Test-Suite `src/verify_feature_migration.py` erstellt (10/10 Tests PASSED).
- **AP2 & AP4 (Master-Orchestrierung & psutil-Benchmarks):** `src/run_overnight.py` als einheitlicher V3.6 Master-Runner mit `PipelineBenchmarkTracker` (RAM-Delta & CPU-Messung pro Schritt) und automatischer HTML/Markdown-Berichterstellung implementiert.
- **AP3 (Verbose-Modus & Clipping-Diagnostik):** `ClippingTracker` in `src/simulation_v3.py` integriert; protokolliert Capping von Motivation, Integration, Overload-Penalty (Deckelung bei 0.15) und Support-Boost in `output_dl/diagnostics/clipping_report.json`.
- **AP5 (Backbone Sanity Check & Benchmark):** `src/benchmark_backbone_sanity_check.py` ausgeführt. Bit-identische Äquivalenz (0.0 Diff) aller 7 Support-/CP-Merkmale über 812.143 Zeilen bewiesen; DuckDB liefert 1.92x Speedup.
- **AP7 (Autoregressive Next-Exam-Vorhersage):** `src/autoregressive_next_exam.py` mit Dual-Head Multi-Task Architektur (GRU-Encoder + Late-Fusion) implementiert. Erreicht ROC-AUC = 0.9202 für Prüfungsbestehen und $R^2 = 0.4618$ für Noten-Regression auf 114k Test-Prüfungen.
- **AP8 (Strukturelle Mediationsanalyse):** `src/structural_mediation_analysis.py` implementiert (Imai/Pearl Framework). Zerlegt Support-Effekte in direkte (ADE) und vermittelte Leistungs-Pfade (ACME).
- **AP9 (Dokumentation & Changelog):** Vollständige Aktualisierung von `CHANGELOG.md`, `walkthrough.md` und `variablen_kausalitaet_und_temporalitaet.md`.

---

## [3.5.0] - 2026-08-23

### Hinzugefügt
- `src/feature_builder.py`: Zentrale Feature-Factory mit 5 Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`).
- `src/run_feature_grid_experiments.py`: Grid-Runner für 4 Modellklassen über alle 5 Modi.
- Theoretische Vorhersagbarkeits-Schranke: $R^2_{\max} = 0.7816$, Bayes-Risiko $= 0.0348$, $\text{AUC}^* = 0.8974$.
- `Artifacts/projekt_evolution_und_methodenvergleich.md`: Dokumentation der Projektgeschichte über alle Phasen.
- `Artifacts/project_index.md` & `Artifacts/dokumentation_der_dokumentation.md`.

### Behoben
- Oracle-Feature-Bug: Hardcodierte `0.5`-Werte in `aggregate.py`, `feature_builder.py` und `extended_cox_delta.py` korrigiert; dynamische $t-1$ Latenzen wiederhergestellt.
- Mermaid-Diagramm-Rendering in `projekt_evolution_und_methodenvergleich.md` repariert.

---

## [3.3.0] - 2026-08-22

### Hinzugefügt
- Universen F, G, H zur Isolation von Confounder-Strukturen (`simulate_universes_fgh.py`).
- Cross-Modal Causal Transformer-DML Pipeline (`train_transformer_dml.py`).
- Erweiterte Survival-Delta-Modelle (`extended_cox_delta.py`, `extended_deep_survival_delta.py`, `recurrent_exam_survival_delta.py`).
- 27-stufige Orchestrierung in `run_retrain_all.py`.

---

## [3.0.0] - 2026-08-20

### Hinzugefügt
- Simulation V3 mit stochastischem Zeitbudget, Überlastungsmechanismus und 5 Paralleluniversen A–E (`simulation_v3.py`).
- Counterfactual Ground Truth Berechnung (`oracle_lift.py`, `compute_macro_effects.py`).
