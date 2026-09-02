# DeepSupport Backlog (Stand: 23. August 2026)

Dieses Dokument sammelt alle **zurückgestellten** Projekte, die nicht Teil des aktuellen Implementation Plans sind, aber für zukünftige Iterationen relevant bleiben.

---

## Priorität A — Nächste Iteration (nach V3.6-Nachtlauf)

| # | Thema | Quelle | Status | Notizen |
|:--|:------|:-------|:-------|:--------|
| A1 | **DGP-Sensitivitätsanalyse** (Parameter-Grid über Simulationsgewichte) | ToDo.md, IP10, Annotation | Zurückgestellt | Grid-Search über `gewicht_support_boost`, `gewicht_rauschen`, `support_effect_multiplier` etc. Wird relevanter nach Clipping-Diagnostik (AP1). |
| A2 | **Finetuning der Simulationsmechanik** | Annotation | Zurückgestellt | Basiert auf Clipping-Diagnostik-Ergebnissen. Ziel: Künstliche Caps reduzieren, wo sie unnötig verzerren. |
| A3 | **DuckDB-Produktiv-Migration** (Backend-Swap in `feature_builder.py`) | ToDo.md, duckdb_architecture_analysis.md, Annotation | Zurückgestellt | Architektur-Design existiert, Benchmark zeigt 10.6× Speedup. Migration nach Feature-Builder-Konsolidierung. |
| A4 | **Erweiterte Kausalmodelle (MSM)** | Annotation, IP-Diskussion | Zurückgestellt | Marginal Structural Models mit IPTW, g-computation. Erfordert zeitveränderliche Treatment-Modellierung. |

---

## Priorität B — Mittelfristig

| # | Thema | Quelle | Status | Notizen |
|:--|:------|:-------|:-------|:--------|
| B1 | **PyTorch / PyCox Portierung** | ToDo.md | Zurückgestellt | Kompletter Refaktor des TF/Keras-Stacks. Ermöglicht GPU-Training auf Windows (CUDA). PyCox bietet LogisticHazard, DeepHit, CoxPH als fertige Klassen. |
| B2 | **Dashboard-Reparatur** (`dashboard_educational.py`, `dashboard_survival_dl.py`) | Annotation, mehrfach | Zurückgestellt | Streamlit-Dashboards existieren, sind aber nicht auf V3.3-Datenstruktur aktualisiert. |
| B3 | **Dropout-Regularisierung vs. L2** | ToDo.md | Zurückgestellt | Systematischer Vergleich von Dropout vs. L2/Weight-Decay über alle Keras-Modelle. Lernkurven-Analyse als Grundlage. |
| B4 | **Präsentations-Update** (`DeepSupport.tex`) | Annotation | Zurückgestellt | Bildet die Baseline der Projektabgabe. Update nach Abschluss aller Analysen. |
| B5 | **Calibration-Analyse** für alle Survival-Modelle | plot_calibration_curves.py | Teilweise vorhanden | Nur für Landmark-Modelle implementiert. Erweiterung auf Sequenzmodelle ausstehend. |

---

## Priorität C — Nice-to-Have / Langfristig

| # | Thema | Quelle | Status | Notizen |
|:--|:------|:-------|:-------|:--------|
| C1 | **CausalGAN / Counterfactual Generative Modelling** | survival_analysis_evaluation.md | Konzept | Generative kontrafaktische Studienverlaufs-Synthese. |
| C2 | **Fachbereichsklima-Modellierung** | LIMITATIONEN_FUTURE_WORK.md | Konzept | Domänenspezifische Motivationsverläufe nach DZHW/CHE-Surveys. |
| C3 | **Parquet-Partitionierung** | duckdb_architecture_analysis.md | Konzept | Multi-Universe Parquet-Layout: `output_dl/data/universe=*/exams.parquet`. |
| C4 | **Zero-Copy TF Streaming** via `tensorflow_io` + DuckDB Arrow | duckdb_architecture_analysis.md | Konzept | Erfordert `tensorflow-io` Kompatibilität mit TF 2.x auf Windows. |
| C5 | **Dual-Head Architektur** für bestehende Modelle | Annotation (aktuell) | Konzept | Erweiterung bestehender Survival-Modelle um parallelen Regressionskopf. |
| C6 | **Branched/Fused Networks** (statisch + sequentiell) | Annotation (aktuell) | Konzept | Separate Encoder für statische Features und Verlaufsdaten, Fusion Layer. |

---

## Erledigte Items (Referenz)

| Thema | Erledigt in | Datum |
|:------|:-----------|:------|
| Oracle-Feature-Bug-Fix (hardcodierte 0.5-Werte) | walkthrough9.md | 23. Aug 2026 |
| Feature-Grid 5-Modi-Runner | run_feature_grid_experiments.py | 22. Aug 2026 |
| 8-Universen Dual-Strand Ground Truth (F, G, H) | simulate_universes_fgh.py | 22. Aug 2026 |
| Unified Feature Engine (`feature_builder.py`) | feature_builder.py | 22. Aug 2026 |
| Theoretisches Predictability Limit (Gauss-Markov) | theoretical_predictability_bound.md | 23. Aug 2026 |
| Leakage-Warnung Semester-Sequence Regression | feature_grid_results.md | 23. Aug 2026 |
| Mermaid-Fix in projekt_evolution_und_methodenvergleich.md | — | 23. Aug 2026 |

---

## Changelog

| Datum | Änderung |
|:------|:---------|
| 23. Aug 2026 | Initiale Erstellung des separaten Backlog-Dokuments |
