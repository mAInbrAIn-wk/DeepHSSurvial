# DeepSupport: Dokumentations-Index & Wissensbasis

Dieses Verzeichnis bündelt alle architektonischen Entscheidungen (ADRs), Kausalsynthesen, Evaluierungsberichte, Master-Pläne und Entwicklungs-Protokolle des DeepSupport-Projekts.

---

## 🧭 Inhaltsverzeichnis

### 1. [01_master_plans/](01_master_plans/) — Strategie & Roadmaps
- [`02_Methodische_Evolution_und_Synthese.md`](01_master_plans/02_Methodische_Evolution_und_Synthese.md): Historischer und methodischer Reifegrad (Selection Bias → Causal Panels → Competing Risks → Attention).
- [`deepsupport_master_topology_and_roadmap.md`](01_master_plans/deepsupport_master_topology_and_roadmap.md): Master-Topologie des Frameworks.
- [`refactoring_plan_evaluation_pipeline1.md`](01_master_plans/refactoring_plan_evaluation_pipeline1.md): Masterplan zur Vereinheitlichung der 5 Evaluator-Klassen.
- [`pytorch_pycox_port_plan.md`](01_master_plans/pytorch_pycox_port_plan.md): Portierungs- und Migrationsstrategie für PyTorch & PyCox.

---

### 2. [02_architectures_and_models/](02_architectures_and_models/) — Modellierung & Data Backbone
- [`feature_builder_map.md`](02_architectures_and_models/feature_builder_map.md): Architektur des zentralen Feature Builders (5 Modi: `standard`, `gradeblind`, `blind`, `oracle`, `realistic`).
- [`duckdb_architecture_analysis.md`](02_architectures_and_models/duckdb_architecture_analysis.md): Hochperformante Voraggregation via DuckDB.
- [`model_architectures.md`](02_architectures_and_models/model_architectures.md): Übersicht aller Keras- und Survival-Architekturen.
- [`script_registry.md`](02_architectures_and_models/script_registry.md): Vollständiges Inventar aller Modell- und Trainingsskripte.

---

### 3. [03_evaluations_and_benchmarks/](03_evaluations_and_benchmarks/) — Synopsen & Benchmarks (V4.2)
- **Gesamtsynthesen:**
  - 🌟 [`master_synopse_v4_gesamt.md`](03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md): **Master-Synthese über alle 15 Szenarien und 225 DL-Modelle** des Feature Grids.
  - 🌟 [`synopse_heavy_suite_s01_s07_s08.md`](03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md): **Heavy Deep Suite Gesamtauswertung** (Dual-Head GRU vs. Deep Transformer, Fail PR-AUC, Landmark Representation Learning).
- **Isolierte Sensitivitäts-Synopsen:**
  - [`synopse_supportwirkung_s01_s02_s03.md`](03_evaluations_and_benchmarks/synopse_supportwirkung_s01_s02_s03.md): Variation der Supportwirkung (0.5× bis 2.0×).
  - [`synopse_notenboost_s01_s04_s05_s06.md`](03_evaluations_and_benchmarks/synopse_notenboost_s01_s04_s05_s06.md): Isolierte Analyse des Notenboosts auf den fachlichen Support.
  - [`synopse_rauschen_s01_s07_s08.md`](03_evaluations_and_benchmarks/synopse_rauschen_s01_s07_s08.md): Rausch-Resilienz und Brier-Score-Analyse.
  - [`synopse_zeitkosten_s01_s09_s10.md`](03_evaluations_and_benchmarks/synopse_zeitkosten_s01_s09_s10.md): Zeitkosten-Dimension und Modulabwurf-Effekte.
  - [`synopse_rct_selektion_s01_s11.md`](03_evaluations_and_benchmarks/synopse_rct_selektion_s01_s11.md): Das RCT-Selektionsparadoxon (Zufall vs. Bedarfsallokation).
  - [`synopse_overload_s01_s12_s13_s14.md`](03_evaluations_and_benchmarks/synopse_overload_s01_s12_s13_s14.md): Überlastungs-Penalty und Kalibrierungs-Robustheit.
  - [`synopse_kombination_s01_s15.md`](03_evaluations_and_benchmarks/synopse_kombination_s01_s15.md): Resilienz im Extrem-Szenario (Kombination von Hoch-Effekt und Doppel-Kosten).

---

### 4. [04_causal_and_simulation/](04_causal_and_simulation/) — Kausale Inferenz & Data-Generating Process
- [`04_Kausale_Vergleichsanalyse.md`](04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md): Empirische Gesamtauswertung der 8 Parallelwelten (A–H) vs. Kausal-Schätzer.
- [`03_Uebersicht_Kausale_Ansaetze.md`](04_causal_and_simulation/03_Uebersicht_Kausale_Ansaetze.md): Methodischer Vergleich von Naive vs. FWL-Partialling vs. DML vs. Oracle-Mediation.

---

### 5. [06_misc/](06_misc/) — Infrastruktur & Hardware
- [`system_and_hardware_stack.md`](06_misc/system_and_hardware_stack.md): Umfassende Dokumentation der Recheninfrastruktur (HP EliteDesk G5 Workstation vs. Lenovo ThinkCentre M70q Homeserver Cluster Node unter Proxmox VE / Debian LXC).

---

### 6. [07_conversation_logs/](07_conversation_logs/) — Entwicklungs-Historie & User-Annotationen
- [`00_Historisches_Gesamtprotokoll.md`](07_conversation_logs/00_Historisches_Gesamtprotokoll.md): Chronologisches Archiv der bisherigen 299 Iterationen.
- [`01_History_Selection_Bias_and_Confounding.md`](07_conversation_logs/01_History_Selection_Bias_and_Confounding.md): Historischer Diskurs zur Genese der Parallelwelten und des Dropout-Paradoxons.
- [`2026_09_02_Master_Refactoring.md`](07_conversation_logs/2026_09_02_Master_Refactoring.md): Protokoll des großen Code- und Archiv-Refactorings (V3.6 → V4).
- [`2026_09_02_Submodules_and_Synthesis.md`](07_conversation_logs/2026_09_02_Submodules_and_Synthesis.md): Portfolio-Architektur via Git Submodules und LFS-Force-Push.
- [`2026_09_04_Full_Grid_Run_and_Heavy_Suite.md`](07_conversation_logs/2026_09_04_Full_Grid_Run_and_Heavy_Suite.md): Vollendung des 225-Modelle-Grid-Runs, ThinkCentre Cluster Execution & Heavy Suite Synopse.