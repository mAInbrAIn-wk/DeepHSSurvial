---
created: 2026-08-23
last_updated: 2026-09-09
status: aktiv
tags: [backlog, master-plan, roadmap, deep-transformer, pytorch, v5]
---

# DeepSupport Backlog & Forschungs-Roadmap (Stand: 09. September 2026 / V4.2.5)

Dieses Dokument buendelt alle aktiven, geplanten und zurueckgestellten Vorhaben des DeepSupport-Projekts. Es dient als strategische Steuerungsbasis fuer Modellentwicklungen, biostatistische Kausalanlaysen und die anstehende DGP-Version 5.

---

## Prioritaet 1 — Akut (Deep Transformer Evolution & Nachtlauf)

| # | Thema | Referenz / Modul | Status | Geplante Umsetzung |
|:--|:------|:-----------------|:-------|:-------------------|
| **T1** | **Deep Transformer Modernisierung** | `docs/06_misc/benefit_analyse_deep_transformer_suite.md`, `src/deep_transformer_regression.py` | In Vorbereitung | Reduktion von $d=128 \to 64$, 4 Heads, Integration von `SinCosPositionalEncoding`, L2-Weight-Decay (`kernel_regularizer=l2(1e-4)`) zur Beseitigung des Overfittings; Halbierung der Rechenzeit. |
| **T2** | **Sideproject A: Regularisierungs-Benchmark** | `docs/01_master_plans/modellarchitektur_uebersicht_und_ueberarbeitungsplan.md` (Modul 4) | In Vorbereitung | Systematischer Vergleich: Reine L2-Regularisierung ($\lambda \in \{10^{-5}, 10^{-4}, 10^{-3}\}$) vs. Dropout ($p \in \{0.1, 0.2, 0.3\}$) vs. ElasticNet auf Tabellen- und Sequenzdaten. |
| **T3** | **Sideproject B: Asymmetrischer Focal Loss Grid** | `docs/01_master_plans/modellarchitektur_uebersicht_und_ueberarbeitungsplan.md` (Modul 5) | In Vorbereitung | Fokussierung auf die seltene Dropout-Klasse ($\approx 3{,}8\,\%$ Baserate) via $\gamma \in \{1.0, 2.0\}$ und $\alpha \in \{0.25, 0.50\}$ zur Maximierung der Minority PR-AUC. |
| **T4** | **Baseline S01 Re-Run (Integrationstest)** | `ToDo.md` | Offen | Vollstaendiger Durchlauf aller modernisierten Modelle auf Universum A mit den 5 neuen OOP-Evaluatoren. |

---

## Prioritaet 2 — Naechste Iteration (PyTorch Fork & Kausale Mediation)

| # | Thema | Referenz | Status | Geplante Umsetzung |
|:--|:------|:---------|:-------|:-------------------|
| **P1** | **PyTorch / PyCox Fork** | `docs/01_master_plans/pytorch_pycox_port_plan.md` | Geplant (morgen) | Eigenstaendiger paralleler Modellstrang in PyTorch/PyCox: `LogisticHazard`, `DeepHit`, PyTorch-basierte Transformer mit Treatment-Effekt-Koepfen fuer ITE-Schaetzung. Keine destruktive Migration, sondern Portfolio-Erweiterung. |
| **P2** | **Re-Run Kausale Mediation auf V4-Daten** | `docs/01_master_plans/analyseplan_mediation_confounding.md`, `docs/04_causal_and_simulation/04_Kausale_Vergleichsanalyse.md` | Offen | Aktualisierung der Imai/Pearl-Mediationsanalyse (ACME/ADE fuer Fachlich, Ueberfachlich, Psychosozial) von V3.6 auf die finalen V4.1/V4.2 S01 Daten. |
| **P3** | **Submodul-Review `DeepLearning/README.md`** | `ToDo.md` | Offen | Bereinigung veralteter Pfad- und Leakage-Hinweise im Submodul. |

---

## Prioritaet 3 — Version 5 Roadmap (DGP-Rekalibrierung & Code-Härtung)

| Phase | Thema | Referenz | Status | Geplante Umsetzung |
|:---|:------|:---------|:-------|:-------------------|
| **V5.1** | **Config-Bereinigung** | `docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md` | Spezifiziert | Entfernung der Zombie-Parameter (`gewicht_erwerb` etc.); Explizierung der 5 heimlichen `.get()`-Defaults (`overload_penalty_factor` etc.) in `CONFIG`. |
| **V5.2** | **Engine-Parametrisierung** | `config_audit_und_v5_roadmap.md` | Spezifiziert | Auslagerung aller 25+ hartcodierten Konstanten (*Magic Numbers*) aus `engine.py` in typisierte Config-Dataclasses. |
| **V5.3** | **Empirische Startverteilungen** | `config_audit_und_v5_roadmap.md` | Spezifiziert | DZHW/DSW-Kalibrierung: Motivation auf $\kappa=9{,}0$ ($\sigma \approx 0{,}15$), HZB-Note $\kappa=6{,}5$, Zero-Inflated Erwerb (37 % bei 0h, 20h BAföG-Knick), Destatis-Geschlechtermatrizen. |
| **V5.4** | **V5-Validierungslauf** | `config_audit_und_v5_roadmap.md` | Konzept | $N=50.000$ Simulationslauf der rekalibrierten Engine und Re-Benchmarking aller Modelle. |

---

## Prioritaet 4 — Langfristig / Nice-to-Have

| # | Thema | Quelle | Status | Notizen |
|:--|:------|:-------|:-------|:--------|
| **C1** | **Dynamische Motivations-Trajektorien** | `config_audit_und_v5_roadmap.md` | Konzept | Akkumulierter Entfremdungsprozess nach Eccles & Wigfield (Erwartungs-Wert-Theorie). |
| **C2** | **Realism-Mode: Tinto-Peer-Netzwerk** | `LIMITATIONEN_FUTURE_WORK.md` | Konzept | Informelle Lerngruppen und Peer-Netzwerkbildung zur Workload-Pufferung. |
| **C3** | **Marginal Structural Models (MSM)** | `analyseplan_mediation_confounding.md` | Konzept | IPTW-Schaetzung fuer zeitvariierende Confounder mit Rueckkopplung. |
| **C4** | **Parquet-Partitionierung** | `docs/02_architectures_and_models/duckdb_architecture_analysis.md` | Konzept | Multi-Universe Parquet-Layout: `output_dl/data/universe=*/exams.parquet`. |

---

## Erledigte Meilensteine (Historische Referenz)

| Thema | Dokument / Artefakt | Datum |
|:------|:--------------------|:------|
| **V4.2 Master Sensitivity Grid (S01–S15)** | `docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md` (225 Modelle, N=50.000) | 04. Sep 2026 |
| **Heavy Deep Suite (S01, S07, S08)** | `docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md` | 04. Sep 2026 |
| **5 OOP Evaluator-Klassen** | `src/deepsupport/evaluation/metrics_logger.py` (Smoke-Tests 5/5 PASSED) | 06. Sep 2026 |
| **Windows Application Control & Whitelisted Venv** | `C:\GitHub_public\.venv`, `AGENTS.md`, `GEMINI.md` | 06. Sep 2026 |
| **Systematische Verteilungsanalyse & S16** | `docs/04_causal_and_simulation/systematische_verteilungsanalyse_v36_vs_v41.md` | 09. Sep 2026 |
| **Config-Audit & V5-Roadmap** | `docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md` | 09. Sep 2026 |
| **Relationales 11-Tabellen Mermaid ERD** | `docs/04_causal_and_simulation/datenarchitektur_und_eda_v4.md` | 09. Sep 2026 |
| **Sunburst I & II Re-Run & Visual Exploration** | `docs/04_causal_and_simulation/visuelle_datenexploration_v4.md` | 09. Sep 2026 |
| **Methodische Grundlagen der Survival-Analyse** | `docs/04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md` | 09. Sep 2026 |
| **Abschluss-Praesentation** | `Praesi/DeepSupport.tex` (Kurs am 31.07.2026 mit ausgezeichnetem Erfolg beendet) | 31. Jul 2026 |

---

## Changelog

| Datum | Aenderung |
|:------|:---------|
| 23. Aug 2026 | Initiale Erstellung des Backlogs (Stand V3.6) |
| 09. Sep 2026 | Vollstaendige Modernisierung auf V4.2.5; Uebernahme aller Meilensteine (Grid, OOP Evaluatoren, ERD, Survival-Grundlagen); Strukturierung von Deep Transformer Update, PyTorch Fork und V5-Phasenplan |
