---
created: 2026-09-02
last_updated: 2026-09-11
status: aktiv
tags: [todo, roadmap, tasks, v5-roadmap, deep-transformer, pytorch]
---

# Aktuelle Baustellen (DeepSupport)

## 🔄 Laufende & Nächste Schritte (Akut)
- [x] **Cluster Grid Run (V4.2 Master Sensitivity Grid):** Alle 15 Szenarien (S01–S15) × 15 Modelle = 225 DL-Modelle erfolgreich trainiert und evaluiert (N=50.000, Seed 99999).
- [x] **Cross-Szenario-Synthese (S01–S15):** Vollständige metrische Synopse über alle 15 Szenarien und 225 Modelle generiert ([`master_synopse_v4_gesamt.md`](docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md)).
- [x] **Heavy Deep Suite (S01, S07, S08):** Vollständige Ausführung (GRU, Deep Transformer, Fail PR-AUC, Landmark Representation Learning) und Synopse ([`synopse_heavy_suite_s01_s07_s08.md`](docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md)).
- [x] **Klassenspezifische Evaluatoren implementieren (V4.2.2):** Vollständige Ablösung des manuellen Logging-Boilerplates durch 5 modulare OOP-Evaluator-Klassen (`SurvivalEvaluator`, `RegressionEvaluator`, `MulticlassEvaluator`, `CausalEvaluator`, `DualHeadEvaluator`), PR-AUC für alle Klassen, Dual-CIs, Rollout auf alle 14 Modellskripte und verifizierte Smoke-Test-Suite.
- [x] **Systematischer Config-Audit & V5-Roadmap:** Vollständiger Codeabgleich (`CONFIG` vs. `engine.py`), Identifikation aller Zombies und Magic Numbers sowie empirischer Kalibrierungsplan ([`config_audit_und_v5_roadmap.md`](docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md)).
- [x] **Neues ERD (Entity Relationship Diagram):** Vollständiges 11-Tabellen Mermaid-ERD in [`datenarchitektur_und_eda_v4.md`](docs/04_causal_and_simulation/datenarchitektur_und_eda_v4.md).
- [x] **Interaktive EDA & Visualisierungen:** Sunburst I & II Re-Run auf V4-Daten, 6 Publikationsgrafiken in [`visuelle_datenexploration_v4.md`](docs/04_causal_and_simulation/visuelle_datenexploration_v4.md).
- [x] **Methodische Grundlagen Survival-Analyse:** Zensierungsmathematik, Greenwood-Herleitung und Competing Risks in [`grundlagen_survival_analyse_und_zensierung.md`](docs/04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md).
- [x] **Deep Transformer Modernisierung:** Überarbeitung von `src/deep_transformer_regression.py` ($d=64$, 4 Heads, `SinCosPositionalEncoding`, L2-Regularisierung, Anbindung an OOP-Evaluatoren).
- [x] **Sideproject A (Regularisierungs-Benchmark):** Systematischer Vergleich L2 vs. Dropout vs. ElasticNet (Hybrid konvergiert 3x schneller, Overfitting beseitigt).
- [x] **Sideproject B (Focal Loss Grid):** Asymmetrischer Loss im Vergleich zu BCE (BCE liefert saubere Brier-Kalibrierung).
- [x] **Nachtlauf S01 Baseline (Fast + Heavy Suite):** Integrationstest aller modernisierten Modelle auf Universum A.
- [ ] **Gradeblind Oracle Exploration:** Theoretische Obergrenze der GPA-Vorhersage rein aus latenten DGP-Ressourcen ohne Noten.

---

## 🔬 Kausalinferenz & Nächste Iteration

- [x] **PyTorch / PyCox Fork & LXC Benchmark:** Aufbau der parallelen PyTorch-Suite (`LogisticHazard`, `DeepHit`, `CoxPH` Breslow, `CoxTime`, `DeepHitCompetingRisks`, `ExamTransformerRegressor`, `CausalExamSurvival`, hybride Autoregressoren `NextExamGRU` / `Transformer`, sequentielle Semester-Survival GRU/Transformer) und vollständiger LXC-Lauf über 6 Szenarien ([`pytorch_lxc_benchmark_evaluation_v42.md`](docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md)).
- [x] **PyTorch Causal Suite (DML, MSM, G-Computation):** Native Implementierung in `src/deepsupport/causal/` mit 5-Fold Cross-Fitting, stabilisierten IPTW-Gewichten, cluster-robusten Standardfehlern, kontrafaktischer Pfadsimulation und Ground-Truth-Validierung gegen Universum B.
- [ ] **LXC Causal Nachtlauf:** Ausführung von `src/run_torch_causal_lxc.py` auf dem Debian ThinkCentre Node über 6 Szenarien (S01, S02, S03, S07, S08, S11) mit direkter Ground-Truth-Validierung.
- [ ] **Ablationsstudie Keras vs. PyTorch (Optional nach Kausal-Lauf):** Empirische Prüfung des $2^4$-Plans ([`ablationsplan_keras_vs_pytorch_architekturhypothesen.md`](docs/01_master_plans/ablationsplan_keras_vs_pytorch_architekturhypothesen.md)).
- [x] **Re-Run Kausale Mediation auf V4-Daten:** 4-Stufen-Prüfplan der Imai/Pearl-Mediationsanalyse (Selektions-Audit $d=-0{,}945$, Realistische Mediation $OR=1{,}195$, Oracle-Entzauberung $OR \le 0{,}999$; [`kausale_mediationsanalyse_v42.md`](docs/04_causal_and_simulation/kausale_mediationsanalyse_v42.md)).
- [ ] **Submodul DeepLearning README prüfen:** Review und Bereinigung historischer Pfad- und Leakage-Hinweise.
- [ ] **MoE / Stacking Router:** Optionales Ensembling kontrafaktischer Universen.

---

## 📦 Infrastruktur, Containerisierung & Deployment

- [ ] **Docker-Kapselung & Proxmox-Deployment:**
  - Erstellung eines reproduzierbaren `Dockerfile` basierend auf `requirements.txt` und Python 3.12 (Debian Slim mit PyTorch CPU/CUDA).
  - `docker-compose.yml` mit standardisierten Volume-Mounts für `data_v4_grid/` und `src/output_lxc_causal/`.
  - Proxmox VE Architektur-Check: Betrieb von Docker im Debian LXC Container (`features: nesting=1,keyctl=1`) zur Kapselung aller C++- und Systemabhängigkeiten.

---

## 🚀 Version 5 Roadmap & DGP-Refactoring
- [x] **Audit & Spezifikation:** Dokumentation aller Leerstellen, Zombies und empirischen Kalibrierungsziele erstellt ([`config_audit_und_v5_roadmap.md`](docs/04_causal_and_simulation/config_audit_und_v5_roadmap.md)).
- [ ] **Phase 1 (Config-Bereinigung):** Zombies archivieren (`gewicht_erwerb`, `gewicht_motivation_rauschen`, `gewicht_integration_rauschen`), 5 heimliche Defaults (`overload_penalty_factor`, `support_kosten_faktor`, etc.) explizit in `CONFIG` aufnehmen.
- [ ] **Phase 2 (Engine-Parametrisierung):** Alle 25+ hartcodierten Konstanten aus `engine.py` in strukturierte Config-Dataclasses auslagern.
- [ ] **Phase 3 (Empirische Startverteilungen):**
  - Motivation: $\kappa_{\text{Motivation}} = 9{,}0$ ($\sigma \approx 0{,}15$, Behebung des V4.1-Varianzkollapses).
  - HZB-Note: $\kappa = 6{,}5$ als konfigurierbarer Parameter.
  - Erwerbstätigkeit: Zero-Inflated kontinuierliche Verteilung (37 % bei 0h, 63 % erwerbstätig mit Modus 15h, 20h BAföG-Knick) nach 22. DSW-Sozialerhebung.
  - Migrationshintergrund: Anpassung auf bundesweite DSW-Quote (28 %).
  - Geschlechterverteilung: Fachbereichsgenaue Matrizen pro Studiengang aus Destatis Fachserie 11.
- [ ] **Phase 4 (V5-Validierungslauf):** Head-to-Head-Simulationslauf $N=50.000$ und Re-Benchmarking der Kausal- und Survivalmodelle.
- [ ] **Nice-to-Have / Backlog (Dynamische Trajektorien):** Modellierung des Studienverlaufs als Desillusions- und Entfremdungsprozess (Eccles & Wigfield; Heublein et al. 2017/2022).
- [ ] **Nice-to-Have / Backlog (Realism-Mode):** Stochastischer Peer-Graph pro Kohorte zur Abbildung von Lerngruppen-Synergien und Isolationsrisiken (Tinto-Netzwerkmodell).
