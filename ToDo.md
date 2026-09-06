---
created: 2026-09-02
last_updated: 2026-09-06
status: aktiv
tags: [todo, roadmap, tasks]
---

# Aktuelle Baustellen (DeepSupport)

## 🔄 Laufende & Nächste Schritte
- [x] **Cluster Grid Run (V4.2 Master Sensitivity Grid):** Alle 15 Szenarien (S01–S15) × 15 Modelle = 225 DL-Modelle erfolgreich trainiert und evaluiert (N=50.000, Seed 99999).
- [x] **Cross-Szenario-Synthese (S01–S15):** Vollständige metrische Synopse über alle 15 Szenarien und 225 Modelle generiert ([`master_synopse_v4_gesamt.md`](docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md)).
- [x] **Heavy Deep Suite (S01, S07, S08):** Vollständige Ausführung (GRU, Deep Transformer, Fail PR-AUC, Landmark Representation Learning) und Synopse ([`synopse_heavy_suite_s01_s07_s08.md`](docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md)).
- [x] **Klassenspezifische Evaluatoren implementieren (V4.2.2):** Vollständige Ablösung des manuellen Logging-Boilerplates durch 5 modulare OOP-Evaluator-Klassen (`SurvivalEvaluator`, `RegressionEvaluator`, `MulticlassEvaluator`, `CausalEvaluator`, `DualHeadEvaluator`), PR-AUC für alle Klassen, Dual-CIs, Rollout auf alle 14 Modellskripte und verifizierte Smoke-Test-Suite.
- [ ] **Nachtlauf S01 Baseline (Fast + Heavy Suite):** Voller Integrationstest der neuen Evaluator-Pipeline auf der Baseline-Welt.
- [ ] **DeepLearning README prüfen:** Review der neu hinzugefügten README im Submodul DeepLearning (Fehler, Leakage-Disclaimer).
- [ ] **MoE / Stacking Router:** Router basierend auf kontrafaktischen Universen trainieren.
- [ ] **Dashboard Erweitern:** Tabs 2-5 (Causal & Stress-Test Reports) in das interaktive HTML SVG Dashboard integrieren.

## 📊 Daten & Visualisierung
- [ ] **Neues ERD (Entity Relationship Diagram):** Ein aktuelles ERD für die finale V4 Datenarchitektur erstellen (das alte aus Projekt_DE ist veraltet).
- [ ] **Interaktive EDA / Dashboards:** EDA auf Basis der neuen, finalen Daten re-runnen und interaktiv (Dashboards) für die finale Präsentation aufbereiten.

## 🧠 Modellierung, Evaluierung & Tuning
- [ ] **PyTorch / PyCox Portierung:** Modelle auf PyTorch umstellen ([`pytorch_pycox_port_plan.md`](docs/01_master_plans/pytorch_pycox_port_plan.md)).
- [ ] **Regularisierung:** Dropout-Regularisierung überprüfen, evtl. L2-Regulierung testen. Lernkurven analysieren (Finetuning-Potential).
