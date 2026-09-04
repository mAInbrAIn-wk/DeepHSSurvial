# Aktuelle Baustellen (DeepSupport)

## 🔄 Laufende & Nächste Schritte
- [x] **Cluster Grid Run (V4.2 Master Sensitivity Grid):** Alle 15 Szenarien (S01–S15) × 15 Modelle = 225 DL-Modelle erfolgreich trainiert und evaluiert (N=50.000, Seed 99999).
- [ ] **Cross-Szenario-Synthese (S01–S15):** Vollständige metrische Synopse über alle 15 Szenarien und 225 Modelle generieren (Erweiterung der Synopsen zu Overload, RCT und Kombi-Effekt).
- [ ] **DeepLearning README prüfen:** Review der neu hinzugefügten README im Submodul DeepLearning (Fehler, Leakage-Disclaimer).
- [ ] **MoE / Stacking Router:** Router basierend auf kontrafaktischen Universen trainieren.
- [ ] **Dashboard Erweitern:** Tabs 2-5 (Causal & Stress-Test Reports) in das interaktive HTML SVG Dashboard integrieren.

## 📊 Daten & Visualisierung
- [ ] **Neues ERD (Entity Relationship Diagram):** Ein aktuelles ERD für die finale V4 Datenarchitektur erstellen (das alte aus Projekt_DE ist veraltet).
- [ ] **Interaktive EDA / Dashboards:** EDA auf Basis der neuen, finalen Daten re-runnen und interaktiv (Dashboards) für die finale Präsentation aufbereiten.

## 🧠 Modellierung, Evaluierung & Tuning
- [ ] **Klassenspezifische Evaluatoren implementieren:** Ersatz des manuellen Logging-Boilerplates durch 5 modulare Evaluator-Klassen (`SurvivalEvaluator`, `RegressionEvaluator`, `MulticlassEvaluator`, `CausalEvaluator`, `DualHeadEvaluator`) inkl. automatischer Baseline-Linie $\pi_0$ und Brier-Skill-Score.
  - Dokumentation: [`docs/01_master_plans/refactoring_plan_evaluation_pipeline1.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/01_master_plans/refactoring_plan_evaluation_pipeline1.md)
  - Audit & Statusbericht: [`docs/03_evaluations_and_benchmarks/evaluation_pipeline_und_modularisierungsbericht_v41.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/evaluation_pipeline_und_modularisierungsbericht_v41.md)
- [ ] **PyTorch / PyCox Portierung:** Modelle auf PyTorch umstellen.
- [ ] **Regularisierung:** Dropout-Regularisierung überprüfen, evtl. L2-Regulierung testen. Lernkurven analysieren (Finetuning-Potential).
