# ✅ Abgeschlossene Aufgaben

## 2026-09-04: Master Grid Run Vollendung (225 Modelle) & Heavy Suite Härtung
- [x] **Vollendung V4.2 Master Sensitivity Grid (S01–S15):**
  - Alle 15 Szenarien (S01 bis S15) × 3 Architekturen (`grid_semester_gru`, `grid_semester_transformer`, `grid_exam_gru`) × 5 Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`) = **225 DL-Modelle** erfolgreich trainiert, evaluiert und in `output_v4_models/` persistiert (Gesamtrechenzeit: ~23,5 Stunden, Exit-Code 0).
  - Volle Metriken (PR-AUC, ROC-AUC, Brier Score, partielle & isolierte Counterfactual Relative Risks) für alle 225 Modelle vorliegend.
- [x] **Heavy Suite Refactoring (`heavy_suite.py`):**
  - Robuste Pfadauflösung (Repo-Root 3 Ebenen up), I/O-Trennung (`output_v4_heavy/`), automatische DuckDB-Voraggregation und CLI-Szenarienauswahl (`--scenarios`).
  - Einbindung der modularen Architekturen (`autoregressive_gru`, `autoregressive_transformer`, `landmark_prediction`).
  - Getestet, synchronisiert und auf GitHub gepusht.
- [x] **Infrastruktur-Dokumentation:**
  - `docs/06_misc/system_and_hardware_stack.md` erstellt (HP EliteDesk G5 Workstation vs. Lenovo ThinkCentre M70q LXC Debian).
  - Evaluator-Klassen-Refactoring als priorisierte Aufgabe in `ToDo.md` verlinkt.

## 2026-09-03: Kausal-Synthese & V4 Grid Zwischenauswertungen
- [x] **Kausale & Historische Dokumentation:**
  - `01_History_Selection_Bias_and_Confounding.md`: Vollständige Aufarbeitung des Dropout-Paradoxons, Immortal-Time Bias und der Genese der Paralleluniversen.
  - `03_Uebersicht_Kausale_Ansaetze.md`: Differenzierung zwischen Sandbox (A-H), Imai/Pearl Struktureller Mediation und Oracle-Diagnose.
  - `04_Kausale_Vergleichsanalyse.md`: Empirisch verifizierte Gesamtauswertung aus den V3.6 JSONs/CSVs (Dropout RRs, kausaler Notenboost von -0.09 GPA vs. +0.22 naivem Confounding, Oracle-Mediation).
- [x] **Synoptische Zwischenberichte (S01–S08):**
  - `synopse_supportwirkung_s01_s02_s03.md`: Auswertung der Supportwirkung (0.5× vs. 1.0× vs. 2.0×).
  - `synopse_notenboost_s01_s04_s05_s06.md`: Verifikation der selektiven Notenboost-Wirkung auf den Fachsupport.
  - `synopse_rauschen_s01_s07_s08.md`: Rausch-Resilienz und Modellstabilität über alle Zielgrößen.
- [x] **V4 Grid Runner:** S01 bis S09 zu 100% fehlerfrei trainiert und persistiert (S10 läuft).

## 2026-09-02: V4 Master Refactoring & Portfolio
- [x] **Submodule Portfolio:** Archive, DataAnalysis, DataEngineering und DeepLearning als saubere Git-Submodule integriert.
- [x] **Git LFS Architektur:** 25 GB Rohdaten komplett aus dem Code-Repo extrahiert und in LFS-gestütztes Archiv migriert.
- [x] **Requirements Update:** 
equirements.txt exakt an die Cluster-Umgebung (inkl. scikit-survival, DuckDB, Tensorflow 2.21) angepasst.
- [x] **Orchestrierung:** grid_runner.py I/O-sicher gemacht (Trennung von data_root und output_root).
- [x] **Doku-Struktur:** Markdown-Dateien systematisch in docs/ einsortiert und Historisches Protokoll (299 Iterationen) generiert.

## Frühere Meilensteine
- [x] Leakage-Fix in den 5 ML-Skripten (Student-Level Split implementiert)
- [x] Future Leakage Fix: cp_rueckstand nutzt nun cp_cum_prev in Exam-Formaten
- [x] Oracle Feature Extension: hidden_overload und hidden_zeit_puffer in Feature Builder ergänzt
- [x] Feature Counts in README.md & Co. aktualisieren
- [x] SQL Backend implementieren (DuckDB & Arrow Ansatz evaluiert und in Architektur integriert)
