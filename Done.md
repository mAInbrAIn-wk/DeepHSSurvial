# ✅ Abgeschlossene Aufgaben

## 2026-09-04: Master Grid Run Vollendung (225 Modelle) & Heavy Suite Härtung
- [x] **Vollendung V4.2 Master Sensitivity Grid (S01–S15):**
  - Alle 15 Szenarien (S01 bis S15) × 3 Architekturen (`grid_semester_gru`, `grid_semester_transformer`, `grid_exam_gru`) × 5 Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`) = **225 DL-Modelle** erfolgreich trainiert, evaluiert und in `output_v4_models/` persistiert (Gesamtrechenzeit: ~23,5 Stunden, Exit-Code 0).
  - Volle Metriken (PR-AUC, ROC-AUC, Brier Score, partielle & isolierte Counterfactual Relative Risks) für alle 225 Modelle vorliegend.
- [x] **Heavy Suite Refactoring (`heavy_suite.py`):**
  - Robuste Pfadauflösung (Repo-Root 3 Ebenen up), I/O-Trennung (`output_v4_heavy/`), automatische DuckDB-Voraggregation und CLI-Szenarienauswahl (`--scenarios`).
  - Einbindung der modularen Architekturen (`autoregressive_gru`, `autoregressive_transformer`, `landmark_prediction`).
  - Getestet, synchronisiert und auf GitHub gepusht.
- [x] **Heavy Deep Suite & Synopse (S01, S07, S08):**
  - Autonome Ausführung der rechenintensiven Exam-Level-Pipelines auf dem Homeserver LXC-Node (ThinkCentre M70q) und Re-Evaluation von Step 2 & 4 auf der Workstation.
  - Next-Exam Dual-Head GRU vs. Deep Transformer mit Sin/Cos Positional Encoding evaluiert: Transformer übertrifft GRU konsistent um $+0.08$ bis $+0.25$ $R^2$ in der Notenvorhersage.
  - Fail-Focus PR-AUC (Minderheitenklasse Nicht-Bestehen) und Landmark Representation Learning (Ende Sem 2: 79.5% 4-Klassen Status-Acc, $R^2 = 0.76$ auf die finale Abschlussnote) vollständig berechnet.
  - Synoptischer Gesamtbericht erstellt: [`docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/synopse_heavy_suite_s01_s07_s08.md).
- [x] **Vollständige Cross-Szenario-Synopsen V4.2 (S01–S15):**
  - Alle verbleibenden Dimensionen vollständig ausgewertet und dokumentiert:
    - Zeitkosten (S01, S09, S10): [`synopse_zeitkosten_s01_s09_s10.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/synopse_zeitkosten_s01_s09_s10.md)
    - RCT-Selektionsparadoxon (S01, S11): [`synopse_rct_selektion_s01_s11.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/synopse_rct_selektion_s01_s11.md)
    - Overload-Penalty Kalibrierung (S01, S12, S13, S14): [`synopse_overload_s01_s12_s13_s14.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/synopse_overload_s01_s12_s13_s14.md)
    - Kombi-Effekt-Resilienz (S01, S15): [`synopse_kombination_s01_s15.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/synopse_kombination_s01_s15.md)
    - Master-Synopse über alle 225 Modelle: [`master_synopse_v4_gesamt.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md)
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
