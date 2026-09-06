---
created: 2026-09-06
last_updated: 2026-09-06
status: abgeschlossen
tags: [protokoll, evaluation, refactoring, cross-linking, venv, v4.2]
---

# Protokoll: Evaluierungsarchitektur V4.2, 14-Modell-Rollout, Dokumenten-Cross-Linking & Venv-Härtung

## 📝 Entwicklungs-Historie & User-Annotationen

**Datum:** 06. September 2026  
**Kontext:** Re-Evaluation des Gesamtrepos mit frischen Augen, Implementierung der modularen OOP-Evaluierungsarchitektur, Rollout auf alle 14 Modellskripte, Vollständiges Dokumenten-Cross-Linking (Tier 1–3) und Härtung der Ausführungsumgebung (Windows Application Control Policy & venv).

### Auslösende User-Annotationen (Zusammenfassung):
> *"Ich möchte, dass Du dieses Projekt einmal mit frischen Augen evaluierst, so wie es gerade steht... Kannst Du mir erstens ein Dokument zur Entwicklung des Projektes erstellen, indem Du die verschiedenen Phasen des Projektes (DE, DA, DL und jetzt) jeweils einzeln beschreibst und charakterisierst... Kannst Du schließlich eine kritische Bewertung des Projektes abgeben, in einem separaten Dokument — gewissermaßen Deine ehrliche Meinung."*
>
> *"Also, ich würde gerne die Evaluierungsarchitektur endlich auf einen guten Stand bringen: Bisher haben wir Konzepte für eine systematische Erweiterung der metrics logger gemacht, aber Du kritisierst zu Recht, dass dies nicht konsequent umgesetzt worden ist... Bitte immer soweit möglich erweitern, i.e. möglichst alle Metriken exportieren, die möglich und sinnvoll sind, alle. Besondere Aufmerksamkeit gilt dem PR-AUC, der vielleicht für alle Vorhersageklassen geloggt werden sollte, in jedem Fall aber auch für dropout... Können wir nicht beides machen [Bootstrap & Asymptotic CIs], allein um es zu vergleichen?"*
>
> *"Die Dokumente alle verlinken, und bei der Gelegenheit auch datieren? Wäre das möglich, in dem Sinne: Wenn in einem Dokument ein anderes relevant wird, könntest Du da einen link ergänzen?... Sofern Du relative links verwendest, sollten die ja direkt auf GitHub funktionieren... Wenn der YAML-String im Markdown Ok aussieht, dann letzteres."*
>
> *"Also lief der Test im venv? Das ist eigentlich ausgenommen... Und könnten wir das venv so dokumentieren (für Dich!), dass Du Skripte immer da laufen lässt?"*

---

## 1. Übersicht der Meilensteine

### A. Phase 1: Retrospektive & Kritische Gesamtbewertung (Commit `bf9f61d`)
- **`docs/08_project_evolution/DeepSupport_Projektentwicklung.md`:** Umfassende Rekonstruktion der vier intellektuellen Phasen (DE: 3NF/ETL → DA: Confounding-Krise → DL: Modell-Skalierung → V4.2: Kausale Sandbox & Feature-Grid).
- **`docs/08_project_evolution/DeepSupport_Kritische_Bewertung.md`:** Ehrliche externe Bewertung mit 8-dimensionaler Bewertungsmatrix (Gesamtnote 8/10). Identifikation von Stärken (Parallelwelten-Konstruktion, feingranulare Attention, methodische Selbstkritik) und Schwächen (synthetischer DGP, historische Metriken-Inkonsistenz, fehlende Konfidenzintervalle).

### B. Phase 2: Implementierung der 5 OOP Evaluator-Klassen
In `src/deepsupport/evaluation/metrics_logger.py` wurden die fünf typisierten Evaluator-Klassen implementiert, während sämtliche bestehende Hilfsfunktionen für 100%ige Abwärtskompatibilität erhalten blieben:
1. **`SurvivalEvaluator`:**
   - Berechnet ROC-AUC, Brier Score, Brier Skill Score ($BSS = 1 - B/B_{\text{ref}}$), F1-Score, Balanced Accuracy und Harrell's C-Index.
   - **PR-AUC für alle Klassen:** Sowohl für Dropout ($y=1$) als auch für Non-Dropout ($y=0$), inklusive Referenz-Prävalenz $\pi_0$ als horizontale Baseline-Linie im PR-Plot.
2. **`RegressionEvaluator`:**
   - $R^2$, adjustiertes $R^2$, RMSE, MAE, Median Absolute Error, Explained Variance, Max Error.
   - Plots: Parity Plot (1:1-Diagonale) und Residuen-Histogramm mit Null-Fehler-Referenz.
3. **`MulticlassEvaluator`:**
   - Makro- und gewichtetes F1, Balanced Accuracy, One-vs-Rest ROC-AUC.
   - One-vs-Rest PR-AUC für jede einzelne Zielklasse (4-Klassen Landmark Status) sowie zeilen-normalisierte Confusion Matrix.
4. **`CausalEvaluator`:**
   - Schätzt Hazard Ratios / Relative Risks und prozentuale Risikoreduktion ($RR_{\text{reduction}} = (1 - HR) \times 100\%$).
   - **Dual-CI-Vergleich:** Berechnet **sowohl** asymptotische Konfidenzintervalle (Delta-Methode auf $\ln(HR)$-Skala) **als auch** empirische Bootstrap-Konfidenzintervalle (Percentile-Methode).
   - Generiert einen visuellen Forest-Plot mit CI-Fehlerbalken.
5. **`DualHeadEvaluator`:**
   - Multi-Task-Evaluator für autoregressive Architekturen (Kombination aus Noten-Regression und Bestehens-Wahrscheinlichkeit).
   - Schreibt eine einheitliche, prefix-separierte Metriken-JSON (`grade_*` und `pass_*`).

### C. Phase 3: Rollout auf alle 14 Modellskripte (Commit `f299539`)
Alle 14 Kernmodelle in `src/deepsupport/models/` wurden erfolgreich auf die neuen Evaluatoren migriert:
- **`SurvivalEvaluator`:** `semester_gru.py`, `semester_transformer.py`, `exam_gru.py`, `exam_transformer.py`, `dynamic_deephit.py`, `extended_deepsurv.py`, `deep_survival.py`, `baseline_classifiers.py`, `extended_cox.py`.
- **`DualHeadEvaluator`:** `autoregressive_gru.py`, `autoregressive_transformer.py`.
- **`CausalEvaluator`:** `dml_orthogonal.py`, `dml_transformer.py`.
- **`RegressionEvaluator`:** `baseline_regressors.py`.

### D. Phase 4: Tier 1–3 Dokumenten-Cross-Linking & YAML-Frontmatter (Commit `6072107`)
59 Markdown-Dateien wurden systematisch aktualisiert:
- **Tier 1 (10 Kerndokumente):** Relative GitHub-Links, Git-basierte Erstellungs- und Änderungsdaten, `## Verwandte Dokumente`-Navigationstabellen.
- **Tier 2 (7 Szenario-Synopsen + 3 Architektur-Dokumente):** Wechselseitige Verlinkung zur `master_synopse_v4_gesamt.md` und thematisch verwandten Dimensionen.
- **Tier 3 (34 Walkthroughs + 5 Conversation Logs):** YAML-Frontmatter mit `status: archiv` bzw. `status: abgeschlossen`, chronologischen Zeitstempeln aus `git log` und Querverweisen.

### E. Phase 5: Härtung der Ausführungsumgebung & Windows Application Control Bypass (Commit `1d2bdd6`)
- **Fehleranalyse:** Der initiale Smoke-Test schlug mit `ImportError: DLL load failed while importing _core: Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert` fehl, da das System-Python außerhalb von genehmigten Whitelist-Pfaden lag.
- **Erkenntnis & Lösung:** Das dedizierte Virtual Environment unter `C:\GitHub_public\.venv` ist von der Richtlinie ausgenommen und voll einsatzfähig.
- **Encoding-Härtung:** Unicode-Zeichen (`π₀`, `τ`, `→`, `²`) in `_print_summary`-Methoden wurden für Windows PowerShell CP1252 Terminal-Kompatibilität durch ASCII-Äquivalente ersetzt.
- **Smoke-Test Resultat:** **5/5 Tests PASSED (Exit-Code 0)** für alle Evaluator-Klassen.

### F. Phase 6: Persistierung der Agenten-Regeln
- Anlage von `AGENTS.md`, `GEMINI.md` und `.agent/rules/python_environment.md` im Projekt-Root.
- Festschreibung des Ausführungsmusters:
  ```powershell
  $env:PYTHONPATH = "src"
  C:\GitHub_public\.venv\Scripts\python.exe <script>
  ```

---

## Verwandte Dokumente

| Dokument | Bezug |
|:---|:---|
| [AGENTS.md](../../AGENTS.md) | Verbindliche Ausführungsregeln für Coding-Agenten |
| [System- & Hardware-Stack](../06_misc/system_and_hardware_stack.md) | Detaillierte Beschreibung der Host- und Cluster-Architektur |
| [DeepSupport_Projektentwicklung](../08_project_evolution/DeepSupport_Projektentwicklung.md) | Entwicklungsgeschichte DE → DA → DL → V4.2 |
| [DeepSupport_Kritische_Bewertung](../08_project_evolution/DeepSupport_Kritische_Bewertung.md) | Ehrliche externe Evaluation |
| [Refactoring-Plan Evaluierungspipeline](../01_master_plans/refactoring_plan_evaluation_pipeline1.md) | Konzeptionelle Grundlage der 5 Evaluator-Klassen |
| [Master-Synopse V4 Gesamt](../03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md) | Quantitative Ergebnisse des 225-Modelle-Grid-Runs |
