# Conversation Log & ADR: Submodules & Methodische Synthese
**Datum:** 2026-09-02 (Abend-Session)

## Kontext
Nach dem grossen V4 Master-Refactoring stellte sich die architektonische Frage, wie die riesigen archive-Daten (25GB) sowie die alten Legacy-Projekte (DataEngineering, DataAnalysis, DeepLearning) sauber mit dem schlanken Hauptprojekt verknuepft werden koennen, ohne die Festplatte zu duplizieren oder GitHub zu sprengen.

## Entscheidungen (ADR)
1. **Portfolio-Architektur via Git Submodules:**
   * Es wurde sich gegen einen monolithischen Ansatz (Nested Git Repos ohne .gitmodules) und gegen Git Subtrees entschieden. 
   * **Entscheidung:** Das Haupt-Repo buendelt die alten Kursprojekte als saubere, Read-Only Git-Submodule unter legacy_projects/.
   * **Vorteil:** Maximale Transparenz fuer Recruiter (klickbare Verlinkungen auf GitHub) bei minimaler Groesse des Haupt-Repos.
2. **Umgang mit GitHub LFS Grenzen:**
   * Da GitHub LFS-Objekte strikt pro Repository (nicht global per Hash) verwaltet, musste das neu geschaffene Archiv-Repository komplett neu (inklusive LFS-Upload der 25GB) auf GitHub hochgeladen werden.
3. **Dokumentations-Konsolidierung (Die Methodische Synthese):**
   * Alte methodische Ueberlegungen aus den Legacy-Projekten wurden gescannt und zentral im Hauptprojekt (docs/04_causal_and_simulation/Methodische_Reflexion.md) konserviert.
   * Ein neues Master-Dokument (docs/01_master_plans/02_Methodische_Evolution_und_Synthese.md) zeichnet den methodischen Reifegrad des Projekts nach (Dropout-Paradoxon -> Causal Panels -> Competing Risks -> Masked Sequence Models).

## Angestossene Operationen
- archive wird per LFS-Force-Push auf GitHub transferiert.
- legacy_projects/ (DataAnalysis, DataEngineering, DeepLearning) sind vollstaendig initialisiert und als Submodule eingebunden.
- Der **Grid-Run (S02-S15)** wurde gestartet, um die Deep-Learning-Modelle auf Robustheit (Stress-Test gegen Pandemie/Noteninflation) zu evaluieren.
