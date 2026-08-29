# Refactoring der V4 Simulation Engine (Performance Optimierung)

Dieses Dokument beschreibt den Plan zur signifikanten Beschleunigung der `simuliere_verlaeufe`-Funktion in `simulation_v4.py`, um den Code reif für künftige Grid-Searches zu machen, ohne die Kausalmechanik (DGP) anzutasten.

## User Review Required

> [!IMPORTANT]
> **Warum nicht DuckDB / NumPy im Inner-Loop?**
> Die Simulation ist als *Agent-Based Model* (OOP) geschrieben, d.h. wir simulieren jeden Studenten individuell durch die Zeit. 
> * **DuckDB** ist genial für relationale Massendaten, wäre aber innerhalb einer dynamischen Python-Schleife für einzelne Studenten viel zu langsam (Overhead für Query-Parsing pro Student). 
> * **NumPy** ließe sich nutzen, wenn wir die Simulation *komplett vektorisieren* würden (alle Studenten gleichzeitig als Matrizen durch die Semester schieben). Das würde aber einen gigantischen Rewrite erfordern, bei dem die Gefahr extrem hoch ist, dass sich Nuancen im DGP (z.B. Abbruchreihenfolgen oder Anomalien) unbemerkt verändern.
> 
> **Die Lösung:** Wir bleiben beim exakt gleichen Agent-Based-Code, ersetzen aber die extrem langsamen Pandas-Zugriffe (`.iterrows()`) durch **Native Python Hashmaps (Dictionaries und Sets)**. Da Hash-Lookups in Python in Nanosekunden ablaufen, erreichen wir exakt dasselbe Ziel (Massive Beschleunigung) mit Null Risiko für die Logik.

## Proposed Changes

### Simulation Engine

#### [MODIFY] src/simulation_v4.py
Die Änderungen beschränken sich ausschließlich auf die Funktion `simuliere_verlaeufe`.
1. **Pre-Computation Phase (Vor der Hauptschleife):**
   * Umwandlung von `sg_module` in native Python-Listen (`sg_module_list`).
   * Vorabberechnung der Pflichtmodule als schnelles Python-`set`: `pflicht_module_set = set(...)`
   * Umwandlung des `support_df` in eine Liste von Dictionaries (`support_list = support_df.to_dict("records")`).
2. **Inner-Loop (Innerhalb der Semester-Schleife):**
   * Ersetzen von `for _, row in sg_module.iterrows():` durch `for row in sg_module_list:`
   * Ersetzen von `for _, angebot in support_df.iterrows():` durch `for angebot in support_list:`
   * Ersetzen des teuren Listen-Abgleichs bei `studi.alle_pflicht_bestanden` durch einen simplen Set-Abgleich.

## Verification Plan

### Automatisierte Tests
* Vor dem Refactoring messen wir die Ausführungszeit von `run_v4_test.py` (N=2000).
* Nach dem Refactoring messen wir die Zeit erneut (Ziel: Reduktion der Laufzeit von mehreren Minuten auf wenige Sekunden).
* Wir stellen sicher, dass der V4-Tracker (Zeitbudget-Overloads) weiterhin identisch funktioniert und die Logik unangetastet blieb.
