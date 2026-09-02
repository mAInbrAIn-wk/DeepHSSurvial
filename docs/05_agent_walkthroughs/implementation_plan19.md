# Implementierungsplan: V4 Simulations-Gridsearch & Sensitivitätsanalyse

Dieser Plan definiert den Versuchsaufbau für eine systematische Sensitivitätsanalyse der **V4-Simulations-Engine**. Ziel ist es, quantitativ zu messen, wie robust und sensitiv die Ground-Truth-Makroeffekte (Dropout-Raten, Relative Risiken der Universen A–H, First-Gen- und Migrations-Gaps sowie Superadditivitäts-Synergien) auf systematische Variationen der Simulationsparameter reagieren.

---

## 1. Übersicht & Zielsetzung

Wir untersuchen, wie sich Variationen in drei Kern-Dimensionen (sowie zwei vorgeschlagenen Zusatz-Dimensionen) auf die kontrafaktischen Universen A–H auswirken:
* **Seed-Konsistenz:** Alle Szenarien nutzen dieselbe initiale Studierenden-Kohorte (`population_seed = 99999`) und denselben Simulator-Zufallsstrom (`sim_seed = 100099`), sodass Unterschiede rein kausal auf die Parameter zurückzuführen sind.
* **Vollständige 8-Universen-Abbildung:** Pro Szenario werden alle 8 Universen (A: Full Support bis H: Only Psychosozial) simuliert.
* **Multiprocessing-Beschleunigung:** Durch parallele Ausführung über CPU-Cores wird die Gesamtlaufzeit der ~12–14 Szenarien von mehreren Stunden auf ca. 20–30 Minuten reduziert.

---

## 2. Der Parameter-Grid (12–14 Szenarien)

Wir strukturieren den Grid als **One-at-a-Time (OAT) Sensitivitäts-Matrix** um die V4-Baseline herum, ergänzt um ausgewählte Extremszenarien:

| Szenario-ID | Parameter-Dimension | Konfiguration / Multiplikator | Konkreter Wert im Code | Forschungsfrage / Hypothese |
| :--- | :--- | :---: | :---: | :--- |
| `S01_baseline` | **Baseline V4** | Normal ($1.0\times$) | Mult=1.0, Boost=0.08, Noise=0.18, Cost=30h | Referenzpunkt |
| `S02_supp_half` | Support-Wirkung | Halbiert ($0.5\times$) | `support_effect_multiplier = 0.5` | Wie stark dämpft schwacher Support den Gesamteffekt? |
| `S03_supp_double` | Support-Wirkung | Verdoppelt ($2.0\times$) | `support_effect_multiplier = 2.0` | Sättigt der Support-Effekt oder verdoppelt sich das Relative Risiko? |
| `S04_grade_half` | Noten-Gewicht Fachlich | Halbiert ($0.5\times$) | `gewicht_support_boost = 0.04` | Fällt fachlicher Support hinter überfachlichen Support zurück? |
| `S05_grade_double` | Noten-Gewicht Fachlich | Verdoppelt ($2.0\times$) | `gewicht_support_boost = 0.16` | Wird fachlicher Support zum dominanten Einzelfaktor? |
| `S06_grade_quad` | Noten-Gewicht Fachlich | Vervierfacht ($4.0\times$) | `gewicht_support_boost = 0.32` | Erreicht fachlicher Notenboost eine Determinanz über Klausurerfolg? |
| `S07_noise_half` | Stochastisches Rauschen | Halbiert ($0.5\times$) | `gewicht_rauschen = 0.09` | Treten kausale Effekte in deterministischerer Welt schärfer hervor? |
| `S08_noise_double` | Stochastisches Rauschen | Verdoppelt ($2.0\times$) | `gewicht_rauschen = 0.36` | Verwässert hohes Prüfungsrauschen die Support-Wirkung? |
| `S09_cost_zero` *(Vorschlag)* | Support-Zeitkosten | Kostenlos ($0\text{ h}$) | `kosten_h = 0` | Wieviel Support-Effekt wurde bisher durch Workload-Overload aufgefressen? |
| `S10_cost_high` *(Vorschlag)* | Support-Zeitkosten | Verdoppelt ($60\text{ h}$) | `kosten_h = 60` | Kippt Support bei hoher Belastung in negative Effekte (Overload-Trap)? |
| `S11_rct_uptake` *(Vorschlag)* | Selektions-Endogenität | RCT (Zufällige Zuweisung) | $p = \text{const} = 0.20$ (ohne Noten/Mot-Bias) | Wie stark ist die reine Selektionsverzerrung in der DGP? |
| `S12_high_synergy` *(Vorschlag)* | Extrem: Starker Hebel | Multiplier=2.0 + Boost=0.16 | Mult=2.0, Boost=0.16, Cost=15h | Maximales Potenzial bei optimierten Rahmenbedingungen |

---

## 3. Vorgeschlagene Änderungen an den Komponenten

### [NEW] [`src/run_v4_simulation_grid.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_v4_simulation_grid.py)
* Haupt-Orchestrierungs-Skript für den Gridsearch.
* Nutzt `concurrent.futures.ProcessPoolExecutor`, um die Universen bzw. Szenarien parallel über alle verfügbaren CPU-Kerne zu rechnen.
* Speichert die Ergebnisse pro Szenario strukturiert unter `output_v4_grid/<szenario_id>/`.
* Berechnet automatisiert für jedes Szenario:
  1. Dropout-Raten aller 8 Universen (A bis H)
  2. Relative Risiken ($RR_{B \dots H}$)
  3. First-Gen und Migrations-Gaps (Equalizer-Effekt)
  4. Superadditivitäts-Metrik (Synergie-Interaktion in Prozentpunkten)

### [NEW] [`src/analyze_v4_grid_sensitivity.py`](file:///C:/GitHub_public/Abschlussprojekt/src/analyze_v4_grid_sensitivity.py)
* Aggregiert alle JSON-Ergebnisse des Grids in eine synoptische Matrix.
* Erstellt vergleichende Markdown-Tabellen und Plots:
  * Sensitivitäts-Elastizitäten: $\frac{\Delta RR}{\Delta \text{Parameter}}$
  * Heatmap der relativen Risiken über die Szenarien
* Exportiert den finalen Bericht nach `sensitivitaetsanalyse_v4_grid.md`.

---

## 4. Verifikations- und Auswertungsplan

### Automatisierte Überprüfungen
* **Sanity-Check:** Überprüfung, dass $RR_B > 1.0$ in allen Szenarien gilt (Support schützt immer).
* **Monotonie-Prüfung:** Prüfen, ob $RR_B(\text{S03}) > RR_B(\text{S01}) > RR_B(\text{S02})$ (Höhere Support-Wirkung $\rightarrow$ größerer Abstand zwischen Universum A und Universum B).
* **Overload-Check:** Prüfen, ob bei $60\text{ h}$ Zeitkosten mehr Module abgeworfen werden (`tracker_modules_dropped`).

### Generierte Artefakte
* `sensitivitaetsanalyse_v4_grid.md`: Vollständiger, vergleichender Synthesebericht mit allen 12 Szenarien.
* `plots_v4_sensitivity_grid.png`: Visualisierung der Elastizitäten und Dropout-Spreads über alle 8 Universen.

---

## 5. Fragen & Optionen zur Abstimmung

> [!IMPORTANT]
> **Stichprobengröße pro Universum:**
> * Option 1 (Empfohlen): $N = 25.000$ Studierende pro Universum $\rightarrow$ Sehr präzise Statistiken ($p < 0.0001$), Gesamtlaufzeit für alle 12 Szenarien parallel: **ca. 15–20 Minuten**.
> * Option 2: $N = 50.000$ Studierende pro Universum $\rightarrow$ Maximale Datenmenge identisch zur Hauptstudie, Gesamtlaufzeit parallel: **ca. 35–45 Minuten**.
