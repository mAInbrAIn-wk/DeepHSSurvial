# Nachtlauf-Plan V4.1

## Übersicht

Vollständiger Neustart aller Simulationen mit V4.1 (korrigierte RNG-Synchronisierung).
Alle bisherigen V4-Ergebnisse sind durch RNG-Desynchronisierung kontaminiert und
werden ersetzt.

---

## Phase 1: Simulation (13 Szenarien × 8 Universen = 104 Runs)

Alle Runs mit `population_seed = 99999`, `N = 25.000`, `save_csv = True`.

### Szenarien

| ID | Dimension | Beschreibung | Override |
| :--- | :--- | :--- | :--- |
| **S01** | Baseline | Standardkonfiguration | — |
| **S02** | Support-Wirkung | Halbiert (mult=2.5) | `support_effect_multiplier: 2.5` |
| **S03** | Support-Wirkung | Verdoppelt (mult=10.0) | `support_effect_multiplier: 10.0` |
| **S04** | Notenboost | Halbiert (0.04) | `gewicht_support_boost: 0.04` |
| **S05** | Notenboost | Verdoppelt (0.16) | `gewicht_support_boost: 0.16` |
| **S06** | Notenboost | Vervierfacht (0.32) | `gewicht_support_boost: 0.32` |
| **S07** | Rauschen | Halbiert (0.09) | `gewicht_rauschen: 0.09` |
| **S08** | Rauschen | Verdoppelt (0.36) | `gewicht_rauschen: 0.36` |
| **S09** | Zeitkosten | Kostenlos (0h) | `support_kosten_override: 0` |
| **S10** | Zeitkosten | Hohe Belastung (60h) | `support_kosten_override: 60` |
| **S11** | Selektion | RCT kalibriert | `rct_support_uptake: True` |
| **S12** | Overload | Penalty halbiert (0.05) | `overload_penalty_factor: 0.05` |
| **S13** | Overload | Penalty verdoppelt (0.2) | `overload_penalty_factor: 0.2` |

> [!NOTE]
> S02 jetzt korrekt: `mult=2.5` = echte Halbierung von Baseline `5.0`.

### Geschätzte Laufzeit

- Pro Run: ~45s (N=25.000 mit per-Student-Seeds = etwas langsamer)
- 104 Runs × 45s = ~78 Min mit ProcessPoolExecutor(max_workers=5)
- Geschätzt: **~80–100 Minuten**

---

## Phase 2: Auswertung pro Szenario (automatisiert)

Für **jedes der 13 Szenarien** wird automatisch erzeugt:

### 2a. Synoptische Tabelle der 8 Universen

| Uni | Dropout-Rate | Dropout-N | Absolventen | Ø Note | Ø Dauer | Support-Teiln. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| A (Full) | ... | ... | ... | ... | ... | ... |
| B (None) | ... | ... | ... | ... | ... | ... |
| ... | | | | | | |

### 2b. Relative Risiken vs. A und vs. B

- A als Referenz für B, C, D, E (Blockade-Welten)
- B als Referenz für F, G, H (isolierte Support-Welten)
- ARR, RRR, NNT

### 2c. Intra-Szenario Migrationsanalyse

Für jede Paarung (A↔B, A↔C, …, A↔E, B↔F, B↔G, B↔H):
- Verloren (Abschluss→Dropout)
- Gerettet (Dropout→Abschluss)
- Netto
- Subgruppen (Erstakademiker, Migration, HZB)

---

## Phase 3: Quervergleiche zwischen Szenarien (pro Parameterdimension)

### 3a. Support-Wirkung (S02 vs S01 vs S03)

| Metrik | S02 (½) | S01 (Baseline) | S03 (2×) |
| :--- | :---: | :---: | :---: |
| Dropout A | ... | ... | ... |
| Dropout B | ... | ... | ... |
| ARR (A vs B) | ... | ... | ... |
| Schutzeffekt netto | ... | ... | ... |

### 3b. Notenboost (S04 vs S01 vs S05 vs S06)

Gleiche Tabelle. Dazu: Notenimpact-Analyse (Notendifferenz bei Supportnutzern,
die in allen Szenarien bestanden haben).

### 3c. Rauschen (S07 vs S01 vs S08)
### 3d. Zeitkosten (S09 vs S01 vs S10)
### 3e. Selektion (S01 vs S11)
### 3f. Overload-Penalty (S12 vs S01 vs S13)

### 3g. Cross-Szenario Migrationsanalyse

Für identisch geseedete Universen (z.B. Uni A in S01 vs Uni A in S03):
- Wie viele Studis wechseln den Status?
- In welche Richtung?
- Subgruppen-Aufschlüsselung

> [!IMPORTANT]
> Bei identischem Seed und synchronisierten RNG-Streams sollten Cross-Szenario-Migrationen
> jetzt ausschließlich durch den veränderten Parameter verursacht sein. Die „Verloren"-Spalte
> sollte bei parametrisch stärkerem Support **nahe null** sein.

---

## Phase 4: Verteilungsplots

### 4a. V4.1 vs V3.6 Verteilungsvergleich
- HZB-Note (Beta vs. Normal?)
- Alter (Beta vs. Uniform?)
- Motivation (Beta vs. Normal?)
- Soziale Integration (Beta vs. Normal?)
- Erwerbstätigkeit (Kategorisch)

### 4b. Ergebnis-Verteilungen (pro Szenario, Uni A)
- Studiendauer (Absolventen vs Dropout)
- Abschlussnoten
- Prüfungsversuche pro Modul
- Module dropped (Zeitkonto-Effekt)

### 4c. Sensitivitäts-Spider/Heatmap
- Zusammenfassung aller Parametervariation in einer Übersichtsgrafik

---

## Phase 5: V3.6 ↔ V4.1 Versionsvergleich

- Vollständiger Code-Diff (wird separat als Artefakt geliefert)
- Ergebnis-Vergleich: V3.6 (seed=99999) vs V4.1 (seed=99999), identische Populationen
  - Dropout-Raten pro Universum
  - Verteilungen

---

## Skripte

### Bestehende Skripte (zu flexibilisieren)

| Skript | Status | Anpassung nötig |
| :--- | :--- | :--- |
| [`run_v4_simulation_grid.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_v4_simulation_grid.py) | ✅ Grid-Runner existiert | Szenarien auf 13 erweitern |
| [`analyze_v4_grid_sensitivity.py`](file:///C:/GitHub_public/Abschlussprojekt/src/analyze_v4_grid_sensitivity.py) | ✅ Analyse existiert | Auf V4.1-Ergebnisse anpassen |
| [`full_migration_zeitkosten.py`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/scratch/full_migration_zeitkosten.py) | ✅ Migrationsanalyse existiert | Generalisieren auf alle Szenarien |
| Notenimpact-Analyse | ✅ Existiert | Auf alle Szenarien mit CSV erweitern |

### Neue Skripte (zu erstellen)

| Skript | Zweck |
| :--- | :--- |
| `nachtlauf_v41.py` | Orchestrierung: Grid-Run → Analyse → Plots → Bericht |
| `cross_scenario_migration.py` | Cross-Szenario-Migrationsanalyse (Uni X in S_i vs Uni X in S_j) |
| `compare_v36_v41.py` | Verteilungsvergleich V3.6 ↔ V4.1 |

---

## Ablauf

```
1. Kurztest (✅ läuft gerade): N=5000 Baseline + Cap-Vergleich + RNG-Validierung
2. Versionsvergleich V3.6 ↔ V4.1 (✅ läuft als Subagent)  
3. [Nutzergenehmigung]
4. Grid-Runner auf 13 Szenarien erweitern
5. Nachtlauf starten (~80-100 Min)
6. Automatisierte Auswertung (Phase 2-4)
7. Synthese-Bericht erstellen
```

## Offene Fragen

> [!IMPORTANT]
> 1. **Overload-Cap als Szenario-Variante?** Aktuell ist die Cap entfernt (per Nutzer-Entscheidung). 
>    S12/S13 variieren den `overload_penalty_factor`. Soll ein S14 mit Cap (0.15) dazu?
> 2. **V3.6-Vergleichsrun:** Soll V3.6 mit demselben Seed (99999) nochmal laufen, oder reichen 
>    die bestehenden V3.6-Daten?
> 3. **Rauschen-Szenarien:** Bei synchronisierten RNG-Streams wird das Prüfungsrauschen jetzt
>    deterministisch per Student/Modul generiert. S07/S08 ändern `gewicht_rauschen`, was die
>    *Amplitude* des Rauschens ändert — die Rauschrichtung (positiv/negativ) bleibt identisch.
>    Das ist korrekt so?
