# RNG-Synchronisierungsproblem in der V4-Simulation

## Befund

> [!CAUTION]
> Die Universen A–H teilen denselben RNG-Seed (`population_seed + 100`), aber die RNG-Streams **divergieren ab dem ersten Semester**, weil die Support-Nutzungsschleife in verschiedenen Universen unterschiedlich viele `rng.random()`-Aufrufe erzeugt.

### Empirischer Beweis

**Student STUD000001, gleicher Seed, Uni A vs B:**

| Prüfung | Uni A (Full Support) | Uni B (Kein Support) |
| :--- | :---: | :---: |
| MOD0036, WS2017 | 3,7 ✓ | 3,7 ✓ |
| MOD0035, WS2017 | **4,0** | **5,0** ❌ |
| MOD0039, SS2018 | **2,7** | **5,0** ❌ |

Bereits ab der **zweiten Prüfung** divergieren die Noten — nicht wegen Support-Wirkung, sondern weil die Support-Schleife in Uni A **12 zusätzliche `rng.random()`-Aufrufe** pro Semester erzeugt (je einen pro Angebot, Zeile 335), die in Uni B null sind.

### Quantifizierung der Divergenz

| Datensatz | Gleicher Status | **Unterschiedlicher Status** | Davon: A besser | Davon: B besser |
| :--- | :---: | :---: | :---: | :---: |
| **V4 Universes ($N = 50.000$)** | 34.245 (68,5%) | **15.755 (31,5%)** | 8.022 | 4.420 |
| **Grid S01 ($N = 25.000$)** | 18.720 (74,9%) | **6.280 (25,1%)** | 4.034 | 2.246 |

> [!WARNING]
> **4.420 von 50.000 Studierenden** schließen in Uni B (kein Support) **erfolgreich ab, obwohl sie in Uni A (voller Support) scheitern**. Das ist kein realer Effekt — es ist RNG-Rauschen. Der beobachtete „Netto-Schutzeffekt" (3.602 gerettete) ist eine **Mischung aus echtem kausalem Effekt und RNG-Artefakt**.

### Ursache im Code

In [`simulation_v4.py` Zeile 297–335](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v4.py#L297-L335):

```
for angebot in support_list:         # 12 Angebote in Uni A, 0 in Uni B
    ...
    if rng.random() < p:             # <-- Hier divergiert der RNG-Stream
        teilgenommene_angebote.append(...)
```

Danach in Zeile 152 (Prüfungsrauschen):
```
rng.normal(0, cfg["gewicht_rauschen"])   # <-- Zieht eine komplett andere Zahl
```

## Lösungsvorschläge

### Option A: Pro-Funktions-RNG-Streams (Empfohlen)

Separate RNG-Generatoren für verschiedene Zwecke ableiten:

```python
# Am Anfang von simuliere_verlaeufe:
rng_support = np.random.default_rng(rng.integers(2**63))   # Für Support-Entscheidungen
rng_exams = np.random.default_rng(rng.integers(2**63))     # Für Prüfungsrauschen  
rng_dropout = np.random.default_rng(rng.integers(2**63))   # Für Dropout-Entscheidungen
```

**Problem:** Die RNG-Seeds wären in jedem Universum identisch generiert, aber die `rng_support`-Sequenz divergiert immer noch, was `rng_exams` und `rng_dropout` nicht beeinflusst. → **Prüfungsnoten und Dropout-Rauschen wären perfekt synchronisiert.**

### Option B: Draw-Auffüllung (Pad-Draws)

In der Support-Schleife immer exakt gleich viele Zufallszahlen ziehen, egal ob das Angebot blockiert ist:

```python
for angebot in ALL_support_list:       # Immer alle 12 durchlaufen
    draw = rng.random()                 # Immer ziehen
    if angebot in active_support_list:  # Nur verwenden, wenn nicht blockiert
        if draw < p:
            teilgenommene_angebote.append(...)
```

**Problem:** Reicht nicht, weil auch die Dropout- und Motivations-Boosts bedingte Draws erzeugen.

### Option C: Per-Student-Seed (Maximal sauber)

Für jeden Studierenden einen eigenen Seed ableiten:

```python
for idx, studi in enumerate(studierende):
    studi_rng = np.random.default_rng(base_seed + idx * 1000)
```

**Vorteil:** Perfekte Isolation. Kein Student kann den RNG-Stream eines anderen beeinflussen.
**Nachteil:** ~25.000 RNG-Generatoren, Overhead.

## Empfehlung

**Option A (Pro-Funktions-Streams)** ist der beste Kompromiss: Minimaler Codeeingriff, die Prüfungs- und Dropout-Ergebnisse wären exakt synchronisiert, nur die Support-Teilnahme variiert. Dadurch wäre jede Statusdifferenz zwischen Universen **kausal eindeutig** auf die Support-Konfiguration zurückzuführen.

### Erwarteter Effekt

Mit synchronisierten RNG-Streams:
- Die „Verloren"-Spalte in der Migrationsanalyse sollte **drastisch sinken** (nahe null für B vs A)
- Die „Gerettet"-Spalte zeigt dann den **reinen kausalen Effekt** des Supports
- Die Netto-Schutzrate wird vermutlich **höher** ausfallen, weil der Rausch-Gegenverkehr wegfällt
