# Diagnose & Korrekturplan: V4 Sensitivitäts-Gridsearch

## Zusammenfassung der Befunde

> [!CAUTION]
> Der Gridsearch hat **einen kritischen Konfigurations-Bug** und **ein Design-Problem** im RCT-Szenario. Die Haupttabelle vergleicht dadurch Äpfel mit Birnen. Ein korrigierter Re-Run ist nötig.

---

## 1. Bug: `support_effect_multiplier` Baseline = 5.0, nicht 1.0

### Was passiert ist

In [`config.py` Zeile 36](file:///C:/GitHub_public/Abschlussprojekt/src/config.py#L36) steht:

```python
'support_effect_multiplier': 5.0
```

Die Szenarien in [`run_v4_simulation_grid.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_v4_simulation_grid.py#L37-L109) wurden aber geschrieben, als wäre der Default `1.0`:

| Szenario | Gesetzer Wert | Intention | **Tatsächlich vs. Baseline** |
| :--- | :---: | :--- | :--- |
| **S01_baseline** | *(kein Override)* → **5.0** | „Normal" | **Baseline** |
| **S02_supp_half** | **0.5** | „Halbiert" → 0.5× von 1.0 | **0.1× der Baseline (1/10!)** |
| **S03_supp_double** | **2.0** | „Verdoppelt" → 2.0× von 1.0 | **0.4× der Baseline (weniger als halb!)** |
| **S12_high_synergy** | **2.0** + Boost 0.16 | Synergie-Optimum | Effektiv geschwächt vs. Baseline |

### Die Konsequenz

- **S03 hat *weniger* Schutzwirkung als S01** — nicht weil Verdopplung schadet, sondern weil `2.0` tatsächlich eine **Reduktion auf 40% der Baseline (5.0)** ist.
- **S02 mit `0.5` liegt fast bei Null-Support** — weil `0.5` effektiv ein Zehntel der Baseline ist.
- Der Text „Gesamt-Schutz steigt von 20.44% auf 9.35%" ist daher exakt korrekt in den Zahlen, aber die **Interpretation ist invertiert**: S03 ist eine *Abschwächung*, keine Verstärkung.

### Die Lösung

Die Szenario-Overrides müssen relativ zum tatsächlichen Baseline-Wert (`5.0`) definiert werden:

```python
# Korrekte Szenarien:
{"support_effect_multiplier": 2.5},  # 0.5× der Baseline (5.0 * 0.5)
{"support_effect_multiplier": 10.0}, # 2.0× der Baseline (5.0 * 2.0)
```

---

## 2. RCT-Szenario: Volumen-Explosion, nicht nur Selektions-Elimination

### Was passiert ist

Im normalen Modus ist die Support-Nutzung **reaktiv und selektiv**:

- **Psychosozial:** $p = 0.01 + (0.5 - \text{soz\_int}) \times 0.12$. Für durchschnittliche Studierende ($\text{soz\_int} \approx 0.65$): $p \leq 0$. Nur Krisenstudis nutzen das.
- **Überfachlich:** $p = 0.05 + (0.5 - \text{motivation}) \times 0.15$. Für durchschnittliche Studierende ($\text{motivation} \approx 0.65$): $p \approx 0.03$.

Im RCT-Modus wird **pauschal $p = 0.20$ pro Angebot** gesetzt. Bei 3 überfachlichen + 3 psychosozialen Angeboten:

$$P(\text{mindestens 1 überfachlich}) = 1 - 0.8^3 = 48.8\%$$
$$P(\text{mindestens 1 psychosozial}) = 1 - 0.8^3 = 48.8\%$$

### Empirischer Beweis (Mini-Simulation, $N = 1.000$)

| Modus | Support-Teilnahmen (gesamt) | Studierende mit ≥1 Support | Dropout |
| :--- | :---: | :---: | :---: |
| **Normal (Baseline)** | **2.697** | **78.5%** | **24.2%** |
| **RCT ($p=0.20$ flat)** | **10.119** (×3.75!) | **99.0%** | **16.2%** |

Die Zahl der Teilnahmen **vervierfacht** sich nahezu. Das ist kein reiner Selektionseffekt — der RCT-Modus flutet die gesamte Population mit Support.

### Warum das problematisch ist

In einem echten RCT würde man die **gleiche Gesamtmenge** an Support-Teilnahmen randomisiert verteilen. Hier wird stattdessen die Nutzung massiv hochgefahren. Der „48.49% Schutz" ist daher primär ein **Volumeneffekt**, nicht der Nachweis von Selektionsbias.

### Die Lösung

Wenn man den Selektionseffekt isolieren will, muss der RCT-Modus die **Baseline-Teilnahmeanzahl beibehalten**, aber die Zuordnung randomisieren. Ansatz:

1. In der Baseline-Welt die durchschnittliche Teilnahme-Rate pro Angebotstyp pro Semester messen
2. Im RCT-Modus diese Rate als pauschale $p$ verwenden (statt 0.20)

---

## 3. Rauschdimension: Warum die niedrigste Dropout-Rate in Uni A?

### Erklärung

Das Rauschen (`gewicht_rauschen = 0.18`) beeinflusst **beide** Welten (A und B):

| Szenario | Drop Uni A | Drop Uni B | Differenz |
| :--- | :---: | :---: | :---: |
| **S07 (Rauschen 0.09)** | **25.44%** | **30.89%** | 5.45 %p |
| **S01 (Rauschen 0.18)** | **27.84%** | **34.99%** | 7.15 %p |
| **S08 (Rauschen 0.36)** | **31.49%** | **38.67%** | 7.18 %p |

Das ist tatsächlich plausibel:
- **Weniger Rauschen** → weniger zufälliges Scheitern → fähige Studierende bestehen zuverlässiger → **niedrigere Grundrate** in allen Universen.
- Der **relative Schutz** ($RR \approx 1.21$) bleibt dabei stabil.
- Uni A hat die niedrigste Rate, weil hier Support *plus* wenig Zufallspech zusammenwirken.

> [!NOTE]
> Dieses Ergebnis ist in sich konsistent. Es zeigt, dass das Rauschen primär das **absolute Niveau** verschiebt, aber die **Elastizität des Supports** (das Relative Risiko) wenig beeinflusst.

---

## 4. First-Gen Gain: Was genau berechnet wird

### Formel ([`run_v4_simulation_grid.py` Zeilen 290–293](file:///C:/GitHub_public/Abschlussprojekt/src/run_v4_simulation_grid.py#L290-L293)):

```python
fg_gap_A = dropout_rate_firstgen_A - dropout_rate_non_firstgen_A   # Gap in Welt A (mit Support)
fg_gap_B = dropout_rate_firstgen_B - dropout_rate_non_firstgen_B   # Gap in Welt B (ohne Support)
equalizer_gain = (fg_gap_B - fg_gap_A) * 100                       # In Prozentpunkten
```

### Interpretation

- **Positiver Equalizer-Gain** = Support verringert die Bildungsungleichheit (First-Gen Dropout-Gap schrumpft oder dreht sich um)
- Im Baseline: `fg_gap_B = +2.18 %p` (First-Gen brechen 2.18 %p häufiger ab), `fg_gap_A = -2.43 %p` (First-Gen brechen 2.43 %p **seltener** ab mit Support)
- Gain = $2.18 - (-2.43) = +4.61$ Prozentpunkte: Support dreht die Bildungsungleichheit um

> [!IMPORTANT]
> Die Umkehrung des Gaps (First-Gen dropout *sinkt* unter Nicht-First-Gen bei vollem Support) ist ein starkes Simulationsergebnis, das darauf hindeutet, dass der Support gezielt die Risikogruppe erreicht (Erstakademiker erhalten +5% Nutzungswahrscheinlichkeit via Zeile 329).

---

## 5. Zeitkosten-Dimension: Detaillierte Analyse

Die Zeitkostendimension ist **das sauberste Ergebnis** des Gridsearch, da sie nicht vom Multiplier-Bug betroffen ist:

| Szenario | Module abgeworfen (Uni A) | Dropout A | Gesamt-Schutz |
| :--- | :---: | :---: | :---: |
| **S09 (0h Kosten)** | **72.233** | **27.12%** | **22.51%** |
| **S01 (Default ~15-30h)** | **81.164** | **27.84%** | **20.44%** |
| **S10 (60h Kosten)** | **92.127** | **28.45%** | **18.70%** |

$\Delta$ zwischen kostenlos und hoher Belastung: **+19.894 abgeworfene Prüfungen** (+ 27.5%!) und **−3.81 Prozentpunkte Schutzwirkung**.

> [!TIP]
> Hier lohnt sich eine Analyse auf Studierendenebene: Welche Subgruppen (Erwerbstätige, Erstakademiker) sind besonders betroffen?

---

## 6. Vorschlag: Korrigierter Re-Run

### Änderungen am Szenario-Set

| # | Szenario | Override | Intention |
| :--- | :--- | :--- | :--- |
| S01 | Baseline | `{}` (mult=5.0) | Referenzpunkt |
| S02 | Support-Wirkung **halbiert** | `{"support_effect_multiplier": 2.5}` | 50% der Baseline |
| S03 | Support-Wirkung **verdoppelt** | `{"support_effect_multiplier": 10.0}` | 200% der Baseline |
| S04–S06 | Notenboost | *(unverändert, korrekt)* | |
| S07–S08 | Rauschen | *(unverändert, korrekt)* | |
| S09–S10 | Zeitkosten | *(unverändert, korrekt)* | |
| S11 | **Kalibriertes RCT** | `{"rct_support_uptake": True}` + kalibrierte $p$ | Gleiches Volumen, andere Zuordnung |
| S12 | Synergie-Optimum | `{"support_effect_multiplier": 10.0, "gewicht_support_boost": 0.16, "support_kosten_override": 15}` | Korrigierter Multiplikator |

### Offene Design-Entscheidung

> [!IMPORTANT]
> **Soll der Default-Multiplikator in `config.py` auf `1.0` normiert werden?**
> Das wäre sauberer (Baseline = 1.0, dann sind 0.5 und 2.0 intuitiv). Dann müssten aber die Basis-Boosts (`0.02`, `0.015`, `0.035`) in `simulation_v4.py` mit 5 multipliziert werden, um das gleiche Verhalten zu erzeugen. Oder wir lassen alles wie es ist und berechnen die Grid-Werte einfach relativ zum tatsächlichen Default.
