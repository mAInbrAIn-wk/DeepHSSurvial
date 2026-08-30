# Sensitivitätsanalyse V4.1 — Vollständige Ergebnisse

> [!IMPORTANT]
> **Alle 15 Szenarien × 8 Universen = 120 Runs abgeschlossen.**
> N = 50.000 pro Universum, seed = 99999. Gesamtlaufzeit: 14,6 Stunden.

---

## 1. Synoptische Übersicht

| # | Szenario | A | B | C | D | E | F | G | H | ARR | NNT |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **S01** | **Baseline** | **29,2%** | **37,1%** | 32,1% | 31,7% | 30,8% | 33,6% | 34,0% | 34,8% | **7,9pp** | **12,6** |
| | *Support-Wirkung* | | | | | | | | | | |
| S02 | Support ½ | 32,7% | 37,1% | 34,4% | 34,0% | 33,8% | 35,2% | 35,6% | 35,8% | 4,4pp | 22,7 |
| S03 | Support 2× | 25,3% | 37,1% | 29,0% | 28,8% | 27,2% | 31,5% | 31,5% | 33,6% | 11,8pp | 8,5 |
| | *Notenboost* | | | | | | | | | | |
| S04 | Boost ½ | 30,6% | 37,1% | 32,1% | 33,1% | 32,4% | 35,2% | 34,0% | 34,8% | 6,5pp | 15,4 |
| S05 | Boost 2× | 27,6% | 37,1% | 32,1% | 29,7% | 29,0% | 31,5% | 34,0% | 34,8% | 9,5pp | 10,5 |
| S06 | Boost 4× | 27,1% | 37,1% | 32,1% | 29,2% | 28,5% | 30,8% | 34,0% | 34,8% | 10,0pp | 10,0 |
| | *Rauschen* | | | | | | | | | | |
| S07 | Rauschen ½ | 26,7% | 33,0% | 29,1% | 28,7% | 28,1% | 30,4% | 30,6% | 31,3% | 6,3pp | 15,8 |
| S08 | Rauschen 2× | 33,2% | 41,0% | 35,7% | 35,7% | 34,8% | 37,8% | 37,6% | 38,7% | 7,9pp | 12,7 |
| | *Zeitkosten* | | | | | | | | | | |
| S09 | Kosten 0 | 28,6% | 37,1% | 32,0% | 31,1% | 30,3% | 33,1% | 33,9% | 34,8% | 8,5pp | 11,8 |
| S10 | Kosten 2× | 29,7% | 37,1% | 32,2% | 32,1% | 31,3% | 34,0% | 34,1% | 34,9% | 7,4pp | 13,5 |
| | *Selektion* | | | | | | | | | | |
| S11 | RCT | 32,6% | 37,1% | 33,7% | 34,1% | 34,4% | 35,9% | 35,5% | 35,1% | 4,5pp | 22,5 |
| | *Overload-Penalty* | | | | | | | | | | |
| S12 | Overload ½ | 26,0% | 34,1% | 29,0% | 28,6% | 27,6% | 30,4% | 30,9% | 31,9% | 8,1pp | 12,3 |
| S13 | Overload 2× | 34,6% | 41,8% | 37,1% | 36,9% | 36,3% | 38,8% | 39,0% | 39,7% | 7,3pp | 13,8 |
| S14 | Overload Cap | 26,7% | 35,0% | 29,9% | 29,3% | 28,3% | 31,3% | 31,8% | 32,8% | 8,4pp | 12,0 |
| | *Kombi* | | | | | | | | | | |
| **S15** | **Kosten+Wirkung 2×** | **25,8%** | **37,1%** | 29,0% | 29,3% | 27,7% | 31,9% | 31,5% | 33,6% | **11,3pp** | **8,9** |

> **Legende:** A = Alle Supports, B = Kein Support, C = −Fachlich, D = −Überfachlich, E = −Psychosozial,
> F = Nur Fachlich, G = Nur Überfachlich, H = Nur Psychosozial.
> ARR = Absolute Risk Reduction (B−A), NNT = Number Needed to Treat.

---

## 2. Validierung ✅

> [!IMPORTANT]
> **Universum B = 37,1% in allen 10 Szenarien ohne globale Parameteränderung.**
> Nur Rauschen (S07/08) und Overload (S12/13/14) verändern B — korrekt, weil diese
> Parameter alle Prüfungen bzw. alle Workloads betreffen. Perfekte RNG-Synchronisation.

---

## 3. Sensitivitätsranking

| Rang | Parameter | Δ bei Verdopplung | Δ bei Halbierung | Spannweite | ARR-Stabilität |
| :---: | :--- | ---: | ---: | ---: | :---: |
| 🥇 | **Overload-Penalty** | +5,4pp | −3,1pp | **8,5pp** | stabil (7,3–8,4) |
| 🥈 | **Support-Wirkung** | −3,8pp | +3,5pp | **7,3pp** | variabel (4,4–11,8) |
| 🥉 | **Rauschen** | +4,0pp | −2,4pp | **6,4pp** | stabil (6,3–7,9) |
| 4 | **Selektion (RCT)** | +3,5pp | — | **3,5pp** | — |
| 5 | **Notenboost** | −1,6pp | +1,4pp | **3,0pp** | variabel (6,5–10,0) |
| 6 | **Zeitkosten** | +0,6pp | −0,6pp | **1,2pp** | stabil (7,4–8,5) |

> [!TIP]
> **Entscheidende Unterscheidung:** Overload und Rauschen verschieben das **Gesamtniveau**
> (alle Universen), während Support-Wirkung und Notenboost den **relativen ARR** steuern.
> Für die kausale Evaluation des Support-Programms ist die ARR-Stabilität bei Overload
> eine gute Nachricht — der relative Schutzeffekt ist robust gegenüber Kalibrierungsunsicherheit.

---

## 4. S15 Kombi-Szenario: Kosten + Wirkung verdoppelt

| Metrik | S03 (nur Wirkung 2×) | S15 (Kosten+Wirkung 2×) | Differenz |
| :--- | ---: | ---: | ---: |
| Dropout A | 25,3% | 25,8% | +0,5pp |
| ARR | 11,8pp | 11,3pp | −0,5pp |
| NNT | 8,5 | 8,9 | +0,4 |
| Netto Migration | +1.923 | +1.669 | −254 |

> [!NOTE]
> **S15 zeigt: Die Kostenverdopplung „kostet" bei gleichzeitiger Wirkungsverdopplung
> nur 0,5pp.** Der Großteil der Wirkung bleibt erhalten (11,3pp vs. 11,8pp ARR).
> Das bestätigt erneut, dass die Zeitkosten des Supports ein untergeordneter Faktor
> sind — die Wirkung dominiert klar über die Kosten.

---

## 5. Cross-Szenario-Differenzen (vs. Baseline)

| Szenario | ΔA | ΔB | ΔC | ΔD | ΔE | ΔF | ΔG | ΔH |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S02 Support ½ | +3,5 | **0,0** | +2,3 | +2,4 | +3,0 | +1,6 | +1,6 | +1,0 |
| S03 Support 2× | −3,8 | **0,0** | −3,1 | −2,8 | −3,7 | −2,2 | −2,5 | −1,2 |
| S04 Boost ½ | +1,4 | **0,0** | **0,0** | +1,5 | +1,6 | +1,6 | **0,0** | **0,0** |
| S05 Boost 2× | −1,6 | **0,0** | **0,0** | −1,9 | −1,8 | −2,2 | **0,0** | **0,0** |
| S06 Boost 4× | −2,1 | **0,0** | **0,0** | −2,5 | −2,4 | −2,8 | **0,0** | **0,0** |
| S07 Rauschen ½ | −2,4 | −4,1 | −3,0 | −2,9 | −2,8 | −3,3 | −3,4 | −3,6 |
| S08 Rauschen 2× | +4,0 | +3,9 | +3,6 | +4,0 | +4,0 | +4,1 | +3,6 | +3,9 |
| S09 Kosten 0 | −0,6 | **0,0** | −0,1 | −0,5 | −0,5 | −0,5 | −0,1 | 0,0 |
| S10 Kosten 2× | +0,6 | **0,0** | +0,1 | +0,4 | +0,5 | +0,4 | +0,1 | 0,0 |
| S11 RCT | +3,5 | **0,0** | +1,6 | +2,4 | +3,5 | +2,3 | +1,5 | +0,3 |
| S12 Overload ½ | −3,1 | −3,0 | −3,1 | −3,1 | −3,2 | −3,2 | −3,1 | −2,9 |
| S13 Overload 2× | +5,4 | +4,7 | +5,0 | +5,2 | +5,5 | +5,2 | +5,0 | +4,8 |
| S14 Overload Cap | −2,5 | −2,1 | −2,2 | −2,4 | −2,5 | −2,4 | −2,3 | −2,1 |
| S15 Kombi | −3,3 | **0,0** | −3,0 | −2,4 | −3,2 | −1,7 | −2,5 | −1,2 |

---

## 6. Migrationsanalyse (Universum A)

| Szenario | Gleich | Gerettet | Verloren | Netto | Ratio |
| :--- | ---: | ---: | ---: | ---: | :---: |
| S03 Support 2× | 95,3% | **1.964** | 41 | **+1.923** | 48:1 |
| S15 Kombi | 94,9% | **1.898** | 229 | **+1.669** | 8:1 |
| S12 Overload ½ | 95,8% | 1.634 | 64 | +1.570 | 26:1 |
| S14 Overload Cap | 96,5% | 1.281 | 48 | +1.233 | 27:1 |
| S07 Rauschen ½ | 93,8% | 1.846 | 624 | +1.222 | 3:1 |
| S06 Boost 4× | 97,4% | 1.061 | 23 | +1.038 | 46:1 |
| S05 Boost 2× | 98,0% | 821 | 29 | +792 | 28:1 |
| S09 Kosten 0 | 98,1% | 482 | 207 | +275 | 2:1 |
| S10 Kosten 2× | 98,0% | 227 | 506 | −279 | 1:2 |
| S04 Boost ½ | 98,0% | 30 | 751 | −721 | 1:25 |
| S11 RCT | 93,9% | 359 | 2.106 | −1.747 | 1:6 |
| S02 Support ½ | 95,4% | 31 | 1.800 | −1.769 | 1:58 |
| S08 Rauschen 2× | 89,4% | 1.265 | 3.265 | −2.000 | 1:3 |
| S13 Overload 2× | 92,7% | 89 | 2.801 | −2.712 | 1:31 |

> [!NOTE]
> **S15 (Kombi) zeigt ein geringeres Ratio (8:1) als S03 (48:1).** Die 229 „Verluste"
> kommen daher, dass die verdoppelten Kosten den Curricular-Pfad-Schmetterlingseffekt
> stärker anregen — mehr Studis an der Schwelle haben veränderte Zeitbudgets, was
> mehr Pfadvariationen erzeugt. Netto ist der Effekt aber klar positiv (+1.669).

---

## 7. Schlussfolgerungen

### Parametersensitivität — Was die Auswertung robust macht
1. **ARR ist robust gegenüber Overload-Kalibrierung** (7,3–8,4pp, Δ<1,1pp)
2. **ARR ist robust gegenüber Rauschen** (6,3–7,9pp, Δ<1,6pp)
3. **ARR ist sensitiv auf Support-Wirkung** (4,4–11,8pp) — das ist erwünscht
4. **ARR ist sensitiv auf Selektion** (4,5pp unter RCT vs 7,9pp) — quantifiziert Bias

### Was die Simulation nicht auflöst
- **Zeitkosten sind zu schwach** (±0,6pp) — reicht der Zeitbudget-Ansatz?
- **Rauschen ist der zweitwichtigste Niveautreiber** — hier lohnt empirische Kalibrierung
- **Erwerbstätigkeit × Overload** — Kreuzprodukt nicht getestet, vermutlich überadditiv

### Handlungsempfehlungen für die nächste Iteration
1. **Overload-Penalty empirisch kalibrieren** — der sensitivste Parameter
2. **Erwerbstätigkeit dynamisieren** — Rückkopplung auf Studienbelastung
3. **Rauschen aus empirischen Notenverteilungen ableiten** — statt Annahme
4. **S15 bestätigt:** Wirkung dominiert Kosten → bei Investitionen in Support
   lohnt sich Effektivitätssteigerung mehr als Kostensenkung
