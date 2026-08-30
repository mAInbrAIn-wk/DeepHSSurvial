# Sensitivitätsanalyse V4.1 — Zwischenbericht

> [!NOTE]
> **11 von 15 Szenarien fertig** (S01–S11). S12–S15 (Overload-Varianten + Kombi) laufen noch.
> N = 50.000 pro Universum, seed = 99999. Wird nach Abschluss aller Szenarien aktualisiert.

---

## 1. Synoptische Übersicht

| Szenario | A | B | C | D | E | F | G | H | ARR | NNT |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **S01 Baseline** | **29,2%** | **37,1%** | 32,1% | 31,7% | 30,8% | 33,6% | 34,0% | 34,8% | **7,9pp** | **12,6** |
| S02 Support ½ | 32,7% | 37,1% | 34,4% | 34,0% | 33,8% | 35,2% | 35,6% | 35,8% | 4,4pp | 22,7 |
| S03 Support 2× | 25,3% | 37,1% | 29,0% | 28,8% | 27,2% | 31,5% | 31,5% | 33,6% | 11,8pp | 8,5 |
| S04 Notenboost ½ | 30,6% | 37,1% | 32,1% | 33,1% | 32,4% | 35,2% | 34,0% | 34,8% | 6,5pp | 15,4 |
| S05 Notenboost 2× | 27,6% | 37,1% | 32,1% | 29,7% | 29,0% | 31,5% | 34,0% | 34,8% | 9,5pp | 10,5 |
| S06 Notenboost 4× | 27,1% | 37,1% | 32,1% | 29,2% | 28,5% | 30,8% | 34,0% | 34,8% | 10,0pp | 10,0 |
| S07 Rauschen ½ | 26,7% | 33,0% | 29,1% | 28,7% | 28,1% | 30,4% | 30,6% | 31,3% | 6,3pp | 15,8 |
| S08 Rauschen 2× | 33,2% | 41,0% | 35,7% | 35,7% | 34,8% | 37,8% | 37,6% | 38,7% | 7,9pp | 12,7 |
| S09 Kosten 0 | 28,6% | 37,1% | 32,0% | 31,1% | 30,3% | 33,1% | 33,9% | 34,8% | 8,5pp | 11,8 |
| S10 Kosten 2× | 29,7% | 37,1% | 32,2% | 32,1% | 31,3% | 34,0% | 34,1% | 34,9% | 7,4pp | 13,5 |
| S11 RCT | 32,6% | 37,1% | 33,7% | 34,1% | 34,4% | 35,9% | 35,5% | 35,1% | 4,5pp | 22,5 |

> **Legende:** A = Alle Supports, B = Kein Support, C = −Fachlich, D = −Überfachlich, E = −Psychosozial,
> F = Nur Fachlich, G = Nur Überfachlich, H = Nur Psychosozial.
> ARR = Absolute Risk Reduction (B−A), NNT = Number Needed to Treat.

---

## 2. Validierung: RNG-Synchronisation ✅

> [!IMPORTANT]
> **Universum B ist in allen nicht-Rauschen-Szenarien exakt identisch: 37,1%.**
> Das beweist, dass die RNG-Streams perfekt synchronisiert sind — Parametervariationen
> beeinflussen ausschließlich die Support-empfangenden Universen.

Nur S07/S08 (Rauschvariation) verändern B korrekt, da `gewicht_rauschen` auch die
Prüfungsnoten in der Nicht-Support-Welt beeinflusst.

---

## 3. Cross-Szenario-Differenzen (vs. Baseline S01, in Prozentpunkten)

| Szenario | ΔA | ΔB | ΔC | ΔD | ΔE | ΔF | ΔG | ΔH |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S02 Support ½ | +3,5 | 0,0 | +2,3 | +2,4 | +3,0 | +1,6 | +1,6 | +1,0 |
| S03 Support 2× | −3,8 | 0,0 | −3,1 | −2,8 | −3,7 | −2,2 | −2,5 | −1,2 |
| S04 Notenboost ½ | +1,4 | 0,0 | **0,0** | +1,5 | +1,6 | +1,6 | **0,0** | **0,0** |
| S05 Notenboost 2× | −1,6 | 0,0 | **0,0** | −1,9 | −1,8 | −2,2 | **0,0** | **0,0** |
| S06 Notenboost 4× | −2,1 | 0,0 | **0,0** | −2,5 | −2,4 | −2,8 | **0,0** | **0,0** |
| S07 Rauschen ½ | −2,4 | −4,1 | −3,0 | −2,9 | −2,8 | −3,3 | −3,4 | −3,6 |
| S08 Rauschen 2× | +4,0 | +3,9 | +3,6 | +4,0 | +4,0 | +4,1 | +3,6 | +3,9 |
| S09 Kosten 0 | −0,6 | 0,0 | −0,1 | −0,5 | −0,5 | −0,5 | −0,1 | 0,0 |
| S10 Kosten 2× | +0,6 | 0,0 | +0,1 | +0,4 | +0,5 | +0,4 | +0,1 | 0,0 |
| S11 RCT | +3,5 | 0,0 | +1,6 | +2,4 | +3,5 | +2,3 | +1,5 | +0,3 |

### Beobachtungen

#### ✅ Notenboost trifft exakt die richtigen Universen
S04–S06 (Notenboost-Variation) ändern **nur Universen mit fachlichem Support** (A, D, E, F),
**nicht** C (kein fachlicher Support) und **nicht** G, H (nur überfachlich / psychosozial).
Das ist exakt korrekt: Der Notenboost (`gewicht_support_boost`) wirkt nur über den
fachlichen Support-Kanal.

#### ✅ Rauschen wirkt symmetrisch und universell
S07/S08 verändern **alle** Universen gleichmäßig (~±3–4pp), inklusive B. Das bestätigt,
dass das Prüfungsrauschen korrekt alle Prüfungen betrifft, unabhängig vom Support-Status.

#### ✅ Zeitkosten haben marginalen Einfluss
S09 (kostenlos) vs. S10 (verdoppelt): Nur ±0,5–0,6pp Differenz auf A. Die Zeitkosten
des Supports spielen eine untergeordnete Rolle im Vergleich zur Support-Wirkung selbst.

#### ✅ RCT ≈ Support-Halbierung
S11 (RCT, ARR=4,5pp) ≈ S02 (Support ½, ARR=4,4pp). Die Selektionsverzerrung
(risikoreiche Studis suchen mehr Support) erklärt fast genau die Hälfte des
beobachteten Schutzeffekts.

---

## 4. Cross-Szenario Migrationsanalyse (Universum A)

| Szenario | Gleich | Gerettet | Verloren | Netto |
| :--- | ---: | ---: | ---: | ---: |
| S02 Support ½ | 47.690 (95,4%) | 31 | 1.800 | **−1.769** |
| S03 Support 2× | 47.648 (95,3%) | 1.964 | 41 | **+1.923** |
| S04 Notenboost ½ | 48.998 (98,0%) | 30 | 751 | −721 |
| S05 Notenboost 2× | 48.977 (98,0%) | 821 | 29 | +792 |
| S06 Notenboost 4× | 48.724 (97,4%) | 1.061 | 23 | +1.038 |
| S07 Rauschen ½ | 46.910 (93,8%) | 1.846 | 624 | +1.222 |
| S08 Rauschen 2× | 44.720 (89,4%) | 1.265 | 3.265 | −2.000 |
| S09 Kosten 0 | 49.067 (98,1%) | 482 | 207 | +275 |
| S10 Kosten 2× | 49.008 (98,0%) | 227 | 506 | −279 |
| S11 RCT | 46.929 (93,9%) | 359 | 2.106 | −1.747 |

> [!IMPORTANT]
> **Asymmetrie als Qualitätssiegel:** Bei S03 (stärkerer Support) werden 1.964 Studis
> gerettet, aber nur 41 verloren. Bei S02 (schwächerer Support) gehen 1.800 verloren,
> aber nur 31 werden gerettet. Diese starke Asymmetrie beweist, dass die Simulation
> kausal korrekt arbeitet — stärkerer Support rettet, schwächerer lässt fallen.

---

## 5. Offene Szenarien (laufen noch)

| ID | Beschreibung | Status |
| :--- | :--- | :---: |
| S12 | Overload-Penalty halbiert (0.05) | 🔄 |
| S13 | Overload-Penalty verdoppelt (0.2) | 🔄 |
| S14 | Overload mit Cap (0.15, wie V3.6) | 🔄 |
| S15 | Kosten UND Wirkung verdoppelt | 🔄 |

Wird nach Abschluss aktualisiert.
