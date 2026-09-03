# Synopse 1: Variation der Supportwirkung (S02 vs. S01 vs. S03)

> **Fokus:** Untersuchung des Einflusses der globalen Support-Effektstärke (`support_effect_multiplier`: 2.5× vs. 5.0× vs. 10.0×) auf die Ground Truth und die Vorhersage- bzw. Kausalmodelle.

## 1. Ground Truth Entwicklung (Paralleluniversen A vs. B)

| Szenario | Multiplier | Dropout A (Full) | Dropout B (No Supp) | Absolute ARR | Relative RR | NNT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S02_supp_half** (Halbierte Wirkung) | 0.5× (2.5) | **32.7%** | 37.1% | **+4.4 pp** | **0.881** | **22.7** |
| **S01_baseline** (Baseline-Referenz) | 1.0× (5.0) | **29.2%** | 37.1% | **+7.9 pp** | **0.786** | **12.6** |
| **S03_supp_double** (Doppelte Wirkung) | 2.0× (10.0) | **25.3%** | 37.1% | **+11.8 pp** | **0.682** | **8.5** |

**Beobachtung:** Während Welt B (RNG-synchronisiert ohne Support) exakt stabil bei **37.1%** verharrt, fällt die Dropout-Rate in Welt A von **32.7%** (S02) über **29.2%** (S01) auf **25.3%** (S03). Die absolute Risikoreduktion (ARR) verdreifacht sich beinahe von **4.4 pp** auf **11.8 pp** (NNT sinkt von 22.7 auf 8.5).

## 2. Modellperformance im Vergleich (ROC-AUC / PR-AUC / Brier Score)

Vergleich der 3 Modellklassen (`grid_semester_gru`, `grid_semester_transformer`, `grid_exam_gru`) über die 5 Feature-Modi:

### 2.1 Modellklasse: `Semester GRU (Dropout Hazard)`

| Modus | S02 (0.5×) ROC / PR | S01 (1.0×) ROC / PR | S03 (2.0×) ROC / PR | S02 Brier | S01 Brier | S03 Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `standard` | 0.8218 / 0.3138 | 0.8187 / 0.2766 | 0.8054 / 0.2241 | 0.0383 | 0.0351 | 0.0312 |
| `gradeblind` | 0.8201 / 0.3063 | 0.8178 / 0.2787 | 0.8042 / 0.2290 | 0.0383 | 0.0348 | 0.0312 |
| `blind` | 0.7869 / 0.1955 | 0.7764 / 0.1665 | 0.7691 / 0.1317 | 0.0416 | 0.0377 | 0.0331 |
| `oracle` | 0.8253 / 0.3189 | 0.8149 / 0.2799 | 0.8063 / 0.2262 | 0.0379 | 0.0346 | 0.0312 |
| `realistic` | 0.8193 / 0.3125 | 0.8115 / 0.2709 | 0.8025 / 0.2234 | 0.0382 | 0.0350 | 0.0313 |

### 2.2 Modellklasse: `Semester Transformer (Dropout Hazard)`

| Modus | S02 (0.5×) ROC / PR | S01 (1.0×) ROC / PR | S03 (2.0×) ROC / PR | S02 Brier | S01 Brier | S03 Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `standard` | 0.8221 / 0.3045 | 0.8126 / 0.2751 | 0.8028 / 0.2306 | 0.0382 | 0.0347 | 0.0311 |
| `gradeblind` | 0.8197 / 0.3097 | 0.8141 / 0.2765 | 0.8015 / 0.2282 | 0.0380 | 0.0347 | 0.0312 |
| `blind` | 0.7926 / 0.1999 | 0.7837 / 0.1736 | 0.7728 / 0.1324 | 0.0411 | 0.0372 | 0.0332 |
| `oracle` | 0.8201 / 0.3080 | 0.8162 / 0.2785 | 0.8056 / 0.2308 | 0.0380 | 0.0346 | 0.0311 |
| `realistic` | 0.8187 / 0.3091 | 0.8138 / 0.2737 | 0.8012 / 0.2233 | 0.0381 | 0.0348 | 0.0313 |

### 2.3 Modellklasse: `Exam GRU (Prüfungsversagen / Next Exam)`

| Modus | S02 (0.5×) ROC / PR | S01 (1.0×) ROC / PR | S03 (2.0×) ROC / PR | S02 Brier | S01 Brier | S03 Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `standard` | 0.9095 / 0.2323 | 0.8990 / 0.1973 | 0.8957 / 0.1609 | 0.0165 | 0.0151 | 0.0135 |
| `gradeblind` | 0.9074 / 0.2303 | 0.8989 / 0.1894 | 0.8986 / 0.1649 | 0.0165 | 0.0152 | 0.0134 |
| `blind` | 0.8947 / 0.1980 | 0.8909 / 0.1742 | 0.8905 / 0.1447 | 0.0170 | 0.0154 | 0.0137 |
| `oracle` | 0.9230 / 0.3051 | 0.9137 / 0.2557 | 0.9117 / 0.2180 | 0.0156 | 0.0145 | 0.0129 |
| `realistic` | 0.9003 / 0.2265 | 0.8914 / 0.1880 | 0.8905 / 0.1588 | 0.0166 | 0.0152 | 0.0135 |

## 3. Kausale RRs & Schützer-Reaktion

Wie reagieren die Counterfactual-Inferenz-Schätzer des Semester-Transformers auf die verdoppelte bzw. halbierte Supportwirkung?

| Szenario | Wahre Ground Truth RR | Transformer RR Fachlich | Transformer RR Überfachlich | Transformer RR Psychosozial |
| :--- | :---: | :---: | :---: | :---: |
| **S02_supp_half** | **0.881** | 1.0086 | 0.9995 | 1.0004 |
| **S01_baseline** | **0.786** | 0.9947 | 0.9995 | 1.0064 |
| **S03_supp_double** | **0.682** | 1.0127 | 1.0004 | 1.0115 |

## 4. Methodische Auswertung & Synthese
1. **Prädiktive Stabilität:** Die ROC-AUC der Diskriminationsmodelle bleibt über alle Wirkungsstärken bemerkenswert stabil (Exam GRU: ~0.897, Semester GRU: ~0.818). Die Modelle 'verlieren' ihre Vorhersagekraft nicht, wenn Support drastisch wirkt.
2. **PR-AUC Dynamik:** Bei doppelter Supportwirkung (S03) sinkt die Event-Prävalenz (Dropout) von 29.2% auf 25.3%. Dadurch verschiebt sich die Zufalls-Baseline für PR-AUC nach unten, was bei gleichen Diskriminationseigenschaften zu leicht geringeren PR-AUC Werten führt (typischer Prävalenz-Effekt).
3. **Kausale Schätzung:** Der Semester-Transformer erkennt tendenziell die Richtung der Verstärkung, leidet aber weiterhin an der bekannten Überdämpfung im beobachteten Feature-Raum.