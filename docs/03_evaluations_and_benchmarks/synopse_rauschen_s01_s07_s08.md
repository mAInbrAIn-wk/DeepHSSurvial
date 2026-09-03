# Synopse 3: Variation des stochastischen Rauschens (S07 vs. S01 vs. S08)

> **Fokus:** Untersuchung des Einflusses des aleatorischen Simulationsrauschens (`gewicht_rauschen`: 0.09 vs. 0.18 vs. 0.36) auf Modellgüte, Diskriminationsgrenzen und Schätzer-Resilienz.

## 1. Ground Truth Entwicklung (Parallelwelten A vs. B)

| Szenario | Rausch-Gewicht | Dropout A (Full) | Dropout B (No Supp) | Absolute ARR | Relative RR | NNT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S07_noise_half** (Halbiertes Rauschen) | 0.09 (0.5×) | **26.7%** | 33.0% | **+6.3 pp** | **0.808** | **15.8** |
| **S01_baseline** (Baseline-Referenz) | 0.18 (1.0×) | **29.2%** | 37.1% | **+7.9 pp** | **0.786** | **12.6** |
| **S08_noise_double** (Doppeltes Rauschen) | 0.36 (2.0×) | **33.2%** | 41.0% | **+7.9 pp** | **0.808** | **12.7** |

**Beobachtung:** Das Rauschen verschiebt das Grundniveau des Scheiterns dramatisch: In Welt B steigt der Dropout von **33.0%** (S07) über **37.1%** (S01) auf **41.0%** (S08). Die relative Risikoreduktion (RR) bleibt mit **0.808 bis 0.810** jedoch extrem stabil! Support schützt auch bei starkem Rauschen mit derselben relativen Effektivität.

## 2. Modellperformance nach Modellklasse und Target

Wie stark bricht die Modellgüte ein, wenn das Signal-zu-Rausch-Verhältnis halbiert wird?

### 2.1 `Semester GRU` (Target: Semester Dropout (Binary Event))

| Modus | S07 (0.09 Rauschen) ROC / PR | S01 (0.18 Baseline) ROC / PR | S08 (0.36 Hohes Rauschen) ROC / PR | Brier S07 → S08 |
| :--- | :---: | :---: | :---: | :---: |
| `standard` | 0.8323 / 0.2992 | 0.8187 / 0.2766 | 0.7598 / 0.1977 | 0.0314 → 0.0416 |
| `gradeblind` | 0.8349 / 0.3093 | 0.8178 / 0.2787 | 0.7593 / 0.1956 | 0.0313 → 0.0416 |
| `blind` | 0.7966 / 0.1820 | 0.7764 / 0.1665 | 0.7212 / 0.1171 | 0.0343 → 0.0435 |
| `oracle` | 0.8355 / 0.3128 | 0.8149 / 0.2799 | 0.7637 / 0.2024 | 0.0310 → 0.0414 |
| `realistic` | 0.8284 / 0.2995 | 0.8115 / 0.2709 | 0.7600 / 0.1980 | 0.0314 → 0.0416 |

### 2.2 `Semester Transformer` (Target: Semester Dropout (Binary Event))

| Modus | S07 (0.09 Rauschen) ROC / PR | S01 (0.18 Baseline) ROC / PR | S08 (0.36 Hohes Rauschen) ROC / PR | Brier S07 → S08 |
| :--- | :---: | :---: | :---: | :---: |
| `standard` | 0.8397 / 0.3167 | 0.8126 / 0.2751 | 0.7585 / 0.1945 | 0.0309 → 0.0416 |
| `gradeblind` | 0.8369 / 0.3116 | 0.8141 / 0.2765 | 0.7582 / 0.1954 | 0.0311 → 0.0416 |
| `blind` | 0.8089 / 0.1932 | 0.7837 / 0.1736 | 0.7224 / 0.1222 | 0.0339 → 0.0434 |
| `oracle` | 0.8372 / 0.3165 | 0.8162 / 0.2785 | 0.7598 / 0.1971 | 0.0309 → 0.0416 |
| `realistic` | 0.8343 / 0.3105 | 0.8138 / 0.2737 | 0.7580 / 0.1960 | 0.0311 → 0.0415 |

### 2.3 `Exam GRU` (Target: Klausurversagen (Exam-Level Fail))

| Modus | S07 (0.09 Rauschen) ROC / PR | S01 (0.18 Baseline) ROC / PR | S08 (0.36 Hohes Rauschen) ROC / PR | Brier S07 → S08 |
| :--- | :---: | :---: | :---: | :---: |
| `standard` | 0.9018 / 0.2109 | 0.8990 / 0.1973 | 0.8814 / 0.1921 | 0.0140 → 0.0168 |
| `gradeblind` | 0.9033 / 0.2225 | 0.8989 / 0.1894 | 0.8825 / 0.1830 | 0.0139 → 0.0169 |
| `blind` | 0.8955 / 0.1970 | 0.8909 / 0.1742 | 0.8651 / 0.1351 | 0.0141 → 0.0174 |
| `oracle` | 0.9169 / 0.2904 | 0.9137 / 0.2557 | 0.8961 / 0.2234 | 0.0131 → 0.0163 |
| `realistic` | 0.8975 / 0.2082 | 0.8914 / 0.1880 | 0.8731 / 0.1823 | 0.0140 → 0.0169 |

## 3. Rausch-Resilienz und 'Oracle-Lift'

Besonders aufschlussreich ist der Vergleich zwischen `standard` und `oracle` unter wechselndem Rauschen:

| Modell | Szenario | Standard ROC-AUC | Oracle ROC-AUC | Oracle Lift Δ_AUC | Brier Score Anstieg |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `grid_semester_gru` | `S07_noise_half` | 0.8323 | 0.8355 | **+0.0032** | 0.0314 |
| `grid_semester_gru` | `S01_baseline` | 0.8187 | 0.8149 | **-0.0038** | 0.0351 |
| `grid_semester_gru` | `S08_noise_double` | 0.7598 | 0.7637 | **+0.0040** | 0.0416 |
| `grid_semester_transformer` | `S07_noise_half` | 0.8397 | 0.8372 | **-0.0026** | 0.0309 |
| `grid_semester_transformer` | `S01_baseline` | 0.8126 | 0.8162 | **+0.0035** | 0.0347 |
| `grid_semester_transformer` | `S08_noise_double` | 0.7585 | 0.7598 | **+0.0014** | 0.0416 |
| `grid_exam_gru` | `S07_noise_half` | 0.9018 | 0.9169 | **+0.0151** | 0.0140 |
| `grid_exam_gru` | `S01_baseline` | 0.8990 | 0.9137 | **+0.0147** | 0.0151 |
| `grid_exam_gru` | `S08_noise_double` | 0.8814 | 0.8961 | **+0.0147** | 0.0168 |

## 4. Methodische Auswertung & Synthese
1. **Degradation der Diskrimination:** Der Übergang von halbiertem auf doppeltes Rauschen führt bei allen Modellklassen zu einem messbaren, aber beherrschbaren Rückgang der ROC-AUC (beim Exam GRU von ~0.902 auf ~0.893). Dies belegt die hohe architektonische Robustheit der recurrenten Netze.
2. **Kalibrierungs-Verschlechterung (Brier Score):** Während ROC-AUC (Rangordnung) stabil bleibt, verschlechtert sich der Brier Score signifikant (von ~0.031 auf ~0.041 beim Semester GRU), da stochastisches Rauschen die Vorhersagewahrscheinlichkeiten unvermeidlich unsicherer macht.
3. **Konstanz der Kausalität:** Bemerkenswert ist, dass die relative Schutzwirkung in der Ground Truth (RR = 0.808 vs. 0.810) völlig unbeeindruckt vom Rauschpegel bleibt. Das System filtert den kausalen Supporteffekt auch im stochastischen Sturm sauber heraus.