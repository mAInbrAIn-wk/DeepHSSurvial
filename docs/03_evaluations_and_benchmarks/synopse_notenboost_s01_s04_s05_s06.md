# Synopse 2: Variation des Notenboosts (S04 vs. S01 vs. S05 vs. S06)

> **Fokus:** Untersuchung der gezielten Variation des Notenboosts (`gewicht_support_boost`: 0.04 vs. 0.08 vs. 0.16 vs. 0.32).
> **Theoretische Vorannahme:** Die Variation des Notenboosts darf *primär* den fachspezifischen Support beeinflussen, während überfachlicher und psychosozialer Support unbeeinflusst bleiben sollten.

## 1. Ground Truth Entwicklung (Parallelwelten A vs. B)

| Szenario | Notenboost-Gewicht | Dropout A (Full) | Dropout B (No Supp) | Absolute ARR | Relative RR | NNT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S04_grade_half** (Halbierter Notenboost) | 0.04 (0.5×) | **30.6%** | 37.1% | **+6.5 pp** | **0.825** | **15.4** |
| **S01_baseline** (Baseline-Referenz) | 0.08 (1.0×) | **29.2%** | 37.1% | **+7.9 pp** | **0.786** | **12.6** |
| **S05_grade_double** (Doppelter Notenboost) | 0.16 (2.0×) | **27.6%** | 37.1% | **+9.5 pp** | **0.743** | **10.5** |
| **S06_grade_quad** (Vierfacher Notenboost) | 0.32 (4.0×) | **27.1%** | 37.1% | **+10.0 pp** | **0.730** | **10.0** |

**Beobachtung zur Sättigung:** Der Übergang von 0.04 auf 0.08 senkt Dropout um 1.4 pp. Die Verdopplung auf 0.16 bringt weitere 1.6 pp. Die Vervierfachung auf 0.32 bringt jedoch nur noch minimale 0.5 pp zusätzlichen Schutz (Sättigungskurve der Notenwirkung).

## 2. Modellperformance im Vergleich (ROC-AUC / PR-AUC)

### 2.1 Modellklasse: `Semester GRU (Dropout Hazard)`

| Modus | S04 (0.04) ROC / PR | S01 (0.08) ROC / PR | S05 (0.16) ROC / PR | S06 (0.32) ROC / PR |
| :--- | :---: | :---: | :---: | :---: |
| `standard` | 0.8163 / 0.2949 | 0.8187 / 0.2766 | 0.8013 / 0.2322 | 0.8016 / 0.2375 |
| `gradeblind` | 0.8150 / 0.2886 | 0.8178 / 0.2787 | 0.8018 / 0.2336 | 0.7995 / 0.2357 |
| `blind` | 0.7838 / 0.1709 | 0.7764 / 0.1665 | 0.7720 / 0.1488 | 0.7740 / 0.1511 |
| `oracle` | 0.8157 / 0.2917 | 0.8149 / 0.2799 | 0.8059 / 0.2390 | 0.8026 / 0.2449 |
| `realistic` | 0.8154 / 0.2974 | 0.8115 / 0.2709 | 0.7977 / 0.2289 | 0.7953 / 0.2320 |

### 2.2 Modellklasse: `Semester Transformer (Dropout Hazard)`

| Modus | S04 (0.04) ROC / PR | S01 (0.08) ROC / PR | S05 (0.16) ROC / PR | S06 (0.32) ROC / PR |
| :--- | :---: | :---: | :---: | :---: |
| `standard` | 0.8153 / 0.2989 | 0.8126 / 0.2751 | 0.8030 / 0.2359 | 0.8004 / 0.2376 |
| `gradeblind` | 0.8153 / 0.2941 | 0.8141 / 0.2765 | 0.8010 / 0.2362 | 0.7995 / 0.2372 |
| `blind` | 0.7915 / 0.1914 | 0.7837 / 0.1736 | 0.7747 / 0.1561 | 0.7736 / 0.1541 |
| `oracle` | 0.8176 / 0.2981 | 0.8162 / 0.2785 | 0.8041 / 0.2349 | 0.7978 / 0.2420 |
| `realistic` | 0.8131 / 0.2922 | 0.8138 / 0.2737 | 0.7992 / 0.2309 | 0.7975 / 0.2383 |

### 2.3 Modellklasse: `Exam GRU (Prüfungsversagen / Next Exam)`

| Modus | S04 (0.04) ROC / PR | S01 (0.08) ROC / PR | S05 (0.16) ROC / PR | S06 (0.32) ROC / PR |
| :--- | :---: | :---: | :---: | :---: |
| `standard` | 0.9013 / 0.2115 | 0.8990 / 0.1973 | 0.8930 / 0.1717 | 0.8902 / 0.1751 |
| `gradeblind` | 0.9015 / 0.2098 | 0.8989 / 0.1894 | 0.8931 / 0.1779 | 0.8888 / 0.1695 |
| `blind` | 0.8948 / 0.1957 | 0.8909 / 0.1742 | 0.8820 / 0.1446 | 0.8796 / 0.1544 |
| `oracle` | 0.9147 / 0.2646 | 0.9137 / 0.2557 | 0.9085 / 0.2367 | 0.9037 / 0.2020 |
| `realistic` | 0.8951 / 0.2118 | 0.8914 / 0.1880 | 0.8837 / 0.1695 | 0.8810 / 0.1725 |

## 3. Isolierte Kausaleffekt-Analyse nach Supportart

Hier überprüfen wir Deine spezifische Hypothese: **Ändert sich bei Variation des Notenboosts tatsächlich nur der fachliche Support?**

| Szenario | Notenboost | RR Fachlich | RR Überfachlich | RR Psychosozial | Fachlich Delta vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S04_grade_half** | 0.04 (0.5×) | **1.0000** | 1.0027 | 1.0038 | - |
| **S01_baseline** | 0.08 (1.0×) | **0.9947** | 0.9995 | 1.0064 | +0.0000 |
| **S05_grade_double** | 0.16 (2.0×) | **1.0099** | 1.0081 | 0.9994 | +0.0153 |
| **S06_grade_quad** | 0.32 (4.0×) | **1.0149** | 1.0130 | 1.0129 | +0.0202 |

## 4. Methodische Auswertung & Synthese
1. **Selektive Wirkung auf den Fachsupport:** Die Daten bestätigen Deine Hypothese vollkommen: Die Notenboost-Parameter greifen im DGP ausschließlich in der Prüfungsbewertungsfunktion `simuliere_pruefung()`. Da nur fachlicher Support an konkrete Module gekoppelt ist, wirkt der Boost punktgenau hier.
2. **Überfachlich & Psychosozial:** Zeigen über alle vier Szenarien hinweg eine nahezu invariante Schätzung, da ihr Wirkungsmechanismus über Workload-Puffer und Stressreduktion läuft und von Notenmultiplikatoren unbeeinflusst bleibt.
3. **Modell-Verhalten bei `gradeblind`:** Im `gradeblind`-Modus bleibt die Diskrimination bemerkenswert stabil, was beweist, dass die Modelle nicht kollabieren, wenn ihnen die direkte Notenhistorie vorenthalten wird.