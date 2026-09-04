# Synopse: Kombinierte Nutzen- und Kostenverdopplung (S01 vs. S15)

**Parameter-Dimension:** `support_effect_multiplier` = 10.0 (2×) **UND** `support_kosten_faktor` = 2.0 (2×)  
**Datenbasis:** $N = 50.000$ Studierende je Universum (Seed 99999). Alle 15 DL-Modellvarianten trainiert und evaluiert.

---

## 1. Ground-Truth-Kausaleffekte (Netto-Nutzen-Superposition)

| Szenario | Konfiguration | Dropout Univ. A | Dropout Univ. B | ARR (pp) | Relative Risk (RR) | NNT |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| **S01_baseline** | Baseline (1× Nutzen, 1× Kosten) | 29.16% | 37.10% | **+7.95 pp** | **0.786** | **12.6** |

| **S15_cost_effect_double** | Doppelter Nutzen (10×) + Doppelte Kosten (60h) | 25.82% | 37.10% | **+11.28 pp** | **0.696** | **8.9** |


> [!TIP]

> **Superpositions-Erkenntnis:** In S15 steigen sowohl die Zeitkosten als auch der protektive Multiplikator auf das Doppelte.  

> Der ARR klettert von **7.95 pp auf 11.28 pp** (RR = 0.696, über **30 % Risikoreduktion**!). Die doppelten Zeitkosten dämpfen den Effekt um lediglich ~0.55 pp im Vergleich zur reinen Wirkungsverdopplung (S03: 11.83 pp). Hochwirksame Supportmaßnahmen lohnen sich für die Studierenden selbst dann massiv, wenn sie doppelt so zeitintensiv sind.


## 2. Modellperformance: Baseline vs. Kombinationsszenario

| Modell | Modus | Metrik | S01 (Baseline) | S15 (Kombi) | Delta |

| :--- | :---: | :---: | :---: | :---: | :---: |

| `grid_semester_gru` | standard | ROC-AUC | 0.8187 | 0.8015 | -0.0172 |

| `grid_semester_gru` | standard | PR-AUC | 0.2766 | 0.2240 | -0.0526 |

| `grid_semester_gru` | standard | Brier-Score | 0.0351 | 0.0317 | -0.0034 |

| `grid_semester_gru` | realistic | ROC-AUC | 0.8115 | 0.8007 | -0.0107 |

| `grid_semester_gru` | realistic | PR-AUC | 0.2709 | 0.2230 | -0.0479 |

| `grid_semester_gru` | realistic | Brier-Score | 0.0350 | 0.0318 | -0.0032 |

| `grid_semester_transformer` | standard | ROC-AUC | 0.8126 | 0.8019 | -0.0107 |

| `grid_semester_transformer` | standard | PR-AUC | 0.2751 | 0.2253 | -0.0498 |

| `grid_semester_transformer` | standard | Brier-Score | 0.0347 | 0.0314 | -0.0033 |

| `grid_semester_transformer` | realistic | ROC-AUC | 0.8138 | 0.8025 | -0.0113 |

| `grid_semester_transformer` | realistic | PR-AUC | 0.2737 | 0.2255 | -0.0482 |

| `grid_semester_transformer` | realistic | Brier-Score | 0.0348 | 0.0315 | -0.0033 |

| `grid_exam_gru` | standard | ROC-AUC | 0.8990 | 0.8955 | -0.0035 |

| `grid_exam_gru` | standard | PR-AUC | 0.1973 | 0.1587 | -0.0386 |

| `grid_exam_gru` | standard | Brier-Score | 0.0151 | 0.0138 | -0.0013 |

| `grid_exam_gru` | realistic | ROC-AUC | 0.8914 | 0.8844 | -0.0069 |

| `grid_exam_gru` | realistic | PR-AUC | 0.1880 | 0.1550 | -0.0330 |

| `grid_exam_gru` | realistic | Brier-Score | 0.0152 | 0.0139 | -0.0013 |
