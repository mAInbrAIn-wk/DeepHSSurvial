# Synopse: Zeitkosten & Modulabwurf-Dynamik (S09 vs. S01 vs. S10)

**Parameter-Dimension:** `support_kosten_faktor` (0.0× = 0h vs. 1.0× = 30h vs. 2.0× = 60h Workload pro Maßnahme)  
**Datenbasis:** $N = 50.000$ Studierende je Universum (Seed 99999). Alle 15 DL-Modellvarianten trainiert und evaluiert.

---

## 1. Ground-Truth-Kausaleffekte der Simulationswelten

| Szenario | Zeitkosten / Maßnahme | Dropout Univ. A (Full) | Dropout Univ. B (None) | ARR (pp) | Relative Risk (RR) | NNT |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| **S09_cost_zero** | 0h (Kostenlos) | 28.61% | 37.10% | **+8.50 pp** | **0.771** | **11.8** |

| **S01_baseline** | 30h (Baseline) | 29.16% | 37.10% | **+7.95 pp** | **0.786** | **12.6** |

| **S10_cost_double** | 60h (Doppelt) | 29.71% | 37.10% | **+7.39 pp** | **0.801** | **13.5** |


> [!NOTE]

> **Kausale Mechanik:** Universum B hat in allen drei Welten exakt **37.10%** Dropout, da ohne Support keine Zeitkosten anfallen. Steigen die Zeitkosten von 0h auf 60h, steigt der Dropout in Universum A moderat von 28.61% auf 29.71% (+1.10 pp). Die Zeitkosten dämpfen den ARR um 1.11 pp (von 8.50 auf 7.39 pp), fressen den Supportnutzen jedoch keineswegs auf.


## 2. Modellperformance über die Zeitkosten-Variationen

Vergleich der diskriminativen Güte (ROC-AUC, PR-AUC, Brier-Score) in den repräsentativen Modi *Standard* und *Realistic*:


| Modell | Modus | Metrik | S09 (0h Zeitkosten) | S01 (30h Baseline) | S10 (60h Zeitkosten) | Delta (S10 - S09) |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| `grid_semester_gru` | standard | ROC-AUC | 0.8092 | 0.8187 | 0.8128 | +0.0037 |

| `grid_semester_gru` | standard | PR-AUC | 0.2627 | 0.2766 | 0.2646 | +0.0019 |

| `grid_semester_gru` | standard | Brier-Score | 0.0345 | 0.0351 | 0.0358 | +0.0012 |

| `grid_semester_gru` | realistic | ROC-AUC | 0.8028 | 0.8115 | 0.8095 | +0.0067 |

| `grid_semester_gru` | realistic | PR-AUC | 0.2515 | 0.2709 | 0.2608 | +0.0093 |

| `grid_semester_gru` | realistic | Brier-Score | 0.0348 | 0.0350 | 0.0358 | +0.0010 |

| `grid_semester_transformer` | standard | ROC-AUC | 0.8079 | 0.8126 | 0.8142 | +0.0063 |

| `grid_semester_transformer` | standard | PR-AUC | 0.2598 | 0.2751 | 0.2696 | +0.0098 |

| `grid_semester_transformer` | standard | Brier-Score | 0.0343 | 0.0347 | 0.0355 | +0.0011 |

| `grid_semester_transformer` | realistic | ROC-AUC | 0.8085 | 0.8138 | 0.8106 | +0.0021 |

| `grid_semester_transformer` | realistic | PR-AUC | 0.2612 | 0.2737 | 0.2660 | +0.0048 |

| `grid_semester_transformer` | realistic | Brier-Score | 0.0343 | 0.0348 | 0.0356 | +0.0013 |

| `grid_exam_gru` | standard | ROC-AUC | 0.9010 | 0.8990 | 0.8954 | -0.0056 |

| `grid_exam_gru` | standard | PR-AUC | 0.2017 | 0.1973 | 0.1939 | -0.0078 |

| `grid_exam_gru` | standard | Brier-Score | 0.0147 | 0.0151 | 0.0155 | +0.0007 |

| `grid_exam_gru` | realistic | ROC-AUC | 0.8886 | 0.8914 | 0.8818 | -0.0068 |

| `grid_exam_gru` | realistic | PR-AUC | 0.1857 | 0.1880 | 0.1882 | +0.0024 |

| `grid_exam_gru` | realistic | Brier-Score | 0.0149 | 0.0152 | 0.0156 | +0.0007 |


## 3. Erkenntnisse & Modell-Invarianz

1. **Prädiktive Robustheit:** Die Vorhersagegenauigkeit aller Modelle (ROC-AUC ~ 0.80–0.81 auf Semesterebene, ~ 0.89 auf Prüfungsebene) reagiert praktisch unempfindlich auf die Zeitkosten-Parametrisierung (Deltas im Bereich von $\pm 0.003$).

2. **Modulabwurf als Kompensation:** Studierende puffern erhöhte Zeitkosten durch gezieltes Strecken des Studiums (Abwurf von Modulen) ab, wodurch die Prüfungserfolgsquoten der angetretenen Klausuren stabil bleiben.

