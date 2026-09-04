# Synopse: Randomisierte Zuteilung vs. Beobachtete Selbstselektion (S01 vs. S11)

**Parameter-Dimension:** `rct_support_uptake` (False = Beobachtungsdaten mit Confounding by Indication vs. True = Randomisierte RCT-Zuweisung)  
**Datenbasis:** $N = 50.000$ Studierende je Universum (Seed 99999). Alle 15 DL-Modellvarianten trainiert und evaluiert.

---

## 1. Ground-Truth-Kausaleffekte (Die Kosten ungerichteter Zuteilung)

| Szenario | Zuweisungs-Mechanismus | Dropout Univ. A (Full) | Dropout Univ. B (None) | ARR (pp) | Relative Risk (RR) | NNT |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| **S01_baseline** | Confounded Selbstselektion (Bedarf) | 29.16% | 37.10% | **+7.95 pp** | **0.786** | **12.6** |

| **S11_rct_calibrated** | Randomisierte Zuteilung (RCT-Uptake) | 32.65% | 37.10% | **+4.45 pp** | **0.880** | **22.5** |


> [!IMPORTANT]

> **Zentrale bildungsökonomische Entdeckung:** Unter randomisierter Zuweisung (S11) sinkt die absolute Risikoreduktion (**ARR**) von **7.95 pp auf 4.45 pp** (ein Rückgang um **44.0 %**), und die **NNT verdoppelt sich von 12.6 auf 22.5**!  

> **Ursache:** In der Baseline (S01) nehmen Studierende mit Leistungsdefiziten Support gezielt in Anspruch (*Confounding by Indication*). Dort erzielt der Support die höchste Hebelwirkung. Bei einer ungerichteten Gleichverteilung (RCT) erhalten viele robuste Studierende Support, deren Dropout-Risiko ohnehin minimal ist. Dies belegt quantitativ den immensen gesellschaftlichen und ökonomischen Mehrwert von prädiktiven Frühwarnsystemen für gezielte Interventionen.


## 2. Modellperformance: Beobachtungsdaten vs. RCT-Daten

| Modell | Modus | Metrik | S01 (Selbstselektion) | S11 (RCT-Zuteilung) | Delta (RCT - Baseline) |

| :--- | :---: | :---: | :---: | :---: | :---: |

| `grid_semester_gru` | standard | ROC-AUC | 0.8187 | 0.8260 | +0.0073 |

| `grid_semester_gru` | standard | PR-AUC | 0.2766 | 0.3091 | +0.0325 |

| `grid_semester_gru` | standard | Brier-Score | 0.0351 | 0.0388 | +0.0038 |

| `grid_semester_gru` | realistic | ROC-AUC | 0.8115 | 0.8225 | +0.0110 |

| `grid_semester_gru` | realistic | PR-AUC | 0.2709 | 0.3091 | +0.0382 |

| `grid_semester_gru` | realistic | Brier-Score | 0.0350 | 0.0386 | +0.0036 |

| `grid_semester_gru` | blind | ROC-AUC | 0.7764 | 0.7896 | +0.0132 |

| `grid_semester_gru` | blind | PR-AUC | 0.1665 | 0.1894 | +0.0230 |

| `grid_semester_gru` | blind | Brier-Score | 0.0377 | 0.0422 | +0.0045 |

| `grid_semester_transformer` | standard | ROC-AUC | 0.8126 | 0.8255 | +0.0128 |

| `grid_semester_transformer` | standard | PR-AUC | 0.2751 | 0.3050 | +0.0299 |

| `grid_semester_transformer` | standard | Brier-Score | 0.0347 | 0.0386 | +0.0039 |

| `grid_semester_transformer` | realistic | ROC-AUC | 0.8138 | 0.8219 | +0.0081 |

| `grid_semester_transformer` | realistic | PR-AUC | 0.2737 | 0.3059 | +0.0321 |

| `grid_semester_transformer` | realistic | Brier-Score | 0.0348 | 0.0385 | +0.0038 |

| `grid_semester_transformer` | blind | ROC-AUC | 0.7837 | 0.7963 | +0.0126 |

| `grid_semester_transformer` | blind | PR-AUC | 0.1736 | 0.2016 | +0.0280 |

| `grid_semester_transformer` | blind | Brier-Score | 0.0372 | 0.0415 | +0.0043 |

| `grid_exam_gru` | standard | ROC-AUC | 0.8990 | 0.9044 | +0.0054 |

| `grid_exam_gru` | standard | PR-AUC | 0.1973 | 0.2270 | +0.0297 |

| `grid_exam_gru` | standard | Brier-Score | 0.0151 | 0.0165 | +0.0014 |

| `grid_exam_gru` | realistic | ROC-AUC | 0.8914 | 0.8948 | +0.0034 |

| `grid_exam_gru` | realistic | PR-AUC | 0.1880 | 0.2254 | +0.0374 |

| `grid_exam_gru` | realistic | Brier-Score | 0.0152 | 0.0166 | +0.0014 |

| `grid_exam_gru` | blind | ROC-AUC | 0.8909 | 0.8989 | +0.0079 |

| `grid_exam_gru` | blind | PR-AUC | 0.1742 | 0.2185 | +0.0443 |

| `grid_exam_gru` | blind | Brier-Score | 0.0154 | 0.0167 | +0.0013 |
