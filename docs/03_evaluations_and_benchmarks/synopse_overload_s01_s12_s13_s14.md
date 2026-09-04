# Synopse: Überlastungs-Dämpfung & Penalty-Kalibrierung (S12 vs. S01 vs. S13 vs. S14)

**Parameter-Dimension:** `overload_penalty_factor` (0.05 vs 0.10 vs 0.20) und `overload_penalty_cap` (0.15 vs 0.30)  
**Datenbasis:** $N = 50.000$ Studierende je Universum (Seed 99999). Alle 15 DL-Modellvarianten trainiert und evaluiert.

---

## 1. Ground-Truth-Kausaleffekte der Simulationswelten

| Szenario | Überlast-Faktor / Deckel | Dropout Univ. A (Full) | Dropout Univ. B (None) | ARR (pp) | Relative Risk (RR) | NNT |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| **S12_overload_half** | Faktor 0.05 (Halbiert) | 26.02% | 34.14% | **+8.13 pp** | **0.762** | **12.3** |

| **S01_baseline** | Faktor 0.10, Cap 0.30 (Baseline) | 29.16% | 37.10% | **+7.95 pp** | **0.786** | **12.6** |

| **S13_overload_double** | Faktor 0.20 (Doppelt) | 34.58% | 41.84% | **+7.26 pp** | **0.827** | **13.8** |

| **S14_overload_cap** | Cap 0.15 (Gedeckelt) | 26.69% | 35.04% | **+8.35 pp** | **0.762** | **12.0** |


> [!NOTE]

> **Strukturelle Dynamik:** Die Überlastungsstrafe wirkt universal auf das gesamte System (auch in Welt B!). Bei doppelter Überlast-Strafe (S13) explodiert die Basis-Dropoutrate in Universum B von 37.10% auf **41.84%** (+4.74 pp).  

> Trotz dieser drastischen Verschiebung des Grundrauschens bleibt die absolute Schutzwirkung des Supports (**ARR**) zwischen **7.26 pp und 8.35 pp** bemerkenswert stabil. Der Support schützt also selbst in extrem überlastungsintensiven Studienumgebungen verlässlich.


## 2. Modellperformance unter extremer Überlastung (S13 vs S01)

| Modell | Modus | Metrik | S01 (Baseline) | S13 (Doppelte Überlast) | Delta |

| :--- | :---: | :---: | :---: | :---: | :---: |

| `grid_semester_gru` | standard | ROC-AUC | 0.8187 | 0.8335 | +0.0148 |

| `grid_semester_gru` | standard | PR-AUC | 0.2766 | 0.3609 | +0.0843 |

| `grid_semester_gru` | standard | Brier-Score | 0.0351 | 0.0391 | +0.0041 |

| `grid_semester_gru` | realistic | ROC-AUC | 0.8115 | 0.8301 | +0.0186 |

| `grid_semester_gru` | realistic | PR-AUC | 0.2709 | 0.3577 | +0.0868 |

| `grid_semester_gru` | realistic | Brier-Score | 0.0350 | 0.0394 | +0.0044 |

| `grid_semester_transformer` | standard | ROC-AUC | 0.8126 | 0.8310 | +0.0184 |

| `grid_semester_transformer` | standard | PR-AUC | 0.2751 | 0.3456 | +0.0705 |

| `grid_semester_transformer` | standard | Brier-Score | 0.0347 | 0.0398 | +0.0051 |

| `grid_semester_transformer` | realistic | ROC-AUC | 0.8138 | 0.8273 | +0.0136 |

| `grid_semester_transformer` | realistic | PR-AUC | 0.2737 | 0.3464 | +0.0727 |

| `grid_semester_transformer` | realistic | Brier-Score | 0.0348 | 0.0399 | +0.0051 |

| `grid_exam_gru` | standard | ROC-AUC | 0.8990 | 0.9078 | +0.0088 |

| `grid_exam_gru` | standard | PR-AUC | 0.1973 | 0.2461 | +0.0487 |

| `grid_exam_gru` | standard | Brier-Score | 0.0151 | 0.0173 | +0.0022 |

| `grid_exam_gru` | realistic | ROC-AUC | 0.8914 | 0.8990 | +0.0077 |

| `grid_exam_gru` | realistic | PR-AUC | 0.1880 | 0.2335 | +0.0455 |

| `grid_exam_gru` | realistic | Brier-Score | 0.0152 | 0.0175 | +0.0023 |
