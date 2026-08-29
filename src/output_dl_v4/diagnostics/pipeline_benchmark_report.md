# Pipeline Benchmark & Execution Report (V3.6)

**Generiert am:** 2026-08-28 22:04:43  
**Gesamtlaufzeit:** 195.36 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Extended Cox Proportional Hazards (Statsmodels PHReg) | PASSED | 11.31 | 202.9 | 452.9 | +250.0 |
| 2. Extended DeepSurv & Logistic Hazard (Panel Breslow) | PASSED | 223.8 | 392.0 | 655.9 | +263.8 |
| 3. Recurrent Semester Survival GRU | PASSED | 164.54 | 655.9 | 793.6 | +137.7 |
| 4. Dynamic DeepHit Competing Risks (Dropout & Abschluss) | PASSED | 193.3 | 793.6 | 955.9 | +162.3 |
| 5. Causal Semester Transformer Survival | PASSED | 58.69 | 955.9 | 1090.4 | +134.5 |
| 6. Recurrent Exam Survival GRU | PASSED | 369.22 | 1090.4 | 1189.8 | +99.4 |
| 7. Causal Exam Transformer Survival | PASSED | 782.47 | 1189.8 | 1429.5 | +239.6 |
| 8. Landmark Baseline Classifiers (RF, SVM, NaiveBayes, MLP) | PASSED | 182.13 | 1430.6 | 1479.4 | +48.9 |
| 9. Landmark Abschlussnoten-Regression (Ridge, SVR, RF, MLP) | PASSED | 70.37 | 1479.4 | 1613.5 | +134.0 |
| 10. Double Machine Learning (DML Orthogonalized Survival) | PASSED | 52.96 | 1613.5 | 1535.3 | -78.2 |
| 11. Deep Transformer-DML Pipeline | PASSED | 452.2 | 1535.3 | 1691.4 | +156.1 |
| 12. Semester Timeseries LSTM GPA Regression | PASSED | 124.99 | 1691.4 | 1717.1 | +25.7 |
| 13. Semester Timeseries Transformer Abschlussnoten-Regression | PASSED | 222.18 | 1717.1 | 1785.2 | +68.1 |
| 14. Exam Timeseries GRU Grade Regression | PASSED | 340.68 | 1785.2 | 1904.8 | +119.6 |
| 15. Exam Timeseries Transformer Grade Regression | PASSED | 530.84 | 1904.8 | 2002.0 | +97.2 |
| 16. Oracle Models (Theoretischer Maximum Lift) | PASSED | 77.41 | 2002.0 | 2001.7 | -0.3 |
| 17. DSGVO Realistic Models (Feature Blindness Analysis) | PASSED | 41.79 | 2001.7 | 2028.3 | +26.6 |
| 18. Deep Transformer Suite (Enlarged Capacity) | PASSED | 5866.87 | 2028.3 | 2599.3 | +571.0 |
| 19. Autoregressive Next-Exam Prediction (Dual-Head Multi-Task) | PASSED | 1948.03 | 2599.3 | 2671.6 | +72.2 |
| 20. Strukturelle Mediationsanalyse (Imai / Pearl Framework) | PASSED | 7.72 | 2671.6 | 2616.2 | -55.3 |
