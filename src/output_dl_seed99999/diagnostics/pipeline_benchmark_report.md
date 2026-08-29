# Pipeline Benchmark & Execution Report (V3.6)

**Generiert am:** 2026-08-27 17:01:28  
**Gesamtlaufzeit:** 234.39 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Extended Cox Proportional Hazards (Statsmodels PHReg) | PASSED | 9.67 | 211.5 | 567.5 | +356.1 |
| 2. Extended DeepSurv & Logistic Hazard (Panel Breslow) | PASSED | 258.2 | 526.6 | 780.7 | +254.1 |
| 3. Recurrent Semester Survival GRU | PASSED | 178.34 | 780.7 | 920.9 | +140.3 |
| 4. Dynamic DeepHit Competing Risks (Dropout & Abschluss) | PASSED | 168.66 | 920.9 | 1082.0 | +161.1 |
| 5. Causal Semester Transformer Survival | PASSED | 95.91 | 1082.0 | 1192.6 | +110.6 |
| 6. Recurrent Exam Survival GRU | PASSED | 514.07 | 1192.6 | 1351.6 | +159.0 |
| 7. Causal Exam Transformer Survival | PASSED | 875.69 | 1351.6 | 1551.7 | +200.2 |
| 8. Landmark Baseline Classifiers (RF, SVM, NaiveBayes, MLP) | PASSED | 240.11 | 1553.2 | 1659.5 | +106.3 |
| 9. Landmark Abschlussnoten-Regression (Ridge, SVR, RF, MLP) | PASSED | 62.51 | 1659.5 | 1766.3 | +106.8 |
| 10. Double Machine Learning (DML Orthogonalized Survival) | PASSED | 63.53 | 1766.3 | 1753.5 | -12.8 |
| 11. Deep Transformer-DML Pipeline | PASSED | 515.01 | 1753.5 | 1755.6 | +2.2 |
| 12. Semester Timeseries LSTM GPA Regression | PASSED | 139.53 | 1755.6 | 1848.3 | +92.7 |
| 13. Semester Timeseries Transformer Abschlussnoten-Regression | PASSED | 236.79 | 1848.3 | 1933.5 | +85.2 |
| 14. Exam Timeseries GRU Grade Regression | PASSED | 236.81 | 1933.5 | 2027.1 | +93.6 |
| 15. Exam Timeseries Transformer Grade Regression | PASSED | 520.06 | 2027.1 | 2140.5 | +113.4 |
| 16. Oracle Models (Theoretischer Maximum Lift) | PASSED | 84.24 | 2140.5 | 2088.3 | -52.3 |
| 17. DSGVO Realistic Models (Feature Blindness Analysis) | PASSED | 44.72 | 2088.3 | 2205.0 | +116.7 |
| 18. Deep Transformer Suite (Enlarged Capacity) | PASSED | 6001.74 | 2205.0 | 2747.6 | +542.6 |
| 19. Autoregressive Next-Exam Prediction (Dual-Head Multi-Task) | PASSED | 3810.14 | 2747.6 | 2625.5 | -122.1 |
| 20. Strukturelle Mediationsanalyse (Imai / Pearl Framework) | PASSED | 7.4 | 2625.5 | 2707.3 | +81.9 |
