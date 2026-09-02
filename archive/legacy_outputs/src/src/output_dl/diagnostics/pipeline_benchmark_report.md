# Pipeline Benchmark & Execution Report (V3.6)

**Generiert am:** 2026-08-24 04:40:27  
**Gesamtlaufzeit:** 80.28 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0. Simulation V3 (5 Universen, Salted Seeds & Clipping Tracker) | PASSED | 1112.38 | 184.3 | 834.8 | +650.4 |
| 1. Extended Cox Proportional Hazards (Statsmodels PHReg) | PASSED | 10.22 | 809.1 | 990.6 | +181.5 |
| 2. Extended DeepSurv & Logistic Hazard (Panel Breslow) | PASSED | 247.48 | 782.0 | 888.9 | +106.9 |
| 3. Recurrent Semester Survival GRU | PASSED | 132.06 | 888.9 | 1076.5 | +187.6 |
| 4. Dynamic DeepHit Competing Risks (Dropout & Abschluss) | PASSED | 201.35 | 1076.5 | 1137.8 | +61.2 |
| 5. Causal Semester Transformer Survival | PASSED | 107.31 | 1137.8 | 1237.9 | +100.1 |
| 6. Recurrent Exam Survival GRU | PASSED | 370.93 | 1237.9 | 1395.3 | +157.4 |
| 7. Causal Exam Transformer Survival | PASSED | 909.92 | 1395.3 | 1580.1 | +184.8 |
| 8. Landmark Baseline Classifiers (RF, SVM, NaiveBayes, MLP) | PASSED | 206.42 | 1581.5 | 1658.1 | +76.7 |
| 9. Landmark Abschlussnoten-Regression (Ridge, SVR, RF, MLP) | PASSED | 90.39 | 1658.2 | 1603.0 | -55.2 |
| 10. Double Machine Learning (DML Orthogonalized Survival) | PASSED | 64.81 | 1603.3 | 1701.3 | +98.0 |
| 11. Deep Transformer-DML Pipeline | PASSED | 603.09 | 1701.3 | 1805.0 | +103.7 |
| 12. Semester Timeseries LSTM GPA Regression | PASSED | 269.28 | 1805.1 | 1777.5 | -27.5 |
| 13. Semester Timeseries Transformer Abschlussnoten-Regression | FAILED | 4.65 | 1777.8 | 1941.8 | +164.0 |
| 14. Exam Timeseries GRU Grade Regression | PASSED | 353.59 | 1883.9 | 1938.7 | +54.8 |
| 15. Exam Timeseries Transformer Grade Regression | FAILED | 4.87 | 1938.9 | 2155.3 | +216.4 |
| 16. Oracle Models (Theoretischer Maximum Lift) | PASSED | 77.67 | 1964.6 | 1954.3 | -10.2 |
| 17. DSGVO Realistic Models (Feature Blindness Analysis) | PASSED | 39.71 | 1954.3 | 1985.8 | +31.4 |
| 18. Deep Transformer Suite (Enlarged Capacity) | FAILED | 4.39 | 1985.8 | 2119.6 | +133.8 |
| 19. Autoregressive Next-Exam Prediction (Dual-Head Multi-Task) | FAILED | 0.0 | 2063.7 | 2063.7 | +0.0 |
| 20. Strukturelle Mediationsanalyse (Imai / Pearl Framework) | PASSED | 6.16 | 2063.7 | 2091.7 | +28.0 |
