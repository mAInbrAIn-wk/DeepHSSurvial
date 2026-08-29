# Pipeline Benchmark & Execution Report (V3.6)

**Generiert am:** 2026-08-24 12:39:17  
**Gesamtlaufzeit:** 295.26 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Extended Cox Proportional Hazards (Statsmodels PHReg) | PASSED | 10.3 | 208.4 | 557.4 | +349.0 |
| 2. Extended DeepSurv & Logistic Hazard (Panel Breslow) | PASSED | 269.52 | 515.5 | 776.8 | +261.3 |
| 3. Recurrent Semester Survival GRU | PASSED | 78.19 | 776.8 | 916.0 | +139.2 |
| 4. Dynamic DeepHit Competing Risks (Dropout & Abschluss) | PASSED | 210.89 | 916.0 | 1071.5 | +155.5 |
| 5. Causal Semester Transformer Survival | PASSED | 96.71 | 1071.5 | 1171.3 | +99.8 |
| 6. Recurrent Exam Survival GRU | PASSED | 411.77 | 1171.3 | 1357.9 | +186.6 |
| 7. Causal Exam Transformer Survival | PASSED | 948.14 | 1357.9 | 1556.7 | +198.8 |
| 8. Landmark Baseline Classifiers (RF, SVM, NaiveBayes, MLP) | PASSED | 210.76 | 1558.2 | 1599.7 | +41.5 |
| 9. Landmark Abschlussnoten-Regression (Ridge, SVR, RF, MLP) | PASSED | 64.43 | 1599.7 | 1692.2 | +92.5 |
| 10. Double Machine Learning (DML Orthogonalized Survival) | PASSED | 63.71 | 1692.2 | 1576.2 | -116.0 |
| 11. Deep Transformer-DML Pipeline | PASSED | 677.1 | 1576.2 | 1737.5 | +161.4 |
| 12. Semester Timeseries LSTM GPA Regression | PASSED | 248.93 | 1737.5 | 1832.6 | +95.0 |
| 13. Semester Timeseries Transformer Abschlussnoten-Regression | PASSED | 344.79 | 1832.6 | 1918.4 | +85.9 |
| 14. Exam Timeseries GRU Grade Regression | PASSED | 601.99 | 1918.4 | 2020.8 | +102.4 |
| 15. Exam Timeseries Transformer Grade Regression | PASSED | 952.15 | 2021.0 | 2138.2 | +117.3 |
| 16. Oracle Models (Theoretischer Maximum Lift) | PASSED | 107.55 | 2138.2 | 2070.7 | -67.5 |
| 17. DSGVO Realistic Models (Feature Blindness Analysis) | PASSED | 52.38 | 2070.7 | 2187.3 | +116.6 |
| 18. Deep Transformer Suite (Enlarged Capacity) | PASSED | 10335.05 | 2187.4 | 2723.6 | +536.2 |
| 19. Autoregressive Next-Exam Prediction (Dual-Head Multi-Task) | PASSED | 2025.37 | 2723.6 | 2607.0 | -116.6 |
| 20. Strukturelle Mediationsanalyse (Imai / Pearl Framework) | PASSED | 6.14 | 2607.0 | 2710.5 | +103.5 |
