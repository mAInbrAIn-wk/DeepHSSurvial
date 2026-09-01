# Fast Core Suite Benchmark Report

**Generiert am:** 2026-09-01 20:56:19  
**Gesamtlaufzeit:** 239.79 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Extended Cox [standard] | PASSED | 13.02 | 19.4 | 455.5 | +436.2 |
| Extended DeepSurv [standard] | PASSED | 276.57 | 223.1 | 656.2 | +433.1 |
| Recurrent Survival GRU [standard] | PASSED | 121.66 | 656.2 | 800.6 | +144.4 |
| Dynamic DeepHit Competing Risks [standard] | PASSED | 89.59 | 800.6 | 960.9 | +160.3 |
| Transformer Survival [standard] | PASSED | 128.95 | 960.9 | 1091.7 | +130.8 |
| Recurrent Exam Survival GRU [standard] | PASSED | 454.92 | 1091.7 | 1193.7 | +102.1 |
| Transformer Exam Survival [standard] | PASSED | 934.87 | 1193.7 | 1429.5 | +235.8 |
| Landmark Baseline Classifiers [standard] | PASSED | 177.96 | 1429.5 | 1479.6 | +50.1 |
| Landmark Regression [standard] | PASSED | 92.99 | 1479.6 | 1619.7 | +140.1 |
| DML Orthogonal Survival [standard] | PASSED | 57.09 | 1619.7 | 1541.5 | -78.2 |
| Transformer DML [standard] | PASSED | 500.89 | 1541.5 | 1687.5 | +146.0 |
| Timeseries Semester LSTM [standard] | PASSED | 255.73 | 1687.5 | 1717.6 | +30.0 |
| Timeseries Semester Transformer [standard] | PASSED | 261.69 | 1717.6 | 1808.8 | +91.2 |
| Timeseries Exam GRU [standard] | PASSED | 379.37 | 1808.8 | 1907.5 | +98.7 |
| Timeseries Exam Transformer [standard] | PASSED | 535.35 | 1907.5 | 2009.9 | +102.4 |
| Extended Cox [gradeblind] | PASSED | 11.13 | 2009.9 | 2258.9 | +249.0 |
| Extended DeepSurv [gradeblind] | PASSED | 287.73 | 2032.7 | 2098.9 | +66.2 |
| Recurrent Survival GRU [gradeblind] | PASSED | 217.61 | 2098.9 | 2188.5 | +89.6 |
| Dynamic DeepHit Competing Risks [gradeblind] | PASSED | 116.93 | 2188.5 | 2378.6 | +190.2 |
| Transformer Survival [gradeblind] | PASSED | 128.82 | 2378.6 | 2495.1 | +116.5 |
| Recurrent Exam Survival GRU [gradeblind] | PASSED | 462.05 | 2495.1 | 2548.6 | +53.5 |
| Transformer Exam Survival [gradeblind] | PASSED | 918.66 | 2548.6 | 2832.3 | +283.6 |
| Landmark Baseline Classifiers [gradeblind] | PASSED | 184.57 | 2832.3 | 2889.2 | +56.9 |
| Landmark Regression [gradeblind] | PASSED | 102.63 | 2889.2 | 2966.7 | +77.5 |
| DML Orthogonal Survival [gradeblind] | PASSED | 62.03 | 2966.7 | 2945.0 | -21.7 |
| Transformer DML [gradeblind] | PASSED | 640.6 | 2945.0 | 3005.5 | +60.5 |
| Timeseries Semester LSTM [gradeblind] | PASSED | 588.58 | 3005.5 | 3065.8 | +60.3 |
| Timeseries Semester Transformer [gradeblind] | PASSED | 536.52 | 3065.8 | 3162.1 | +96.3 |
| Timeseries Exam GRU [gradeblind] | PASSED | 877.42 | 3162.1 | 3279.4 | +117.3 |
| Timeseries Exam Transformer [gradeblind] | PASSED | 1001.76 | 3279.4 | 3381.5 | +102.1 |
| Oracle Models (Lift Analysis) | PASSED | 98.03 | 3381.5 | 3347.9 | -33.6 |
| DSGVO Realistic Models | PASSED | 47.94 | 3347.9 | 3322.8 | -25.1 |
| Strukturelle Mediationsanalyse | PASSED | 6.63 | 3322.8 | 3357.9 | +35.1 |
| Deep Survival Landmark (LH & DS) | PASSED | 43.6 | 3357.9 | 3459.0 | +101.0 |
| Plot Calibration Curves | FAILED | 0.0 | 3459.0 | 3459.0 | +0.1 |
| Feature Grid Experiments (Cross-Mode) | PASSED | 3720.49 | 3459.1 | 4003.2 | +544.2 |
| Counterfactual HR Analyzer (Extended Cox/Panel) | PASSED | 39.07 | 4003.2 | 4002.6 | -0.6 |
| Counterfactual DeepHit Competing Risks | PASSED | 13.65 | 4002.6 | 4007.7 | +5.1 |
| Counterfactual Grade Transformer | FAILED | 0.02 | 4007.7 | 4007.7 | +0.0 |
| Counterfactual Oracle Logistic Hazard | FAILED | 0.0 | 4007.7 | 4007.7 | +0.0 |
