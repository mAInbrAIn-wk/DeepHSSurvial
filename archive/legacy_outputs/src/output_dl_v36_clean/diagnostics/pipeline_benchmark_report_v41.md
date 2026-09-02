# Pipeline Benchmark & Execution Report (V4.1)

**Generiert am:** 2026-08-31 21:03:22  
**Gesamtlaufzeit:** 1535.57 Minuten

| Schritt | Status | Dauer (s) | RAM Start (MB) | RAM Ende (MB) | RAM Delta (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Extended Cox [standard] | PASSED | 14.61 | 19.6 | 468.7 | +449.1 |
| 2. Extended DeepSurv [standard] | PASSED | 340.79 | 217.5 | 657.7 | +440.2 |
| 3. Recurrent Survival GRU [standard] | PASSED | 117.29 | 657.7 | 797.1 | +139.5 |
| 4. Dynamic DeepHit Competing Risks [standard] | PASSED | 133.87 | 797.1 | 958.0 | +160.8 |
| 5. Transformer Survival [standard] | PASSED | 106.32 | 958.0 | 1088.3 | +130.3 |
| 6. Recurrent Exam Survival GRU [standard] | PASSED | 438.1 | 1088.3 | 1203.0 | +114.8 |
| 7. Transformer Exam Survival [standard] | PASSED | 872.75 | 1203.0 | 1450.0 | +247.0 |
| 8. Landmark Baseline [standard] | PASSED | 234.4 | 1450.0 | 1476.2 | +26.2 |
| 9. Landmark Regression [standard] | PASSED | 64.21 | 1476.2 | 1605.6 | +129.4 |
| 10. DML Orthogonal Survival [standard] | PASSED | 71.47 | 1605.6 | 1607.3 | +1.7 |
| 11. Transformer DML [standard] | PASSED | 622.31 | 1607.3 | 1683.8 | +76.5 |
| 12. Timeseries Semester LSTM [standard] | PASSED | 174.84 | 1683.8 | 1711.1 | +27.3 |
| 13. Timeseries Semester Transformer [standard] | PASSED | 247.19 | 1711.1 | 1816.7 | +105.6 |
| 14. Timeseries Exam GRU [standard] | PASSED | 341.73 | 1816.7 | 1908.7 | +92.1 |
| 15. Timeseries Exam Transformer [standard] | PASSED | 514.41 | 1908.7 | 2001.5 | +92.7 |
| 18. Deep Transformer Suite [standard] | PASSED | 6237.5 | 2001.5 | 2531.1 | +529.6 |
| 1. Extended Cox [gradeblind] | PASSED | 10.86 | 2531.1 | 2784.1 | +253.0 |
| 2. Extended DeepSurv [gradeblind] | PASSED | 363.87 | 2548.6 | 2652.1 | +103.5 |
| 3. Recurrent Survival GRU [gradeblind] | PASSED | 118.81 | 2652.1 | 2741.3 | +89.2 |
| 4. Dynamic DeepHit Competing Risks [gradeblind] | PASSED | 119.11 | 2741.3 | 2890.8 | +149.5 |
| 5. Transformer Survival [gradeblind] | PASSED | 107.88 | 2890.8 | 2886.5 | -4.3 |
| 6. Recurrent Exam Survival GRU [gradeblind] | PASSED | 476.36 | 2886.5 | 2982.3 | +95.9 |
| 7. Transformer Exam Survival [gradeblind] | PASSED | 1058.93 | 2982.3 | 3155.6 | +173.3 |
| 8. Landmark Baseline [gradeblind] | PASSED | 263.08 | 3155.6 | 3207.4 | +51.8 |
| 9. Landmark Regression [gradeblind] | PASSED | 113.88 | 3207.4 | 3243.9 | +36.5 |
| 10. DML Orthogonal Survival [gradeblind] | PASSED | 72.28 | 3243.9 | 3366.9 | +123.0 |
| 11. Transformer DML [gradeblind] | PASSED | 656.16 | 3366.9 | 3314.2 | -52.8 |
| 12. Timeseries Semester LSTM [gradeblind] | PASSED | 315.34 | 3314.2 | 3365.4 | +51.2 |
| 13. Timeseries Semester Transformer [gradeblind] | PASSED | 197.42 | 3365.4 | 3458.8 | +93.5 |
| 14. Timeseries Exam GRU [gradeblind] | PASSED | 235.22 | 3458.8 | 3532.3 | +73.4 |
| 15. Timeseries Exam Transformer [gradeblind] | PASSED | 480.31 | 3532.3 | 3644.9 | +112.7 |
| 18. Deep Transformer Suite [gradeblind] | PASSED | 6921.74 | 3644.9 | 1042.5 | -2602.4 |
| 1. Extended Cox [blind] | PASSED | 9.16 | 1042.6 | 1327.4 | +284.7 |
| 2. Extended DeepSurv [blind] | PASSED | 188.71 | 1108.4 | 1328.1 | +219.8 |
| 3. Recurrent Survival GRU [blind] | PASSED | 256.61 | 1328.2 | 1436.7 | +108.6 |
| 4. Dynamic DeepHit Competing Risks [blind] | PASSED | 291.81 | 1436.7 | 1561.2 | +124.5 |
| 5. Transformer Survival [blind] | PASSED | 87.49 | 1561.2 | 1736.1 | +174.9 |
| 6. Recurrent Exam Survival GRU [blind] | PASSED | 590.36 | 1736.1 | 2392.0 | +655.9 |
| 7. Transformer Exam Survival [blind] | PASSED | 932.54 | 2392.0 | 2511.7 | +119.7 |
| 8. Landmark Baseline [blind] | PASSED | 228.73 | 2511.7 | 2538.0 | +26.3 |
| 9. Landmark Regression [blind] | PASSED | 87.04 | 2538.0 | 2581.3 | +43.3 |
| 10. DML Orthogonal Survival [blind] | PASSED | 55.81 | 2581.3 | 2699.9 | +118.6 |
| 11. Transformer DML [blind] | PASSED | 684.64 | 2699.9 | 2912.1 | +212.1 |
| 12. Timeseries Semester LSTM [blind] | PASSED | 666.39 | 2912.1 | 2966.2 | +54.2 |
| 13. Timeseries Semester Transformer [blind] | PASSED | 427.14 | 2966.2 | 3055.8 | +89.5 |
| 14. Timeseries Exam GRU [blind] | PASSED | 513.09 | 3055.8 | 3158.1 | +102.4 |
| 15. Timeseries Exam Transformer [blind] | PASSED | 917.11 | 3158.1 | 3132.9 | -25.2 |
| 18. Deep Transformer Suite [blind] | PASSED | 10133.73 | 3132.9 | 3483.2 | +350.3 |
| 1. Extended Cox [oracle] | PASSED | 11.95 | 3483.2 | 3851.6 | +368.5 |
| 2. Extended DeepSurv [oracle] | PASSED | 432.13 | 3580.1 | 3685.1 | +104.9 |
| 3. Recurrent Survival GRU [oracle] | PASSED | 604.42 | 3685.1 | 3775.9 | +90.8 |
| 4. Dynamic DeepHit Competing Risks [oracle] | PASSED | 520.79 | 3775.9 | 3892.4 | +116.4 |
| 5. Transformer Survival [oracle] | PASSED | 196.99 | 3892.4 | 4066.0 | +173.6 |
| 6. Recurrent Exam Survival GRU [oracle] | PASSED | 1051.33 | 4066.0 | 4024.6 | -41.4 |
| 7. Transformer Exam Survival [oracle] | PASSED | 1890.58 | 4024.6 | 4207.4 | +182.8 |
| 8. Landmark Baseline [oracle] | PASSED | 268.76 | 4207.4 | 4152.1 | -55.3 |
| 9. Landmark Regression [oracle] | PASSED | 146.74 | 4152.1 | 4201.4 | +49.2 |
| 10. DML Orthogonal Survival [oracle] | PASSED | 92.57 | 4201.4 | 4363.1 | +161.7 |
| 11. Transformer DML [oracle] | PASSED | 1280.97 | 4363.1 | 4506.6 | +143.5 |
| 12. Timeseries Semester LSTM [oracle] | PASSED | 800.74 | 4506.6 | 4560.4 | +53.8 |
| 13. Timeseries Semester Transformer [oracle] | PASSED | 693.85 | 4560.4 | 4646.1 | +85.7 |
| 14. Timeseries Exam GRU [oracle] | PASSED | 718.34 | 4646.1 | 4751.3 | +105.2 |
| 15. Timeseries Exam Transformer [oracle] | PASSED | 1083.17 | 4751.3 | 4711.8 | -39.5 |
| 18. Deep Transformer Suite [oracle] | PASSED | 11857.58 | 4711.8 | 3159.1 | -1552.7 |
| 1. Extended Cox [realistic] | PASSED | 8.93 | 3159.1 | 3444.7 | +285.7 |
| 2. Extended DeepSurv [realistic] | PASSED | 473.47 | 3226.7 | 3356.0 | +129.3 |
| 3. Recurrent Survival GRU [realistic] | PASSED | 435.65 | 3356.0 | 3433.0 | +77.0 |
| 4. Dynamic DeepHit Competing Risks [realistic] | PASSED | 420.82 | 3433.0 | 3539.7 | +106.6 |
| 5. Transformer Survival [realistic] | PASSED | 169.23 | 3539.7 | 3649.1 | +109.4 |
| 6. Recurrent Exam Survival GRU [realistic] | PASSED | 1250.98 | 3649.1 | 3187.0 | -462.1 |
| 7. Transformer Exam Survival [realistic] | PASSED | 2328.67 | 3187.0 | 3471.0 | +284.0 |
| 8. Landmark Baseline [realistic] | PASSED | 244.29 | 3471.0 | 3706.8 | +235.8 |
| 9. Landmark Regression [realistic] | PASSED | 122.03 | 3706.8 | 3742.9 | +36.1 |
| 10. DML Orthogonal Survival [realistic] | PASSED | 92.81 | 3742.9 | 3804.0 | +61.1 |
| 11. Transformer DML [realistic] | PASSED | 1437.59 | 3804.0 | 3971.0 | +167.0 |
| 12. Timeseries Semester LSTM [realistic] | PASSED | 716.7 | 3971.0 | 4045.3 | +74.3 |
| 13. Timeseries Semester Transformer [realistic] | PASSED | 815.13 | 4045.3 | 4141.5 | +96.1 |
| 14. Timeseries Exam GRU [realistic] | PASSED | 831.1 | 4141.5 | 4230.7 | +89.2 |
| 15. Timeseries Exam Transformer [realistic] | PASSED | 1121.45 | 4230.7 | 4155.5 | -75.1 |
| 18. Deep Transformer Suite [realistic] | PASSED | 14095.53 | 4155.5 | 1137.9 | -3017.6 |
| 16. Oracle Models | PASSED | 143.55 | 1138.0 | 1392.2 | +254.2 |
| 17. Erwerb Blind Models | PASSED | 70.93 | 1392.2 | 1432.1 | +39.9 |
| 19. Autoregressive Next-Exam | PASSED | 3542.93 | 1432.1 | 3373.2 | +1941.1 |
| 20. Structural Mediation Analysis | PASSED | 6.87 | 3373.2 | 3376.3 | +3.1 |
| 21. Deep Survival | PASSED | 65.1 | 3376.3 | 3488.3 | +112.1 |
| 22. Dynamic DeepHit Delta Model | FAILED | 0.0 | 3488.3 | 3488.3 | +0.0 |
| 23. Extended Deep Survival Delta | FAILED | 0.0 | 3488.3 | 3488.3 | +0.0 |
| 24. Extended Exam Survival | FAILED | 0.0 | 3488.3 | 3488.4 | +0.1 |
| 25. Extended Cox Delta | FAILED | 0.0 | 3488.4 | 3488.4 | +0.0 |
| 26. Recurrent Survival Model Delta | FAILED | 0.0 | 3488.4 | 3488.4 | +0.0 |
| 27. Recurrent Exam Survival Delta | FAILED | 0.0 | 3488.4 | 3488.4 | +0.0 |
| 28. Recurrent Exam Survival V2 | FAILED | 0.0 | 3488.4 | 3488.4 | +0.0 |
| 29. Autoregressive Deep Transformer | FAILED | 0.0 | 3488.4 | 3488.5 | +0.1 |
| 30. Landmark Prediction | FAILED | 4.48 | 3488.5 | 3489.2 | +0.7 |
| 31. Plot Calibration Curves | FAILED | 0.02 | 3489.2 | 3489.3 | +0.2 |
| 32. Feature Grid Experiments | PASSED | 3971.8 | 3489.3 | 4143.9 | +654.6 |
| 33. Counterfactual HR Delta | FAILED | 0.02 | 4143.9 | 4143.9 | +0.0 |
| 34. Counterfactual RR Logistic Hazard Delta | FAILED | 0.0 | 4143.9 | 4143.9 | +0.0 |
| 35. Counterfactual RR DeepHit Delta | FAILED | 0.0 | 4143.9 | 4143.9 | +0.0 |
| 36. Counterfactual Inference Semester Transformer | FAILED | 0.0 | 4143.9 | 4143.9 | +0.0 |
| 37. Counterfactual RR Exam RNN Delta | FAILED | 0.0 | 4143.9 | 4143.9 | +0.0 |
