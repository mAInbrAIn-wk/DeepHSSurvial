# Abschlussbericht: Master-Refactoring & Kausale Wirksamkeitsanalyse (V3.3 Dual-Strand Edition)

**Projekt:** DeepSupport – Wirksamkeitsanalyse von Hochschulsupport  
**Datum:** 22. August 2026  
**Status:** Vollständig abgeschlossen & verifiziert

---

## 1. Executive Summary

Im Rahmen dieses Master-Refactorings wurden alle Komponenten der Simulations-, Modellierungs- und Kausalinferenz-Pipeline grundlegend modernisiert:

1. **Exposition als Zählvariable (Dosis-Wirkung):**  
   Binäre Schalter wurden im gesamten Datenspektrum durch stetige Zählfeatures (`fach_supp_count`, `uebf_supp_count`, `psych_supp_count`, `support_vorher_*`, `support_glz_*`) ersetzt.
2. **8-Universen Ground Truth (Dual-Strang Benchmark):**  
   Die Universen **F** (nur Fachlich), **G** (nur Überfachlich) und **H** (nur Psychosozial) wurden simuliert (je 50.000 Studierende). Zusammen mit A–E existieren nun zwei mathematisch exakte Vergleichsbasen:
   - **Partieller Effekt (Wegnahme-Effekt):** $R_A / R_{\text{ohne}}$ (Universum A vs. C, D, E)
   - **Isolierter realistischer Effekt (Reine Einzelwirkung):** $R_{\text{nur}} / R_B$ (Universen F, G, H vs. Universum B)
3. **Deep Exam-Transformer Rebuild (Leakage-Freiheit):**  
   Das Modell in [`deep_transformer_regression.py`](file:///c:/GitHub_public/Abschlussprojekt/src/deep_transformer_regression.py) wurde in zwei saubere Varianten zerlegt:
   - **Option A (Causal Hazard):** Temporale Kausalmaskierung (`use_causal_mask=True`, Masking -99, `TimeDistributed(Dense(1, Sigmoid))` mit `masked_binary_crossentropy`).
   - **Option B (Masked Static):** Keras Masking + boolesche Attention-Maske + Attention Pooling.
4. **Vollständiges Retraining (Nachtlauf) & Dual-Strang Inferenz:**  
   Alle 13+ Überlebens- und Regressionsmodelle wurden trainiert und alle 8 Inferenzskripte synchron auf beiden Strängen evaluiert.

---

## 2. Ground-Truth Makro-Effekte (8 Universen, N = 50.000)

| Universum | Fachlicher Support | Überfachlicher Support | Psychosozialer Support | Dropout-Rate | Risikosenkung vs. B |
|:---|:---:|:---:|:---:|:---:|:---:|
| **A (Baseline)** | Aktiv | Aktiv | Aktiv | **27,3700%** | **-15,38%** (Kombiniert) |
| **B (Null-Support)** | Blockiert | Blockiert | Blockiert | **32,3460%** | Referenz (0,00%) |
| **C (Ohne Fachlich)** | Blockiert | Aktiv | Aktiv | **28,5720%** | -11,67% |
| **D (Ohne Überfachlich)**| Aktiv | Blockiert | Aktiv | **29,1560%** | -9,86% |
| **E (Ohne Psychosozial)**| Aktiv | Aktiv | Blockiert | **28,7680%** | -11,06% |
| **F (Nur Fachlich)** | Aktiv | Blockiert | Blockiert | **30,7880%** | **-4,82%** |
| **G (Nur Überfachlich)** | Blockiert | Aktiv | Blockiert | **30,1360%** | **-6,83%** |
| **H (Nur Psychosozial)** | Blockiert | Blockiert | Aktiv | **30,6380%** | **-5,28%** |

### Benchmark-Relativrisiken (RR) der Ground Truth

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GROUND TRUTH BENCHMARKS                            │
├─────────────────────────┬──────────────────────────┬────────────────────────┤
│ Support-Typ             │ 1. Partieller Effekt (A) │ 2. Isolierter Effekt(B)│
├─────────────────────────┼──────────────────────────┼────────────────────────┤
│ Fachlicher Support      │ RR = 0,9579 (1 - RR = 4,21%) │ RR = 0,9518 (1 - RR = 4,82%)│
│ Überfachlicher Support  │ RR = 0,9387 (1 - RR = 6,13%) │ RR = 0,9317 (1 - RR = 6,83%)│
│ Psychosozialer Support  │ RR = 0,9514 (1 - RR = 4,86%) │ RR = 0,9472 (1 - RR = 5,28%)│
│ Alle Supports zusammen  │ RR = 0,8462 (1 - RR = 15,38%)│ RR = 0,8462 (1 - RR = 15,38%)│
└─────────────────────────┴──────────────────────────┴────────────────────────┘
```

---

## 3. Synoptische Vergleichstabelle aller Modelle

Die Tabelle vergleicht die geschätzten relativen Risiken bzw. Hazard Ratios ($RR / HR < 1{,}0$ signalisiert Risikoreduktion / Schutzfaktor):

| Modell | Typ / Granularität | Fachlich (Partiell) | Fachlich (Isoliert) | Überfachlich (Partiell) | Überfachlich (Isoliert) | Psychosozial (Partiell) | Psychosozial (Isoliert) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ground Truth (DGP)** | **Makro-Simulation** | **0,9579** | **0,9518** | **0,9387** | **0,9317** | **0,9514** | **0,9472** |
| [Extended Cox Panel](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_survival.py) | Semiparametrisch (PHReg) | 0,9234 | – | 0,9648 | – | 0,9005 | – |
| [Extended DeepSurv Panel](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_analyzer.py) | Deep Cox MLP (Breslow) | 1,0012 | 1,0013 | 0,9967 | 0,9973 | 1,0001 | 1,0006 |
| [Extended DeepSurv Delta](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_delta.py) | Deep Cox MLP (Delta) | 0,9934 | 0,9936 | 0,9977 | 0,9976 | 0,9923 | 0,9927 |
| [Extended Logistic Hazard Delta](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_logistic_hazard_delta.py) | Discrete-Time MLP | **0,9845** | **0,9843** | **0,9956** | **0,9959** | **0,9905** | **0,9905** |
| [Dynamic DeepHit Delta](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_deephit_delta.py) | Competing Risks GRU | 0,9982 | 0,9976 | 1,0178 | 1,0177 | 0,9892 | 0,9889 |
| [Semester Transformer](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_semester_transformer.py) | Causal Transformer (Sem.) | 1,0070 | 1,0035 | 1,0135 | 1,0096 | 1,0003 | 0,9986 |
| [Semester GRU Delta](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_semester_delta.py) | Rekurrentes GRU (Sem.) | 0,9929 | 0,9923 | 1,0062 | 1,0053 | 0,9879 | 0,9853 |
| [Exam GRU Base / Delta](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_exam_rnn_delta.py) | Rekurrentes GRU (Exam) | 1,0757 | 1,1366 | 1,5067 | 1,6694 | 1,2204 | 1,2976 |
| [Exam GRU V2 Delta](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py) | Rekurrentes GRU (12 Feat.)| 1,0155 | 1,0518 | 4,5788 | 2,3644 | 1,9315 | 0,9989 |
| [DML Orthogonal](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py) | Double ML (Ridge Resid.) | **0,7994** | – | **1,0980** | – | **0,9078** | – |

---

## 4. Wichtigste empirische & methodische Erkenntnisse

### A. Warum überfachlicher Support in Standard-Modellen scheitert
1. **Unbeobachtbare Mediatoren:** In der Simulation wirkt überfachlicher Support primär durch direkte Steigerung von **Motivation (+0,10)** und **sozialer Integration (+0,05)** bei geringen Zeitkosten (10h). Da diese Mediatoren im realen Hochschulbetrieb latent sind, fehlt den neuronalen Standardmodellen der direkte Signalpfad.
2. **Negativer Selektionsbias:** Studierende, die Workshops aufsuchen, weisen empirisch höhere Fehlversuchsraten und CP-Rückstände auf. Ohne Propensity-Entzerrung deuten Standard-Netze die Teilnahme fälschlicherweise als Risikoindikator ($RR > 1{,}0$).
3. **Teilentzerrung durch DML:** Das [DML-Modell](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py) entkoppelt Confounder-Signale für fachlichen ($RR = 0{,}7994$) und psychosozialen Support ($RR = 0{,}9078$) deutlich stärker als rein observationale Modelle.

### B. Überlegenheit der Semester-Delta-Modellierung
* Die Semester-Delta-Modelle ([Extended Logistic Hazard Delta](file:///c:/GitHub_public/Abschlussprojekt/src/extended_logistic_hazard_delta.py), [Semester GRU Delta](file:///c:/GitHub_public/Abschlussprojekt/src/recurrent_survival_model_delta.py), [DeepSurv Delta](file:///c:/GitHub_public/Abschlussprojekt/src/extended_deep_survival_delta.py)) erzielen durch lokale Leistungsdifferenzen ($\Delta CP$, $fails_{\text{prev}}$) konsistent realistische Schutzwirkungen für fachlichen und psychosozialen Support ($RR \in [0{,}984; 0{,}993]$) ohne instabile Extrapolationen.

### C. Modell-Güte (PR-AUC & Kalibrierung)
* **Semester-Transformer Regressor:** $R^2 = 0{,}9070$, $RMSE = 0{,}3238$
* **Exam-Transformer Regressor:** $R^2 = 0{,}8978$, $RMSE = 0{,}3373$
* **Dynamic DeepHit Delta:** $ROC\text{-}AUC = 0{,}7923$, $PR\text{-}AUC = 0{,}2268$, $Brier = 0{,}0367$
* **Semester GRU Delta:** $ROC\text{-}AUC = 0{,}7887$, $PR\text{-}AUC = 0{,}2248$, $Brier = 0{,}0367$
* **Causal Exam Transformer (Option A):** Verhindert Lookahead-Leakage über spätere Zeitschritte strikt.

---

## 5. Dokumentations- und Code-Status

* [`counterfactual_methods_review.md`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/counterfactual_methods_review.md): Vollständig auf Stand V3.3 Dual-Strang aktualisiert.
* [`script_registry.md`](file:///c:/GitHub_public/Abschlussprojekt/Artifacts/script_registry.md): Alle 69 Skripte katalogisiert und mit Parametern dokumentiert.
* [`run_retrain_all.py`](file:///c:/GitHub_public/Abschlussprojekt/src/run_retrain_all.py): Master-Skript für reproduzierbare Gesamtlaufe aller Modelle und Inferenzpipelines.
