# Heavy Deep Suite: Synoptische Auswertung (S01 vs. S07 vs. S08)

> **Fokus:** Umfassende Analyse der sequentiellen Deep-Learning-Architekturen (**Autoregressives Dual-Head GRU**, **Deep Autoregressive Transformer mit Sinusoidal Positional Encoding**, **Fail PR-AUC** und **Landmark Representation Learning**) auf den V4-Szenarien unter variierendem aleatorischen Rauschen.

---

## 1. Executive Summary & Modellarchitekturen

Die **Heavy Deep Suite** repräsentiert die rechenintensivste Säule des DeepSupport-Frameworks. Sie operiert auf der feinsten Granularitätsstufe der Prüfungshistorie (Exam-Level-Sequenzen) und umfasst vier aufeinander aufbauende Stufen:

1. **Step 1: Autoregressive Next-Exam Prediction (Dual-Head GRU)**
   - Sequentielles Modell (GRU mit 64 Hidden Units) zur simultanen Vorhersage der Note der nächsten Prüfung (MSE/Regression) und des Bestehens (Binary Crossentropy/Klassifikation).
2. **Step 2: Fail-Focus Evaluation (PR-AUC)**
   - Dedizierte Evaluation der Minderheitsklasse *Nicht-Bestehen* (Fail), berechnet aus dem Dual-Head-Modell gegen die reale Prävalenz.
3. **Step 3: Deep Autoregressive Transformer**
   - Multi-Head Self-Attention Architektur mit analytischem **Sinusoidal Positional Encoding** ($\sin/\cos$) zur Modellierung langreichweitiger Abhängigkeiten und Modul-Voraussetzungen.
4. **Step 4: Landmark Representation Learning (Landmark Ende Semester 2)**
   - Extraktion der gefrorenen 64-dimensionalen latenten Repräsentationen aus dem Transformer nach genau 2 Fachsemestern. Anschließend Downstream-Vorhersage des finalen Studienausgangs (4-Klassen Status) und der finalen Abschlussnote mittels Gradient Boosting.

### Getestete Szenarien der Rausch-Dimension

| Szenario | `gewicht_rauschen` | Charakteristik |
| :--- | :---: | :--- |
| **S07_noise_half** | `0.09` (0.5×) | Hohes Signal-zu-Rausch-Verhältnis; Prüfungsnoten spiegeln Kompetenz und Vorbereitung sehr direkt wider. |
| **S01_baseline** | `0.18` (1.0×) | Realistische Baseline; balancierte stochastische Schwankungen des Klausuralltags. |
| **S08_noise_double** | `0.36` (2.0×) | Sehr hohes Rauschen; extreme Zufallseinflüsse auf Klausurleistungen (Tagesform, Prüfervarianz). |

---

## 2. Transformer vs. GRU: Der Benchmark sequentieller Architekturen

### 2.1 Notenvorhersage der nächsten Prüfung ($R^2$, RMSE, MAE)

| Modell | Metrik | S07 (0.09 Rauschen) | S01 (0.18 Baseline) | S08 (0.36 Rauschen) |
| :--- | :--- | :---: | :---: | :---: |
| **Autoregressive Dual-Head GRU** | **Note $R^2$** | 0.6135 | 0.5659 | 0.3051 |
| | **Note RMSE** | 0.8105 | 0.8998 | 1.2354 |
| | **Note MAE** | 0.5867 | 0.6652 | 0.9530 |
| **Deep Transformer (Sin/Cos)** | **Note $R^2$** | **0.8623** | **0.6996** | **0.3825** |
| | **Δ $R^2$ (Transformer vs. GRU)** | **+0.2488 (+40.5%)** | **+0.1337 (+23.6%)** | **+0.0774 (+25.4%)** |

```
Next-Exam Note R² Score Vergleich:
S07 (Halbes Rauschen):
  GRU:         [====================                ] 0.6135
  Transformer: [============================        ] 0.8623  (+0.25 R²)

S01 (Baseline):
  GRU:         [==================                  ] 0.5659
  Transformer: [======================              ] 0.6996  (+0.13 R²)

S08 (Doppeltes Rauschen):
  GRU:         [==========                          ] 0.3051
  Transformer: [============                        ] 0.3825  (+0.08 R²)
```

> **Kernbefund:** Der **Deep Transformer mit Sinusoidal Positional Encoding schlägt das GRU in allen drei Szenarien signifikant und konsistent** um $+0.08$ bis $+0.25$ $R^2$-Punkte!
> 
> *Begründung:* Ein Recurrent Neural Network (GRU) leidet unter dem sequentiellen Flaschenhals: Frühe Modulleistungen müssen durch den kontinuierlichen Hidden-State transportiert werden und werden durch spätere Prüfungen partiell überschrieben. Der Transformer hingegen kann mittels **Multi-Head Self-Attention** direkt auf inhaltlich verwandte Vorläuferklausuren (z. B. *Mathematik I* für *Statistik II*) fokussieren, unabhängig davon, wie viele Zwischenprüfungen absolviert wurden.

---

### 2.2 Klausurbestehens-Vorhersage (Pass ROC-AUC, PR-AUC, Brier Score)

| Modell | Metrik | S07 (0.09 Rauschen) | S01 (0.18 Baseline) | S08 (0.36 Rauschen) |
| :--- | :--- | :---: | :---: | :---: |
| **Autoregressive Dual-Head GRU** | **Pass ROC-AUC** | 0.9685 | 0.9327 | 0.8306 |
| | **Pass PR-AUC** | 0.9945 | 0.9859 | 0.9492 |
| | **Brier Score** | 0.0722 | 0.0768 | 0.1180 |
| **Deep Transformer (Sin/Cos)** | **Pass ROC-AUC** | **0.9782** | **0.9410** | **0.8363** |
| | **Δ ROC-AUC (Transf. vs. GRU)** | **+0.0097** | **+0.0083** | **+0.0057** |

Die Diskrimination zwischen Bestehen und Nicht-Bestehen erreicht bei beiden Modellen exzellente Werte (> 0.93 in S01; > 0.96 in S07). Auch hier behauptet der Transformer einen beständigen Vorteil.

---

## 3. Fail-Focus Evaluation: Detektion des Nicht-Bestehens (Step 2)

In akademischen Verlaufsdaten ist das Bestehen einer Klausur die Mehrheitsklasse (~80–86%), während das Scheitern die kritische, interventionsbedürftige Minderheitsklasse darstellt. Die Evaluierung von `eval_autoregressive_fail.py` isoliert die **PR-AUC der Fail-Klasse** (`bestanden == 0`):

| Metrik | S07 (0.09 Rauschen) | S01 (0.18 Baseline) | S08 (0.36 Rauschen) |
| :--- | :---: | :---: | :---: |
| **Prävalenz Nicht-Bestehen (Fail Rate)** | 14.58% | 16.38% | 19.68% |
| **Zufalls-Baseline (No-Skill PR-AUC)** | 0.1458 | 0.1638 | 0.1968 |
| **Next_Exam_Fail_PR_AUC** | **0.6573** | **0.3801** | **0.4170** |
| **Relativer Precision-Lift ($\frac{\text{PR-AUC}}{\text{Prävalenz}}$)** | **4.51× Lift** | **2.32× Lift** | **2.12× Lift** |

> **Erkenntnis für Frühwarnsysteme:**
> - Bei halbiertem Rauschen (S07) erreicht die Fail-Vorhersage eine überragende PR-AUC von **0.6573** (ein **4.5-facher Lift** gegenüber Zufall).
> - In der Baseline (S01) erreicht das System eine solide PR-AUC von **0.3801** (**2.3-facher Lift**).
> - Bei doppeltem Rauschen steigt die Basis-Durchfallquote auf fast 20% an; die PR-AUC verharrt bei **0.4170** (2.1-facher Lift).

---

## 4. Landmark Representation Learning: Prognose ab Semester 2 (Step 4)

Im Landmark-Design friert der Transformer nach genau **2 absolvierten Semestern** ein. Aus der Sequenz aller bis dahin abgelegten Prüfungen wird ein kompakter 64-dimensionaler Feature-Vektor (Embedding) extrahiert. Dieser Vektor speist zwei Downstream-Modelle:
1. **Status-Klassifikator (4 Klassen):** Absolviert vs. Abbruch (Freiwillig) vs. Exmatrikulation (Zwang) vs. Zeitüberschreitung.
2. **Abschlussnoten-Regressor:** Prognose der finalen Bachelor-Gesamtnote (ausschließlich für spätere Absolventen).

### 4.1 Ergebnisse der Landmark-Modelle

| Aufgabe | Metrik | S07 (0.09 Rauschen) | S01 (0.18 Baseline) | S08 (0.36 Rauschen) |
| :--- | :--- | :---: | :---: | :---: |
| **4-Klassen Statusprognose** | **Accuracy (Gesamt)** | **82.40%** | **79.48%** | **75.23%** |
| | *Absolviert F1-Score* | 0.92 | 0.90 | 0.87 |
| | *Abbruch (Freiwillig) F1-Score* | 0.35 | 0.32 | 0.27 |
| | *Exmatrikulation (Zwang) F1-Score* | 0.25 | 0.23 | 0.19 |
| **Finale Abschlussnote** | **Graduation Grade $R^2$** | **0.8675** | **0.7645** | **0.5045** |

```
Landmark Prognosegüte (nach 2 Semestern):
Status-Genauigkeit (4 Klassen):
  S07: [==================================  ] 82.4%
  S01: [================================    ] 79.5%
  S08: [==============================      ] 75.2%

Abschlussnote R² (nur Absolventen):
  S07: [=================================== ] 0.8675
  S01: [==============================      ] 0.7645
  S08: [====================                ] 0.5045
```

> **Strategische Bedeutung für Studienberatung und Intervention:**
> Bereits **am Ende des zweiten Fachsemesters** erklärt das latente Embedding des Deep Transformers **76.5% der Varianz der späteren finalen Bachelor-Abschlussnote** (in S01) bzw. **86.8%** (in S07)!
> 
> Das System identifiziert künftige Absolventen bereits im 1. Studienjahr mit einem **F1-Score von 0.90**. Dies belegt, dass die ersten beiden Fachsemester die weichenstellende Phase des Studiums bilden.

---

## 5. Hardware- & Runtime-Effizienz: Benchmark des Homeserver-Laufs

Die Heavy Deep Suite wurde autonom im Debian LXC-Container des Homeservers (Lenovo ThinkCentre M70q, Intel Core i5-10400T 6C/12T, 16 GB RAM, CPU-only Keras/TensorFlow) gerechnet.

| Teilschritt | Dauer S01 | Dauer S07 | Dauer S08 | RAM Start / Peak |
| :--- | :---: | :---: | :---: | :---: |
| **Step 1: Next-Exam GRU** | 16.1 min (966s) | 28.4 min (1704s) | 18.1 min (1086s) | 0.6 GB → 1.9 GB |
| **Step 2: Fail PR-AUC** | 1.0 min (59s) | 1.0 min (57s) | 1.0 min (58s) | 1.7 GB → 3.9 GB |
| **Step 3: Deep Transformer** | 90.9 min (5455s) | 88.8 min (5327s) | 92.5 min (5548s) | 2.7 GB → 2.3 GB |
| **Step 4: Landmark Learning** | 1.9 min (114s) | 1.9 min (116s) | 1.9 min (112s) | 1.7 GB → 2.3 GB |
| **Gesamt pro Szenario** | **~1.83 h** | **~2.00 h** | **~1.89 h** | **Peak RAM: < 4.6 GB** |

- **Effizienz:** Der RAM-Bedarf überschritt zu keinem Zeitpunkt 4.6 GB; das System arbeitete im 16-GB-Container stabil und ohne Swap-Auslagerung.
- **Heterogene Architektur:** Während die Workstation den massiven 225-Modelle-Grid-Run (S01–S15) meisterte, demonstrierte der Homeserver, dass auch anspruchsvolle Deep-Learning-Pipelines (Transformer mit Self-Attention auf 50.000 Studierenden-Verläufen) ressourcenschonend und headless auf Standard-Serverhardware lauffähig sind.

---

## 6. Fazit und Synthese

1. **Transformer-Dominanz:** Der Wechsel von rekurrenter Modellierung (GRU) zu Self-Attention (Transformer mit $\sin/\cos$ Positional Encoding) bringt einen enormen Qualitätsgewinn bei der kontinuierlichen Notenprädiktion ($+0.13$ bis $+0.25$ $R^2$).
2. **Resilienz gegen Rauschen:** Während einfaches Rauschen Vorhersagen verwischt (Absinken des Noten-$R^2$ von 0.70 auf 0.38 bei doppeltem Rauschen), bleiben die Diskrimination des Bestehens (ROC-AUC > 0.83) und die Identifikation von Risiko-Studierenden (Fail PR-AUC 2.1× Lift) selbst unter extremem Rauschen intakt.
3. **Frühzeitige Prognosekraft (Landmark Sem 2):** Die Repräsentationsanalyse beweist, dass ein Transformer-Encoder bereits nach 2 Semestern die zentralen Trajektorieninformationen verdichtet hat, um Abschlussstatus (79.5% Genauigkeit) und Abschlussnote ($R^2 = 0.76$) zielsicher zu prognostizieren.
