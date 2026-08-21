# Walkthrough: Master-Nachtlauf V3.3 – Ergebnisse & Realistische Evaluierung

Der Master-Nachtlauf (Dauer: 2,8 Stunden) auf dem **Simulation V3.3 Datensatz** (mit **perfekter RNG-Synchronisation**, Carry-over ⅔ und verdoppeltem Support-Boost) wurde vollständig durchgeführt. Alle 21 Modellarchitekturen, Kausalschätzer, Oracle-Analysen und Zeitreihenmodelle wurden trainiert und ausgewertet.

---

## 1. Makro-Kausaleffekte (Ground Truth aus 5 Universen – V3.3 Datensatz)

Auf Ebene der Gesamtkohorte (50.000 Studierende über 5 parallele Welten mit **deterministisch synchronisierten Würfel-Streams**) ergeben sich folgende stochastisch saubere makroskopische Effekte:

| Universum | Bedingung | Dropout-Quote | Relatives Risiko (RR) vs. A | Netto-Gerettete vs. A | Kausale Wirkung auf Makro-Ebene |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Universum A** | Baseline (Alle Support-Typen) | **27,37 %** | **1,0000** | — | Ausgangslage |
| **Universum B** | Kein Support (komplett blockiert) | **32,35 %** | **0,8462** | **+2.488** | **-15,38 % Risikoreduktion** (Support-Gesamtsystem schützt) |
| **Universum C** | Kein fachlicher Support | **28,57 %** | **0,9579** | **+601** | **-4,21 % Risikoreduktion** (Fachlicher Support schützt) |
| **Universum D** | Kein überfachlicher Support | **29,16 %** | **0,9387** | **+893** | **-6,13 % Risikoreduktion** (Überfachlicher Support schützt) |
| **Universum E** | Kein psychosozialer Support | **28,77 %** | **0,9514** | **+699** | **-4,86 % Risikoreduktion** (Psychosozialer Support schützt) |

> [!NOTE]
> **Erkenntnis V3.3:** Das Gesamtsystem schützt die Studierenden effektiv (2.488 weniger Abbrüche). Der fachliche Support stellt auf Prüfungsebene zwar den stärksten Notengewinn dar, führt auf Kohortenebene aber durch die Zeitkosten zu einer moderateren Netto-Risikoreduktion (-4,21 %).

---

## 2. Kausale Entzerrung & DML-Schätzer (Realistischer Befund)

Die Evaluierung der Kausalschätzer zeigt die deutlichen Grenzen beobachtender Machine-Learning-Verfahren:

| Evaluierter Support-Typ | Ground Truth RR (V3.3) | Standard DML (Tabular Cox) | Deep Causal Transformer-DML | Methodischer Befund |
| :--- | :---: | :---: | :---: | :--- |
| **Fachlicher Support (`fach`)** | **0,9579** | **0,7899** (Starke Überschätzung) | **1,0172** (Verfehlt Signal, leicht negativ) | **Problem:** Fachlicher Support wird von Standard-DML wegen des Selektionsbias massiv überschätzt ($RR=0.79$). Deep Transformer-DML dämpft den Ausschlag zwar ab, schätzt den Effekt jedoch leicht im negativen Bereich ($1.017$), da Sequenz-Embeddings die Zeitkosten-Interaktion nicht vollständig von der Notenwirkung trennen können. |
| **Überfachlicher Support (`uebf`)** | **0,9387** | **1,0460** (Falsches Vorzeichen) | **0,9957** (Nahe Neutralität) | Standard-DML verzerrt den Effekt in Richtung Schädlichkeit ($RR=1.046$). Transformer-DML behebt die Richtung, unterschätzt die Stärke aber deutlich ($0.996$). |
| **Psychosozialer Support (`psych`)** | **0,9514** | **0,9078** | **0,9569** | Akzeptabler Treffer nahe der Ground Truth. |

> [!WARNING]
> **Ehrliches Fazit:** Die Kausalmodelle haben nach wie vor erhebliche Mühe, die ehrliche Wirkung des fachlichen Supports korrekt abzubilden. Man kann hier nicht von einem "gelösten Problem" sprechen. Der V3.3-Datensatz dient als wertvolle Benchmark-Grundlage, die veranschaulicht, wo DML-Verfahren an ihre Grenzen stoßen.

---

## 3. Modell-Portfolio Performance-Übersicht

### A) Survival- & Hazard-Modelle (Abbruch- & Zeitpunkt-Vorhersage)

| Modell-Kategorie | Modell-Name | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: |
| **Deep Exam-Transformer Survival** | Exam Sequence ($d=128$, Attn) | **0,9999** | **0,9998** | **0,0007** |
| **Exam-Level Survival** | Extended Logistic Hazard Exam Delta | **0,8636** | 0,1757 | 0,0169 |
| **Landmark Hazard** | Discrete-Time Logistic Hazard Landmark | **0,8597** | **0,7146** | — |
| **Exam-Level Survival** | Recurrent Exam Survival GRU Delta | 0,8504 | 0,1389 | 0,0175 |
| **Exam-Level Survival** | Recurrent Exam Survival GRU (Base) | 0,8453 | 0,1420 | 0,0174 |
| **Sequence Survival** | Transformer Survival (Semester) | 0,7909 | 0,2284 | 0,0365 |
| **Sequence Survival** | Recurrent Survival GRU (Semester) | 0,7898 | 0,2234 | 0,0368 |
| **Panel Survival** | Dynamic DeepHit Delta (Dropout) | 0,7898 | 0,2234 | 0,0366 |
| **Panel Survival** | Recurrent Survival Model Delta | 0,7893 | 0,2257 | 0,0367 |
| **Panel Survival** | Extended Logistic Hazard Delta | 0,7694 | 0,2081 | 0,0370 |
| **Causal Panel** | DML Orthogonalized Survival | 0,7694 | 0,2081 | 0,0370 |

---

### B) Regressions-Modelle (Noten- & GPA-Vorhersage)

| Modell-Typ | Modell-Name | $R^2$ Score | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **Sequenziell (Exam)** | **Deep Exam-Transformer Regressor** | **0,9991** | **0,0223** | **0,0162** |
| **Sequenziell (Semester)** | **Semester-LSTM Regressor** | **0,9144** | 0,3108 | 0,2352 |
| **Sequenziell (Semester)** | Semester-Transformer Regressor | 0,9084 | 0,3215 | 0,2448 |
| **Sequenziell (Exam)** | Exam-GRU Regressor | 0,9029 | 0,3289 | 0,2480 |
| **Statisch / Punktuell** | Keras MLP Regression | 0,8694 | **0,2272** | **0,1731** |
| **Statisch / Punktuell** | SVR (Support Vector Regression) | 0,8668 | 0,2294 | 0,1752 |
| **Statisch / Punktuell** | Random Forest Regression | 0,8484 | 0,2448 | 0,1857 |
| **Statisch / Punktuell** | Linear Ridge Regression | 0,8461 | 0,2466 | 0,1914 |

---

## 4. Durchgeführte Audits & Artefakte

1. **[Model Uniformity Audit](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/model_uniformity_audit.md)**: Kritische, ungeschönte Bewertung aller Modelle.
2. **[Implementation Plan Rev. 2](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/implementation_plan.md)**: Refactoring-Protokoll der RNG-Entkopplung.
3. **[Hypothesis Evolution](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/hypothesis_evolution.md)**: Chronologisches Protokoll aller Test-Phasen.
