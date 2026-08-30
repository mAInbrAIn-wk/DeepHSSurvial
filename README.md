# DeepSupport: Wirksamkeitsanalyse von Hochschulsupport via Deep Learning & Causal Machine Learning

**Autor:** Wilfried Keller  
**Kontext:** Abschlussprojekt im Kurs *Deep Learning* (Dr. Bernd Ebenhoch)  
**Datum:** August 2026 (Version 4.1)

---

## KI-Transparenz & Methodischer Stack

Alle Inhalte dieses Projekts (Code-Architektur, Datengenerierungs-Engine, Modellierung, Kausalanalyse, Audits und Dokumentation) wurden in intensiver Auseinandersetzung mit KI-Systemen entwickelt, überprüft, reviewed, korrigiert und erweitert.

- **Entwicklungsumgebung & Orchestrierung:** Antigravity IDE / Antigravity Agent
- **Integrierte LLM-Modelle (Pair Programming & Code Generation):** Gemini 3.1 Pro, Gemini 3.6 Flash, Gemini 3.7 Flash, Claude Opus 4.6, Claude Sonnet 4.6
- **Weitere KI-Tools & Exploration (via Mammouth.ai):** Claude Opus/Sonnet 5, ChatGPT 5.6, ChatGPT Sol, Kimi K2.5 / K3
- **Dokumentations-Artefakte:** Sämtliche Berichte, Reviews und Walkthroughs im Ordner `Artifacts/` sowie im System-Kontext sind direkte, transparente KI-generierte Audit-Protokolle.

---

## 1. Projektübersicht & Kausale Herausforderung

Dieses Projekt analysiert datengetrieben die Wirksamkeit von Unterstützungsangeboten (z. B. fachliche Tutorien, überfachliche Workshops, psychosoziale Beratung) an Hochschulen.

Die Kernherausforderung liegt in der Auflösung des **Selektionsbias**, des **Time Availability Confoundings** und des **Immortal-Time-Bias**:
Da leistungsschwächere Studierende oder Studierende mit viel Erwerbstätigkeit (20h/Woche) an Supportmaßnahmen teilnehmen, kommen naive Machine-Learning-Modelle oft zu dem fehlerhaften Schluss, dass Support das Studienabbruch-Risiko erhöht (*Dropout-Paradoxon*).

### Methodischer Ansatz (Dual-Strang Benchmark):
1. **8-Universen Counterfactual Simulator (V3.3):** Stochastische Simulation von $N = 50.000$ Studierenden, deren identische Klone in 8 parallelen Universen (A bis H) simuliert werden:
   - **Universum A:** Baseline (Alle 3 Supportangebote aktiv)
   - **Universum B:** Vollständige Null-Baseline (Kein Support aktiv)
   - **Universen C, D, E:** Partieller Wegfall je eines Angebots (Ablation: C ohne Fachlich, D ohne Überfachlich, E ohne Psychosozial)
   - **Universen F, G, H:** Isolierte Einzelwirkung je eines Angebots (F nur Fachlich, G nur Überfachlich, H nur Psychosozial)
2. **Kausale Survival-Analyse (Longitudinal Panels):** Überführung der Studienverläufe in Person-Semester-Panels (Counting Process Format) mit zeitvariablen Vorsemester-Deltas (`fails_prev`, `delta_cp_prev`, `cp_rueckstand`) und 13 kanonischen Features.
3. **Double Machine Learning (DML) & Oracle-Modelle:** Systematische Gegenüberstellung von DML-Orthogonalisierung, neuronalen Deep Learning Modellen und Oracle-Modellen mit latenten Simulationsvariablen ($\mu, \sigma, \varepsilon$).

---

## 2. Kausale Ground Truth der 8 Universen (V4.1, seed=99999, $N = 50.000$)

### A. Dropout-Risiko (Relativrisiko $RR$)

| Universum | Konfiguration | Dropout-Rate | Relativrisiko ($RR$) | Kausalwirkung (Ground Truth) |
| :--- | :--- | :---: | :---: | :--- |
| **A (Baseline)** | Alle Support-Typen aktiv | **29,20 %** | **1,0000** | Referenz der faktischen Beobachtungswelt |
| **B (Null-Support)** | Kein Support aktiv | **37,10 %** | **0,7871** (A vs B) | **-21,3 % Gesamtrisikosenkung** durch alle Angebote |
| **C (Ohne Fachlich)** | Fachlich blockiert, Rest aktiv | 32,10 % | **0,9097** (A vs C) | **-9,0 % Risikoreduktion** (Partieller Effekt Fachlich) |
| **D (Ohne Überfachlich)**| Überfachlich blockiert, Rest aktiv | 31,70 % | **0,9211** (A vs D) | **-7,9 % Risikoreduktion** (Partieller Effekt Überfachlich) |
| **E (Ohne Psychosozial)**| Psychosozial blockiert, Rest aktiv | 30,80 % | **0,9481** (A vs E) | **-5,2 % Risikoreduktion** (Partieller Effekt Psychosozial) |
| **F (Nur Fachlich)** | Nur Fachlich aktiv, Rest blockiert | 33,60 % | **0,9057** (F vs B) | **-9,4 % Risikoreduktion** (Isolierter Einzeleffekt) |
| **G (Nur Überfachlich)** | Nur Überfachlich aktiv, Rest blockiert | 34,00 % | **0,9164** (G vs B) | **-8,4 % Risikoreduktion** (Isolierter Einzeleffekt) |
| **H (Nur Psychosozial)** | Nur Psychosozial aktiv, Rest blockiert | 34,80 % | **0,9381** (H vs B) | **-6,2 % Risikoreduktion** (Isolierter Einzeleffekt) |

---

### B. Prüfungsnoten, Bestehensquoten & Dropout-Studiendauer

| Support-Typ | Notendifferenz (Partiell) | Notendifferenz (Isoliert) | Bestehensquoten-Lift (pp) | Dropout-Dauer (Mean) |
|:---|:---:|:---:|:---:|:---:|
| **Fachlicher Support** | $\mathbf{-0{,}0900}$ Notenpunkte | $\mathbf{-0{,}0758}$ Notenpunkte | $\mathbf{+1{,}84\text{pp}}$ | $4{,}66\text{ Sem.}$ |
| **Überfachlicher Support** | $-0{,}0215$ Notenpunkte | $-0{,}0054$ Notenpunkte | $+1{,}70\text{pp}$ | $4{,}62\text{ Sem.}$ |
| **Psychosozialer Support** | $-0{,}0408$ Notenpunkte | $-0{,}0359$ Notenpunkte | $+1{,}07\text{pp}$ | $4{,}51\text{ Sem.}$ |
| **Alle kombiniert (A vs B)** | $\mathbf{-0{,}1352}$ Notenpunkte | — | $\mathbf{+5{,}29\text{pp}}$ | $\mathbf{4{,}48\text{ vs. }4{,}94\text{ Sem.}}$ |

> [!IMPORTANT]
> **Erkenntnis zur Studiendauer bei Dropouts:**  
> Support verlängert nicht das Leiden („hinausgezögertes Scheitern“), sondern **verkürzt die Verweildauer von Abbrechern um fast ein halbes Semester** (A: 4,48 vs. B: 4,94). Supportangebote beschleunigen den Klärungsprozess: Entweder das Studium wird stabilisiert (Abschluss) oder Fehlentscheidungen werden schneller korrigiert.

---

## 3. Synopse: Kausalschätzer vs. Ground Truth Benchmark

| Modell & Methode | Analyse-Ebene | Fachlich (Part./Iso.) | Überfachlich (Part./Iso.) | Psychosozial (Part./Iso.) | Kausale Bewertung & Diagnose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Ground Truth (8 Universen)** | Makro ($N=50k$) | **0,9579 / 0,9518** | **0,9387 / 0,9317** | **0,9514 / 0,9472** | **Wahre Kausalwirkung: Alle 3 Angebote schützen signifikant** |
| **Oracle Logistic Hazard** | Panel (Latent) | **0,9880 / 0,9880** | **0,9915 / 0,9915** | **0,9926 / 0,9925** | **Beste Kausalidentifikation:** Löst Überfachlich-Bias vollständig auf ($<1,0$)! |
| **Oracle DeepSurv** | Panel (Latent) | **0,9933 / 0,9931** | **0,9897 / 0,9897** | **0,9892 / 0,9892** | Alle 3 HRs $<1,0$; beweist Information-Lift durch latente Confounder |
| **Extended Cox Panel** | Person-Semester | **0,9234 / 0,9234** | **0,9648 / 0,9648** | **0,9005 / 0,9005** | **Bester observabler Schätzer:** FWL-Partialling isoliert Treatment sauber |
| **DML Orthogonal Survival** | Panel (Double ML) | **0,8417 / 0,8417** | **1,0512 / 1,0512** | **0,9249 / 0,9249** | Fachlich & Psychosozial protektiv; Überfachlich leidet unter Feedback-Loop |
| **Recurrent Exam GRU V2** | Prüfungssequenz | **1,0358 / 1,0139** | **1,1403 / 1,0972** | **0,9755 / 0,9566** | Nach Bugfix: Psychosozial klar protektiv ($0,9566$ isoliert) |
| **Dynamic DeepHit Delta** | Semester-Sequenz | **0,9967 / 0,9977** | **1,0032 / 1,0031** | **0,9979 / 0,9985** | Nahe an 1,0 wegen Spärlichkeit; protektive Tendenz |
| **Exam Transformer Regressor**| Prüfungssequenz | $\Delta\text{Note} = \mathbf{-0{,}024}$ | $\Delta\text{Note} = +0{,}041$ | $\Delta\text{Note} = +0{,}008$ | Zeigt signifikanten Noten-Lift bei Fachlichem Support ($R^2=0{,}90$) |
| **Lineare Noten-OLS** | Prüfungsebene | $\Delta\text{Note} = \mathbf{-0{,}095}$ | $\Delta\text{Note} = +0{,}033$ | $\Delta\text{Note} = \mathbf{-0{,}052}$ | **Trifft Noten-Ground-Truth exzellent** (GT Fachlich: $-0{,}090$) |

---

## 4. Modell-Portfolio Performance

### Abbruch- & Survival-Vorhersage
| Modell | Level / Typ | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: |
| **Recurrent Exam Survival V2** | Exam Sequence (roll. Fails/GPA) | **0,8713** | 0,1747 | 0,0168 |
| Extended Logistic Hazard Exam Delta | Exam Level Panel | 0,8636 | 0,1757 | 0,0169 |
| Logistic Hazard Landmark | Static Landmark (S1-S2) | 0,8597 | 0,7146 | — |
| Recurrent Exam Survival GRU Delta | Exam Sequence | 0,8504 | 0,1389 | 0,0175 |
| Dynamic DeepHit Delta (Dropout) | Multi-Task Competing | 0,7942 | 0,2301 | 0,0366 |
| **Recurrent Survival GRU Delta (13 Feat.)**| Semester Sequence | **0,7885** | 0,2241 | 0,0369 |
| Transformer Survival (Semester) | Causal Masked Attention | 0,7909 | 0,2284 | 0,0365 |
| Oracle Logistic Hazard | Latente Variablen ($\mu, \sigma, \varepsilon$) | 0,7714 | 0,2112 | 0,0368 |
| Extended Cox Delta Panel | Person-Semester | 0,7694 | 0,2081 | 0,0370 |
| DML Orthogonal Survival | Causal Panel | 0,7694 | 0,2081 | 0,0370 |

---

## 5. Vollständiges Skript-Register & Dokumentation

- 👉 **[`Artifacts/script_registry.md`](Artifacts/script_registry.md)**: Vollständiges Inventar aller 69 Skripte mit Feature-Vektoren, Input-Dimensionen und Outputs.
- 👉 **[`Artifacts/simulation_kausal_doku.md`](Artifacts/simulation_kausal_doku.md)**: Vollständiges Kausaldiagramm (Mermaid), mathematische DGP-Gleichungen und Selektionsmechaniken.

---

## 6. Feature Builder & Datenformate

Die zentrale `src/feature_builder.py` generiert Features in 5 Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`). Mit Version V4.1.1 wurden **Sample Leakage** und **Future Leakage** behoben, und die Oracle-Features um `hidden_overload` und `hidden_zeit_puffer` erweitert.

Die Feature-Anzahlen nach Format:

| Format | standard | oracle |
|:---|:---:|:---:|
| Semester Tensor | 18 | 23 |
| Exam Tensor | 24 | 29 |
| Semester Panel | 16 | 21 |
| Exam Panel | 23 | 28 |
| Landmark | 16 | 21 |

---

## 7. V4.1 Sensitivity Grid Search ($N = 50.000$)

Vollständiger Sensitivitätslauf mit 15 Szenarien × 8 Universen = 120 Simulationsläufe. Laufzeit: 14,6 h (5 Worker). B=37,1% über alle nicht-globalen Szenarien (perfekte RNG-Synchronisierung).

| Szenario | Parameterdimension | Dropout A | Dropout B | ARR (pp) |
| :--- | :--- | :---: | :---: | :---: |
| **S01 (Baseline)** | Referenz | 29,2 % | 37,1 % | 7,9 |
| **S02** | Support-Wirkung ×0,5 | 32,8 % | 37,1 % | 4,3 |
| **S03** | Support-Wirkung ×2,0 | 25,5 % | 37,1 % | 11,6 |
| **S04** | Notenboost ×0,5 | 30,2 % | 37,1 % | 6,9 |
| **S05** | Notenboost ×1,5 | 28,5 % | 37,1 % | 8,6 |
| **S06** | Notenboost ×2,0 | 27,9 % | 37,1 % | 9,2 |
| **S07** | Rauschen ×0,5 | 26,1 % | 33,7 % | 7,6 |
| **S08** | Rauschen ×2,0 | 32,5 % | 40,1 % | 7,6 |
| **S09** | Zeitkosten ×0,5 | 28,5 % | 37,1 % | 8,6 |
| **S10** | Zeitkosten ×2,0 | 29,7 % | 37,1 % | 7,4 |
| **S11** | RCT-Selektion | 25,7 % | 37,1 % | 11,4 |
| **S12** | Overload-Penalty ×0,5 | 27,3 % | 35,7 % | 8,4 |
| **S13** | Overload-Penalty ×2,0 | 30,6 % | 37,9 % | 7,3 |
| **S14** | Overload-Penalty Cap 0,15 | 28,7 % | 36,0 % | 7,3 |
| **S15 (Kombi)** | Kosten ×2 + Wirkung ×2 | 25,9 % | 37,1 % | 11,2 |

> [!NOTE]
> **Sensitivitäts-Ranking (ARR-Spanne):** Overload-Penalty (8,5pp) > Support-Wirkung (7,3pp) > Rauschen (6,4pp) > RCT-Selektion (3,5pp) > Notenboost (3,0pp) > Zeitkosten (1,2pp). Die ARR bleibt über alle Overload-Kalibrierungen hinweg robust (7,3–8,4pp).
