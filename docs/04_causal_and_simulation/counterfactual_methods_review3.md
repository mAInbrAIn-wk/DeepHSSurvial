# Review: Mathematische Analyse der Counterfactual-Methoden (V3.3 Dual-Strand Edition)

**Projekt:** DeepSupport – Wirksamkeitsanalyse von Hochschulsupport  
**Datum:** 22. August 2026  
**Dokumenttyp:** Methodisches Code-Review & mathematische Modellvergleiche

---

## Inhalt

1. [Korrekte Zeitkosten und Wirkungsmechanismen](#1-korrekte-zeitkosten-und-wirkungsmechanismen)
2. [Methode 0: Ground Truth (8-Universen-Simulation A–H)](#2-methode-0-ground-truth-8-universen-simulation)
3. [Methode 1: Statistischer Cox-Regressor (Extended Cox Delta)](#3-methode-1-statistischer-cox-regressor)
4. [Methode 2: Neuronale Cox-Modelle (DeepSurv Panel & Delta)](#4-methode-2-neuronale-cox-modelle)
5. [Methode 3: Discrete-Time-Modelle (Logistic Hazard, DeepHit)](#5-methode-3-discrete-time-modelle)
6. [Methode 4: Sequenzmodelle (Transformer, GRU) & Leakage-Freiheit](#6-methode-4-sequenzmodelle)
7. [Methode 5: Double Machine Learning (DML)](#7-methode-5-double-machine-learning)
8. [Synoptischer Vergleich aller Methoden & Dual-Teststränge](#8-synoptischer-vergleich-aller-methoden)
9. [Diagnose: Warum überfachlicher Support falsch geschätzt wird](#9-diagnose-warum-überfachlicher-support-falsch-geschätzt-wird)
10. [Umgesetzte Lösungen & Zusammenfassung](#10-umgesetzte-lösungen)

---

## 1. Korrekte Zeitkosten und Wirkungsmechanismen

Die Zeitkosten der Support-Angebote sind **nicht** einheitlich 30h, sondern unterscheiden sich erheblich. Aus [`config.py`](file:///c:/GitHub_public/Abschlussprojekt/src/config.py) (`SUPPORT_ANGEBOTE`):

### A. Fachlicher Support: **30 h / Semester**
| Angebot | Kosten |
|:---|:---:|
| SUP01 Mathe-Tutorium | 30 h |
| SUP02 Programmier-Lerngruppe | 30 h |
| SUP03 Statistik-Repetitorium | 30 h |
| SUP04 Schreibwerkstatt | 30 h |
| SUP05 Sprachkurs Englisch | 30 h |
| SUP06 Examenscoaching | 30 h |

**Wirkungsweise im DGP:**
- Wirkt **nicht** direkt auf Motivation oder soziale Integration.
- Wirkt ausschließlich über **Notenverbesserung** bei zugeordneten Prüfungen in `simuliere_pruefung`:
  $$\text{boost} = \min\left(0{,}40 \cdot \left(\sum_{\text{aktuell}} w_m + \frac{2}{3} \sum_{\text{carryover}} w_m\right), \; 1{,}0\right)$$
  $$\text{note} = \text{clip}(\text{erwartete\_note} - \text{boost} + \epsilon_{\text{exam}}, 1{,}0, 5{,}0)$$
- Durch die Notenverbesserung sinkt die Durchfallquote, was kumulativ `fails_prev` und `cp_rueckstand` verringert und erst dadurch das Dropout-Risiko senkt.

### B. Überfachlicher Support: **10 h / Semester**
| Angebot | Kosten |
|:---|:---:|
| SUP07 Zeitmanagement-Workshop | 10 h |
| SUP08 Lerncoaching | 10 h |
| SUP09 Mentoring-Programm | 10 h |

**Wirkungsweise im DGP:**
- Direkter Motivationsboost **+0,10** und sozialer Integrationsboost **+0,05** pro Teilnahme (linear stapelbar bis zu 3 Angebote).
- **Kein direkter Einfluss auf Prüfungsnoten:** Motivation und Integration verändern *nicht* die Prüfungsleistung $leistung$, sondern speisen sich direkt in die strukturelle Dropout-Entscheidungsfunktion $P(\text{dropout})$ ein.

### C. Psychosozialer Support: **5–15 h / Semester** (Durchschnitt: 8,3 h)
| Angebot | Kosten |
|:---|:---:|
| SUP10 Psychologische Beratung | 5 h |
| SUP11 Studienberatung | 5 h |
| SUP12 Peer-Support-Gruppe | 15 h |

**Wirkungsweise im DGP:**
- Direkter Motivationsboost **+0,075** und starker sozialer Integrationsboost **+0,175** pro Teilnahme.
- Speist sich direkt in die strukturelle Dropout-Funktion ein.

> [!IMPORTANT]
> **Struktureller Unterschied:** Überfachlicher und psychosozialer Support kosten nur **⅕ bis ⅓ der Zeit** des fachlichen Supports und wirken über motivationale und soziale Resilienz direkt im Entscheidungszeitpunkt des Studienabbruchs. Fachlicher Support schützt hingegen über harte Leistungsnachweise (CP-Erwerb, Nicht-Durchfallen), verursacht aber hohe Zeitkosten (30h), die bei überlasteten Studierenden Abwürfe provozieren können.

---

## 2. Methode 0: Ground Truth (8-Universen-Simulation)

### Funktionsweise
Die Simulation in [`simulation_v3.py`](file:///c:/GitHub_public/Abschlussprojekt/src/simulation_v3.py) erzeugt 50.000 identische Studierenden-Klone, die in 8 parallelen Universen simuliert werden (`POPULATION_SEED = 12345`). Alle Zufallsziehungen sind über 4 getrennte RNG-Streams synchronisiert.

| Universum | `block_fach` | `block_uebf` | `block_psych` | Konfiguration | Dropout-Rate |
|:---|:---:|:---:|:---:|:---|:---:|
| **A (Baseline)** | ✗ | ✗ | ✗ | Alle Supports aktiv | **27,37%** |
| **B (Null-Support)** | ✓ | ✓ | ✓ | Kein Support | **32,35%** |
| **C (Ohne Fachlich)** | ✓ | ✗ | ✗ | Nur fachlich blockiert | **28,57%** |
| **D (Ohne Überfachlich)** | ✗ | ✓ | ✗ | Nur überfachlich blockiert | **29,16%** |
| **E (Ohne Psychosozial)** | ✗ | ✗ | ✓ | Nur psychosozial blockiert | **28,77%** |
| **F (Nur Fachlich)** | ✗ | ✓ | ✓ | Nur fachlich aktiv | **30,79%** |
| **G (Nur Überfachlich)** | ✓ | ✗ | ✓ | Nur überfachlich aktiv | **30,14%** |
| **H (Nur Psychosozial)** | ✓ | ✓ | ✗ | Nur psychosozial aktiv | **30,64%** |

### Duale Ground-Truth Relativrisiken

#### 1. Partieller Effekt (Entzugseffekt vs. Universum A):
$$\text{RR}_{\text{partial}} = \frac{R_A}{R_{\text{ohne}}}$$
- **Fachlich:** $\text{RR} = 27{,}370\% / 28{,}572\% = \mathbf{0{,}9579}$ (Risikosenkung: $4{,}21\%$)
- **Überfachlich:** $\text{RR} = 27{,}370\% / 29{,}156\% = \mathbf{0{,}9387}$ (Risikosenkung: $6{,}13\%$)
- **Psychosozial:** $\text{RR} = 27{,}370\% / 28{,}768\% = \mathbf{0{,}9514}$ (Risikosenkung: $4{,}86\%$)

#### 2. Isolierter Effekt (Reine Einzelwirkung vs. Universum B):
$$\text{RR}_{\text{isolated}} = \frac{R_{\text{nur}}}{R_B}$$
- **Fachlich (F):** $\text{RR} = 30{,}788\% / 32{,}346\% = \mathbf{0{,}9518}$ (Risikosenkung: $4{,}82\%$)
- **Überfachlich (G):** $\text{RR} = 30{,}136\% / 32{,}346\% = \mathbf{0{,}9317}$ (Risikosenkung: $6{,}83\%$)
- **Psychosozial (H):** $\text{RR} = 30{,}638\% / 32{,}346\% = \mathbf{0{,}9472}$ (Risikosenkung: $5{,}28\%$)

---

## 3. Methode 1: Statistischer Cox-Regressor

**Skripte:** [`extended_cox_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_delta.py), [`extended_cox_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_survival.py)  
**Modelltyp:** Semiparametrischer Cox Proportional-Hazards (PHReg)

$$h(t \mid X) = h_0(t) \cdot \exp\left(\sum_{k=1}^{K} \beta_k X_k\right)$$

Schätzung durch Maximierung der Breslow-Log-Partial-Likelihood. Koeffizienten $\beta$ auf den stetigen Zählvariablen `fach_supp_count`, `uebf_supp_count`, `psych_supp_count`.

---

## 4. Methode 2: Neuronale Cox-Modelle (DeepSurv)

**Skripte:**  
- [`counterfactual_hr_analyzer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_analyzer.py) → Extended DeepSurv Panel  
- [`counterfactual_hr_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_hr_delta.py) → Extended DeepSurv Delta

$$h(t \mid X) = h_0(t) \cdot \exp\left(g_\theta(X)\right)$$

### Dual-Strang Kontrafaktische Inferenz
1. **Partiell:** $X_{\text{Ziel}} = 0$ vs. $X_{\text{Ziel}} = \text{beobachtet}$ (andere Kovariaten bleiben beobachtet).
2. **Isoliert Realistisch:** Alle Support-Zähler auf $0$ vs. $X_{\text{Ziel}} = \text{beobachtet}$ (andere auf $0$).

$$\text{HR}_i = \exp\left(g_\theta(X_i^{(1)}) - g_\theta(X_i^{(0)})\right)$$

---

## 5. Methode 3: Discrete-Time-Modelle (Logistic Hazard, DeepHit)

**Skripte:**  
- [`counterfactual_rr_logistic_hazard_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_logistic_hazard_delta.py)  
- [`counterfactual_rr_deephit_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_deephit_delta.py)

$$p_{i,t} = P(E_{i,t} = 1 \mid T_i \ge t, X_{i,t})$$

Kontrafaktischer Relativrisiko-Quotient:
$$\text{RR}_i = \frac{p_{1,i}}{p_{0,i}}$$

---

## 6. Methode 4: Sequenzmodelle & Leakage-Freiheit

### A. Deep Exam-Transformer Rebuild ([`deep_transformer_regression.py`](file:///c:/GitHub_public/Abschlussprojekt/src/deep_transformer_regression.py))
Um das zuvor identifizierte **Sequenzlängen- und Notenleakage** vollständig zu eliminieren, wurden zwei saubere Architekturen implementiert:
1. **Option A (Causal Hazard):** `use_causal_mask=True`, Masking(-99), `TimeDistributed(Dense(1, Sigmoid))` trainiert mit `masked_binary_crossentropy`. Verhindert Lookahead-Leakage über spätere Zeitschritte strikt.
2. **Option B (Masked Static):** Keras `Masking(-99)` mit boolescher Attention-Maskierung, gelerntem `AttentionPooling` und statischem Event-Kopf.

### B. GRU & Transformer Sequenz-Inferenz
- [`counterfactual_inference_semester_transformer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_semester_transformer.py)
- [`counterfactual_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py)
- [`counterfactual_rr_exam_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_exam_rnn_delta.py)
- [`counterfactual_rnn_semester_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_semester_delta.py)

---

## 7. Methode 5: Double Machine Learning (DML)

**Skript:** [`dml_orthogonal_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py)

1. **Stufe 1 (Propensity-Residualisierung via Ridge):**
   Für die stetigen Zählvariablen $A_k \in \{\text{fach\_cnt}, \text{uebf\_cnt}, \text{psych\_cnt}\}$:
   $$\hat{e}_k(W) = \text{Ridge}(\alpha=1.0) \cdot W, \quad \tilde{A}_k = A_k - \hat{e}_k(W)$$
2. **Stufe 2 (Orthogonales Modell):**
   Neuronales Netz $f_\theta(W, \tilde{A})$ trainiert mit Binary Cross-Entropy.
3. **Stufe 3 (Duale Kontrafaktische Auswertung):**
   Berechnung sowohl partieller als auch isolierter realistischer Relativrisiken.

---

## 8. Synoptischer Vergleich aller Methoden

| Methode | Modelltyp | Zielvariable | Support-Repräsentation | Kontrafaktischer Modus |
|:---|:---|:---|:---|:---|
| **Ground Truth** | Stochastische Simulation | Makro-Dropout $R$ | Kategorische Blockierung | Partiell ($R_A/R_X$) & Isoliert ($R_{\text{nur}}/R_B$) |
| **Extended Cox** | Semiparametrisch | Log-Hazard | Semester-Zählung | Parametrisch ($\exp(\hat{\beta})$) |
| **DeepSurv Panel/Delta** | Deep Cox MLP | Log-Hazard | Semester-Zählung | Dual: Partiell + Isoliert |
| **Logistic Hazard** | Discrete-Time MLP | Step-Hazard | Semester-Zählung | Dual: Partiell + Isoliert |
| **DeepHit Delta** | Competing Risks GRU | Step-Hazards | Lokale Semester-Zählung | Dual: Partiell + Isoliert |
| **Semester Transformer** | Causal Transformer | Step-Hazard | Lokale Semester-Zählung | Dual: Partiell + Isoliert (alle 3 Typen) |
| **Exam GRU / Delta / V2** | Rekurrentes GRU | Step-Hazard | Prüfungsexposition (vorher/glz) | Dual: Partiell + Isoliert |
| **Deep Exam-Transformer** | Causal / Masked | Step-Hazard / Event | Prüfungsexposition (12 Features) | Dual: Causal Option A & Masked Option B |
| **DML Orthogonal** | Neyman-Orthogonal | Event Probability | Ridge-Residuen auf Zählung | Dual: Partiell + Isoliert |

---

## 9. Diagnose: Warum überfachlicher Support von observationalen Modellen unterschätzt wird

### Der fundamentale Mechanismus
1. **Unbeobachtbarkeit der Hauptmediatoren:** Überfachlicher Support entfaltet seine Schutzwirkung primär über **Motivation (+0,10)** und **soziale Integration (+0,05)**. Beide Variablen sind in realen Hochschuldaten nicht direkt messbar (latent).
2. **Selektionsbias (Reverse Causality):** Studierende, die Workshops für Zeitmanagement oder Lerncoaching aufsuchen, weisen im Schnitt mehr Vorsemester-Fehlversuche und CP-Rückstände auf. Observationale Modelle ohne Confounder-Korrektur interpretieren die Teilnahme fälschlicherweise als Risikoindikator.
3. **Fehlende Notensignatur:** Da überfachliche Angebote keine Modulprüfungsnoten verändern, fehlt der sichtbare positive Noten-Kompensationseffekt, den der fachliche Support hinterlässt.

---

## 10. Umgesetzte Lösungen

1. **Vollständige Zähl-Exposition:** Alle Modelle wurden von binären Schaltern auf quantitative Expositionszählungen (`*_count`, `support_vorher_*`, `support_glz_*`) umgestellt.
2. **Duale Teststränge:** Alle Inferenzskripte reporten konsistent:
   - **Partiell:** $X_{\text{Ziel}} = 0$ vs. beobachtet (entspricht Ground Truth Universum A vs. C/D/E)
   - **Isoliert Realistisch:** Alle $0$ vs. nur Ziel beobachtet (entspricht Ground Truth Universum B vs. F/G/H)
3. **Ground Truth F, G, H:** Die 8-Universen-Matrix liefert die exakten mathematischen Benchmark-Punkte für beide Inferenzstränge.
