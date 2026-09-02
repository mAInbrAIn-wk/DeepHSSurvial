# Implementation Plan V4: Oracle-Counterfactuals, Feature-Enrichment & Noteneffekte

## Zusammenfassung

Dieser Plan integriert die Selektionsbias-Maßnahmen (Priorität 1), Oracle-Counterfactuals, Feature-Erweiterung der rekurrenten Modelle, Noteneffekt-Analyse und Bugfixes in einem kohärenten Nachtlauf.

---

## Klärung der Annotationen

### 1. Überfachlicher Support – Feedback-Schleife & Oracle-Option

Sie haben richtig beobachtet: Überfachlicher Support wirkt über **exakt denselben** Pfad wie psychosozialer (beide heben Motivation und soz. Integration an), aber beim überfachlichen gibt es einen stärkeren **Selektionsbias-Feedback-Loop**: $p_{\text{uebf}} = 0{,}05 + (0{,}5 - \mu) \cdot 0{,}15$ — niedrige Motivation *verursacht* die Supportinanspruchnahme UND das Dropout-Risiko. Das macht die kausale Identifikation schwieriger als beim psychosozialen Support, wo die Selektionsvariable ($\sigma$) im DGP einen geringeren Gewichtskoeffizienten hat.

Die Oracle-Option wird diesen Effekt besonders eindrücklich zeigen: Wenn das Oracle-Modell mit beobachteter $\mu(t-1)$ die Konfundierung auflöst, sollte der überfachliche HR deutlich <1,0 werden.

### 2. „Hinausgezögertes Leiden" – Dropout-Dauer (Ergebnis)

Die Analyse zeigt einen **klaren und überraschenden Effekt**:

| Universum | Dropout-Dauer (Mean) | Diff. vs. A |
|:---|:---:|:---:|
| **A (Alle aktiv)** | **4,479 Sem.** | Referenz |
| **B (Kein Support)** | **4,944 Sem.** | **+0,465 Sem.** |
| C (Ohne Fachlich) | 4,663 | +0,183 |
| D (Ohne Überfachlich) | 4,615 | +0,136 |
| E (Ohne Psychosozial) | 4,509 | +0,030 |
| F (Nur Fachlich) | 4,677 | +0,198 |
| G (Nur Überfachlich) | 4,741 | +0,262 |
| H (Nur Psychosozial) | 4,868 | +0,389 |

> [!IMPORTANT]
> **Kein „hinausgezögertes Leiden" — das Gegenteil!** Support *verkürzt* die Dropout-Dauer um fast ein halbes Semester (A: 4,48 vs. B: 4,94). Studierende mit Support, die dennoch abbrechen, tun dies *schneller*. Das deutet darauf hin, dass Support den Entscheidungsprozess beschleunigt: Entweder das Studium wird stabilisiert (Abschluss) oder die Erkenntnis kommt schneller, dass es nicht passt.

### 3. Bestehensquoten-Modelle

Ja, bestehende Modelle können auf Bestehensquoten angesetzt werden! Die Modifikationen sind **gering**:
- Der **Deep Exam Transformer Regressor** sagt bereits Noten voraus → Bestanden = (Note ≤ 4,0)
- Ein einfaches **logistisches Modell** auf dem Prüfungspanel kann die Bestehenswahrscheinlichkeit direkt schätzen
- Die kontrafaktische Logik ist identisch (Support nullsetzen, Bestehensrate vergleichen)

### 4. Median = 1,0: Verifiziert & Erklärt

Die Zahlen sind **verifiziert** — alle Modelle (außer DML) zeigen Median $= 1{,}0000$ exakt. Der Grund:

| Datensatz | Support-Häufigkeit (nonzero) |
|:---|:---:|
| `support_glz_fachlich` | **4,2%** der Prüfungen |
| `support_glz_ueberfachlich` | **8,8%** der Prüfungen |
| `support_glz_psychosozial` | **7,0%** der Prüfungen |
| `support_vorher_fachlich` | 6,4% |
| `support_vorher_ueberfachlich` | 22,9% |
| `support_vorher_psychosozial` | 19,5% |

Da in der kontrafaktischen Berechnung $RR_i = p(\text{treated}_i) / p(\text{control}_i)$ gilt und für >90% der Semester $X_{\text{treated}} = X_{\text{control}} = 0 \implies RR_i = 1{,}0$, ist der Median notwendigerweise 1,0. Das DML-Modell ist die Ausnahme (Median ≠ 1,0), weil es mit **Residuen** arbeitet ($\tilde{A} = A - \hat{E}[A|W]$), die auch bei $A=0$ von Null verschieden sind.

Trotz der Spärlichkeit findet der **Mean** dennoch einen Effekt, weil die wenigen Studierenden mit Support (4–23%) deutlich veränderte Hazards aufweisen. Das ist analog zu einer klinischen Studie, in der nur ein Teil der Kohorte behandelt wird.

### 5. Feature-Erweiterung: Sichere Proxies (Punkt 9 aus §4.2)

Die Feature-Gap-Analyse hat konkret identifiziert, was **ohne Leakage** ergänzt werden kann:

**Aktuell fehlend in Semester-GRU/Transformer, aber sicher hinzufügbar:**

| Variable | Proxy für | Leakage-Sicherheit | Bereits verfügbar in |
|:---|:---|:---|:---|
| `erstakademiker` | $\sigma(0)$, Selektionsbias | ✅ Baseline ($t=0$) | `agg_abschluesse.csv` |
| `cp_rueckstand` | Dropout-Treiber | ✅ Gelaggt ($t-1$) | Nur in `_delta`-Varianten |
| `cum_fails_vorher` | Akkumulierter Leistungsstress | ✅ Gelaggt ($t-1$) | Berechenbar |
| `delta_gpa_hzb` | Motivations-Proxy (Notenüberraschung) | ✅ Gelaggt ($t-1$) | Berechenbar |

**Zusätzlich möglich, aber mit Vorsicht:**

| Variable | Status |
|:---|:---|
| `cum_support_uebf_prev` | Kumulierte Hilfesuche → Motivationsproxy, aber Feedback-Loop! |
| `is_inactive_prev` | Inaktivitätsflag → Motivationsdrain-Proxy |
| `migrationshintergrund` | Baseline-Proxy für $\sigma(0)$ |

Die Positional Encodings im Transformer erfassen die Semester-Zeitinformation bereits korrekt. Die Support-Features sind in Semester-Modellen aktuell nur `glz` (gleichzeitig), während Exam-Modelle **beide** (`vorher` + `glz`) nutzen.

---

## Proposed Changes (nach Priorität geordnet)

### Phase 0: Bugfix Exam GRU V2 Delta (sofort)

#### [MODIFY] [`src/counterfactual_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py#L64-L72)
- Feature-Indizes korrigieren: `ALL_SUPP_IDXS = [3,4,5,6,7,8]` statt `[6,7,8,9,10,11]`
- Fachlich = (3,4), Überfachlich = (5,6), Psychosozial = (7,8)
- Neulauf erzeugt korrekte RR-Werte

---

### Phase 1: Oracle-Counterfactuals (Priorität 1)

#### [NEW] `src/counterfactual_oracle_logistic_hazard.py`
- Trainiert ein **Oracle Logistic Hazard** mit den 3 hidden Features (`hidden_motivation_prev`, `hidden_soziale_integration_prev`, `hidden_erwartete_note_prev`) zusätzlich zum Standard-Featureset
- Führt dieselbe Dual-Strang kontrafaktische Evaluation durch wie [`counterfactual_rr_logistic_hazard_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rr_logistic_hazard_delta.py)
- Speichert: `output_dl/metrics/counterfactual_oracle_logistic_hazard_metrics.json`
- Erwartung: Die Oracle-HRs sollten deutlich näher an der Ground Truth liegen → Beweist den kausalen Identifikationslift

#### [MODIFY] [`src/train_oracle_models.py`](file:///c:/GitHub_public/Abschlussprojekt/src/train_oracle_models.py)
- Modell speichern: `output_dl/models/oracle_logistic_hazard.keras` + Preprocessor (für spätere Counterfactual-Nutzung)
- Analoge Speicherung für Oracle DeepSurv

#### [NEW] `src/counterfactual_oracle_deepsurv.py`
- Oracle DeepSurv HR-Analyse (analog zu `counterfactual_hr_delta.py`)
- Speichert: `output_dl/metrics/counterfactual_oracle_deepsurv_metrics.json`

---

### Phase 2: Feature-Erweiterung der Semester-Sequenzmodelle

#### [MODIFY] [`src/recurrent_survival_model_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/recurrent_survival_model_delta.py)
- Feature-Vektor von 9 → 13 Features erweitern:
  - `[9]` `erstakademiker` (Baseline)
  - `[10]` `cum_fails_vorher` (kumulierte Fehlversuche, gelaggt $t-1$)
  - `[11]` `delta_gpa_hzb` ($\text{GPA}_{t-1} - \text{hzb\_note}$, Motivations-Proxy)
  - `[12]` `migrationshintergrund` (Baseline)
- Keine Änderung der GRU-Architektur nötig (Inputdimension passt sich an)

#### [MODIFY] [`src/transformer_survival_model.py`](file:///c:/GitHub_public/Abschlussprojekt/src/transformer_survival_model.py)
- Analog: Support-Features von binär auf Zählung umstellen + dieselben 4 Proxy-Features ergänzen
- `cp_rueckstand` hinzufügen (fehlt komplett im Transformer, ist aber in `_delta` vorhanden)
- Feature-Vektor: 8 → 14 Features

#### [MODIFY] [`src/counterfactual_rnn_semester_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_semester_delta.py)
- Indizes der Counterfactual-Manipulation an erweiterten Featurevektor anpassen

#### [MODIFY] [`src/counterfactual_inference_semester_transformer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_inference_semester_transformer.py)
- Indizes anpassen

---

### Phase 3: Noteneffekt-Analyse

#### [NEW] `src/counterfactual_grade_transformer.py`
- Kontrafaktische Noteninferenz mit dem trainierten Exam Transformer Regressor ($R^2 = 0{,}91$)
- Berechne $\Delta\text{Note} = \overline{\hat{y}(\text{treated})} - \overline{\hat{y}(\text{control})}$
- Dual-Strang (Partiell + Isoliert)
- Speichert: `output_dl/metrics/counterfactual_grade_transformer_metrics.json`

#### [NEW] `src/grade_effect_linear.py`
- OLS-Regression auf dem Prüfungspanel
- Outcome: Prüfungsnote
- Treatment: `support_vorher_*`, `support_glz_*` (Zählvariablen)
- Confounder: `schwierigkeit`, `cp`, `versuch`, `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`
- Koeffizienten direkt als Noteneffekt interpretierbar

#### [NEW] `src/pass_rate_analysis.py`
- Bestehensquoten-Modell: Logistisches Modell auf Prüfungsebene
- Outcome: `bestanden` (0/1)
- Gleiche Treatment- und Confounder-Struktur wie `grade_effect_linear.py`
- Kontrafaktisch: Support nullsetzen → Bestehensrate vergleichen
- Speichert: `output_dl/metrics/pass_rate_counterfactual_metrics.json`

---

### Phase 4: Fehlende Werte auffüllen

#### [MODIFY] [`src/dml_orthogonal_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py)
- Neulauf → erzeugt die bereits implementierten Isoliert-Werte

#### [MODIFY] [`src/extended_cox_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_survival.py)
- Isolierte HR-Werte explizit im JSON exportieren (identisch mit partiellen: log-linear additiv)

---

### Phase 5: Orchestrierung & Dokumentation

#### [MODIFY] [`src/run_retrain_all.py`](file:///c:/GitHub_public/Abschlussprojekt/src/run_retrain_all.py)
- Neue Schritte einfügen:
  - Schritt 13b: Oracle-Modelle speichern (Keras + Preprocessor)
  - Schritt 23: `counterfactual_oracle_logistic_hazard.py`
  - Schritt 24: `counterfactual_oracle_deepsurv.py`
  - Schritt 25: `counterfactual_grade_transformer.py`
  - Schritt 26: `grade_effect_linear.py`
  - Schritt 27: `pass_rate_analysis.py`

#### [MODIFY] [`Artifacts/script_registry.md`](file:///c:/GitHub_public/Abschlussprojekt/Artifacts/script_registry.md)
- Neue Skripte eintragen:
  - `counterfactual_oracle_logistic_hazard.py` — Oracle-Counterfactual LH
  - `counterfactual_oracle_deepsurv.py` — Oracle-Counterfactual DeepSurv
  - `counterfactual_grade_transformer.py` — Noteneffekt Transformer
  - `grade_effect_linear.py` — Lineares Notenmodell
  - `pass_rate_analysis.py` — Bestehensquoten-Analyse
- Feature-Erweiterung der GRU/Transformer dokumentieren

#### [MODIFY] [`README.md`](file:///c:/GitHub_public/Abschlussprojekt/README.md)
- Synopse-Tabelle auf V4 aktualisieren (8 Universen, Oracle-Ergebnisse, Noteneffekte)
- 8-Universen-Design dokumentieren (F, G, H)
- Kausales Diagramm aus der Selektionsbias-Analyse einbinden

#### [NEW] [`Artifacts/simulation_kausal_doku.md`](file:///c:/GitHub_public/Abschlussprojekt/Artifacts/simulation_kausal_doku.md)
- Formale Dokumentation der Simulation mit dem Kausaldiagramm und mathematischen Formeln
- Direkte Übernahme aus [`selektionsbias_analyse.md`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/selektionsbias_analyse.md) §1

---

## Ground Truth: Noten-, Bestehens- & Dropout-Dauer-Effekte

### Noteneffekte (Absolventen)

| Support-Typ | Partieller Effekt (A vs. ohne) | Isolierter Effekt (nur vs. B) |
|:---|:---:|:---:|
| **Fachlich** | **–0,0900 Notenpunkte** | **–0,0758** |
| **Überfachlich** | –0,0215 | –0,0054 |
| **Psychosozial** | –0,0408 | –0,0359 |
| **Alle zusammen (A vs. B)** | **–0,1352** | — |

### Bestehensquoten

| Support-Typ | Partieller Effekt (pp) | Isolierter Effekt (pp) |
|:---|:---:|:---:|
| **Fachlich** | +1,84pp | +2,33pp |
| **Überfachlich** | +1,70pp | +2,16pp |
| **Psychosozial** | +1,07pp | +1,46pp |
| **Alle zusammen** | **+5,29pp** | — |

### Dropout-Dauer

| Universum | Dropout-Dauer (Mean) | Diff. vs. A |
|:---|:---:|:---:|
| A (Alle aktiv) | 4,479 Sem. | Referenz |
| B (Kein Support) | 4,944 Sem. | +0,465 Sem. |

---

## Verification Plan

### Automatisierte Tests
```bash
# Phase 0: Bugfix
python src/counterfactual_rnn_delta.py

# Phase 1: Oracle
python src/counterfactual_oracle_logistic_hazard.py
python src/counterfactual_oracle_deepsurv.py

# Phase 2: Feature-Erweiterung → Retrain + Counterfactual
python src/recurrent_survival_model_delta.py   # retrain
python src/transformer_survival_model.py       # retrain
python src/counterfactual_rnn_semester_delta.py # re-evaluate
python src/counterfactual_inference_semester_transformer.py

# Phase 3: Noten
python src/counterfactual_grade_transformer.py
python src/grade_effect_linear.py
python src/pass_rate_analysis.py

# Phase 4: Fehlende Werte
python src/dml_orthogonal_survival.py
```

### Erwartete Validierung
- Oracle LH-HRs sollten deutlich näher an Ground Truth liegen als Standard-LH
- Exam GRU V2 nach Bugfix: RR-Werte im Bereich [0,95; 1,05]
- Noteneffekte: Fachlich-Koeffizient ≈ –0,09 (Ground Truth)
- Feature-erweiterte GRU/Transformer: Bessere Überfachlich-Schätzung durch Erstakademiker-Proxy
