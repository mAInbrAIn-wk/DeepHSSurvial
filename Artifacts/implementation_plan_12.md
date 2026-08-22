# Implementation Plan: Noten- und Studiendauer-Effekte des Supports

## Hintergrund

Neben dem Dropout-Risiko (bisheriger Fokus) hat Support auch Effekte auf **Prüfungsnoten** und **Studiendauer**. Die Ground Truth dafür steht durch die 8 simulierten Universen bereits fest und wurde extrahiert (gespeichert in [`grade_duration_ground_truth.json`](file:///c:/GitHub_public/Abschlussprojekt/src/output_dl/metrics/grade_duration_ground_truth.json)).

---

## Ground Truth: Noten- & Studiendauer-Effekte (8 Universen, $N = 50.000$)

### Noteneffekte (Absolventen-Abschlussnote)

| Universum | Abschlussnote (Mean) | Diff. vs. A (Partiell) | Diff. vs. B (Isoliert) |
|:---|:---:|:---:|:---:|
| **A (Alle aktiv)** | **2,0277** | Referenz | –0,1352 |
| **B (Kein Support)** | **2,1629** | +0,1352 | Referenz |
| C (Ohne Fachlich) | 2,1177 | +0,0900 | — |
| D (Ohne Überfachlich) | 2,0492 | +0,0215 | — |
| E (Ohne Psychosozial) | 2,0685 | +0,0408 | — |
| **F (Nur Fachlich)** | 2,0871 | — | –0,0758 |
| **G (Nur Überfachlich)** | 2,1575 | — | –0,0054 |
| **H (Nur Psychosozial)** | 2,1270 | — | –0,0359 |

**Interpretation:** Fachlicher Support verbessert die Abschlussnote am stärksten (–0,09 Notenpunkte partiell, –0,08 isoliert), gefolgt von psychosozialem Support (–0,04). Überfachlicher Support hat minimal direkten Noteneffekt (–0,02 partiell, –0,005 isoliert) — das ist konsistent mit seinem indirekten Wirkungspfad über Motivation.

### Bestehensquoten (alle Prüfungen)

| Universum | Bestehensquote | Diff. vs. A (pp) | Diff. vs. B (pp) |
|:---|:---:|:---:|:---:|
| **A** | **88,16%** | Referenz | +5,29pp |
| **B** | **82,87%** | –5,29pp | Referenz |
| C | 86,32% | –1,84pp | — |
| D | 86,46% | –1,70pp | — |
| E | 87,09% | –1,07pp | — |
| **F** | 85,20% | — | +2,33pp |
| **G** | 85,03% | — | +2,16pp |
| **H** | 84,33% | — | +1,46pp |

### Studiendauer (Absolventen)

| Universum | Studiendauer (Mean) | Diff. vs. A |
|:---|:---:|:---:|
| **A** | **8,235** | Referenz |
| **B** | **8,261** | +0,025 |
| C | 8,277 | +0,041 |
| D | 8,245 | +0,010 |
| E | 8,257 | +0,022 |
| F | 8,256 | –0,005 |
| G | 8,291 | +0,030 |
| H | 8,254 | –0,007 |

> [!NOTE]
> **Studiendauer-Effekte sind minimal** (Differenzen <0,05 Semester). Das liegt daran, dass der Median bei allen Universen konstant bei 7,0 Semestern liegt — die Mehrheit der Absolventen studiert in Regelstudienzeit. Support beeinflusst primär, *ob* jemand abschließt, weniger *wann*.

---

## User Review Required

> [!IMPORTANT]
> **Sollen wir die Studiendauer-Analyse trotz der minimalen Ground-Truth-Effekte verfolgen?** Die Noteneffekte sind deutlich aussagekräftiger.

> [!IMPORTANT]
> **Welche bestehenden Modelle sollen für die Notenanalyse priorisiert werden?** Mein Vorschlag:
> 1. **Deep Exam Transformer Regressor** ($R^2 = 0{,}91$) — das beste Notenvorhersagemodell, bereits trainiert
> 2. **Extended Cox mit Notenoutcome** — neue Spezifikation mit der Abschlussnote als Outcome
> 3. **OLS/Ridge-Regression** auf dem aggregierten Panel — einfach, interpretierbar

---

## Proposed Changes

### Phase 1: Ground Truth (✅ Bereits abgeschlossen)

Die Noten- und Studiendauer-Kennzahlen aller 8 Universen wurden extrahiert und in [`grade_duration_ground_truth.json`](file:///c:/GitHub_public/Abschlussprojekt/src/output_dl/metrics/grade_duration_ground_truth.json) gespeichert.

---

### Phase 2: Noteneffekt-Analyse mit bestehenden Modellen

#### [MODIFY] [`src/deep_transformer_regression.py`](file:///c:/GitHub_public/Abschlussprojekt/src/deep_transformer_regression.py)
- Kontrafaktische Noteninferenz hinzufügen: Berechne $\hat{y}_{\text{treated}} - \hat{y}_{\text{control}}$ für jede Prüfung
- Support-Zählvariablen (Indices 3–8) nullsetzen vs. beobachtet lassen → Notendifferenz als Effektmaß
- Dual-Strang (Partiell + Isoliert) analog zur Dropout-Analyse

#### [NEW] `src/counterfactual_grade_transformer.py`
- Kontrafaktisches Inferenzskript für den trainierten Exam Transformer Regressor
- Lädt das gespeicherte Modell, berechnet für jede Prüfung im Testset die vorhergesagte Note mit und ohne Support
- Aggregiert: $\Delta\text{Note}_{\text{support-typ}} = \overline{\hat{y}(\text{treated})} - \overline{\hat{y}(\text{control})}$
- Dual-Strang-Reporting (Partiell + Isoliert Realistisch)
- Speichert Ergebnisse in `output_dl/metrics/counterfactual_grade_transformer_metrics.json`

#### [NEW] `src/grade_effect_linear.py`
- Einfaches lineares Modell (OLS oder Ridge) auf dem aggregierten Panel
- Outcome: Prüfungsnote
- Treatment: `support_vorher_*`, `support_glz_*` (Zählvariablen)
- Confounder: `schwierigkeit`, `cp`, `versuch`, `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`
- Koeffizienten direkt als Effektschätzer interpretierbar

---

### Phase 3: Bugfix Exam GRU V2 Delta

#### [MODIFY] [`src/counterfactual_rnn_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_rnn_delta.py#L64-L72)
- Feature-Indizes korrigieren: `ALL_SUPP_IDXS = [3,4,5,6,7,8]` statt `[6,7,8,9,10,11]`
- Index-Zuordnung: Fachlich = (3,4), Überfachlich = (5,6), Psychosozial = (7,8)
- Neulauf zur Erzeugung korrigierter RR-Werte

---

### Phase 4: Fehlende Werte auffüllen

#### [MODIFY] [`src/dml_orthogonal_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py)
- Neulauf des DML-Skripts mit bereits implementiertem Dual-Strang → erzeugt die fehlenden Isoliert-Werte

#### [MODIFY] [`src/extended_cox_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_survival.py)
- Isolierte HR-Werte explizit im JSON exportieren (identisch mit partiellen, da log-linear additiv)

---

## Verification Plan

### Automated Tests
```bash
python src/counterfactual_grade_transformer.py
python src/grade_effect_linear.py
python src/counterfactual_rnn_delta.py  # korrigiert
python src/dml_orthogonal_survival.py   # Neulauf
```

### Manual Verification
- Vergleich der geschätzten Noteneffekte mit Ground Truth aus [`grade_duration_ground_truth.json`](file:///c:/GitHub_public/Abschlussprojekt/src/output_dl/metrics/grade_duration_ground_truth.json)
- Exam GRU V2 Delta RR-Werte sollten nach Korrektur im Bereich [0,95; 1,05] liegen
- DML Isoliert-Werte sollten nach Neulauf vorhanden sein
