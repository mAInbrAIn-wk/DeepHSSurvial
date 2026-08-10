# Implementation Plan: Vollständige Counterfactual-Analyse & DeepHit Delta

## Bestandsaufnahme: Was existiert, was fehlt

### Aktuelle Counterfactual-Skripte (Zustand vor diesem Plan)

| Skript | Modell | Features | Logging | Problem |
|:---|:---|:---|:---:|:---|
| `counterfactual_inference.py` | `recurrent_exam_survival.keras` (GRU, Prüfungsebene) | Kumulativer `fach_supp_cum` | ✅ JSON | Behandelt nur fachlichen Support; kumulatives Feature |
| `counterfactual_rnn.py` | `recurrent_exam_survival_v2.keras` (GRU v2) | Kumulativer `fach_supp`, `uebf_supp`, `psych_supp` | ❌ kein Logging | Setzt alle 3 Typen simultan auf 0/1 (kein einzelner Effekt); kein Logging |
| `counterfactual_inference_deephit.py` | `dynamic_deephit_competing.keras` | Kumulativer `fach_supp` (Feature-Index 3) | ❌ kein Logging | Nur fachlicher Support; kumulatives Feature; kein Logging |
| `counterfactual_deepsurv.py` | `deepsurv_landmark.keras` (statisches Landmark) | Statische `Fach_supp`, `Uebf_supp`, `Psych_supp` | ❌ kein Logging | Statisches Modell; kein Panel |
| `counterfactual_hr_analyzer.py` | `extended_deepsurv_panel.keras` | Fach-Cum-Flag (vorher fehlende Features) | ✅ JSON+MD | Bereits mit Cum-Features, nicht mit Delta |
| `counterfactual_hr_delta.py` (**neu**) | `extended_deepsurv_delta.keras` | **Delta: `fach_supp_active`, `fails_prev`, `cp_rueckstand`** | ✅ JSON+MD | ✅ Korrekt — Referenzimplementierung |

### Dynamic DeepHit: Was ist bereits da?
Das Modell `dynamic_deephit_model.py` enthält:
- GRU-Backbone mit zwei Output-Köpfen (Dropout-Risiko + Abschlusswahrscheinlichkeit)
- Aber: **Feature-Konstruktion (Zeile 84–86) verwendet kumulativen `cum_fach`, `cum_uebf`, `cum_psych`** — genau dasselbe Problem wie bei den Panel-Modellen
- Format: Sequenz-Tensor `(N, T, F)` mit Padding — nicht das Person-Semester-Panel-Format

**Was fehlt für DeepHit Delta:**
1. `build_competing_risks_dataset_delta()` — Datenkonstruktion mit semester-aktiven Support-Flags und `fails_prev`, `delta_cp_prev`, `cp_rueckstand` als Features
2. `dynamic_deephit_delta_model.py` — Neues Modell trainiert auf Delta-Features
3. `counterfactual_rr_deephit_delta.py` — Wrapper für Relatives Risiko (RR) aus DeepHit Delta

---

## Proposed Changes

> [!IMPORTANT]
> Alle neuen Skripte schreiben in **neue** Dateien. Die bestehenden Counterfactual-Skripte werden **nicht verändert** (historische Vergleichbarkeit).

---

### Ebene A: Fixes & Logging für bestehende Wrapper (ohne Modell-Retrain)

#### [NEW] `counterfactual_rnn_delta.py`
Erneuerter Wrapper für `recurrent_exam_survival_v2.keras`. Wesentliche Änderung gegenüber `counterfactual_rnn.py`:
- Testet jeden der drei Support-Typen **separat** (nicht simultan)
- Gibt für jeden Typ Mean RR, Median RR, Q05–Q95 CI aus
- Logging via `save_metrics("counterfactual_rnn_delta", ...)` als JSON + MD

> [!NOTE]
> Das zugrundeliegende Modell wurde mit kumulativen Features trainiert. Das RR, das dieser Wrapper berechnet, ist also ein Basisvergleich, **keine kausale Schätzung**. Es wird entsprechend im Logging gekennzeichnet.

#### [NEW] `counterfactual_deephit_fixed.py`
Erneuerter Wrapper für `dynamic_deephit_competing.keras` (bestehendes Modell). Änderungen:
- Testet fachlichen, überfachlichen und psychosozialen Support **einzeln**
- Logging via `save_metrics("counterfactual_deephit_fixed", ...)` als JSON + MD
- Gleiche Einschränkung: Basis-Modell mit kumulativen Features

---

### Ebene B: Dynamic DeepHit Delta (Neues Modell + Wrapper)

#### [NEW] `dynamic_deephit_delta_model.py`
Neues Trainings-Skript für das DeepHit Competing-Risks Modell auf Delta-Features:

**Feature-Konstruktion** (in `build_competing_risks_dataset_delta()`):
- Sequenz-Features (pro Semester, gelaggt): `sem_gpa`, `delta_cp_prev`, `fails_prev`, `cp_rueckstand`
- Semester-aktive Treatment-Features: `fach_supp_active`, `uebf_supp_active`, `psych_supp_active`
- Statische Features (wiederholt): `hzb_note`, `erwerbstaetigkeit_std`

**Modell-Architektur:** Identisch mit bestehendem DeepHit (GRU + 2 Heads), aber auf `F_delta` Input-Features

**Output:** Speichert `dynamic_deephit_delta.keras` + Metriken (JSON + MD) + Plots

#### [NEW] `counterfactual_rr_deephit_delta.py`
Wrapper für das neue DeepHit-Delta-Modell. Berechnet für jeden Support-Typ separat das **Relative Risiko** des Dropout-Heads:

$$RR = \frac{\bar{p}_1}{\bar{p}_0}, \quad \text{oder individuell: } RR_i = \frac{p_{1,i}}{p_{0,i}}$$

Gibt Mean RR, Median RR, Q05–Q95 CI für alle 3 Support-Typen aus.
Logging: `counterfactual_rr_deephit_delta_metrics.json` + `.md`

---

### Ebene C: Counterfactual für Extended Logistic Hazard Delta

#### [NEW] `counterfactual_rr_logistic_hazard_delta.py`
Wrapper für `extended_logistic_hazard_delta.keras` (ROC-AUC 0.799 — das stärkste Panel-Modell). Da dieses Modell Wahrscheinlichkeiten $p$ ausgibt, berechnen wir das **Relative Risiko** statt der HR:

$$RR_i = \frac{p^{(1)}_i}{p^{(0)}_i}$$

Alle 3 Support-Typen einzeln. Logging: `counterfactual_rr_logistic_hazard_delta_metrics.json` + `.md`.

---

### Ebene D: Portfolio Quick Wins (Review-Feedback)

1. **`.gitignore` bereinigen**: Hinzufügen von `__pycache__/`, `*.ipynb_checkpoints/`, `*.pyc`. Löschen eventuell eingecheckter Caches.
2. **Archivierung redundanter Skripte**:
   - Verschieben von `run_remaining_experiments.py`, `recurrent_exam_survival_v2.py` und anderen obsoleten Skripten in einen neuen Ordner `src/archive/`.
3. **PH-Annahmen-Test (Schoenfeld-Residuen)**:
   - Erweitern von `extended_cox_delta.py` um einen Schoenfeld-Residuentest am Ende des Skripts mit `results.resid_schoenfeld` (aus `statsmodels`), um die Proportional-Hazards-Annahme nativ zu überprüfen.
4. **README-Update zum Dashboard**:
   - Den Vermerk "Das ehemals verwendete Dash-Dashboard befindet sich derzeit im Umbau." um die ehrliche Kennzeichnung ergänzen, dass es sich um *Work in Progress* handelt und die aktuelle Version buggy/problematisch ist (besonders in Bezug auf die neuen Delta-Modelle).

---

## Gesamtübersicht nach Abschluss

| Wrapper (neu) | Modell | Features | Metrik | Panel? |
|:---|:---|:---|:---|:---|
| `counterfactual_rnn_delta.py` | GRU v2 (kum.) | kumulativ | RR | Prüfungsfolge |
| `counterfactual_deephit_fixed.py` | DeepHit (kum.) | kumulativ | RR | Semesterfolge |
| `counterfactual_rr_logistic_hazard_delta.py` | **Logistic Hazard Δ** (ROC 0.799) | **delta** | **RR** | Person-Semester |
| `counterfactual_hr_delta.py` (**done**) | DeepSurv Δ | **delta** | **HR** | Person-Semester |
| `counterfactual_rr_deephit_delta.py` | **DeepHit Δ** (neu) | **delta** | **RR** | Semesterfolge |
| `dynamic_deephit_delta_model.py` | **DeepHit Δ** (neu trainiert) | **delta** | ROC/PR/Brier | Semesterfolge |

---

## Verification Plan

```bash
python counterfactual_rnn_delta.py                    # RR für GRU v2
python counterfactual_deephit_fixed.py                # RR für DeepHit (kumulativ)
python counterfactual_rr_logistic_hazard_delta.py     # RR für Logistic Hazard Delta
python dynamic_deephit_delta_model.py                 # Training DeepHit Delta
python counterfactual_rr_deephit_delta.py             # RR für DeepHit Delta
```

### Erwartete neue Metrik-Dateien
- `counterfactual_rnn_delta_metrics.json/.md`
- `counterfactual_deephit_fixed_metrics.json/.md`
- `counterfactual_rr_logistic_hazard_delta_metrics.json/.md`
- `dynamic_deephit_delta_metrics.json/.md`
- `counterfactual_rr_deephit_delta_metrics.json/.md`
