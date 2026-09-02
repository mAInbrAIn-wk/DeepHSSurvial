# Modelltraining V4.1 — Aktualisierter Plan

> [!IMPORTANT]
> **Ziel:** Komplette Modellsuite auf V4.1 Baseline (S01, Universum A, N=50.000) trainieren.
> **Alle 5 Feature-Modi:** standard, gradeblind, blind, oracle, realistic.

---

## 1. Datenlage ✅

- **V4.1 Rohdaten:** `src/output_v4_grid_v41/S01_baseline/universe_A/`
- **Aggregation:** `agg_abschluesse.csv` (50.000 Zeilen) und `agg_pruefungen.csv` (852.368 Zeilen) erfolgreich erzeugt via `aggregate.py` (Pandas-Backend)
- **Feature Builder:** Alle 3 Formate getestet und kompatibel:
  - Semester Tensor: (50000, 16, 18) ✅
  - Exam Tensor: (50000, 40, 24) ✅  
  - Landmark: (47973, 81) ✅

## 2. Spaltenkompatibilität ✅

> [!NOTE]
> **Ergebnis der Prüfung:** Die rohen CSVs (`pruefungen.csv`) haben exakt identische
> Spalten in V3.6 und V4.1. Die abgeleiteten Spalten (`support_glz_fachlich`,
> `support_glz_ueberfachlich`, `support_glz_psychosozial`, `support_vorher_*`)
> werden NICHT im Export erzeugt, sondern von `aggregate.py` durch JOINs mit
> `support_teilnahmen.csv` + `support_angebote.csv` + `support_modul_zuordnung.csv`
> **zur Laufzeit** berechnet. Dieser Mechanismus ist identisch für V3.6 und V4.1.
>
> **Fazit:** Keine Namensänderung nötig. Die Aggregation produziert identische Features.

## 3. Split-Konsistenz ✅

> [!NOTE]
> **Audit-Ergebnis:** Alle Modellskripte verwenden `random_state=42` für den Split.
>
> **Gruppenkonsistenz bei Panel-Daten:** ✅ Korrekt implementiert.
> Panel-Modelle splitten auf **`unique_studis`** (Studenten-Ebene), dann filtern
> sie das Panel mit `.isin(train_ids)`. Kein Student erscheint in mehreren Sets.
>
> **Sequenz-Modelle:** ✅ Splitten auf Student-Index (`idx = np.arange(len(studis))`),
> da jeder Student genau ein Eintrag im 3D-Tensor ist.
>
> **Split-Proportionen:** Konsistent 70/15/15 (die meisten) oder 80/20 (einige Landmark).
> Stratifizierung auf `studi_events` wo möglich.

## 4. Vollständige Modellsuite (28 Schritte)

### In `run_overnight.py` bereits enthalten (20 Schritte):

| # | Modell | Skript |
| :---: | :--- | :--- |
| 1 | Extended Cox PH | `extended_cox_survival.py` |
| 2 | Extended DeepSurv & LogHaz (Panel) | `extended_deep_survival.py` |
| 3 | Recurrent Semester Survival GRU | `recurrent_survival_model.py` |
| 4 | Dynamic DeepHit Competing Risks | `dynamic_deephit_model.py` |
| 5 | Causal Semester Transformer Survival | `transformer_survival_model.py` |
| 6 | Recurrent Exam Survival GRU | `recurrent_exam_survival.py` |
| 7 | Causal Exam Transformer Survival | `transformer_exam_survival.py` |
| 8 | Landmark Baselines (RF, SVM, NB, MLP) | `train_mlp_baseline.py` |
| 9 | Landmark Noten-Regression | `train_mlp_regression.py` |
| 10 | DML Orthogonal Survival | `dml_orthogonal_survival.py` |
| 11 | Deep Transformer-DML | `train_transformer_dml.py` |
| 12 | Semester LSTM GPA Regression | `timeseries_semester.py` |
| 13 | Semester Transformer Regression | `timeseries_semester_transformer.py` |
| 14 | Exam GRU Grade Regression | `timeseries_exam.py` |
| 15 | Exam Transformer Grade Regression | `timeseries_exam_transformer.py` |
| 16 | Oracle Models | `train_oracle_models.py` |
| 17 | DSGVO Realistic Models | `train_erwerb_blind_models.py` |
| 18 | Deep Transformer Suite (d=128) | `deep_transformer_regression.py` |
| 19 | Autoregressive Next-Exam (GRU Dual-Head) | `autoregressive_next_exam.py` |
| 20 | Strukturelle Mediationsanalyse | `structural_mediation_analysis.py` |

### Fehlend in `run_overnight.py` — muss ergänzt werden:

| # | Modell | Skript | Quelle |
| :---: | :--- | :--- | :--- |
| 21 | **Deep Landmark DeepSurv & LogHaz** | `deep_survival.py` | `run_all` Schritt 13 |
| 22 | **Dynamic DeepHit Delta** | `dynamic_deephit_delta_model.py` | `run_all` Schritt 12 |
| 23 | **Extended Deep Survival Delta** | `extended_deep_survival_delta.py` | `run_all` Schritt 15 |
| 24 | **Extended Exam Survival** | `extended_exam_survival.py` | `run_all` Schritt 16 |
| 25 | **Extended Cox Delta** | `extended_cox_delta.py` | `run_all` Schritt 16b |
| 26 | **Recurrent Survival Delta** | `recurrent_survival_model_delta.py` | `run_all` Schritt 18 |
| 27 | **Recurrent Exam Survival Delta** | `recurrent_exam_survival_delta.py` | `run_all` Schritt 19 |
| 28 | **Recurrent Exam Survival V2** | `recurrent_exam_survival_v2.py` | `run_all` Schritt 19b |
| 29 | **Deep Transformer Autoregressor** | `autoregressive_deep_transformer.py` | Nicht in run_all |
| 30 | **Kalibrierungskurven** | `plot_calibration_curves.py` | `run_all` Schritt 21 |
| 31 | **Feature Grid Sweep (5 Modi)** | `run_feature_grid_experiments.py` | Separater Runner |

### Counterfactual-Analysen (benötigen Multi-Universum-Daten):

| # | Modell | Skript | Braucht |
| :---: | :--- | :--- | :--- |
| CF1 | Counterfactual HR Delta | `counterfactual_hr_delta.py` | Uni A + B |
| CF2 | Counterfactual RR LogHaz Delta | `counterfactual_rr_logistic_hazard_delta.py` | Uni A + B |
| CF3 | Counterfactual RR DeepHit Delta | `counterfactual_rr_deephit_delta.py` | Uni A + B |
| CF4 | Counterfactual Semester Transformer | `counterfactual_inference_semester_transformer.py` | Uni A + B |
| CF5 | Counterfactual Exam RNN Delta | `counterfactual_rr_exam_rnn_delta.py` | Uni A + B |

> [!WARNING]
> **Counterfactual-Analysen brauchen Daten aus mehreren Universen** (mindestens A + B).
> Dafür muss die Aggregation auch für Universum B (und ggf. C–H) laufen.
> Soll das Teil dieses Laufs sein, oder separat?

## 5. Vorgeschlagene Vorgehensweise

### Schritt 1: `run_overnight.py` erweitern
Die 8 fehlenden Modelle (21–28) und den Deep Transformer Autoregressor (29) 
als zusätzliche Schritte in `run_overnight.py` integrieren.

### Schritt 2: Feature Grid Sweep
`run_feature_grid_experiments.py` separat mit allen 5 Modi starten.
Geschätzte Zusatzzeit: ~2–3h.

### Schritt 3: Counterfactual-Analysen (optional)
Aggregation für Uni B laufen lassen, dann CF1–CF5.

### Geschätzte Gesamtlaufzeit
- Erweiterte Suite (31 Schritte): ~6–8h
- Feature Grid (5 Modi × 4 Modellklassen): ~2–3h
- Counterfactual (optional): ~1h
- **Gesamt: ~10–12h**
