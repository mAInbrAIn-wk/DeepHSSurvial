# Audit: Split-Konsistenz & Feature-Kompatibilität V4.1

---

## 1. Split-Konsistenz: Kritischer Befund

> [!CAUTION]
> **Sample-Leakage in 5 Skripten entdeckt!**
> Die folgenden Skripte splitten auf **Zeilen-/Sample-Ebene** statt auf Studenten-Ebene.
> Mehrere Prüfungspaare desselben Studierenden landen in Train UND Test gleichzeitig.

| Skript | Zeile | Problem |
| :--- | :---: | :--- |
| `autoregressive_next_exam.py` | L.137–138 | Splittet `idx = np.arange(n_samples)` → Exam-Paare (k, k+1) desselben Studis in Train+Test |
| `autoregressive_deep_transformer.py` | L.110–111 | Identisches Problem |
| `eval_autoregressive_fail.py` | L.18–19 | Identisches Problem |
| `run_transfer_learning.py` | L.29–30 | Identisches Problem |
| `counterfactual_deepsurv.py` | L.25–26 | Splittet Panel-Zeilen direkt (nicht `unique_studis`) |

### Betroffene Metriken

Die Autoregressive-Modelle melden **ROC-AUC = 0.9202** für Prüfungsbestehen — 
dieser Wert ist durch Leakage inflationiert. Prüfungspaare desselben Studenten teilen 
demografische Features, GPA-Trends und Support-Exposition, was das Modell im Test 
"erkennen" kann.

### Empfohlener Fix

```python
# VORHER (Leakage):
idx = np.arange(n_samples)
tr_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42)

# NACHHER (Group-konsistent):
unique_studis = df['studierenden_id'].unique()
tr_studis, temp_studis = train_test_split(unique_studis, test_size=0.30, random_state=42)
va_studis, te_studis = train_test_split(temp_studis, test_size=0.50, random_state=42)
tr_mask = df['studierenden_id'].isin(tr_studis)
# ... dann Tensoren über Masken filtern
```

> [!WARNING]
> **Entscheidung nötig:** Soll der Fix VOR dem V4.1-Trainingslauf eingespielt werden?
> Das würde die Metriken der Autoregressive-Modelle senken (realistischer), aber
> die Vergleichbarkeit mit den V3.6-Ergebnissen brechen.

---

## 2. Korrekte Splits (Bestätigt) ✅

**Alle übrigen 28 Skripte** sind korrekt:

| Split-Typ | Skripte | Mechanismus |
| :--- | :--- | :--- |
| **Student-Level Panel** | `extended_deep_survival.py`, `dml_orthogonal_survival.py`, `train_transformer_dml.py`, `train_oracle_models.py`, `train_erwerb_blind_models.py`, `plot_calibration_curves.py`, `run_feature_grid_experiments.py` + 5 CF-Skripte | `unique_studis` → `train_test_split` → `.isin()` |
| **Student-Level Tensor** | `recurrent_survival_model.py`, `recurrent_exam_survival.py`, `transformer_survival_model.py`, `transformer_exam_survival.py`, `dynamic_deephit_model.py`, `deep_transformer_regression.py`, alle Timeseries-Skripte | `idx = np.arange(N)` auf 3D-Tensor (1 Student = 1 Eintrag) |
| **Student-Level Landmark** | `train_mlp_baseline.py`, `train_mlp_regression.py`, `deep_survival.py`, `landmark_prediction.py` | 1 Zeile pro Student bei $t_0=2$ |
| **Voller Datensatz** | `extended_cox_survival.py`, `extended_cox_delta.py`, `grade_effect_linear.py` | Ökonometrische Schätzung, kein Split (cluster-robust) |

**Alle verwenden `random_state=42`.** Survival-Modelle zusätzlich stratifiziert auf `studi_events`.

---

## 3. Spaltenkompatibilität V4.1 ↔ aggregate.py ✅

| CSV-Datei | Erwartete Spalten | V4.1 Match |
| :--- | ---: | :---: |
| `studierende.csv` | 15 Spalten | ✅ 100% |
| `pruefungen.csv` | 15 Spalten | ✅ 100% |
| `abschluesse.csv` | 7 Spalten | ✅ 100% |
| `einschreibungen.csv` | 4 Spalten | ✅ 100% |
| `support_teilnahmen.csv` | 3 Spalten | ✅ 100% |
| `support_angebote.csv` | 5 Spalten | ✅ 100% |
| `support_modul_zuordnung.csv` | 3 Spalten | ✅ 100% |
| `module.csv` | 6 Spalten | ✅ 100% |
| `studiengaenge.csv` | 4 Spalten | ✅ 100% |
| `semester.csv` | 5 Spalten | ✅ 100% |

**Fazit:** V4.1 erzeugt exakt die Rohspalten und Formate, die `aggregate.py` erwartet.
Die Support-Flags (`support_glz_fachlich` etc.) werden korrekt zur Laufzeit durch 
JOINs mit `support_teilnahmen` + `support_angebote` + `support_modul_zuordnung` 
berechnet. Keine Namensänderung nötig.

---

## 4. Feature Builder Spaltenreferenzen

### Aus `agg_pruefungen.csv` (18 Spalten)

| Spalte | Verwendung |
| :--- | :--- |
| `studierenden_id`, `fachsemester`, `pruefung_id` | Gruppierung, Zeitachse, Sortierung |
| `bestanden`, `cp`, `cp_attempted`, `note` | Leistungsmerkmale (cp_earned, is_fail, GPA) |
| `schwierigkeit`, `versuch` | Prüfungskontext (nicht in Realistic-Modus) |
| `support_glz_fachlich/uebf/psych` | Gleichzeitige Support-Exposition |
| `support_vorher_fachlich/uebf/psych` | Historische Support-Exposition |
| `hidden_motivation/integration/note` | Latente DGP-Werte (nur Oracle-Modus) |

### Aus `agg_abschluesse.csv` (16 Spalten)

| Spalte | Verwendung |
| :--- | :--- |
| `studierenden_id`, `status`, `studiendauer_semester` | Zielgrößen (is_dropout, Ereigniszeit) |
| `abschlussnote` | Regressionsziel |
| `stg_name` | One-Hot-Encoding (5 Studiengänge) |
| `hzb_note`, `hzb_typ` | Basisfeatures |
| `migrationshintergrund`, `erstakademiker`, `erwerbstaetigkeit_std` | Demografie (fehlt in Realistic) |
| `motivation_initial`, `soziale_integration_initial`, `hidden_*_initial` | Oracle-Landmark |

---

## 5. Zusätzlich fehlende Skripte (über den bisherigen Plan hinaus)

| Skript | Funktion | In `run_overnight.py` |
| :--- | :--- | :---: |
| `landmark_prediction.py` | XGBoost auf Transformer-Embeddings | ❌ |
| `eval_autoregressive_fail.py` | PR-AUC Eval auf Klausur-Nichtbestehen | ❌ |
| `grade_effect_linear.py` | Lineare OLS Notenregression (cluster-robust) | ❌ |
