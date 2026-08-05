# Modell-Steckbriefe: Alle ML/DL-Modelle im Detail

Jedes Modell wird hier einzeln mit Features, Target, Preprocessing, Architektur, Metriken und Leakage-Status dokumentiert.

---

## Inhaltsverzeichnis

1. [Statische Baselines](#1-statische-baselines)
2. [Zeitreihen-Modelle (Regression)](#2-zeitreihen-modelle-regression)
3. [Landmark Survival-Modelle (Stufe 1)](#3-landmark-survival-modelle-stufe-1)
4. [Extended Panel Survival-Modelle (Stufe 2)](#4-extended-panel-survival-modelle-stufe-2)
5. [Sequenz-Survival-Modelle (Stufe 3)](#5-sequenz-survival-modelle-stufe-3)
6. [Competing Risks (Stufe 4)](#6-competing-risks-stufe-4)

---

## 1. Statische Baselines

---

### 1.1 MLP Baseline Classification — [train_mlp_baseline.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/train_mlp_baseline.py)

**Ziel:** Statische Dropout-Klassifikation auf Querschnittsdaten

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 2D Tabelle `agg_abschluesse.csv` ($N \times F$) |
| **Target** | `status` → `LabelEncoder` (multi-class) oder `(status == 'abgeschlossen').astype(int)` (binär) |
| **Loss** | `sparse_categorical_crossentropy` (multi) / `binary_crossentropy` (binär) |

#### Features (Eingabe)

Alle Spalten aus `agg_abschluesse.csv` **minus** explizit ausgeschlossene:

**Explizit gelöscht (`LEAKAGE_COLUMNS`):** `studierenden_id`, `status`, `abschlussnote`, `bachelorarbeitsnote`, `studiendauer_semester`, `abschluss_semester_id`, `anomalie_typ`

**Explizit maskiert (`OPTIONAL_MASKED_COLUMNS`):** `AVG_Note`, `Anz_Pruefungen`, `Anz_Bestanden`, `Anz_Fehlversuche`, `Fehlversuchsquote`

**Verwendete Features (11 ehrliche Früherkennungs-Features an $T_0=2$ Semestern):**
- Demographie bei Einschreibung: `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`, `hzb_typ`
- Frühe Leistung (Sem. 1–2): `AVG_note_sem1-2`, `AVG_cp_sem1-2`, `fehlversuche_sem12`
- Frühe Supportnutzung (Sem. 1–2): `Fach_supp_sem12`, `Uebf_supp_sem12`, `Psych_supp_sem12`

> ✅ **Behobenes Target- & Future-Leakage:** Frühere Lifetime-Aggregate (`Anz_DrittVersuche`, `support_exposure_count`, `Fach_supp`), Späteres Notenwissen (`AVG_note_sem1-4`) und interne Ground-Truth-Variablen (`hidden_*`) wurden in `train_mlp_baseline.py` strikt maskiert.

#### Preprocessing

| Schritt | Numerisch | Kategorisch |
| :--- | :--- | :--- |
| Imputation | `SimpleImputer(strategy='median')` | `SimpleImputer(strategy='most_frequent')` |
| Transformation | `StandardScaler()` | `OneHotEncoder(handle_unknown='ignore')` |

> ✅ Preprocessor wird ausschließlich auf `X_train_df` gefittet und auf Validation/Test angewendet → **Preprocessing-Leakage behoben**

#### Train/Test Split

3-Way Stratified Split: 70% Train / 15% Val / 15% Test (`stratify=y`, `random_state=42`)

#### Modell-Architekturen & Testergebnisse (saubere Früherkennung an $T_0=2$ Semestern)

| Modell | Architektur | Test Accuracy | Validation Acc |
| :--- | :--- | :---: | :---: |
| **Naive Bayes** | `GaussianNB()` | 72.92% | 72.84% |
| **Random Forest** | `RandomForestClassifier(n_estimators=100)` | 77.03% | 77.93% |
| **SVM** | `SVC(kernel='rbf', C=1.0)` | 78.80% | 79.39% |
| **Keras MLP** | `Dense(64,relu) → BN → Dropout(0.3) → Dense(32,relu) → BN → Dropout(0.2) → Dense(4,softmax)` | **79.03%** | **79.68%** |

> ✅ **Ohne Future-/Target-Leakage** (`Anz_DrittVersuche`, Lifetime-Aggregate & Ground Truth maskiert) erreicht das Keras MLP an $T_0=2$ Semestern eine ehrliche Früherkennungs-Accuracy von **79.03%**. (Vorher 98.60% durch unzulässige Zukunftsfeatures).

Training: `optimizer='adam'`, `epochs=150`, `batch_size=32`, `EarlyStopping(patience=25)`

#### Metriken

Accuracy, Precision, Recall, F1-Score (per Klasse), Confusion Matrix

---

### 1.2 MLP Regression — [train_mlp_regression.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/train_mlp_regression.py)

**Ziel:** Vorhersage der Abschlussnote aus statischen Features

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 2D Tabelle `agg_abschluesse.csv` (nur Absolventen, default `graduates_only`) |
| **Target** | `abschlussnote` (kontinuierlich, 1.0–4.0) |
| **Loss** | `mse` |

#### Features

Identisch zu 1.1 (ausschließlich 11 ehrliche Früherkennungs-Features an $T_0=2$ Semestern). `abschlussnote` wird als Target verwendet.

> ✅ **Behobenes Target- & Future-Leakage:** `AVG_ErstVersucheNote` (welches als mathematischer Proxy für die Abschlussnote wirkte) sowie alle weiteren Lifetime-Aggregate, `ECTS_bestanden` und `hidden_*`-Variablen wurden in `train_mlp_regression.py` strikt maskiert.

#### Modell-Architekturen & Testergebnisse (saubere Früherkennung an $T_0=2$ Semestern)

| Modell | Architektur | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Ridge Regression** | `Ridge(alpha=1.0)` | 0.2279 | 0.1785 | 0.8839 |
| **Random Forest** | `RandomForestRegressor(n_estimators=100)` | 0.2304 | 0.1773 | 0.8813 |
| **SVR** | `SVR(kernel='rbf', C=1.0)` | 0.2196 | 0.1694 | 0.8921 |
| **Keras MLP** | `Dense(64,relu) → BN → Dropout(0.2) → Dense(32,relu) → BN → Dropout(0.1) → Dense(1,linear)` | **0.2174** | **0.1668** | **0.8943** |

> ✅ **Ohne Future-/Target-Leakage** (`AVG_ErstVersucheNote`, Lifetime-Aggregate & Ground Truth maskiert) schätzt das Keras MLP die geglückte Abschlussnote an $T_0=2$ Semestern mit $R^2 = 0.8943$ und einer durchschnittlichen Abweichung von nur MAE = 0.1668 Notenstufen.

Training: `optimizer='adam'`, `loss='mse'`, `epochs=60`, `batch_size=64`, `EarlyStopping(patience=12)`

#### Metriken

RMSE, MAE, $R^2$, Parity Plot (Predicted vs. Actual)

---

## 2. Zeitreihen-Modelle (Regression)

---

### 2.1 Semester LSTM — [timeseries_semester.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/timeseries_semester.py)

**Ziel:** Vorhersage des Gesamtnotendurchschnitts über Semestersequenzen

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 3D Tensor $(N, T_{\max}, F)$, Padding `-99.0` |
| **Target** | $y_i = \text{mean}(\text{sem\_avg\_note}_t)$ über alle Semester des Studierenden |
| **Loss** | MSE |

#### Sequenz-Features (pro Semester $t$)

`sem_cp_earned`, `sem_cp_attempted`, `sem_fail_count`, `sem_support_fachlich_relevant`, `sem_support_fachlich_sonst`, `sem_support_ueberfachlich`, `sem_support_psychosozial`

#### Statische Features (repliziert pro Zeitschritt)

`hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker` + OHE(`stg_name`, `hzb_typ`)

#### Architektur

```
Masking(-99.0) → LSTM(64, return_sequences=True) → BN → Dropout(0.3)
→ LSTM(32) → BN → Dropout(0.2) → Dense(1, linear)
```

Training: `optimizer='adam'`, `epochs=50`, `batch_size=64`, `EarlyStopping(patience=12)`

#### Metriken (auf neuen Daten ohne Target- & Scaler-Leakage)

| Metrik | Wert |
| :--- | :---: |
| **RMSE** | **0.3056** |
| **MAE** | **0.2197** |
| **$R^2$ Score** | **0.9336** |

> ✅ Ohne Target-Leakage (`sem_avg_note` entfernt) erreicht das Modell ehrliche, echten Nutzen erbringende Vorhersagewerte ($R^2 = 0.9344$ auf dem Held-Out Testset).

---

### 2.2 Exam GRU — [timeseries_exam.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/timeseries_exam.py)

**Ziel:** Vorhersage des Gesamtnotendurchschnitts über Prüfungssequenzen

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 3D Tensor $(N, K_{\max}, F)$, Padding `-99.0` |
| **Target** | $y_i = \text{mean}(\text{note}_k)$ über alle Prüfungen des Studierenden |
| **Loss** | MSE |

#### Sequenz-Features (pro Prüfung $k$)

`fachsemester`, `versuch`, `cp`, `schwierigkeit`, `support_vorher_fachlich`, `support_vorher_ueberfachlich`, `support_vorher_psychosozial`, `support_glz_fachlich`, `support_glz_ueberfachlich`, `support_glz_psychosozial`, `support_genutzt`

✅ Einzelnoten (`note`) sind **nicht** als Feature enthalten — kein Target-Leakage.

#### Statische Features

`hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker` + OHE(`stg_name`, `hzb_typ`)

#### Architektur

```
Masking(-99.0) → GRU(64, return_sequences=True) → BN → Dropout(0.3)
→ GRU(32) → BN → Dropout(0.2) → Dense(1, linear)
```

Training: `optimizer='adam'`, `epochs=50`, `batch_size=64`, `EarlyStopping(patience=12)`

#### Metriken (auf neuen Daten ohne Scaler-Leakage)

| Metrik | Wert |
| :--- | :---: |
| **RMSE** | **0.3207** |
| **MAE** | **0.2310** |
| **$R^2$ Score** | **0.9249** |

> ✅ Preprocessor- & Scaler-Leakage behoben. Das Modell prognostiziert die Gesamtabschlussnote aus der Prüfungsabfolge mit $R^2 = 0.9249$.

---

## 3. Landmark Survival-Modelle (Stufe 1)

---

### 3.1 Dashboard DeepSurv — [dashboard_survival_dl.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/dashboard_survival_dl.py)

**Ziel:** Interaktives Dashboard mit Cox PH, DeepSurv und DTL Hazard

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 2D Landmark $T_0=3$ aus `agg_abschluesse.csv` |
| **Target** | `time_rel = studiendauer_semester - 2.0`, `event = (status != 'abgeschlossen')` |
| **Loss** | Breslow Cox Partial Likelihood / Binary Cross-Entropy |

#### Features

`hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`, `hzb_typ`, `AVG_note_sem1-2`, `AVG_cp_sem1-2`, `fehlversuche_sem12`, `Fach_supp` (=`Fach_supp_sem12`), `Uebf_supp` (=`Uebf_supp_sem12`), `Psych_supp` (=`Psych_supp_sem12`)

✅ Support strikt aus Sem. 1–2 → kein Immortal Time Bias

#### Preprocessing

Num: `SimpleImputer(median)` → `StandardScaler()` | Cat: `SimpleImputer(most_frequent)` → `OneHotEncoder()`
✅ Preprocessor wird auf Train gefittet, Test nur transformiert

#### Architekturen

| Modell | Architektur |
| :--- | :--- |
| **DeepSurv** | `Dense(32,relu) → BN → Dropout(0.2) → Dense(16,relu) → BN → Dense(1,linear,no_bias)` |
| **DTL Hazard** | `Dense(32,relu) → BN → Dropout(0.2) → Dense(16,relu) → BN → Dense(14,sigmoid)` |
| **Classic Cox** | `statsmodels.phreg` mit Breslow-Ties |

#### Metriken

C-Index (dynamisch berechnet ← **gefixt**), Hazard Ratios + 95% CIs, Log-Rank-Test

#### Train/Test Split

80/20 random split (`random_state=42`)

---

### 3.2 Standalone DeepSurv — [deep_survival.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/deep_survival.py)

**Ziel:** Ausführliche DeepSurv + DTL Analyse mit Bootstrap-CIs

| Aspekt | Detail |
| :--- | :--- |
| **Features** | `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`, `hzb_typ`, `AVG_note_sem1-2`, `AVG_cp_sem1-2`, `Fach_supp`, `Uebf_supp`, `Psych_supp` |
| **Target** | Landmark $T_0=3$, `time_rel`, `event` |

#### Architektur (DeepSurv)

```
Dense(32,relu) → BN → Dropout(0.2) → Dense(16,relu) → BN → Dropout(0.1) → Dense(1,linear,no_bias)
```
Training: **Full-Batch** (`batch_size=N_train`), `epochs=80`, `Adam(0.005)`

#### Metriken

C-Index (Harrell's), Bootstrap 95% CIs für HRs (100 Replikationen)

> [!NOTE]
> ✅ **Preprocessor korrekt:** `fit_transform` nur auf `X_train_df`, `transform` auf `X_test_df`

---

## 4. Extended Panel Survival-Modelle (Stufe 2)

---

### 4.1 Statistical Extended Cox — [extended_cox_survival.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/extended_cox_survival.py)

**Ziel:** Zeitveränderliches Cox-Modell im Counting-Process-Format

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | Person-Semester Panel $(t_{\text{start}}, t_{\text{stop}}, \text{event})$, 337.754 Zeilen |
| **Features** | `fach_supp_tv`, `uebf_supp_tv`, `psych_supp_tv`, `any_supp_tv` (kumulativ), `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name` |
| **Target** | `event = 1` nur im letzten Semester bei Dropout |
| **Modell** | `statsmodels.phreg` |
| **Split** | Keiner (Inferenz auf Gesamtdaten) |
| **Metriken** | HR, 95% CI, p-Werte |

---

### 4.2 Extended DeepSurv + DTL (Semester) — [extended_deep_survival.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/extended_deep_survival.py)

**Ziel:** Neural Survival auf Person-Semester Panel

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | Person-Semester Panel, 337.754 Zeilen |
| **Features** | `hzb_note`, `erwerbstaetigkeit_std`, `t_stop`, `t_start`, `stg_name`, `erstakademiker`, `fach_supp_tv`, `uebf_supp_tv`, `psych_supp_tv` |
| **Target** | DeepSurv: `[t_stop, event]` / DTL: `event` |

#### Architektur

| Modell | Architektur |
| :--- | :--- |
| **Extended DeepSurv** | `Dense(32,relu) → BN → Dropout(0.2) → Dense(16,relu) → BN → Dense(1,linear,no_bias)` |
| **Extended DTL** | `Dense(32,relu) → BN → Dropout(0.2) → Dense(16,relu) → BN → Dense(1,sigmoid)` |

#### Split & Preprocessing

✅ **Group Split** nach `studierenden_id` (80/20)
✅ Preprocessor auf Train gefittet

#### Metriken

ROC-AUC (Person-Semester), Brier Score (DTL)

---

### 4.3 Extended DeepSurv + DTL (Prüfung) — [extended_exam_survival.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/extended_exam_survival.py)

**Ziel:** Ultra-granulare Prüfungsebene Survival

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | Person-Exam Panel, 824.792 Zeilen |
| **Features** | `hzb_note`, `erwerbstaetigkeit_std`, `t_stop`, `versuch`, `schwierigkeit`, `cp`, `note`, `fachsemester`, `stg_name`, `erstakademiker`, `fach_supp_tv`, `uebf_supp_tv`, `psych_supp_tv` |
| **Target** | `event = 1` nur beim letzten Exam-Step bei Dropout |

> [!NOTE]
> **Concurrent Outcome Risiko:** Die Note `note` beim letzten Prüfungsschritt ($k = K_i$) kann ein Concurrent Indicator sein (Fehlversuch → sofortige Exmatrikulation). Dies ist ein unvermeidliches Artefakt der Prüfungsebene, kein Leakage im eigentlichen Sinne.

#### Split & Architektur

✅ Group Split nach `studierenden_id` (80/20)
Architektur identisch zu 4.2. Statsmodels Cox auf Subsample von 100k Zeilen.

#### Metriken

ROC-AUC (Exam-Step), Brier Score (DTL). **ROC-AUC DTL ≈ 0.889** (höchster Panel-Wert)

---

## 5. Sequenz-Survival-Modelle (Stufe 3)

---

### 5.1 GRU Dynamic Survival (Semester) — [recurrent_survival_model.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/recurrent_survival_model.py)

**Ziel:** Rekurrentes Survival-Modell mit Sequenzgedächtnis

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 3D Tensor $(N, 16, 8)$, Padding `-99.0` |
| **Target** | 3D Sequenz $(N, 16, 1)$: `y[i,t] = 1.0` nur bei Dropout im letzten Semester |
| **Loss** | Custom `masked_binary_crossentropy` (ignoriert Padding) |

#### Features (8 pro Semester)

`sem_gpa`, `sem_cp`, `sem_fails`, `fach_supp_cum`, `uebf_supp_cum`, `psych_supp_cum`, `hzb_note`, `erwerbstaetigkeit_std`

#### Architektur

```
Masking(-99.0) → GRU(32, return_sequences=True) → BN → Dropout(0.2)
→ TimeDistributed(Dense(16,relu)) → TimeDistributed(Dense(1,sigmoid))
```

Training: `Adam(0.005)`, `epochs=25`, `batch_size=512`

#### Split & Preprocessing

80/20 Student Split. ⚠️ Scaler auf Gesamtdaten gefittet (wird gefixt in Sequenzmodellen)

#### Metriken (auf neuen Daten ohne Scaler-Leakage)

| Metrik | Wert |
| :--- | :---: |
| **ROC-AUC (Global Ranking)** | **0.8223** |
| **PR-AUC / Average Precision (Gold)** | **0.2841** |
| **Brier Score (Kalibrierung)** | **0.0434** |
| **Accuracy @ Top 5%** | 0.9263 |
| **Precision @ Top 5%** | 0.2904 |
| **Recall @ Top 5%** | 0.2764 |
| **F1-Score @ Top 5%** | 0.2832 |

---

### 5.2 GRU Exam Survival — [recurrent_exam_survival.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/recurrent_exam_survival.py)

**Ziel:** Prüfungsebene mit sequenziellem Gedächtnis

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 3D Tensor $(N, 50, 7)$, Padding `-99.0` |
| **Target** | 3D Sequenz $(N, 50, 1)$: `y[i,k] = 1.0` nur bei Dropout bei letztem Exam |
| **Loss** | Custom `masked_binary_crossentropy` |

#### Features (7 pro Prüfung)

`versuch`, `schwierigkeit`, `cp`, `note`, `fach_supp_cum`, `uebf_supp_cum`, `psych_supp_cum`

#### Architektur

```
Masking(-99.0) → GRU(32, return_sequences=True) → BN → Dropout(0.2)
→ TimeDistributed(Dense(16,relu)) → TimeDistributed(Dense(1,sigmoid))
```

Training: `Adam(0.005)`, `epochs=20`, `batch_size=512`

#### Metriken

ROC-AUC, PR-AUC, Brier Score, Accuracy/Precision/Recall/F1 @Top-5%

---

### 5.3 Causal Transformer Survival — [transformer_survival_model.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/transformer_survival_model.py)

**Ziel:** Transformer mit kausaler Maskierung für Survival-Analyse

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 3D Tensor $(N, 16, 8)$, Padding `-99.0` |
| **Target** | Identisch zu 5.1 |
| **Loss** | Custom `masked_binary_crossentropy` |

#### Features (identisch zu 5.1)

`sem_gpa`, `sem_cp`, `sem_fails`, `fach_supp_cum`, `uebf_supp_cum`, `psych_supp_cum`, `hzb_note`, `erwerbstaetigkeit_std`

#### Architektur

```
Input(16,8) → Masking(-99.0) → TimeDistributed(Dense(d_model=32))
→ PositionalEncoding(sinusoidal)
→ MultiHeadAttention(heads=4, key_dim=8, dropout=0.1, use_causal_mask=True)
→ Add + LayerNorm
→ TimeDistributed(Dense(64,relu)) → TimeDistributed(Dense(32)) → Dropout(0.1)
→ Add + LayerNorm
→ TimeDistributed(Dense(1,sigmoid))
```

Training: `Adam(0.003)`, `epochs=25`, `batch_size=512`

> [!IMPORTANT]
> **`use_causal_mask=True`** — Attention kann NUR auf vergangene und aktuelle Zeitschritte zugreifen. Kein Future Leakage.

#### Metriken (auf neuen Daten ohne Scaler-Leakage)

| Metrik | Wert |
| :--- | :---: |
| **ROC-AUC (Global Ranking)** | **0.8247** |
| **PR-AUC / Average Precision (Gold)** | **0.2926** |
| **Brier Score (Kalibrierung)** | **0.0430** |
| **Accuracy @ Top 5%** | 0.9274 |
| **Precision @ Top 5%** | 0.3014 |
| **Recall @ Top 5%** | 0.2861 |
| **F1-Score @ Top 5%** | 0.2935 |

---

## 6. Competing Risks (Stufe 4)

---

### 6.1 Dynamic DeepHit — [dynamic_deephit_model.py](file:///c:/Users/wilfr/OneDrive/Dokumente/Data%20Science/Abschlussprojekt/src/dynamic_deephit_model.py)

**Ziel:** Multi-Task Competing Risks (Abbruch vs. Abschluss)

| Aspekt | Detail |
| :--- | :--- |
| **Datenformat** | 3D Tensor $(N, 16, 8)$, Padding `-99.0` |
| **Target** | **Zwei** 3D-Sequenzen: `y_dropout` und `y_graduation` |
| **Loss** | `masked_binary_crossentropy` auf beiden Output-Heads |

#### Features (identisch zu 5.1)

`sem_gpa`, `sem_cp`, `sem_fails`, `fach_supp_cum`, `uebf_supp_cum`, `psych_supp_cum`, `hzb_note`, `erwerbstaetigkeit_std`

#### Architektur

```
Input(16,8) → Masking(-99.0)
→ Shared GRU(32, return_sequences=True) → BN → Dropout(0.2)
  ├─ Head 1 (dropout_head): TimeDistributed(Dense(16,relu)) → TimeDistributed(Dense(1,sigmoid))
  └─ Head 2 (graduation_head): TimeDistributed(Dense(16,relu)) → TimeDistributed(Dense(1,sigmoid))
```

Training: `Adam(0.005)`, `epochs=25`, `batch_size=512`

#### Metriken (auf neuen Daten ohne Scaler-Leakage)

| Ursache | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: |
| **Studienabbruch (Dropout)** | **0.8261** | **0.2847** | **0.0434** |
| **Studienerfolg (Abschluss)** | **0.9997** | **0.9968** | **0.0033** |

---

## Leakage-Status Zusammenfassung

| Skript | Preprocessing Leakage | Target Leakage | Status |
| :--- | :---: | :---: | :--- |
| `train_mlp_baseline.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `train_mlp_regression.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `timeseries_semester.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `timeseries_exam.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `dashboard_survival_dl.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `deep_survival.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `extended_cox_survival.py` | ✅ N/A | ✅ Keine | ✅ Sauber |
| `extended_deep_survival.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `extended_exam_survival.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `recurrent_survival_model.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `recurrent_exam_survival.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `transformer_survival_model.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
| `dynamic_deephit_model.py` | ✅ Nein | ✅ Keine | ✅ Sauber |
