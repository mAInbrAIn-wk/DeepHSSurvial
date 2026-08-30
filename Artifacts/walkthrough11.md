# Walkthrough V4.1.1 — Quality Fixes & Extended Runner

---

## 1. Code-Fixes (10 Dateien, +174/−40 Zeilen)

### 1.1 Sample Leakage Fix ✅

5 Skripte von Row-Level auf Student-Level Split umgestellt:

| Datei | Vorher | Nachher |
| :--- | :--- | :--- |
| [`autoregressive_next_exam.py`](file:///c:/GitHub_public/Abschlussprojekt/src/autoregressive_next_exam.py) | `train_test_split(idx)` | `train_test_split(unique_studis)` + `.isin()` |
| [`autoregressive_deep_transformer.py`](file:///c:/GitHub_public/Abschlussprojekt/src/autoregressive_deep_transformer.py) | dto. | dto. |
| [`eval_autoregressive_fail.py`](file:///c:/GitHub_public/Abschlussprojekt/src/eval_autoregressive_fail.py) | dto. | dto. + refaktorierter predict-Input |
| [`run_transfer_learning.py`](file:///c:/GitHub_public/Abschlussprojekt/src/run_transfer_learning.py) | dto. | dto. |
| [`counterfactual_deepsurv.py`](file:///c:/GitHub_public/Abschlussprojekt/src/counterfactual_deepsurv.py) | `train_test_split(df_raw)` | Split auf `unique_studis` |

### 1.2 Future Leakage Fix ✅

[`feature_builder.py`](file:///c:/GitHub_public/Abschlussprojekt/src/feature_builder.py):
- `cp_rueckstand` in `build_exam_sequence_tensor()` und `build_exam_panel_df()`:
  `cp_cum` (inkl. aktuelle Prüfung) → `cp_cum_prev` (shifted, exkludiert aktuelle Prüfung)
- `temporal='cum'` Modus: Inklusiver cumsum als dokumentierte Design-Entscheidung

### 1.3 Oracle Feature Erweiterung ✅

| Feature | Typ | Default | Quelle |
| :--- | :--- | :--- | :--- |
| `hidden_overload` | Dynamisch/Prüfung | 0.0 | `pruefungen.csv` |
| `hidden_zeit_puffer` | Statisch/Student | 60.0 | `studierende.csv` + `pruefungen.csv` |

Implementiert in allen 5 Build-Funktionen. `build_exam_panel_df()` hat jetzt erstmals einen Oracle-Modus.

### 1.4 Feature-Counts (verifiziert)

| Format | standard | oracle |
| :--- | :---: | :---: |
| Semester Tensor | **18** | **23** (+5) |
| Exam Tensor | **24** | **29** (+5) |
| Semester Panel | **16** | **21** (+5) |
| Exam Panel | **23** | **28** (+5) |
| Landmark | **16** | **21** (+5) |

---

## 2. Neuer Runner ✅

[`run_overnight_v41.py`](file:///c:/GitHub_public/Abschlussprojekt/src/run_overnight_v41.py) (NEU):
- **37 Schritte** (20 aus run_overnight.py + 11 fehlende Modelle + Feature Grid + 5 Counterfactual)
- **5 Feature-Modi** pro Modell (standard, gradeblind, blind, oracle, realistic)
- Try/except pro Schritt → Pipeline bricht nicht ab
- `PipelineBenchmarkTracker` für Timing + Report

**Startbefehl** (noch nicht ausführen):
```bash
cd src
python run_overnight_v41.py --data_dir output_v4_grid_v41/S01_baseline/universe_A --temporal prev
```

---

## 3. Dokumentation ✅

| Datei | Änderung |
| :--- | :--- |
| [`CHANGELOG.md`](file:///c:/GitHub_public/Abschlussprojekt/CHANGELOG.md) | Neuer Abschnitt [V4.1.1 Quality Fixes] |
| [`README.md`](file:///c:/GitHub_public/Abschlussprojekt/README.md) | Sektion 6: Feature Builder & Datenformate |
| [`LIMITATIONEN_FUTURE_WORK.md`](file:///c:/GitHub_public/Abschlussprojekt/LIMITATIONEN_FUTURE_WORK.md) | temporal='cum' Design-Note, OHE-Inkonsistenz |
| [`Musteranalyse_V41.md`](file:///c:/GitHub_public/Abschlussprojekt/src/Musteranalyse_V41.md) | §1.6 Feature Builder Update |

---

## 4. Offene Nächste Schritte

1. **Runner starten** (nach Ihrem Go): `python run_overnight_v41.py` (~10-12h)
2. **V3.6 Re-Run** mit korrigierten Splits (zum Vergleich)
3. **Post-Training:** Verteilungsplots V3.6 ↔ V4.1
