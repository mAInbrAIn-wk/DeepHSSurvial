# Design-Dokumentation: Feature Engine Architektur

Die **Feature Engine** (`src/feature_builder.py`) bildet die zentrale Schnittstelle zwischen den flachen Rohdaten des Simulators und den hochdimensionalen Input-Tensoren der Deep Learning Modelle. Durch dieses Refactoring wurde eine N:M-Abhängigkeit (jedes Modell baut seine eigenen Features) durch eine standardisierte Architektur ersetzt.

## 1. Ebenen-Architektur (Level of Granularity)

Die Engine exportiert drei zentrale Daten-Builder, die exakt auf die Architekturklassen (1 bis 7) abgestimmt sind:

### A. Landmark (Cross-Sectional) Ebene
- **Funktion:** `build_landmark_dataset(...)`
- **Output:** Flacher DataFrame/Tensor $(N, F)$
- **Zielgruppe:** Klasse 1 (Logistische Regression, SVM), Klasse 2a (MLP), Klasse 4 (Cox PH, Random Survival Forests)
- **Logik:** Aggregiert alle Verlaufsdaten (Noten, CP, Support) bis zu einem fixen Stichtag (Landmark, z.B. Ende Semester 2). Ideal für rein präventive, statische Prognosen.

### B. Semester-Panel & Sequence Ebene
- **Funktionen:** `build_semester_sequence_tensor(...)` und `build_semester_panel_df(...)`
- **Output (Seq):** 3D-Tensor $(N, T_{max}, F)$
- **Output (Panel):** DataFrame im Start-Stop-Format (Multiple Rows pro ID)
- **Zielgruppe:** Klasse 5 (Extended Cox PH, Logistic Hazard), Klasse 6 (Semester GRU/Transformer)
- **Logik:** Bündelt alle Prüfungsereignisse eines Semesters (aggregierte CP, Durchschnittsnote des Semesters) zu einem longitudinalen Zustands-Update.

### C. Exam-Sequence Ebene
- **Funktion:** `build_exam_sequence_tensor(...)`
- **Output:** 3D-Tensor $(N, K_{max}, F)$ wobei $K$ der Prüfungsversuch-Index ist.
- **Zielgruppe:** Klasse 3 (Dynamisches DeepHit), Klasse 7 (Exam GRU/Transformer)
- **Logik:** Höchste Granularität. Jeder Schritt ist eine einzelne Klausur. Ermöglicht Modellen, auf einzelne Modul-Fehlversuche statt nur auf Semester-Durchschnitte zu reagieren.

---

## 2. Das Evaluierungs-Grid (Dynamic Mode Injector)

Anstatt Features in den Modellen durch Nullen zu maskieren (was die Gewichts-Updates stört), schneidet die Engine die Tensoren **physisch** auf das gewählte Szenario zu.

Der Parameter `mode` bestimmt das Set:
1. **`standard`:** Baseline. $F_{full}$ Dimensionen.
2. **`gradeblind`:** Filtert dynamisch `[col for col in cols if "note" not in col and "gpa" not in col]`.
3. **`blind`:** Filtert alle dynamischen Leistungsdaten. Tensor degeneriert de facto zu einer statischen Demographie-Zeitreihe, die nur Support-Counts aufaddiert.
4. **`realistic`:** Greift per Regex auf Listen geschützter Merkmale (z.B. `migrationshintergrund`, `psych_supp`) zu und entfernt diese.
5. **`oracle`:** Injiziert `hidden_*` Merkmale aus dem Simulator.

---

## 3. Dynamisches Index-Mapping für Causal Inference

Da sich die Spalten-Dimension $F$ je nach `mode` ändert (z.B. hat `gradeblind` weniger Spalten als `standard`), verschieben sich die Array-Indices für die Counterfactual-Inferenz.
**Lösung:** Jede Build-Funktion retourniert ein Dictionary `feature_indices`:
```python
X, y, f_names, f_indices = build_semester_sequence_tensor(mode="realistic")
idx_fach = f_indices["fach_supp"]
```
Modelle greifen bei der Counterfactual-Analyse über diesen Index auf den Tensor zu (`X[:, :, idx_fach] = 0.0`), ohne jemals Indices hardcoden zu müssen.

---

## 4. Ausblick: SQL-Backend Migration
Der aktuelle interne Code von `feature_builder.py` stützt sich auf ressourcenintensive Pandas `merge` und `groupby` Operationen. 
Durch die Zentralisierung in dieser einen Datei ist die Basis für ein **Refactoring auf DuckDB / Apache Arrow** gelegt: Die Signatur der Funktionen (`build_*`) bleibt identisch, lediglich die interne Implementierung wird von Pandas-Schleifen auf SQL Window Functions umgeschrieben (siehe *DuckDB Architektur-Analyse*).
