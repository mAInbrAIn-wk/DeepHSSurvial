# Feature Builder — Systematische Feature-Map

> [!NOTE]
> Basierend auf vollständiger Code-Analyse von [`feature_builder.py`](file:///c:/GitHub_public/Abschlussprojekt/src/feature_builder.py) (758 Zeilen).

---

## 0. Temporal-Switch und Support-Granularität

### Temporal-Switch (`temporal='prev'` vs `temporal='cum'`)

Der `temporal`-Parameter ist in **allen 5 Build-Funktionen** vorhanden (Default: **`'prev'`**).
Er steuert **ausschließlich die Verlaufs-Features** — Support bleibt davon unberührt:

| temporal | Was ändert sich | Semester-Verlauf | Exam-Verlauf |
| :--- | :--- | :--- | :--- |
| `'prev'` **(Default)** | Lokales Vorsemester/-prüfung | fails_prev, delta_cp_prev, gpa_prev | fails_prev_exam, cp_earned_prev_exam, note_prev_exam |
| `'cum'` | Kumulierte Historie | cum_fails_vorher, cum_cp_vorher, gpa_cum_vorher | fails_cum, cp_cum, gpa_cum |

### Support: 6 vs. 3 Features — bedingt durch Analyseebene, NICHT durch temporal

Die Differenz kommt **nicht** vom temporal-Switch, sondern von der **Granularität**:

- **Semester-Ebene** (Tensor + Panel): Support wird pro Semester aggregiert.
  Ein einziger Count pro Typ genügt → **3 Features** (`fach_supp_count`, `uebf_supp_count`, `psych_supp_count`).
  Die zeitliche Ordnung (vorher/gleichzeitig) wird durch die Sequenz der Semester implizit erfasst
  (d.h. Support in Semester t−1 steht als Feature in Zeitschritt t−1 des Tensors).

- **Exam-Ebene** (Tensor + Panel): Jede Prüfung hat individuelle Support-Exposition.
  Hier ist die Unterscheidung **innerhalb** eines Zeitschritts relevant:
  `support_vorher_*` = Teilnahmen in **früheren** Semestern (kumulierter Effekt)
  `support_glz_*` = Teilnahmen im **selben** Semester wie die Prüfung (akuter Effekt)
  → **6 Features** (2 × 3 Typen).

## 1. Fünf Datenformate für verschiedene Modellklassen

| Format | Funktion | Shape | Modelle |
| :--- | :--- | :--- | :--- |
| **Semester Tensor** | `build_semester_sequence_tensor()` | (N, 16, F) | GRU Semester, Transformer Semester, DeepHit, LSTM Regression |
| **Exam Tensor** | `build_exam_sequence_tensor()` | (N, 40, F) | GRU Exam, Transformer Exam, Autoregressive |
| **Semester Panel** | `build_semester_panel_df()` | (N×T, F) | Extended Cox, Extended DeepSurv, DML, Transformer-DML |
| **Exam Panel** | `build_exam_panel_df()` | (N×K, F) | Extended Exam Survival |
| **Landmark** | `build_landmark_dataset()` | (N, F) | MLP Baseline, RF/SVM/NB, DeepSurv Landmark, Regression |

---

## 2. Feature-Matrix nach Modus

### Legende:
- ✅ = Enthalten
- ❌ = Entfernt/Geblockt
- 🔶 = Nur in bestimmten Formaten

### 2.1 Statische Features (alle Formate identisch)

| Feature | standard | gradeblind | blind | oracle | realistic |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `hzb_note` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `hzb_typ_ord` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `stg_*` (5 OHE) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `migrationshintergrund` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `erstakademiker` | ✅ | ✅ | ✅ | ✅ | ❌ |
| `erwerbstaetigkeit_std` | ✅ | ✅ | ✅ | ✅ | ❌ |

### 2.2 Verlaufs-Features: Semester-Ebene (Tensor + Panel)

| Feature | Bedeutung | temporal='prev' | temporal='cum' | gradeblind | blind |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `fails_prev` | Durchfallanzahl VorSem | ✅ | — | ✅ | ❌ |
| `delta_cp_prev` | Erw. CP VorSem | ✅ | — | ✅ | ❌ |
| `gpa_prev` | GPA VorSem | ✅ | — | ❌ | ❌ |
| `cum_fails_vorher` | Kum. Durchfälle bis VorSem | — | ✅ | ✅ | ❌ |
| `cum_cp_vorher` | Kum. erw. CP bis VorSem | — | ✅ | ✅ | ❌ |
| `gpa_cum_vorher` | Kum. GPA bis VorSem | — | ✅ | ❌ | ❌ |
| `cp_rueckstand_vorher` | max(0, (sem-1)×30 − cum_cp) | ✅ | ✅ | ✅ | ❌ |
| `sem_cp_attempted` | Versuchte CP im Sem | ✅ | ✅ | ✅ | ❌ |

### 2.3 Verlaufs-Features: Exam-Ebene (Tensor + Panel)

| Feature | Bedeutung | temporal='prev' | temporal='cum' | gradeblind | blind |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `fails_prev_exam` | Vorherige Prüfung durchgef.? | ✅ | — | ✅ | ❌ |
| `cp_earned_prev_exam` | CP der vorigen Prüfung | ✅ | — | ✅ | ❌ |
| `note_prev_exam` | Note der vorigen Prüfung | ✅ | — | ❌ | ❌ |
| `fails_cum` | Kum. Durchfälle bis hierher | — | ✅ | ✅ | ❌ |
| `cp_cum` | Kum. erw. CP bis hierher | — | ✅ | ✅ | ❌ |
| `gpa_cum` | Kum. GPA bis hierher | — | ✅ | ❌ | ❌ |
| `cp_rueckstand` | max(0, (sem-1)×30 − cp_cum) | ✅ | ✅ | ✅ | ❌ |

### 2.4 Prüfungskontext (nur Exam-Formate)

| Feature | Bedeutung | standard | realistic | blind |
| :--- | :--- | :---: | :---: | :---: |
| `fachsemester` | Fachsemester der Prüfung | ✅ | ✅ | ✅ |
| `versuch` | Prüfungsversuch (1, 2, 3) | ✅ | ✅ | ✅ |
| `cp_value` | CP-Gewicht des Moduls | ✅ | ✅ | ✅ |
| `schwierigkeit` | Modul-Schwierigkeitsgrad | ✅ | ❌ | ✅ |

> [!NOTE]
> **`versuch`** ist ein **Leistungsmerkmal** (höherer Versuch = vorheriges Scheitern
> an diesem Modul), nicht nur Kontext. Korrekt als „Verlaufs-Information" einzuordnen.
>
> **`cp_value`** = ECTS-Punkte des Moduls (z.B. 5, 10, 15 CP). Das ist das
> **Modulgewicht**, nicht die erworbenen CP. Wird in allen Exam-Formaten verwendet.

### 2.5 Support-Treatment (alle Formate)

| Feature | Sem Tensor | Exam Tensor/Panel | Sem Panel | Landmark |
| :--- | :---: | :---: | :---: | :---: |
| `fach_supp_count` | ✅ (= glz agg.) | — | ✅ (= glz agg.) | ✅ (`_s1s2`) |
| `uebf_supp_count` | ✅ | — | ✅ | ✅ |
| `psych_supp_count` | ✅ (−realistic) | — | ✅ (−realistic) | ✅ (−realistic) |
| `support_vorher_fachlich` | — | ✅ | — | — |
| `support_glz_fachlich` | — | ✅ | — | — |
| `support_vorher_ueberfachlich` | — | ✅ | — | — |
| `support_glz_ueberfachlich` | — | ✅ | — | — |
| `support_vorher_psychosozial` | — | ✅ (−realistic) | — | — |
| `support_glz_psychosozial` | — | ✅ (−realistic) | — | — |

### 2.6 Oracle/Hidden Features

| Feature | Sem Tensor | Exam Tensor | Sem Panel | Landmark | Datenquelle |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `hidden_motivation` | ✅ (`_prev`) | ✅ (momentan) | ✅ (`_prev`) | ✅ (`_s1s2`) | `pruefungen.csv` |
| `hidden_soziale_integration` | ✅ (`_prev`) | ✅ (momentan) | ✅ (`_prev`) | ✅ (`_s1s2`) | `pruefungen.csv` |
| `hidden_erwartete_note` | ✅ (`_prev`) | ✅ (momentan) | ✅ (`_prev`) | ✅ (`_s1s2`) | `pruefungen.csv` |
| **`hidden_overload`** | ❌ **fehlt** | ❌ **fehlt** | ❌ **fehlt** | ❌ **fehlt** | `pruefungen.csv` ✅ |
| **`hidden_zeit_puffer`** | ❌ **fehlt** | ❌ **fehlt** | ❌ **fehlt** | ❌ **fehlt** | `pruefungen.csv` + `studierende.csv` ✅ |
| `hidden_penalty_capped` | ❌ nicht genutzt | ❌ | ❌ | ❌ | `pruefungen.csv` (bool) |
| `hidden_support_capped` | ❌ nicht genutzt | ❌ | ❌ | ❌ | `pruefungen.csv` (bool) |

> [!WARNING]
> **`hidden_overload`** (Überlastung in Stunden, z.B. 320h) und **`hidden_zeit_puffer`**
> (individueller Puffer, z.B. 73h) sind in den Rohdaten vorhanden, werden aber nicht
> als Oracle-Features im Feature Builder genutzt. Diese sind kausal hochrelevant:
> - `hidden_overload` treibt direkt die Dropout-Wahrscheinlichkeit über die Overload-Penalty
> - `hidden_zeit_puffer` moduliert individuell, ab wann Module abgeworfen werden
> **→ Sollten als Oracle-Features ergänzt werden.**

---

## 3. CP-Variablen: Klärung

| Variable | Berechnung | Bedeutung |
| :--- | :--- | :--- |
| `cp` | Aus `module.csv` | ECTS-Gewicht des Moduls (5, 10, 15) |
| `cp_attempted` | `= cp` (Zeile 259 `aggregate.py`) | **Identisch mit `cp`** — redundant! |
| `cp_earned` | `cp if bestanden else 0` | Tatsächlich erworbene CP |
| `cp_value` | `= cp` (Exam-Formate) | Umbenannte Kopie von `cp` |
| `sem_cp` | `sum(cp_earned)` pro Semester | Erworbene CP im Semester |
| `sem_cp_attempted` | `sum(cp_attempted)` pro Semester | **= `sum(cp)` = Versuchte CP im Semester** |
| `delta_cp_prev` | Shift von `sem_cp` | Erworbene CP im Vorsemester |
| `cum_cp` | Cumsum von `sem_cp` / `cp_earned` | Kumulativ erworbene CP |
| `cp_rueckstand` | `max(0, (sem-1)×30 − cum_cp)` | Rückstand zu Regel-Studienplan |

> [!IMPORTANT]
> **`cp_attempted` vs. `cp`:** Auf Prüfungsebene sind sie identisch — jede Prüfung
> „versucht" genau die CP des Moduls. Der Unterschied entsteht erst bei der
> **Semester-Aggregation:**
> - `sem_cp` = Summe der **bestandenen** CP → was der Student erworben hat
> - `sem_cp_attempted` = Summe **aller versuchten** CP → wie viel er versucht hat
>
> `sem_cp_attempted` IST also sinnvoll: Ein Student, der 30 CP versucht aber nur 15
> besteht, hat `sem_cp=15` aber `sem_cp_attempted=30`. Das zeigt die Ambition/Belastung.
> **Kein Duplikat — semantisch korrekt!**

---

## 4. Counterfactual-Architektur: Kein zweites Universum nötig

> [!NOTE]
> Die Counterfactual-Skripte arbeiten **rein modellbasiert** auf einem einzigen Universum:
>
> 1. Ein trainiertes Survival-Modell (z.B. Extended DeepSurv) wird geladen
> 2. Für jeden Studenten im Test-Set werden **zwei Vorhersagen** gemacht:
>    - **Kontrolle:** Treatment-Variable(n) auf 0 gesetzt
>    - **Treated:** Beobachteter Treatment-Wert
> 3. **Hazard Ratio** = `exp(h_treated − h_control)`
>
> Das simuliert „Was wäre, wenn dieser Student keinen Support bekommen hätte?"
> rein über das Modell — keine echten Kontroll-Universen nötig.
>
> **Partieller Effekt** (≙ A vs. C/D/E): Nur ein Support-Typ auf 0, Rest beobachtet
> **Isolierter Effekt** (≙ B vs. F/G/H): Alle auf 0, nur einer beobachtet
>
> **→ Alle CF-Skripte laufen auf einem einzigen `data_dir` (Universum A).** ✅

---

## 5. Feature-Anzahl pro Modus und Format (temporal='prev')

> [!NOTE]
> **Warum 3 vs. 6 Support-Inputs?**
> - **Semester-Formate** (Tensor + Panel): 3 Features — `fach_supp_count`, `uebf_supp_count`,
>   `psych_supp_count` = gleichzeitige Teilnahmen pro Semester (aggregiert).
>   „Vorher"-Info wird durch das temporale Shift (`_prev`) der gesamten Zeitreihe implizit erfasst.
> - **Exam-Formate** (Tensor + Panel): 6 Features — je `support_vorher_*` + `support_glz_*`
>   pro Typ, weil auf Prüfungsebene der Unterschied „vor dieser Prüfung" vs. „im selben Semester
>   wie diese Prüfung" explizit informativ ist.
>
> **Warum 5 vs. 4 Studiengang-OHE?**
> - Tensor- und Landmark-Formate: 5 OHE (alle Studiengänge, inkl. Informatik)
> - Panel-Formate: 4 OHE (Informatik als Referenzkategorie, `STUDIENGAENGE_LIST[1:]`)

### 5.1 Semester Tensor (GRU, Transformer, DeepHit, LSTM Regression)

| Modus | Statisch | Verlauf | Support | Oracle | **Gesamt** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| standard | 10 | 5 | 3 | — | **18** |
| gradeblind | 10 | 4 | 3 | — | **17** |
| blind | 10 | 0 | 3 | — | **13** |
| oracle | 10 | 5 | 3 | 3 (+2 geplant) | **21** → **23** |
| realistic | 7 | 5 | 2 | — | **14** |

Statisch: hzb_note, hzb_typ_ord, 5×stg_OHE, [migr, erst, erw]
Verlauf: fails_prev, delta_cp_prev, cp_rueckstand_vorher, sem_cp_attempted, [gpa_prev]
Support: fach_supp_count, uebf_supp_count, [psych_supp_count]

### 5.2 Exam Tensor (Exam GRU, Exam Transformer, Autoregressive)

| Modus | Statisch | Kontext | Verlauf | Support | Oracle | **Gesamt** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| standard | 10 | 4 | 4 | 6 | — | **24** |
| gradeblind | 10 | 4 | 3 | 6 | — | **23** |
| blind | 10 | 4 | 0 | 6 | — | **20** |
| oracle | 10 | 4 | 4 | 6 | 3 (+2 geplant) | **27** → **29** |
| realistic | 7 | 3 | 4 | 4 | — | **18** |

Kontext: fachsemester, versuch, cp_value, [schwierigkeit]
Verlauf: fails_prev_exam, cp_earned_prev_exam, cp_rueckstand, [note_prev_exam]
Support: support_vorher_fach + support_glz_fach + vorher_uebf + glz_uebf + [vorher_psych + glz_psych]

### 5.3 Semester Panel (Extended Cox, DeepSurv, DML, Transformer-DML)

| Modus | Statisch | Verlauf | Support | Oracle | **Gesamt** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| standard | 9 | 4 | 3 | — | **16** |
| gradeblind | 9 | 3 | 3 | — | **15** |
| blind | 9 | 0 | 3 | — | **12** |
| oracle | 9 | 4 | 3 | 3 (+2 geplant) | **19** → **21** |
| realistic | 6 | 4 | 2 | — | **12** |

Statisch: hzb_note, hzb_typ_ord, 4×stg_OHE (Ref=Informatik), [migr, erst, erw]
Verlauf: fails_prev, delta_cp_prev, cp_rueckstand, [gpa_prev]
Support: fach_supp_count, uebf_supp_count, [psych_supp_count]

> [!NOTE]
> **Kein `sem_cp_attempted`** im Panel — anders als im Semester-Tensor.
> Das Panel hat dafür `delta_cp_prev` (= erworbene CP im Vorsemester),
> was die gleiche Information aus einer anderen Perspektive liefert.

### 5.4 Exam Panel (Extended Exam Survival)

| Modus | Statisch | Kontext | Verlauf | Support | Oracle | **Gesamt** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| standard | 9 | 4 | 4 | 6 | — | **23** |
| gradeblind | 9 | 4 | 3 | 6 | — | **22** |
| blind | 9 | 4 | 0 | 6 | — | **19** |
| oracle | — | — | — | — | — | **⚠️ nicht implementiert** |
| realistic | 6 | 3 | 4 | 4 | — | **17** |

> [!WARNING]
> **`build_exam_panel_df()` unterstützt keinen Oracle-Modus** —
> `hidden_*` Features werden dort nicht eingebaut. Das ist eine Lücke,
> die bei der Oracle-Erweiterung mitbehoben werden sollte.

### 5.5 Landmark (MLP Baseline, RF, SVM, DeepSurv Landmark, Regression)

| Modus | Statisch | Verlauf (S1+S2) | Support (S1+S2) | Oracle (S1+S2) | **Gesamt** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| standard | 10 | 3 | 3 | — | **16** |
| gradeblind | 10 | 2 | 3 | — | **15** |
| blind | 10 | 0 | 3 | — | **13** |
| oracle | 10 | 3 | 3 | 3 (+2 geplant) | **19** → **21** |
| realistic | 7 | 3 | 2 | — | **12** |

Verlauf: cp_s1s2, fails_s1s2, [gpa_s1s2] (aggregiert über Semester 1+2)
Support: fach_supp_s1s2, uebf_supp_s1s2, [psych_supp_s1s2]
Oracle: hidden_motivation_s1s2, hidden_soziale_integration_s1s2, hidden_erwartete_note_s1s2

---

## 6. Zusammenfassung: Identifizierte Verbesserungen

| # | Verbesserung | Betrifft | Priorität |
| :---: | :--- | :--- | :---: |
| 1 | **Leakage-Fix:** 5 Skripte auf Student-Level Split umstellen | autoregressive_*.py, eval_*, transfer, cf_deepsurv | 🔴 Hoch |
| 2 | **Oracle-Erweiterung:** `hidden_overload` + `hidden_zeit_puffer` als Features | Alle 5 `build_*` Funktionen | 🟡 Mittel |
| 3 | **Exam Panel Oracle:** Oracle-Modus in `build_exam_panel_df()` nachziehen | feature_builder.py L.564–643 | 🟡 Mittel |
| 4 | **Future Leakage prüfen:** `cp_cum`, `gpa_cum` im Exam-Tensor inkl. aktuelle Prüfung? | feature_builder.py L.312–315 | 🔴 Prüfen! |
