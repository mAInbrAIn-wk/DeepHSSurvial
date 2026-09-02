# Feature-Mapping-Report: Alte vs. Neue Feature-Logik pro Skript

Dieses Dokument begleitet die Feature-Builder-Migration (AP1) und dokumentiert f\u00fcr jedes Skript die **exakte Feature-Zuordnung** zwischen der alten (Inline) und neuen (`feature_builder.py`) Logik.

---

## 1. Vorbemerkung: Feature-Taxonomie

Jedes Feature l\u00e4sst sich in eine von 4 Kategorien einordnen:

| Typ | Symbol | Beschreibung | Beispiel |
|:----|:------:|:-------------|:---------|
| **Static** | S | Zeitunver\u00e4nderlich, 1× pro Student | `hzb_note`, `erstakademiker` |
| **Current** | C | Eigenschaft des aktuellen Zeitschritts | `versuch`, `schwierigkeit`, `cp` (Modul) |
| **Delta** | \u0394 | Lokale Ver\u00e4nderung im Zeitschritt $t$ oder $t\u22121$ | `sem_gpa`, `fails_prev`, `delta_cp_prev`, `is_fail` |
| **Cumulative** | \u03a3 | Kumulierte Historie bis $t\u22121$ | `cum_fails`, `cum_cp`, `gpa_cum`, `cp_rueckstand` |

---

## 2. Befund: `timeseries_semester.py` und die 8 rohen CSVs

### Warum 8 CSVs statt `agg_pruefungen.csv`?

Das Skript berechnet eine **Modul-Matching-Logik**, die in den aggregierten Daten **nicht verf\u00fcgbar** ist:

| Feature | Beschreibung | In `agg_pruefungen.csv`? | In `feature_builder.py`? |
|:--------|:-------------|:------------------------:|:------------------------:|
| `sem_support_fachlich_relevant` | Fachlicher Support, der exakt auf ein im selben Semester gepr\u00fcftes Modul passt | \u274c Nein | \u274c Nein |
| `sem_support_fachlich_sonst` | Fachlicher Support f\u00fcr ein Modul, das in dem Semester **nicht** gepr\u00fcft wurde | \u274c Nein | \u274c Nein |
| `sem_cp_attempted` | CP-Last aller **angemeldeten** Pr\u00fcfungen (bestanden + durchgefallen) | \u274c Nein | \u274c Nein |
| Support in pr\u00fcfungsfreien Semestern | Wenn ein Student kein Examen ablegt, aber Support nutzt | \u26a0\ufe0f Verloren | \u26a0\ufe0f Verloren |

> [!IMPORTANT]
> **Entscheidung f\u00fcr die Migration:** Die Modul-Matching-Unterscheidung (`relevant` vs. `sonst`) ist ein interessantes Feature, das wir im Feature Builder als **optionale Erweiterung (E6)** nachziehen k\u00f6nnten. F\u00fcr die initiale Migration gen\u00fcgt es, das einfachere `fach_supp_count` (Summe beider Typen) zu verwenden, wie es alle anderen Modelle tun. Die `sem_cp_attempted`-Metrik kann als zus\u00e4tzliches Feature neben `sem_cp` erg\u00e4nzt werden.

---

## 3. Delta vs. Cumulative: Ist-Zustand und Strategie

### 3.1 Semester-Modelle: Feature-Vergleich

#### `recurrent_survival_model.py` vs. `recurrent_survival_model_delta.py`

> [!NOTE]
> **\u00dcberraschendes Ergebnis:** Beide Skripte verwenden **identische 13 Features**! Die Benennung als \u201eDelta\u201c ist **irref\u00fchrend** \u2014 beide Modelle nutzen denselben Mix aus lokalen und kumulierten Merkmalen. Der einzige Unterschied sind Variablen-Benennungen im Code.

| # | Feature | Typ | Base (L132\u2013134) | Delta (L101\u2013103) | `feature_builder` |
|:--|:--------|:---:|:---:|:---:|:---:|
| 1 | `sem_gpa` | \u0394 | \u2705 | \u2705 | \u2705 `sem_gpa` |
| 2 | `sem_cp` | \u0394 | \u2705 | \u2705 | \u2705 `sem_cp` |
| 3 | `sem_fails` | \u0394 | \u2705 | \u2705 | \u2705 `sem_fails` |
| 4 | `cp_rueckstand` | \u03a3 | \u2705 | \u2705 | \u2705 `cp_rueckstand_vorher` |
| 5 | `fach_cnt` | \u0394 | \u2705 | \u2705 | \u2705 `fach_supp_count` |
| 6 | `uebf_cnt` | \u0394 | \u2705 | \u2705 | \u2705 `uebf_supp_count` |
| 7 | `psych_cnt` | \u0394 | \u2705 | \u2705 | \u2705 `psych_supp_count` |
| 8 | `hzb_note` | S | \u2705 | \u2705 | \u2705 `hzb_note` |
| 9 | `erwerbstaetigkeit_std` | S | \u2705 | \u2705 | \u2705 `erwerbstaetigkeit_std` |
| 10 | `erstakademiker` | S | \u2705 | \u2705 | \u2705 `erstakademiker` |
| 11 | `cum_fails_vorher` | \u03a3 | \u2705 | \u2705 | \u2705 `cum_fails_vorher` |
| 12 | `delta_gpa` (= `sem_gpa - hzb_note`) | \u0394 | \u2705 | \u2705 | \u274c **(separat als `sem_gpa` + `hzb_note`)** |
| 13 | `migrationshintergrund` | S | \u2705 | \u2705 | \u2705 `migrationshintergrund` |

**\u2192 Fazit:** Migration ist f\u00fcr beide Skripte identisch. `delta_gpa` ist im Feature Builder nicht als berechnetes Feature vorhanden, kann aber trivial erg\u00e4nzt oder aus den bestehenden Spalten abgeleitet werden.

---

#### `extended_cox_survival.py` (Base) vs. `extended_cox_delta.py` (Delta)

Hier ist der Unterschied **real und konzeptionell bedeutsam**:

| Feature | Typ | Base-Cox (L132) | Delta-Cox (L161) | `feature_builder` Panel |
|:--------|:---:|:---:|:---:|:---:|
| `fach_supp_count` | \u0394 | \u2705 | \u2705 | \u2705 |
| `uebf_supp_count` | \u0394 | \u2705 | \u2705 | \u2705 |
| `psych_supp_count` | \u0394 | \u2705 | \u2705 | \u2705 |
| **`cum_cp`** | **\u03a3** | **\u2705** | \u274c | \u2705 (im DataFrame, nicht in Default-`feature_cols`) |
| **`cum_fails`** | **\u03a3** | **\u2705** | \u274c | \u2705 (im DataFrame) |
| **`fails_prev`** | **\u0394** | \u274c | **\u2705** | \u2705 (in Default-`feature_cols`) |
| **`delta_cp_prev`** | **\u0394** | \u274c | **\u2705** | \u2705 (in Default-`feature_cols`) |
| **`cp_rueckstand`** | **\u03a3** | \u274c | **\u2705** | \u2705 (in Default-`feature_cols`) |
| `hzb_note` | S | \u2705 | \u2705 | \u2705 |
| `erstakademiker` | S | \u2705 | \u2705 | \u2705 |
| `erwerbstaetigkeit_std` | S | \u2705 | \u2705 | \u2705 |

> **Ergebnis:** `build_semester_panel_df` stellt **beide** Feature-Typen im DataFrame bereit, selektiert aber per Default die **Delta**-Variante (`fails_prev`, `delta_cp_prev`). F\u00fcr die Base-Cox-Migration muss der R\u00fcckgabewert `feature_cols` konfigurierbar werden.

---

### 3.2 Exam-Modelle: Feature-Vergleich

| Feature | Typ | Base (9F) | V2 (12F) | Delta (12F) | `feature_builder` |
|:--------|:---:|:---:|:---:|:---:|:---:|
| `versuch` | C | \u2705 | \u2705 | \u2705 | \u2705 |
| `schwierigkeit` | C | \u2705 | \u2705 | \u2705 | \u2705 |
| `cp` | C | \u2705 | \u2705 | \u2705 | \u2705 |
| `support_vorher_fachlich` | \u03a3 | \u2705 | \u2705 | \u2705 | \u2705 |
| `support_glz_fachlich` | \u0394 | \u2705 | \u2705 | \u2705 | \u2705 |
| `support_vorher_uebf.` | \u03a3 | \u2705 | \u2705 | \u2705 | \u2705 |
| `support_glz_uebf.` | \u0394 | \u2705 | \u2705 | \u2705 | \u2705 |
| `support_vorher_psych.` | \u03a3 | \u2705 | \u2705 | \u2705 | \u2705 |
| `support_glz_psych.` | \u0394 | \u2705 | \u2705 | \u2705 | \u2705 |
| **`fails_cum`** | **\u03a3** | \u274c | **\u2705** | \u274c | \u2705 |
| **`cp_cum`** | **\u03a3** | \u274c | **\u2705** | \u274c | \u2705 |
| **`gpa_cum`** | **\u03a3** | \u274c | **\u2705** | \u274c | \u2705 |
| **`is_fail`** | **\u0394** | \u274c | \u274c | **\u2705** | \u274c |
| **`hzb_note`** | **S** | \u274c | \u274c | **\u2705** | \u2705 |
| **`erwerbstaetigkeit_std`** | **S** | \u274c | \u274c | **\u2705** | \u2705 |

> **Ergebnis:** V2 erweitert Base um **kumulierte** Features. Delta ersetzt sie durch **lokale** Features + Demografie. `feature_builder` liefert die kumulierten (\u03a3), aber nicht `is_fail` (\u0394). Alle drei Varianten werden nach Migration durch denselben Builder mit unterschiedlicher Feature-Selektion bedient.

---

## 4. Vorgeschlagene Temporal-Strategie im Feature Builder

### Neuer Parameter: `temporal='hybrid'|'delta'|'cum'`

| Wert | Verhalten | Nutzer |
|:-----|:----------|:-------|
| `'hybrid'` (Default) | Liefert **alle** Features (Delta + Cumulative) | Grid-Runner, Analyse |
| `'delta'` | Filtert kumulierte Features aus, beh\u00e4lt nur lokale/Vorsemester | Delta-Modelle |
| `'cum'` | Filtert Delta-Features aus, beh\u00e4lt nur kumulierte | Base-Cox, V2-Exam |

Zus\u00e4tzlich werden fehlende Features erg\u00e4nzt:
- `delta_gpa` = `sem_gpa - hzb_note` (berechenbar)
- `is_fail` = `1 - bestanden` (berechenbar)
- `sem_cp_attempted` (aus `agg_pruefungen` ableitbar, wenn CP des Moduls bekannt)

---

## 5. Klassifikation der Migrations\u00e4nderungstypen

| Typ | Symbol | Beschreibung | Betroffene Skripte |
|:----|:------:|:-------------|:-------------------|
| **Drop-In** | \u2705 | Datenladung ersetzen, Feature-Set identisch | `extended_cox_delta`, `extended_deep_survival_delta`, `dml_orthogonal_survival`, `train_oracle_models`, `recurrent_exam_survival_v2`, `transformer_exam/survival`, `train_transformer_dml` |
| **Feature-Superset** | \u2b06\ufe0f | Neuer Builder liefert mehr Features als altes Skript. Modell erh\u00e4lt zus\u00e4tzliche Infos (Demografie, STG-Dummies etc.) | `recurrent_survival_model`, `recurrent_survival_model_delta`, `recurrent_exam_survival`, `recurrent_exam_survival_delta` |
| **Temporal-Switch** | \u0394\u03a3 | Altes Skript nutzte nur `cum` oder nur `delta`; neuer Builder muss per `temporal`-Flag die richtige Selektion treffen | `extended_cox_survival` (cum), `dynamic_deephit_*` (delta + competing) |
| **Target-Switch** | \ud83c\udfaf | Altes Skript hat ein anderes Regressionstarget (GPA statt Hazard, Multiclass statt bin\u00e4r) | `timeseries_semester`, `timeseries_exam`, `train_mlp_baseline`, `train_mlp_regression`, `deep_survival` |
| **Datenquelle-Wechsel** | \ud83d\udcbe | Altes Skript las rohe relationale CSVs statt `agg_*` | `timeseries_semester.py` (8 CSVs), `timeseries_semester_transformer.py` |
| **Neue Funktion** | \ud83c\udd95 | Ben\u00f6tigt eine Funktion, die im Builder noch nicht existiert | `extended_exam_survival` (`build_exam_panel_df`) |

---

## 6. Survivorship Bias bei `graduates_only`

### Problem

[`train_mlp_regression.py`](file:///C:/GitHub_public/Abschlussprojekt/src/train_mlp_regression.py) filtert mit `graduates_only=True` auf Studierende, die einen Abschluss erreicht haben, und prognostiziert deren `abschlussnote`. Das ist **Survivorship Bias**: Die Selektion auf Absolventen verzerrt die Koeffizienten, da Studierende mit schlechten Prognosen \u00fcberproportional abbrechen.

### Bewertung

| Aspekt | Einsch\u00e4tzung |
|:-------|:----------------|
| Ist das ein Problem f\u00fcr die Kausalanalyse? | \ud83d\udfe1 **Moderat.** Das Modell wird nicht f\u00fcr kausale Inferenz genutzt, sondern als pr\u00e4diktiver Benchmark. |
| Verf\u00e4lscht es die Support-Koeffizienten? | \ud83d\udfe0 **Ja, leicht.** Support korreliert mit Krisenrisiko. Absolventen mit Support sind eine selektierte Gruppe (die es *trotz* Krise geschafft hat). |
| Gibt es Alternativen? | \u2705 **Ja:** (A) Alle Studierenden einschlie\u00dfen, Dropouts mit Note 5.0 imputieren. (B) Heckman-Selektion (2-Stufen-Sch\u00e4tzer mit Probit-Selektionsgleichung). (C) Modell als Hilfsregressor in der Mediationsanalyse nutzen (dort ist die Selektionsstufe explizit). |
| Empfehlung | **Migration durchf\u00fchren, aber `graduates_only` als Flag beibehalten und im Ergebnisbericht als Limitation flaggen.** F\u00fcr die Mediationsanalyse (AP8) wird die Selektionsstufe ohnehin modelliert. |
