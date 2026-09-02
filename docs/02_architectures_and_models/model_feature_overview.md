# Übersicht: Features und Modellklassen im Projekt

Dieses Dokument bietet eine strukturierte Übersicht über alle im Projekt verwendeten Modelle, die Feature-Sets, auf denen sie trainiert wurden, und ihre methodische Zuordnung.

Es gibt im Projekt **vier grundlegende Klassen von Modellen**, die aufgrund ihrer unterschiedlichen Datenstrukturen (Shapes) nur bedingt direkt miteinander verglichen werden können.

---

## 1. Statische Landmark-Modelle (1 Zeile pro Student)
**Datenstruktur:** `(N_Studierende, F_Features)`
Hier wird in die Zukunft geschaut, basierend auf den aggregierten Daten der ersten beiden Semester (Landmark $t=2$).

* **Dateien:** `train_mlp_baseline.py`, `train_mlp_regression.py`, `deep_survival.py`
* **Features:**
  * Demografie: `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`
  * Leistung (Sem 1-2): `AVG_note_sem1-2`, `AVG_cp_sem1-2`
  * Support (Sem 1-2): `Fach_supp_sem12`, `Uebf_supp_sem12`, `Psych_supp_sem12` (Binär: Hat in den ersten 2 Semestern teilgenommen)

---

## 2. Person-Semester Panel-Modelle (Zeitdiskret)
**Datenstruktur:** `(N_Semester_Zeilen, F_Features)` im Format `[t_start, t_stop, event, X(t)]`.
Diese Modelle werten für jedes Semester aus, ob der Student überlebt. Sie haben **kein inhärentes Gedächtnis** für die Vergangenheit, weshalb die Features die Vergangenheit explizit kodieren müssen (Markov-Eigenschaft).

### 2a. Alte "Cum"-Versionen (Confounding Bias)
* **Dateien:** `extended_cox_survival.py`, `extended_deep_survival.py`
* **Features:**
  * `cum_cp`, `cum_fails` (Kumulierte Summen bis Semester $t$)
  * `fach_supp_tv`, `uebf_supp_tv`, `psych_supp_tv` (**Akkumuliert / "Ever-Exposed"**)
  * Demografie: `hzb_note`, `erwerbstaetigkeit_std`, `erstakademiker`, `stg_name`

### 2b. Neue "Delta"-Versionen (Kausaler)
* **Dateien:** `extended_cox_delta.py`, `extended_deep_survival_delta.py`
* **Features:**
  * Leistung (Lokal/Delta): `fails_prev`, `delta_cp_prev`, `cp_rueckstand`
  * Support (Lokal): `fach_supp_active`, `uebf_supp_active`, `psych_supp_active` (**Nur in Semester $t$ aktiv**)
  * Demografie: wie oben.

---

## 3. Rekurrente Sequenz-Modelle auf Prüfungs-Ebene
**Datenstruktur:** `(N_Studierende, T_Prüfungen, F_Features)` mit Padding.
Das Modell prozessiert chronologisch jede einzelne Prüfung. Das Netz (GRU/Transformer) baut sich intern ein "Gedächtnis" (Hidden State) über `cum_cp` etc. auf.

* **Dateien:** `recurrent_exam_survival_v2.py`, `transformer_exam_survival.py`
* **Features:**
  * Prüfungs-Lokal: `modul_schwierigkeit`, `versuch_nr`, `days_since_start`
  * Support: `fach_supp_cum`, `uebf_supp_cum`, `psych_supp_cum` (**Akkumuliert**, was für RNNs problematisch ist, da sie den *Zustandswechsel* lernen sollten).
  * Demografie: `hzb_note`, `erwerb_std`

---

## 4. Rekurrente Sequenz-Modelle auf Semester-Ebene (inkl. DeepHit)
**Datenstruktur:** `(N_Studierende, T_Semester, F_Features)` mit Padding.
Das Modell iteriert über Semester. Auch hier baut das RNN (GRU) intern die akkumulierten Werte selbst auf.

* **Dateien:** `recurrent_survival_model.py` (GRU), `timeseries_semester_transformer.py`, `dynamic_deephit_model.py`
* **Features (aktueller Stand):**
  * Semester-Lokal: `sem_gpa`, `sem_cp`, `sem_fails`
  * Support: `cum_fach`, `cum_uebf`, `cum_psych` (**Akkumuliert**)
  * Demografie: `hzb_note`, `erwerbstaetigkeit_std`

---

## Beantwortung deiner Architektur-Frage zu DeepHit

Du fragtest völlig zurecht: *"Enthält DeepHit auch `cum_cp` oder `cum_fehlversuche`? Die sollten da rein, bzw. ins delta-Modell die Ableitungen."*

Hier zeigt sich der brillante mathematische Unterschied zwischen Panel-Modellen und Sequenz-Modellen (RNNs):
1. **Das Extended Cox Panel** hat *kein Gedächtnis*. Es schaut nur isoliert auf die Zeile `t=4`. Wenn ich ihm nur sage "dieses Semester hat er 10 CP gemacht", weiß es nicht, ob der Student davor 0 oder 90 CP hatte. Daher *mussten* wir im `_delta` Panel explizit `cp_rueckstand` und `fails_prev` als Features hinein-engineeren.
2. **DeepHit (als GRU-Netzwerk)** prozessiert Semester 1, dann Semester 2, dann Semester 3. Wenn ich ihm in jedem Zeitschritt nur die Ableitungen/lokalen Werte `sem_cp` und `sem_fails` gebe, **baut das GRU-Netzwerk in seinem Hidden State automatisch die kumulierten Werte und den Rückstand auf!** Es lernt die Integration (Summe) selbst.

**Der Fehler in DeepHit bisher war:** Das GRU bekam für die Leistung korrekterweise die "Deltas" (`sem_cp`, `sem_fails`), aber für den Support bekam es fälschlicherweise die aufaddierten ("Ever-Exposed") Flags (`cum_fach`). 

### Wie das DeepHit Delta-Modell aussehen wird:
Wenn ich `dynamic_deephit_delta_model.py` baue, werde ich dem Modell geben:
* `sem_gpa`, `sem_cp` (Deltas)
* `sem_fails` (Delta)
* `fach_supp_active` (Delta - nur 1, wenn im Semester aktiv)
* `hzb_note` etc.
* *Optional (um es dem Netz leichter zu machen):* `cp_rueckstand`. Zwar kann ein RNN das selbst aus `sem_cp` lernen, aber harte explizite Features helfen Netzen oft enorm. Ich werde `cp_rueckstand` mit aufnehmen, um die Konsistenz zum Cox-Delta-Modell zu maximieren!
