# Implementation Plan: Trajektorien-Klon (Simulator v2) & Orakel-Modell

Dieser Plan beschreibt die Umsetzung von Version B (Der Trajektorien-Klon) zur Ermittlung des wahren kausalen Makro-Effekts sowie den Bau eines "Orakel"-Modells (Version C), welches auf die versteckten inneren Zustände zugreift.

---

## 1. Der Trajektorien-Klon (Simulator v2)

Das Ziel ist es, für jeden simulierten Studierenden zwei vollkommen identische Paralleluniversen zu erschaffen, die sich **ausschließlich** in der Erlaubnis zur Supportnutzung unterscheiden. So erhalten wir die unbestreitbare Kausal-Wahrheit (die *wahre Hazard Ratio*) der Simulation.

#### [NEW] [`src/simulation_v2.py`](file:///c:/GitHub_public/Abschlussprojekt/src/simulation_v2.py)
- **Klonen der Simulationslogik:** Kopie der bestehenden `simulation.py`, aber mit einem entscheidenden Architektur-Update: Der Zufallsgenerator (`rng`) wird nicht global für den gesamten Durchlauf verwendet, sondern **pro Student deterministisch geseedet** (z.B. basierend auf der Studenten-ID).
- **Zwei parallele Durchläufe:** 
  - *Lauf A (Faktisch):* Support-Nutzung ist normal erlaubt.
  - *Lauf B (Kontrafaktisch):* Die Variable `support_blockiert = True` erzwingt $do(Support = 0)$.
  - Da der RNG für beide Läufe beim Start jedes Studenten exakt gleich initialisiert wird, sind alle externen Rausch-Faktoren, Härte von Klausuren und Krankheiten (Anomalien) in beiden Universen identisch.
- **Auswertung:** Das Skript wird abschließend direkt die Makro-Bestehensquoten und Dropout-Raten der beiden Läufe vergleichen und als `output_dl/metrics/true_macro_causal_effect.json` speichern.
- *(Abwärtskompatibilität: Die alte `simulation.py` bleibt als Original erhalten.)*

---

## 2. Das Orakel-Modell (Hidden Variables im ML)

Hier beweisen wir, wie stark die Prognosekraft ansteigt, wenn Modelle Gedanken lesen können (Zugriff auf `hidden_motivation`, `hidden_soziale_integration`, `hidden_erwartete_note`).

#### [MODIFY] [`src/extended_cox_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_delta.py)
- **Erweiterung des Delta-Panels:** Die Funktion `build_delta_panel()` wird angepasst, um die semesterweisen Durchschnitte der `hidden_`-Variablen aus `pruefungen.csv` zu aggregieren und an das Panel anzuhängen.
- *(Rückwärtskompatibilität: Das Panel enthält diese Spalten künftig einfach zusätzlich. Die alten Modelle ignorieren sie.)*

#### [NEW] [`src/train_oracle_model.py`](file:///c:/GitHub_public/Abschlussprojekt/src/train_oracle_model.py)
- **Training des Orakels:** Ein Skript, das ein neuronales Netzwerk (Logistic Hazard Delta oder ein Keras MLP) trainiert. 
- **Features:** Es verwendet das normale Feature-Set **PLUS** die neuen Hidden-Variables.
- **Evaluierung:** Vergleich der ROC-AUC des Orakels mit der ROC-AUC des normalen Baseline-Modells, um den Wert der "Nicht-Beobachtbarkeit" von Soft-Skills exakt zu beziffern (z.B. "Die Prognosegüte steigt um X Prozentpunkte, wenn wir die Motivation kennen").

---

## Open Questions & Review

> [!IMPORTANT]
> **Zufalls-Synchronisation im Klon (Vorschlag 2):**
> Wenn Lauf A Support nutzt, verbraucht das Klären der Support-Wahrscheinlichkeit Zufallszahlen. Lauf B nutzt keinen Support. Dadurch verschiebt sich theoretisch der Zufalls-Stream für alle *darauffolgenden* Ereignisse (z.B. Klausur-Rauschen). Um das zu verhindern, werde ich die Rausch-Ziehungen an feste Hashes (z.B. `hash(student_id + modul_id)`) binden, statt fortlaufend `rng.random()` zu rufen. Ist dieses Level an Determinismus für den Trajektorien-Klon in Deinem Sinne?

> [!TIP]
> Das Orakel-Modell baue ich als einfaches, schnelles neuronales Panel-Modell auf Keras-Basis (Logistic Hazard Architektur). Das reicht völlig aus, um den Informationsgewinn der Hidden Variables (ROC-AUC Lift) zu beweisen, ohne ein gigantisches DeepHit-Netzwerk neu trainieren zu müssen.

Bitte bestätige den Plan oder gib Feedback, dann starte ich sofort mit der Implementierung von `simulation_v2.py`!
