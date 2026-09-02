# Vergleich: Hidden Ground Truth vs. Modellierte Kontrafaktuale

Dieses Dokument vergleicht die Art und Weise, wie die "Wahrheit" im Datengenerator (`simulation.py`) abgebildet wird, mit der Kausalanalyse unserer Machine-Learning-Modelle (insb. Double ML und Extended Cox). Es geht der Frage nach, ob und wie wir die geloggten `hidden_variablen` tiefergehend auswerten können.

---

## 1. Methodischer Vergleich der Ansätze

### A. Hidden Ground Truth (Der Mikrokosmos)
- **Die Logik:** Der Datengenerator berechnet zum Zeitpunkt einer Prüfung exakt zwei Leistungswerte (Klausurnote) für denselben Studierenden unter identischen Bedingungen (gleiches Semester, gleiches Rauschen, gleiche Motivation). 
  - `note`: inkl. `fachlicher_boost` (Treatment = 1)
  - `note_counterfactual`: exkl. `fachlicher_boost` (Treatment = 0)
- **Die Perspektive:** Es handelt sich um einen **isolierter, momentaner Effekt** (Instantaneous Treatment Effect). Die Auswertung (siehe `calculate_true_effect.py`) liefert uns den exakten, unverzerrten *Average Treatment Effect on the Treated (ATT)* auf Modul-Ebene.

### B. Kontrafaktische Modellierung (Der Makrokosmos)
- **Die Logik:** Unsere Modelle (Extended Cox / Double ML) analysieren das Überleben (Survival/Dropout) über den gesamten Studienverlauf (Makro-Ebene). Sie nutzen Längsschnittdaten (Panel-Deltas) und orthogonale Residuen, um Pfadabhängigkeiten (z.B. *Schlechte Note $\rightarrow$ Frust $\rightarrow$ Support $\rightarrow$ Abbruch*) zu entwirren.
- **Die Perspektive:** Die Modelle schätzen den **Makro-Effekt** (Hazard Ratio bzw. Relative Risk) auf die Lebensdauer (Studiendauer). Sie messen, wie sich die Intervention auf die *gesamte zukünftige Trajektorie* auswirkt.

---

## 2. Vergleich der tatsächlichen Ergebnisse

| Metrik / Erkenntnis | Hidden Ground Truth (Simulator-Mechanik) | Kontrafaktische Modelle (DML / Cox) |
| :--- | :--- | :--- |
| **Effekt-Ebene** | **Mikro:** Modul-Note & Bestehensquote | **Makro:** Dropout-Wahrscheinlichkeit |
| **Ergebnis** | -0.170 Notenpunkte Verbesserung <br> +4.21 % Bestehensquote (ATT) | $RR \approx 0.91$ bzw. $HR \approx 0.88$ (Risikosensenkung für Studienabbruch) |
| **Selektionsbias** | Gar nicht vorhanden. Da wir exakt den Moment des Treatments klonen, ist Confounding physikalisch unmöglich. | Massiv vorhanden (Dropout-Paradoxon). Wurde erst durch DML und zeitveränderliche Deltas erfolgreich entzerrt. |
| **Blindspot** | Erkennt keine *kumulativen* (langfristigen) Effekte der Supportnutzung auf den gesamten Studienverlauf. | Erkennt keine direkten Kausalpfade ("Warum sinkt das Risiko?"). Sieht nur das finale Outcome (Abbruch ja/nein). |

Die Ergebnisse sind höchst konsistent: Die Mikro-Verbesserung der Bestehensquote um $\approx 4\%$ (Ground Truth) übersetzt sich im longitudinalen Makrokosmos in eine relative Risikosenkung (Survival) von $\approx 9-12\%$ (Modelle).

---

## 3. Potenzial der `hidden_variablen`: Vorschläge für tiefere Analysen

Du hast absolut Recht: Aktuell vergleichen wir die makroskopische Hazard-Ratio unserer Modelle mit der mikroskopischen Noten-Differenz der Ground Truth. Wir verschenken das Wissen über `hidden_motivation`, `hidden_soziale_integration` und `hidden_erwartete_note`.

Der `note_counterfactual` Wert im Simulator ist streng auf das momentane Klausurergebnis limitiert. Er erfasst **nicht** den Schmetterlingseffekt: Wenn ein Student im 2. Semester psychologischen Support nutzt, steigt seine Motivation dauerhaft (`studi.motivation += 0.015`). Das verhindert eventuell einen Dropout im 4. Semester. Dieser langfristige Kausaleffekt taucht in `note_counterfactual` nicht auf!

Um den echten Makro-Effekt zu berechnen und das Potenzial der Hidden Variablen zu heben, gibt es folgende Vorschläge:

### Vorschlag 1: Strukturgleichungs-Pfadanalyse (Mediation Analysis)
Wir können die direkten und indirekten Effekte von Support auf das Dropout-Risiko exakt quantifizieren, indem wir die Hidden-Variablen nutzen.
- **Idee:** Wir berechnen, wie viel Prozent des reduzierten Dropout-Risikos (Makro) auf den direkten Weg (*Support $\rightarrow$ bessere Note $\rightarrow$ mehr CP*) und wie viel auf den indirekten Weg (*Support $\rightarrow$ höhere `hidden_motivation` $\rightarrow$ weniger Risiko*) entfällt.
- **Aufwand:** Mittel. Lässt sich gut über Regressionsmodelle mit Interaktionstermen auf dem vorhandenen `pruefungen.csv` Datensatz durchführen.

### Vorschlag 2: Der "Trajektorien-Klon" (True Macroscopic Ground Truth)
Das wäre die ultimative Lösung, um den echten, "wahren" Makro-Effekt (die theoretische Hazard Ratio) des Simulators zu berechnen und den Modellen gegenüberzustellen.
- **Idee:** Wir greifen tief in den Simulator (`simulation.py`) ein. Anstatt nur in der Sekunde der Klausur einen `note_counterfactual` zu berechnen, **klonen wir den kompletten Studenten** bei der Immatrikulation:
  - **Klon A:** Darf Supportangebote ganz normal (reaktiv) nutzen.
  - **Klon B:** Hat exakt denselben Random-Seed, darf aber **niemals** Support nutzen (Intervention $do(Support = 0)$).
- Beide durchlaufen das Studium unabhängig. Am Ende messen wir: Wäre Klon B ohne Support abgebrochen, obwohl Klon A mit Support bestanden hat? 
- **Aufwand:** Höher (Eingriff in die Simulation notwendig), liefert aber die unbestreitbare physikalische Ground Truth für das Dropout-Paradoxon auf Populationsebene.

### Vorschlag 3: Hidden Variables als "Oracle" für die Neural Networks
- **Idee:** Wir trainieren eines unserer Transformer-Modelle nicht nur auf den beobachtbaren Daten (Noten, CP), sondern füttern ihm zusätzlich als Oracle die `hidden_motivation` und `hidden_soziale_integration` Kurven aus der Simulation.
- **Erkenntnis:** Wir können so messen, wie viel Prognose-Genauigkeit (ROC-AUC) uns durch die Nicht-Beobachtbarkeit innerer Zustände in der Realität verloren geht. Dies dient als starkes wissenschaftliches Argument für die Relevanz von Soft-Skills-Tracking.
- **Aufwand:** Gering bis Mittel (Feature-Ergänzung in der Modell-Pipeline).

---

> [!TIP]
> **Empfehlung:** Vorschlag 2 (Der Trajektorien-Klon) ist wissenschaftlich extrem wertvoll, da er den ultimativen Beweis für die Richtigkeit unserer Double Machine Learning Schätzungen liefert. Er schließt die Lücke zwischen dem mikroskopischen `note_counterfactual` und der makroskopischen Überlebenswahrscheinlichkeit.
