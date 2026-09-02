# Präziser Metrik-Audit & Cross-Szenario Status

Du hattest absolut recht mit Deiner Skepsis! Ich habe ein striktes Skript geschrieben, das alle Dateien auf Herz und Nieren (inhaltliche JSON-Keys) prüft und das Dateisystem über alle 15 Szenarien hinweg scannt. 

Das Ergebnis bestätigt Deine Bedenken vollständig: **Wir haben eine asymmetrische Metrik-Matrix und noch überhaupt keine Cross-Szenario-Ergebnisse.**

---

## 1. Cross-Szenario Check (S01 bis S15)

Das wichtigste Ergebnis vorweg: **Es liegen noch keine Modelldaten für S02 bis S15 vor.** 
Das Verzeichnis `universe_A/metrics/` existiert in diesen Ordnern schlichtweg nicht. 
Unsere bisherige Auswerte-Engine (`analyze_cross_scenario_models.py`) hat technisch einwandfrei funktioniert, aber faktisch nur die Daten aus `S01_baseline` eingelesen. 

| Szenario | Metrik-Dateien (universe_A) |
| :--- | :---: |
| `S01_baseline` | 92 Dateien (Vollständig) |
| `S02_supp_half` bis `S15_cost_effect_double` | **0 Dateien (Fehlen komplett)** |

**Folgerung für die Roadmap:** Der "Multi-Szenario-Grid-Run" (Phase 3) ist nicht nur eine Ergänzung, sondern absolut zwingend erforderlich, um den Titel "Cross-Szenario" überhaupt zu rechtfertigen!

---

## 2. Die exakte Metrik-Matrix (S01 Baseline)

Wenn wir die 92 Dateien in `S01_baseline` filtern, bleiben **54 echte Evaluierungs-JSONs** übrig. 
Die restlichen **38 Dateien** sind "Aliase" (alte Dateinamen für dieselben Modelle, z. B. `timeseries_exam_gru` statt `recurrent_exam_survival`), reine Kontrafaktik-Logs oder Diagnose-Lifts ohne klassische ROC/PR-Metriken.

Hier ist die lückenlose Matrix der **54 validen Evaluierungs-Dateien**, aufgeteilt nach Modell-Architektur und Modus. Wie Du siehst, ist die Matrix stark asymmetrisch historisch gewachsen:

### A. Die vollständig evaluierte "Fast Grid Suite"
Diese Modelle wurden über alle 5 Feature-Modi hinweg evaluiert:

* **Semester GRU (Grid):** `standard`, `gradeblind`, `blind`, `oracle`, `realistic` (5 Dateien)
* **Semester Transformer (Grid):** `standard`, `gradeblind`, `blind`, `oracle`, `realistic` (5 Dateien)
* **Exam GRU (Grid):** `standard`, `gradeblind`, `blind`, `oracle`, `realistic` (5 Dateien)

### B. Die asymmetrischen Basis-Modelle
Diese Modelle wurden meist nur in `standard` und `gradeblind` (sowie oft in `cum` und `prev` Temporals) trainiert:

* **Recurrent Exam Survival (Basis):** `standard` (4x), `gradeblind` (2x) $\to$ *Fehlt:* blind, oracle, realistic
* **Transformer Exam Survival:** `standard` (3x), `gradeblind` (2x) $\to$ *Fehlt:* blind, oracle, realistic
* **Timeseries Exam Transformer:** `standard` (2x), `gradeblind` (2x) $\to$ *Fehlt:* blind, oracle, realistic
* **Dynamic DeepHit (Competing Risks):** `standard` (4x), `gradeblind` (2x) $\to$ *Fehlt:* blind, oracle, realistic
* **DML Orthogonal Survival:** `standard` (2x), `gradeblind` (2x) $\to$ *Fehlt:* blind, oracle, realistic
* **Transformer DML:** `standard` (2x), `gradeblind` (2x) $\to$ *Fehlt:* blind, oracle, realistic

### C. Einmal-Tests und Spezialmodelle
Diese Modelle liefen ausschließlich im `standard`-Modus (oder `flat` ohne Temporals):

* **Deep Survival (Breslow):** Nur `standard` (1 Datei)
* **Extended DeepSurv (mit TVCs):** Nur `standard` (3 Dateien: cum, prev, delta)
* **Logistic Hazard Landmark:** Nur `standard` (1 Datei)
* **Extended Logistic Hazard:** Nur `standard` (3 Dateien: cum, prev, delta)
* **Autoregressive Deep Transformer:** Nur `standard` (1 Datei)

---

## 3. Fazit und Auswirkungen auf das Refactoring

Deine Intuition war zu 100 % richtig. Die Codebase liefert unvollständige und benennungstechnisch inkonsistente Matrizen, weil alte Modellläufe (Basis-Modelle) nicht mit den neuen "Grid Suites" (die alle Modi iterieren) harmonisiert wurden.

**Konkreter Plan für Phase 1 (Refactoring & Harmonisierung):**
1. **Aufräumen:** Die 38 Aliase/Legacy-Logs verschieben wir in den `archive/`-Ordner.
2. **Kanonische Namenskonvention:** Ein Modell = Ein Name (z.B. Abschaffung von `recurrent_exam_survival` zugunsten von `grid_exam_gru`).
3. **Ausführung der Lücken (Fill the Matrix):** Wir müssen entscheiden, ob wir z. B. Dynamic DeepHit auch für `blind`, `oracle` und `realistic` nachberechnen wollen, oder ob wir die Matrix absichtlich auf die Fast-Grid-Modelle beschränken.

Wie wollen wir weiter vorgehen? Soll ich zuerst die **Cross-Szenarien (S02-S15)** für die 3 Grid-Modelle berechnen lassen, damit wir echte Cross-Szenario-Daten haben, oder erst das **Code-Refactoring (Phase 1)** durchziehen?
