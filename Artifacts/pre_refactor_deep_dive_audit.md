# Pre-Refactor Deep Dive Audit: Monolithen & Kontrafaktik

Du hast den absolut richtigen Instinkt bewiesen, hier auf die Bremse zu treten! Meine detaillierte Code-Analyse (`ast` Parsing) bestätigt Deine Befürchtungen zu 100 %. Wenn ich das Refactoring ohne diesen Zwischenschritt durchgezogen hätte, hätten wir die gesamte Causal Inference Pipeline zerstört.

Hier sind die brisanten Erkenntnisse:

## 1. Die "Monolith"-Skripte (Mehrere Modelle pro Skript)

Das Prinzip *One Model = One Script* kollidiert aktuell hart mit folgenden 3 Skripten, die heimlich ganze Modell-Suiten in sich tragen:

* **`train_mlp_baseline.py`**: Heißt MLP, importiert und trainiert aber in Wahrheit `RandomForestClassifier`, `SVC` (Support Vector Machine) *und* ein Keras MLP.
* **`train_mlp_regression.py`**: Das gleiche Spiel für Regression (`RandomForestRegressor`, `SVR`).
* **`train_oracle_models.py`**: Baut intern einfach noch einmal ein eigenes `Logistic Hazard` und ein eigenes `DeepSurv` Modell zusammen.

**Lösung für das Refactoring:** Diese Skripte können wir nicht 1:1 umbenennen. Wir müssen sie auftrennen. Ich werde z.B. für Random Forest und SVM eigene Skripte `baseline_rf.py` und `baseline_svm.py` extrahieren, damit der Grid-Runner diese sauber und atomar ansteuern kann.

---

## 2. Die Kontrafaktik-Skripte (Causal Inference Pipeline)

Wir haben exakt **16** `counterfactual_*.py` Skripte. Diese sind *keine* unbrauchbaren Aliase, sondern hochspezialisierte Inferenz-Engines. 

Das massive Problem: **Ihre Modell-Targets sind hartcodiert!**
* `counterfactual_deephit_fixed.py` lädt zwingend `dynamic_deephit_prev.keras`
* `counterfactual_grade_transformer.py` lädt zwingend `timeseries_exam_transformer.keras`
* `counterfactual_rnn.py` lädt zwingend `recurrent_exam_survival_v2.keras`
* `counterfactual_hr_analyzer.py` lädt zwingend `extended_deepsurv_prev_{mode}.keras`

**Die Gefahr:** Hätte ich (wie in meinem ursprünglichen Refactoring-Bericht geplant) `recurrent_exam_survival.py` einfach in `exam_gru.py` umbenannt, hätte das Training zwar funktioniert, aber die Modelle wären unter neuem Namen (z.B. `exam_gru.keras`) gespeichert worden. Alle 16 Kontrafaktik-Skripte wären beim Laden der alten Dateinamen sofort abgestürzt!

**Lösung für das Refactoring:** 
1. Die 16 Skripte dürfen vorerst **nicht** ins `archive/` geschoben werden, sondern ziehen um nach `src/deepsupport/evaluation/`.
2. Bevor wir die Kern-Modelle umbenennen, müssen wir den hartcodierten Lade-Code in diesen 16 Skripten so anpassen, dass sie auf die neuen `.keras`-Dateinamen verweisen (oder besser noch: die Modellnamen als Argument übergeben bekommen).

---

## 3. Die Autoregressiven Modelle (Deine Rückfrage)

Ja, die bleiben absolut getrennt. Mein Skript hat bestätigt:
* `autoregressive_next_exam.py` nutzt eine rekurente Architektur (GRU). Es wird zu `autoregressive_gru.py`.
* `autoregressive_deep_transformer.py` nutzt Self-Attention. Es wird zu `autoregressive_transformer.py`.

### Nächster Schritt

Deine extrem wertvolle Vorsicht hat uns gerade vor einem gewaltigen Code-Bruch gerettet. 

Dank dieses Deep-Dives wissen wir nun, dass wir **alle Causal Inference Skripte** in Bezug auf ihre Lade-Pfade anpassen müssen, wenn wir die Modelle umbenennen.

Soll ich nun (unter Berücksichtigung dieser Lade-Pfade und der Monolithen-Auftrennung) mit den physischen Code-Verschiebungen und dem Splitten beginnen?
