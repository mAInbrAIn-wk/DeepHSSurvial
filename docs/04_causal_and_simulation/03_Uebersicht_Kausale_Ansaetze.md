# Übersicht: Kausale und Kontrafaktische Ansätze

Dieses Dokument bündelt die verschiedenen Methoden und philosophischen Ansätze, mit denen im *DeepHSSurvival*-Projekt kausale Schlüsse (insbesondere die Effekte von Support-Maßnahmen) gezogen werden. Da klassische ML-Methoden (wie XGBoost oder naive neuronale Netze) lediglich Korrelationen lernen (Confounding by Indication), bedient sich das Projekt eines mehrschichtigen Arsenals an Kausalinferenz-Techniken.

## 1. Der Sandbox-Ansatz: Deterministische Paralleluniversen (A-H)
Die **Ground Truth** des Projekts. Da das Data-Generating-Process (DGP) Skript die volle Kontrolle über den Zufall hat (via Seed-Synchronisation), können exakt dieselben Studierenden mehrfach simuliert werden.
* **Methode:** Differenzbildung der Survival-Raten zwischen Universum A (alle Support-Optionen) und Universen B-H (geblockte Support-Optionen).
* **Vorteil:** Die absolut reinste Form der Kausalevaluation, unbeeinflusst von Modellspezifikationen. Es zeigt den "wahren" Effekt *innerhalb der Regeln unserer Simulation*.
* **Limitierung:** In der Realität gibt es keine Paralleluniversen. Dieser Ansatz dient primär als *Sanity-Check*, um zu validieren, ob unsere ML-Modelle überhaupt fähig sind, Kausalität aus reinen Beobachtungsdaten (nur Universum A) zu lernen.

## 2. Kausal-Frameworks (Pearl & Imai)
Um von dem Sandbox-Szenario wegzukommen und Methoden zu entwickeln, die potenziell auf reale Hochschuldaten anwendbar sind, nutzt das Projekt fortgeschrittene Kausalinferenz. Hierzu werden sogenannte *Kontrafaktik-Skripte* eingesetzt, die strukturelle Kausalmodelle (SCMs) auswerten.
* **Methode:** Trennung von **direkten** und **mediierten Effekten** (nach Judea Pearl / Kosuke Imai). Die Modelle versuchen, kontrafaktische Vorhersagen (*Was wäre, wenn der Student keinen Support besucht hätte?*) direkt aus dem gelernten latenten Raum der neuronalen Netze zu extrahieren.
* **Besonderheit:** Die Auswertung läuft **ausschließlich auf Daten von Universum A**. Die Modelle müssen die Kausalität rein aus den zeitlichen Sequenzen und den kontrollierten Confoundern (wie CP-Rückstand) inferieren.

## 3. Der Oracle-Ansatz (Identifikations-Beweis)
Ein Spezial-Diagnose-Werkzeug, das sich die Natur der Simulation zunutze macht.
* **Methode:** Bestimmte Modelle (mit dem Suffix `_oracle`) erhalten exklusiven Zugriff auf die verborgenen Zustandsvariablen der Simulation (z.B. `hidden_motivation`, `hidden_soziale_integration`). 
* **Zweck:** In unseren Evaluierungen wurde nachgewiesen, dass naive Modelle dem überfachlichen Support ein schädliches Vorzeichen gaben (RR > 1). Sobald den Modellen die latenten Variablen übergeben wurden, wendete sich das Vorzeichen zu RR < 1 (protektiv). Der Oracle-Ansatz liefert den formalen mathematischen Beweis, dass vorherige fehlerhafte Schätzungen tatsächlich durch unvollständige Confounder-Kontrolle (Omitted Variable Bias) entstanden sind.

## 4. Statistische Kontrollmechanismen (Der Legacy-Weg)
Schon vor den kontrafaktischen Skripten und Paralleluniversen gab es Versuche, das Dropout-Paradoxon zu lösen.
* **Methode:** In den alten Projekten (*DataAnalysis*, frühes *DeepLearning*) wurde versucht, die Confounder durch Mixed Effects Modelle oder als einfache Kontrollvariablen im klassischen Cox-Proportional-Hazards Modell (bzw. Extended Cox) abzufangen.
* **Limitierung:** Die Feedback-Schleifen (insbesondere beim überfachlichen Support, der direkt an die Motivation gekoppelt ist) waren für diese klassischen Methoden zu komplex. Die statischen Modelle fielen regelmäßig dem *Immortal-Time Bias* oder *Future-Leakage* zum Opfer.

---

**Nächster Schritt:** Eine vollumfängliche, vergleichende **quantitative Analyse** all dieser Ansätze. Wie nah kommt die Pearl/Imai-Mediation des Deep Transformers an die absolute Ground Truth des Universum-B-Vergleichs heran? Wie viel Performance steuert das Oracle-Wissen exakt bei?
