# Methodische Evolution & Projektsynthese

Dieses Dokument zeichnet die intellektuelle und architektonische Reise des Projekts nach. Es kontrastiert die initialen konzeptionellen Fragestellungen (aus den Legacy-Projekten *DataAnalysis* und *DataEngineering*) mit den High-End-Lösungen des aktuellen *DeepHSSurvival*-Ökosystems. 

Es dient als methodische Brücke, um zu verstehen, *warum* das Projekt in seiner heutigen Komplexität existiert.

---

## 1. Entwirrte Fäden: Wie alte Fragen beantwortet wurden

Die frühen Brainstormings warfen fundamentale Fragen zur Messbarkeit von Studienerfolg und der Wirkung von Support-Maßnahmen auf. Viele dieser Fäden wurden im aktuellen Projekt methodisch verknüpft oder gelöst:

### A. Das Dropout-Paradoxon, Selektionsbias & Immortal-Time Bias
* **Der alte Faden:** In der Frühphase zeigte sich ein "Dropout-Paradoxon" – statische Baseline-Modelle schätzten fälschlicherweise eine Risikoerhöhung (Hazard Ratio > 1) durch Support. Gleichzeitig warb eine frühe Präsentation mit einer massiven Risikosenkung (HR ~ 0.37) im Extended Cox Modell. 
* **Die Entwirrung:** Es zeigte sich, dass diese extremen Schwankungen methodische Artefakte waren. Das HR > 1 Paradoxon war stark durch den **Immortal-Time Bias** getrieben (wer ablegt/abbricht, hat keine Zeit mehr für Support). Das fantastische HR ~ 0.37 Resultat wiederum litt unter Data-Leakage und unsauberem Feature-Engineering. Um diese Bias-Fallen (inkl. Confounding by Indication) ein für alle Mal aufzulösen, wurden die Daten in **Person-Semester-Panels** transformiert und eine absolute **kontrafaktische Simulation** als *Sanity-Check* eingeführt. Erst dadurch ließ sich beweisen, dass der reale kausale Effekt sehr viel moderater ist (z.B. median HR ~ 0.88 bei den Deep-Learning-Schätzern).

### B. Dynamik des Lernens: Reibung vs. Scheitern
* **Der alte Faden:** *"Ist Reibung nicht essentiell für echtes Lernen? Reibung heißt nicht Durchfallen!"*
* **Die Einordnung & Grenzen:** Echtes Lernen erfordert Einstellungsänderung und das Überwinden von kognitiven Engpässen. Eine naive Maximierung der Durchflussquote (z.B. durch Noteninflation) würde unqualifizierte Absolventen produzieren. Dieser tiefgreifende pädagogische Vorbehalt steht als mahnender Rahmen über dem Projekt: Die mathematische **Simulation kann "echte" qualitative Lern-Reibung nicht wirklich einfangen**. In der V4-Simulation ist dies stark abstrahiert (Fehlversuche interagieren mit Motivation), aber es bleibt eine fundamentale Grenze der Optimierbarkeit.

### C. Competing Risks (Dropout vs. Abschluss)
* **Der alte Faden:** Schon ganz zu Beginn des Setups war klar, dass es nicht nur um Dropouts geht, sondern dass "Exmatrikulation" und "Erfolgreicher Abschluss" zwei völlig unterschiedliche, konkurrierende Zielzustände sind.
* **Die Lösung:** Während frühe Analysen damit kämpften, dies sauber zu trennen, gipfelte die technische Lösung in der Implementierung von **Dynamic DeepHit**, einer Multi-Task-Netzwerkarchitektur mit geteiltem Backbone (Shared GRU), die beide konkurrierenden Risiken simultan modelliert.

### D. Der Architektur-Bottleneck (CSV vs. DWH)
* **Der alte Faden:** Bedenken bezüglich Performance und Daten-Silos aus Projekt_DE. Das Generieren von Millionen von Zeilen in CSVs und das Verknüpfen über Pandas-Merges führte zu extremen Laufzeiten.
* **Die Lösung (Erfolgreich implementiert):** Dieser Architektur-Faden wurde mit dem V4-Refactoring entwirrt: Durch den Einbau von **DuckDB (In-Memory SQL)** und Parquet/Arrow-Streaming wurde die Datenaggregation massiv beschleunigt und standardisiert.

---

## 2. Neue Fäden: Hinzugekommene methodische Dimensionen

### A. Future-Leakage & Temporale Maskierung
Beim Feature-Engineering zeigte sich, dass maschinelles Lernen bei Zeitreihen hochgradig anfällig für den Blick in die Zukunft ist. Für Deep-Learning-Modelle (Transformer) wurde ein striktes Padding (`-99.0`) sowie **Causal Masking** implementiert, damit das Modell im Semester t keine Informationen aus t+1 "sehen" kann.

### B. Real-World Daten vs. Ground Truth (Der Simulations-Workaround)
Die Unzulänglichkeit echter Hochschuldaten (Datenschutz, Silos wie LMS vs. Prüfungsamt) machte reine Causal-Inference unmöglich. Die V4-Simulation ist ein notwendiger **Workaround**: Durch die Erschaffung deterministischer Paralleluniversen lässt sich die *Ground Truth* der Support-Wirksamkeit (Was wäre passiert, wenn...?) überhaupt erst mathematisch evaluieren. Einige hier entwickelte Techniken (wie das Padding/Masking) können später auf echte Daten übertragen werden.

---

## 3. Offene Fäden: Die aktuellen Baustellen

1. **Parameter-Tuning & Alternativwelten (S02-S15):** 
   Die Alternativwelten wurden nicht für makroökonomische Schocks wie Pandemien entwickelt, sondern variieren zentrale systemische Tuning-Parameter der Simulation. **Offener Faden:** Der finale Grid-Run untersucht gezielt, wie fein die verschiedenen Modelle auf diese systematischen Parameterverschiebungen reagieren.

2. **Individuelle Heterogenität (Hybride Netze & MoE):**
   Die kontrafaktischen Schätzungen zeigen, dass Support hochgradig individuell wirkt. **Offener Faden:** Ein theoretisch noch wenig abgesicherter, aber extrem spannender Gedanke ist ein Mixture-of-Experts (MoE) Router, der individuell entscheidet, welche Maßnahme am besten wirkt. Ansätze hiervon fließen bereits in unsere hybriden Netzwerküberlegungen ein.
