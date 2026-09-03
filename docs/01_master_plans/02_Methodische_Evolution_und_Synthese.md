# Methodische Evolution & Projektsynthese

Dieses Dokument zeichnet die intellektuelle und architektonische Reise des Projekts nach. Es kontrastiert die initialen konzeptionellen Fragestellungen (aus den Legacy-Projekten *DataAnalysis* und *DataEngineering*) mit den High-End-Lösungen des aktuellen *DeepHSSurvival*-Ökosystems. 

Es dient als methodische Brücke, um zu verstehen, *warum* das Projekt in seiner heutigen Komplexität existiert.

---

## 1. Entwirrte Fäden: Wie alte Fragen beantwortet wurden

Die frühen Brainstormings warfen fundamentale Fragen zur Messbarkeit von Studienerfolg und der Wirkung von Support-Maßnahmen auf. Viele dieser Fäden wurden im aktuellen Projekt methodisch meisterhaft verknüpft:

### A. Das Dropout-Paradoxon & Selektionsbias
* **Der alte Faden:** In `kpi.md` wurde gefragt: *Wie können wir messen, ob Support die Wahrscheinlichkeit eines Abschlusses erhöht, wenn doch vor allem Studierende mit Schwierigkeiten (die ohnehin ein höheres Abbruchrisiko haben) den Support aufsuchen?*
* **Die Lösung (Entwirrung):** Dieses Problem (Selektionsbias / Confounding by Indication) wurde als Kernstück von DeepHSSurvival identifiziert. Naive statische Modelle (Stufe 0 & 1) bestätigten das Paradoxon fälschlicherweise (Hazard Ratio > 1). Durch die Transformation der Daten in **Person-Semester-Panels** und den Einsatz **kausaler Survival-Modelle** (Extended Cox, DeepSurv) konnte der Bias herausgerechnet werden. Der wahre, risikosenkende Effekt (HR ~ 0.37) wurde erfolgreich freigelegt.

### B. Dynamik des Lernens: Reibung vs. Scheitern
* **Der alte Faden:** *"Ist Reibung nicht essentiell für echtes Lernen? Reibung heißt nicht Durchfallen!"*
* **Die Lösung:** Diese philosophische Beobachtung wurde mathematisch in die **stochastische Datengenerierung (Simulation V36)** gegossen. Studierende haben nun dynamische Attribute wie `motivation` und `cp_rueckstand`. Ein Fehlversuch (Reibung) führt nicht sofort zum Abbruch, sondern interagiert mit der Motivation und triggert (bei bestimmten Profilen) die Support-Nutzung, was anschließend zu einem Boost der Bestehenswahrscheinlichkeit im Folgeversuch führt.

### C. Real-World Daten vs. Ground Truth
* **Der alte Faden:** Die `Design_Dokumentation.md` warnte vor Daten-Silos (LMS, Prüfungsamt) und Datenschutz-Limits, die eine echte Evaluierung fast unmöglich machen.
* **Die Lösung:** Um Kausalinferenz-Methoden beweisen zu können, wurde eine **kontrafaktische Simulation (Universen A-H)** erschaffen. Da in der Realität nie beobachtet werden kann, was passiert wäre, wenn Student X *keinen* Support besucht hätte, generiert das Projekt deterministische Paralleluniversen. Nur so lässt sich die absolute *Ground Truth* der Modelle mathematisch beweisen.

---

## 2. Neue Fäden: Hinzugekommene methodische Dimensionen

Während der Entwicklung von DeepHSSurvival stieß das Projekt in methodische Tiefen vor, die in den Legacy-Projekten noch nicht absehbar waren:

### A. Competing Risks (Konkurrierende Risiken)
Ursprünglich ging es nur um "Dropout vs. Nicht-Dropout". Das Projekt erkannte jedoch, dass "Exmatrikulation" und "Erfolgreicher Abschluss" zwei konkurrierende Zielzustände sind (wer abschließt, kann nicht mehr abbrechen). Dies führte zur Implementierung von **Dynamic DeepHit**, einer Multi-Task-Netzwerkarchitektur mit geteiltem Backbone (Shared GRU), die beide Risiken simultan modelliert.

### B. Future-Leakage & Temporale Maskierung
Beim Feature-Engineering zeigte sich, dass maschinelles Lernen bei Zeitreihen hochgradig anfällig für *Leakage* (Blick in die Zukunft) ist. Als Konsequenz wurde eine strikte Trennung von *Pre-Landmark*-Features und *zeitvariablen* Features eingeführt. Für Deep-Learning-Modelle (Transformer) wurde ein striktes Padding (`-99.0`) sowie **Causal Masking** implementiert, damit das Modell im Semester t keine Informationen aus t+1 "sehen" kann.

### C. Double Machine Learning (DML)
Um den kausalen Effekt von Support noch schärfer zu isolieren, wurde die Architektur um orthogonales Machine Learning (DML) erweitert. Hierbei wird die Support-Wahrscheinlichkeit (Propensity) unabhängig von der Dropout-Wahrscheinlichkeit gelernt, um Störfaktoren (Confounder) mathematisch herauszuprojizieren.

---

## 3. Offene Fäden: Die aktuellen Baustellen

Trotz der massiven Fortschritte weisen einige konzeptionelle Fäden direkt in die Zukunft (die noch offenen ToDos):

1. **Der Architektur-Bottleneck (CSV vs. DWH):** 
   Die frühen Bedenken bezüglich Daten-Silos aus Projekt_DE holen uns auf technischer Ebene ein. Das aktuelle Projekt generiert Millionen von Zeilen, die in CSVs gespeichert und über teure Pandas-Merges verknüpft werden. **Offener Faden:** Einbau einer eingebetteten relationalen Datenbank (DuckDB/Parquet) als Feature-Store, um Zero-Copy Data Streaming für TensorFlow zu ermöglichen.

2. **Individuelle Heterogenität (MoE Router):**
   Das Extended Cox Modell liefert einen *durchschnittlichen* kausalen Effekt (ATE). Die Deep-Learning-Modelle zeigen in den kontrafaktischen Simulationen jedoch, dass Support hochgradig individuell wirkt. **Offener Faden:** Ein Mixture-of-Experts (MoE) Router, der basierend auf individuellen Studierendenprofilen entscheidet, welches Modell (bzw. welche Support-Maßnahme) am treffsichersten ist.

3. **Der Stress-Test (Universen S02-S15):**
   Die Datengenerierung wurde kürzlich darauf ausgelegt, Strukturbrüche (Pandemie, plötzliche Noteninflation) zu simulieren. **Offener Faden:** Der finale Grid-Run muss nun beweisen, ob die trainierten Deep-Learning-Modelle robuster gegen diese "Real-World-Schocks" sind als klassische statistische Verfahren.
