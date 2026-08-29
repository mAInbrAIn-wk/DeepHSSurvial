# Umsetzungsplan V3.7: Hybride Moving Window Vorhersage (Next-Exam)

## 1. Das Ziel & Die Korrektur
Du hast völlig recht, ich hatte den Fokus falsch gesetzt! Die globale Abschlussnote vorherzusagen, ist – selbst ohne Leakage – nicht die spannendste Fragestellung. 
**Das eigentliche Ziel:** Wir wollen für eine spezifische, anstehende Prüfung ($t_{k+1}$) die **Note** und das **Bestehen** vorhersagen. 
Und zwar nicht aus einer singulären Datenbasis, sondern aus der *kompletten Historie* der vergangenen Prüfungen ($t_{k-W} \dots t_k$), ohne dass dies ein unzulässiges Leakage darstellt (denn die Vergangenheit formt kausal die Zukunft).

## 2. Die hybride Architektur (Zweigleisiges Netz)
Um die anstehende Prüfung $t_{k+1}$ optimal vorherzusagen, greifen wir auf Dein Konzept der hybriden Architektur zurück. Das Netzwerk wird in zwei spezialisierte Datenströme ("Köpfe") aufgeteilt, die später fusionieren:

### Zweig A: Das Sequenz-Modell (Verlaufsdaten)
* **Input:** Ein "Moving Window" der letzten $W$ Prüfungen (z. B. $W=10$ bis $20$).
* **Features:** Historische Noten, erreichte CP, Dauer, Support-Inanspruchnahme in diesen Prüfungen.
* **Verarbeitung:** Ein Transformer-Encoder oder GRU aggregiert diese Historie zu einem dichten Vektor (dem *Student State Embedding*).

### Zweig B: Das statische & asynchrone Kontext-Modell
* **Input:** 
  1. Statische Stammdaten des Studierenden (HZB-Note, Erstakademiker, Alter).
  2. **Asynchrone Modul-Informationen für $t_{k+1}$**: Schwierigkeitsgrad des anstehenden Moduls, Fachsemester-Empfehlung, Versuch-Nummer (ist es ein Zweitversuch?).
* **Verarbeitung:** Ein Multi-Layer-Perceptron (MLP), das diesen Kontext zu einem *Context Embedding* verdichtet.

### Die Fusion (Hybrider Layer)
* Die Embeddings aus Zweig A (Historie) und Zweig B (Ziel-Klausur + Stammdaten) werden konkateniert.
* Ein finaler tiefer MLP-Block berechnet daraus die beiden Targets:
  1. **Kopf 1:** Regressions-Output für die *Note* in $t_{k+1}$.
  2. **Kopf 2:** Sigmoid-Output für das *Bestehen* in $t_{k+1}$.

## 3. Datenbereitstellung (Moving Window Generator)
Wir generieren für jeden Studierenden iterativ Trainingsfenster:
* **Fenster 1:** Nutze $t_0 \dots t_{4}$ als Historie. Ziel ist die Prüfung $t_5$. Der Kontext von $t_5$ fließt in Zweig B.
* **Fenster 2:** Nutze $t_5 \dots t_{14}$ als Historie. Ziel ist die Prüfung $t_{15}$.
* Dadurch kreieren wir Millionen von kleinen, hochrelevanten Trainings-Schnipseln (Windows), die exakt die Fragestellung modellieren: *"Was passiert in der nächsten Klausur X, gegeben die bisherige Reise Y?"*

## 4. Evaluation & Next Steps
Dieser PoC wird zeigen, ob das Modell durch das separate Füttern der "Ziel-Modul-Metadaten" (Zweig B) deutlich besser lernt, ob ein Student an einer spezifischen "Killer-Klausur" scheitert, als wenn man alles in einen großen temporalen Topf wirft.
