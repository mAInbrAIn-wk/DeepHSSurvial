# Forschungsakte: Evolution der Hypothesen & Empirische Evidenz des Support-Paradoxons

**Projekt:** Causal Survival Analysis & Macro/Micro Evaluation  
**Datum:** 12. August 2026  
**Status:** Empirisch verifiziert & Abgeschlossen  

---

## 1. Das Grundproblem (Das Dropout-Paradoxon)

In der 5-Universen-Simulation (50.000 Studierende × 5 Trajektorien-Klone) zeigte sich auf Makro-Ebene (Universum A vs. Universum C) ein erstaunlicher Befund:
- Der **fachliche Support** senkt das Gesamtdropout fast überhaupt nicht (**-0.07 %-Punkte**, RR = 0.9972).
- Dennoch führt die Verfügbarkeit von fachlichem Support dazu, dass **1.064 Studierende (G1)** ihr Studium abbrechen, die *ohne* fachlichen Support (in Universum C) erfolgreich abgeschlossen hätten.
- Gleichzeitig schätzen alle Causal Machine Learning Modelle (inkl. Double Machine Learning - DML) den fachlichen Support als **stark protektiv (RR ~ 0.895)** ein.

---

## 2. Präzisierung der G1-Opfer-Population (Status-Analyse)

Ein exakter Blick in die `abschluesse.csv` der 1.064 G1-Geschädigten zeigt folgende Verteilung:

* **Abgebrochen (Freiwilliger Studienabbruch):** **965 Studierende (90.7 %)**
* **Exmatrikuliert (Leistungsbedingt / endgültig nicht bestanden):** **73 Studierende (6.9 %)**
* **Zeitüberschreitung (Maximalstudienzeit überschritten):** **26 Studierende (2.4 %)**

> [!NOTE]
> Über 90 % der G1-Opfer verlasssen die Hochschule durch **aktiven Abbruch**, da ihre Motivation und ihr CP-Fortschritt durch die induzierten Verzögerungen unter die kritische Schwelle fallen.

---

## 3. Die exakte Code-Mechanik von Overload & Modul-Abwurf

Die vertiefte Code-Recherche in `simulation_v2.py` enthüllt die genauen Formeln und Wirkungskanäle:

### A. Berechnung des Overloads (`simulation_v2.py`, Z. 317–320)
$$\text{total\_workload} = \text{geplanter\_workload} + \text{support\_zeit\_kosten}$$
$$\text{overload} = \max(0, \text{total\_workload} - \text{verfuegbare\_zeit})$$
$$\text{overload\_penalty} = \left(\frac{\text{overload}}{100}\right) \times 0.1$$

### B. Die zwei Wirkungskanäle der `overload_penalty`
Die `overload_penalty` wirkt an zwei Stellen gleichzeitig:
1. **Verschlechterung der Prüfungsleistung (`simuliere_pruefung`, Z. 145):**  
   $$\text{leistung\_base} = \text{startwert} + \dots - \mathbf{overload\_penalty} + \text{rauschen}$$  
   Ein Overload führt unmittelbar zu schlechteren Prüfungsnoten und höheren Durchfallquoten im laufenden Semester.
2. **Direkte Erhöhung der Abbruchwahrscheinlichkeit (`berechne_dropout`, Z. 163):**  
   $$p_{\text{drop}} = \dots + \min(\mathbf{overload\_penalty}, 0.3) \times 0.10$$  
   Ein Overload erhöht das Dropout-Risiko direkt um bis zu **+3 %-Punkte**.

### C. Die 1.4 % (15 Studierende) OHNE Modul-Abwurf
Bei 15 der 1.064 Opfer war der Support (+30h) knapp unter der Schwelle von `+150h`, die einen Modul-Abwurf auslöst. Die 30h erhöhten jedoch die `overload_penalty` exakt so weit, dass $p_{\text{drop}}$ die kritische Schwelle überschritt und den Abbruch auslöste.

### D. Die Ursache des Kaskaden-Effekts (Turnus-Locks & Bachelorarbeit-Gate)
Warum schaukeln sich abgeworfene Module auf durchschnittlich **8.02 blockierte Module** über das Studium an, obwohl die Simulation keinen expliziten Modul-Pre-Req-Graphen hat?
1. **Turnus-Lock (`simulation_v2.py`, Z. 229):** Module sind an Sommer- (SS) oder Wintersemester (WS) gebunden. Wer im SS ein Modul abwirft, kann es im WS nicht nachholen, sondern muss **ein ganzes Jahr (2 Semester) warten**. Das Modul-Slot im WS bleibt leer.
2. **Empfohlenes Fachsemester-Gate (Z. 233):** Verhindert das unbegrenzte Vorziehen von Modulen.
3. **Bachelorarbeit-Gate (Z. 235–237):** Die Abschlussarbeit wird erst freigeschaltet, wenn $\text{CP}_{\text{bestanden}} \ge \text{CP}_{\text{gesamt}} - 18$. Durch abgeworfene Module wird diese Sperre erst verzögert erreicht.

---

## 4. Spezifikation für Simulation V3

Um künstliche Artefakte der starr diskreten 150h-Klippenfunktion in V3 zu beseitigen, vereinbaren wir folgende Anpassungen:

1. **Stochastischer Puffer mit Logging:**  
   Der Zeitpuffer $B_i$ pro Student wird normalverteilt gewürfelt: $B_i \sim \mathcal{N}(60\text{h}, 30\text{h})$. Dieser Wert wird explizit als `hidden_zeit_puffer` im Dataset geloggt.
2. **Kombination aus Puffer & Soft-Thresholding:**  
   Die Modul-Abwurfwahrscheinlichkeit kombiniert den verdeckten Puffer mit einer verrauschten Logistik-Grenze:
   $$P(\text{drop}) = \sigma\left(\frac{\text{total\_workload} - \text{verfuegbare\_zeit} - B_i}{\tau}\right)$$
3. **Verwurf:** Eine feingranulare Skalierung von Vorbereitungszeiten wird verworfen (zu hoher Umbauaufwand).

---

## 5. Ausblick & Timing: Transformer-DML

Wir schlagen vor, das **Transformer-DML Experiment auf dem aktuellen Datensatz V2 durchzuführen**, bevor Simulation V3 gestartet wird:
- **Ziel:** Prüfen, ob eine Transformer-Architektur (über `TimeDistributed` oder Attention-Layers) den latenten `geplanten_workload` auf den V2-Daten erlernen und den DML-Bias (RR 0.89) bereits bereinigen kann.
