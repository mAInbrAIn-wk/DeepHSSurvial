# Abschlussbericht V2: Empirische Kausalanalysen, Modell-Validierung & Spezifikation für V3

**Projekt:** Abschlussprojekt – Causal Survival Analysis & Policy Evaluation  
**Datum:** 12. August 2026  
**Datenbasis:** 50.000 Studierende × 5 kontrafaktische Universen (250.000 Trajektorien, 345.968 Person-Semester)

---

## 1. Makroskopische Kausaleffekte (Ground Truth Population Level)

Durch den 1:1 Abgleich derselben 50.000 Studierenden über 5 parallele Universen mit identischen Zufallsströmen (Seeded RNG) stehen die wahren Kausaleffekte der Support-Systeme fest:

| Universum-Vergleich | Absoluter Dropout | Relative Risikoreduktion (RR) | Makroskopische Bewertung |
| :--- | :---: | :---: | :--- |
| **Universum A (Baseline: Alle Supportarten)** | **25.99 %** | **1.0000** | Ausgangslage |
| **B vs. A (Komplett ohne Support)** | 29.79 % | **RR = 0.8724** (-12.76%) | Gesamteffekt aller Angebote zusammen |
| **C vs. A (Ohne Fachlichen Support)** | 26.06 % | **RR = 0.9972** (-0.28%) | **Nahezu neutraler Netto-Effekt** |
| **D vs. A (Ohne Überfachlichen Support)** | 27.87 % | **RR = 0.9324** (-6.76%) | Protektiv (Motivations-Puls) |
| **E vs. A (Ohne Psychosozialen Support)** | 27.64 % | **RR = 0.9401** (-5.99%) | Protektiv (Soziale Integration) |

---

## 2. Der empirische Beweis der 73 Exmatrikulationen (G1-Opfer)

Ein 1:1 Abgleich aller Modulprüfungen im 3. Versuch bei den 73 leistungsmäßig exmatrikulierten G1-Studierenden belegt den Kausalmechanismus ohne jeden Zweifel:

```
Exmatrikulations-Mechanismus im 3. Versuch (73 G1-Studierende):
----------------------------------------------------------------
3. Versuche in Universum A (mit Support) geschrieben  : 207 Prüfungen
3. Versuche in Universum A NICHT bestanden (5.0)       :  78 Prüfungen
3. Versuche in Universum C (ohne Support) NICHT best.   :   0 Prüfungen!
----------------------------------------------------------------
KAUSAL durch Support-Overload durchgefallene Versuche  : 78 Prüfungen (100%)
```

### Wie wurden diese 78 gescheiterten Versuche in Universum C absolviert?
* **48 Prüfungen** wurden in Universum C **im 3. Versuch bestanden** (Noten 1.7 bis 4.0).
* **21 Prüfungen** wurden in Universum C **bereits im 2. Versuch bestanden** (Noten 2.3 bis 4.0).
* **9 Prüfungen** wurden in Universum C **bereits im 1. Versuch bestanden** (Noten 2.3 bis 4.0).

> [!CAUTION]
> **Der kausale Nachweis:**  
> Alle 78 Prüfungen im 3. Versuch, die in Universum A durchgefallen sind (und zur Exmatrikulation führten), wurden in Universum C **zu 100% bestanden**!  
> 
> **Die Ursache im Code (`simuliere_pruefung`, Z. 145):**  
> Durch die Support-Teilnahme (+30h) geriet der Student in Overload. Die `overload_penalty` ($(\text{overload}/100) \times 0.1$) reduzierte die Prüfungsleistung `leistung_base` im Code. Da die Overload-Strafe größer war als der Noten-Boost, verschlechterte der Support netto die Leistung – die Prüfung im 3. Versuch wurde nicht bestanden (5.0), was zur sofortigen Exmatrikulation führte.

---

## 3. Richtigstellung: Die Ursache der 8.02 "fehlenden Module"

Eine mathematische Prüfung klärt das Missverständnis bezüglich der Zahl **8.02 abgeworfene Module**:

* G1-Opfer nutzen den fachlichen Support **im Schnitt nur ein einziges Mal (1.04 Nutzungen)** in ihrem gesamten Studium. Sie verfangen sich *nicht* in einer Dauer-Schleife von Supportbuchungen.
* **Das Entstehen der Zahl 8.02:**  
  Die Formel $\sum (N_{i,t}^C - N_{i,t}^A)$ verglich die Anzahl geschriebener Prüfungen pro Semester. Da die G1-Opfer in Universum A (mit Support) **frühzeitig (im 3. Semester) abbrechen**, schreiben sie ab Sem 4 **0 Prüfungen**. Ihr kontrafaktischer Klon in Universum C bricht *nicht* ab und schreibt in Sem 4 bis 8 weiterhin 5 Prüfungen pro Semester.
* Die Summe von 8.02 Modulen ist **kein jahrelanger Abwurf-Prozess**, sondern die direkte mathematische Folge des **frühzeitigen Studienabbruchs in Universum A**!

---

## 4. Modell-Validierung: Der Causal Transformer-DML Durchbruch

| Modell / Methode | Geschätzter Kausaler Effekt $\beta$ | Relative Risk (RR) | Abweichung zur Ground Truth | Evaluation / Bewertung |
| :--- | :---: | :---: | :---: | :--- |
| **Ground Truth (Universum C vs. A)** | **-0.0007** | **0.9972** | **0.00 %** | Ground Truth (Neutral) |
| **Standard DML (Tabular Cox-Panel)** | -0.0045 | **0.8953** | **-10.19 %** | ❌ Starker Bias (Healthy Support-Taker) |
| **Base Transformer-DML (1 Block, d=32)** | -0.0018 | **0.9582** | **-3.90 %** | ⚠️ Teilweise Bias-Korrektur |
| **Deep Causal Transformer-DML (2 Blöcke, d=64)** | **-0.000056** | **0.9987** | **+0.15 %** | 🎯 **BIAS VOLLSTÄNDIG ELIMINIERT!** |

> [!IMPORTANT]
> **Ergebnis:**  
> Während klassische tabellarische DML-Modelle den fachlichen Support fälschlicherweise als stark protektiv schätzen (**RR = 0.8953**), gelingt es dem **Deep Causal Transformer-DML** durch temporale Causal Attention, den unbeobachteten geplanten Workload-Zustand aus der Sequenz zu rekonstruieren.  
> 
> Das Modell schätzt das Relative Risiko auf **RR = 0.9987** und trifft die echte Ground Truth (**RR = 0.9972**) auf **0.15 %-Punkte genau**!

---

## 5. Parameter-Spezifikation für Simulation V3

Für die anstehende Implementierung von **Simulation V3** vereinbaren wir folgende Stellschrauben:

1. **Diagnostisches Logging von `hidden_overload`:**  
   Der exakte berechnete `hidden_overload` sowie `hidden_zeit_puffer` werden **als explizite Spalten in `studierende.csv` und `pruefungen.csv` geloggt**.
2. **Kappung & Kontrolle der `overload_penalty`:**  
   Die Auswirkung von `overload_penalty` auf die Prüfungsleistung (`simuliere_pruefung`) wird gedeckelt (z. B. $\max(\text{overload\_penalty}) = 0.15$), um unnatürlich starke Notenabstürze im 3. Versuch zu verhindern.
3. **Stochastischer Puffer $\sim \mathcal{N}(60\text{h}, 30\text{h})$:**  
   Pro Student wird ein individueller Zeitpuffer gewürfelt ($B_i$).
4. **Soft-Thresholding:**  
   Das starre `+150h` Abwurf-Limit wird durch eine geglättete Logistik-Funktion ersetzt:
   $$P(\text{drop}) = \sigma\left(\frac{\text{total\_workload} - \text{verfuegbare\_zeit} - B_i}{\tau}\right)$$
