# Vertiefte Analyse: Amortisation, Schwellenwerte und ML-Blindspots

**Stand:** 11. August 2026  
**Datenquelle:** Simulation V2, Universum A vs. Universum C  

Ihre Kritikpunkte waren präzise und haben den Finger genau in die Wunde der bisherigen Auswertung gelegt. Hier sind die exakten, quantitativen Antworten auf Ihre drei Fragen.

---

## 1. Die Amortisations-Frage: Sparen gerettete Prüfungen netto Zeit?

**Ihre These:** Wenn Support vor einem Fail rettet, spart man sich die Zeit für die Wiederholungsprüfung im Folgesemester (ca. 150h). Warum brechen die 1.064 Studierenden also trotzdem ab?

Wir haben für die "Geschädigten" (G1) und die "Geretteten" (G2) die exakte Netto-Zeitbilanz berechnet:
`Netto-Zeitbilanz = (Gerettete Prüfungen × 150h) - (Support-Teilnahmen × 30h)`

| Gruppe | ø Support-Kosten | ø Zeitgewinn (gesparte Fails) | **ø Netto-Bilanz** | **Amortisiert (>0)** |
|:-------|:----------------:|:-----------------------------:|:------------------:|:--------------------:|
| **G1 (Geschädigte)** | 27.3 h | 42.6 h | **+ 15.3 h** | **Nur 22.3%** |
| **G2 (Gerettete)** | 39.7 h | 82.5 h | **+ 42.8 h** | **40.3%** |

> [!CAUTION]
> **Das Zeit-Paradoxon:**
> Sie haben recht: *Im Durchschnitt* (Mean) ist die Zeitbilanz positiv! Die 30h Support amortisieren sich mathematisch durch die 150h ersparten Wiederholungen. 
> 
> **Warum brechen sie trotzdem ab?**
> 1. **Verzögerte Amortisation:** Der Zeitgewinn (150h) entsteht erst im *Folgesemester* (weil man das Modul nicht nochmal belegen muss). Die Kosten (30h) fallen jedoch im *aktuellen* Semester an.
> 2. **Der Todesstoß durch Overload:** Wenn das Zeitkonto im aktuellen Semester bereits am Limit ist, führen die 30h akut zu einem massiven `overload_penalty` und dem Abwurf anderer Module. Das Dropout-Risiko steigt *sofort*, noch bevor sich die Maßnahme im nächsten Semester auszahlen kann. 
> 3. **Fehlinvestition:** Bei 77.7% der Geschädigten amortisiert sich der Support gar nicht (Netto-Bilanz ≤ 0). Sie haben die 30h investiert, aber die Prüfung trotzdem nicht bestanden (oder hätten sie ohnehin bestanden).

---

## 2. Korrektur der "Ausschließlichkeit" (Erwerbstätigkeit)

**Ihr Einwand:** Das Wort "ausschließlich" (für >17.5h) muss an harten Zahlen geprüft werden.

Sie hatten vollkommen recht. "Ausschließlich" war eine Überspitzung. Hier ist die exakte Verteilung der Erwerbstätigkeit für die 1.064 Geschädigten (G1) im Vergleich zu den Geretteten (G2):

| Schwellenwert | Anteil in G1 (Geschädigte) | Anteil in G2 (Gerettete) |
|:--------------|:--------------------------:|:------------------------:|
| ≥ 10 h/Woche | 86.1% | 78.3% |
| ≥ 15 h/Woche | 76.0% | 65.1% |
| **≥ 17.5 h/Woche** | **63.0%** | **48.4%** |
| ≥ 20 h/Woche | 63.0% | 48.4% |
| ≥ 25 h/Woche | 29.8% | 24.9% |

> [!TIP]
> **Korrektur:** Der negative Effekt trifft nicht *ausschließlich* Studierende mit mehr als 17.5h Erwerbstätigkeit, sondern es ist ein gradueller Risikofaktor. Je höher die Erwerbstätigkeit, desto wahrscheinlicher kippt der Effekt des fachlichen Supports ins Negative. Bei über 17.5h kippt das Verhältnis massiv zu Ungunsten des Supports.

---

## 3. Die ML-Diskrepanz und die "Blind"-Modelle

**Ihr Einwand:** "Blind" bedeutet den Entzug der Noten, nicht der `hidden_`-Variablen. Warum erkennen die ML-Modelle den Netto-Null-Effekt (Gerettete vs. Geschädigte heben sich auf) nicht, sondern schätzen einen starken positiven Effekt?

Die Überprüfung des Codes (`train_mlp_baseline.py`, Zeile 90-92) bestätigt Ihre Aussage zu 100%:
* `hidden_`-Variablen sind für *alle* Baseline-Modelle ohnehin maskiert.
* `blind=True` entfernt zusätzlich **alle Noten** (`note`, `gpa`).

Wir haben die Vorhersagekraft (PR-AUC) verglichen:
* **MLP Baseline (Mit Noten):** PR-AUC = 0.561
* **MLP Baseline BLIND (Ohne Noten):** PR-AUC = 0.535

> [!IMPORTANT]
> **Warum halluzinieren die ML-Modelle einen positiven Support-Effekt?**
> Die Modelle leiden unter einer **Scheinkorrelation (Spurious Correlation)**, die durch die Architektur der Simulation bedingt ist:
> 
> 1. Der fachliche Support verbessert die Note stetig (z.B. von 3.7 auf 2.7 oder von 4.0 auf 3.0).
> 2. Die Modelle lernen aus den Trainingsdaten: *Gute Noten korrelieren extrem stark mit wenig Dropout.*
> 3. Daher folgern die Modelle: *Support -> Bessere Note -> Weniger Dropout.*
> 4. **Der Denkfehler:** In der Simulation (`berechne_dropout()`) haben die Noten *keinen kausalen Effekt* auf den Dropout! Nur binäres Durchfallen (Note > 4.0) erhöht den Penalty. Ob jemand eine 3.7 oder 1.7 hat, ändert sein Dropout-Risiko um exakt 0.0%.
> 
> Die ML-Modelle (die die Noten sehen) werden durch die künstlich geboosteten Noten getäuscht und überschätzen den Dropout-verhindernden Effekt des Supports massiv.
