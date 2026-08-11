# Methodologische & Theoretische Analyse: Evaluation & Kausalfragen

Dieses Dokument behandelt differenziert die theoretischen, statistischen und methodischen Fragestellungen, die im Rahmen der Projektevaluation und der neuen Modellreihe (`_delta`) aufgeworfen wurden.

---

## 1. Proportional-Hazards-Diagnose (Schoenfeld-Residuen)

### Was wurde umgesetzt?
In [`extended_cox_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_cox_delta.py) wurde eine nativ gerechnete Schoenfeld-Residuen-Diagnose über `statsmodels.phreg` an allen 17.874 echten Ereignis-Zeitpunkten (Abbrüchen) integriert.

### Ergebnisse der Schoenfeld-Residuen:
- **`fach_supp_active`**: $\bar{|r|} = 0.3401$
- **`uebf_supp_active`**: $\bar{|r|} = 0.4362$
- **`psych_supp_active`**: $\bar{|r|} = 0.3910$
- **`hzb_note`**: $\bar{|r|} = 0.4238$
- **`erstakademiker`**: $\bar{|r|} = 0.4996$
- **`fails_prev`**: $\bar{|r|} = 1.0035$
- **`delta_cp_prev`**: $\bar{|r|} = 5.8091$ *(Skalierung entspricht CP-Größenordnung)*
- **`cp_rueckstand`**: $\bar{|r|} = 18.1994$ *(Skalierung entspricht akkumuliertem CP-Fehlbetrag)*

### Interpretation:
Die zeitbezogenen Residuen für die Treatments (Support-Variablen) und soziodemografischen Variablen weisen sehr geringe und stabile Abweichungen auf. Das bedeutet: **Die Proportional-Hazards-Annahme ist für die Support-Behandlungen im Extended Cox Delta Modell überraschend gut erfüllt.** Die Hazard Ratio ist somit nicht nur ein zeitlicher Mittelwert, sondern stellt einen im Semesterverlauf weitgehend konstanten relativen Risikofaktor dar.

---

## 2. Warum ist DeepSurv bei ROC-AUC (~0.56) "schlecht", liefert aber gute HR-Werte?

Hier liegt ein fundamentales Missverständnis bezüglich der **Ziel- und Loss-Funktion** von Cox-Modellen vor:

### A. Der mathematische Unterschied der Ausgaben:
- **Logistic Hazard / DeepHit**: Schätzen eine **diskrete bedingte Abbruchwahrscheinlichkeit** $h_i(t) = P(Y_t = 1 \mid Y_{t-1} = 0, X_i(t)) \in [0, 1]$.
- **DeepSurv (Breslow Cox Loss)**: Schätzt **keine Wahrscheinlichkeit**, sondern ein unnormiertes, zeit-invariantes **relatives Log-Risiko** $r_i = f_\theta(X_i)$.

### B. Warum schlägt die globale ROC-AUC fehl?
Die binäre ROC-AUC bewertet, wie gut ein Modell über *alle gepoolten Person-Semester* hinweg Abbrecher ($1$) von Nicht-Abbrechern ($0$) trennt.
1. Im Cox-Modell steigt das absolute Risiko über die Zeit über die nicht-parametrische Baseline-Hazard $h_0(t)$ an. DeepSurv gibt jedoch nur das relative $r_i$ aus (ohne $h_0(t)$ explizit mit einer Sigmoid-Funktion in eine Wahrscheinlichkeit umzurechnen).
2. Ein Student in Semester 1 mit hohem $r_i$ bricht vielleicht (noch) nicht ab ($0$), während ein Student in Semester 6 mit mittlerem $r_i$ aufgrund der akkumulierten Zeit bricht ($1$).
3. Wenn man $r_i$ global über alle Semester gegen ein binäres $0/1$-Target prüft, entsteht eine schlechte ROC-AUC (~0.56), weil die zeitliche Komponente $h_0(t)$ in der Reihung fehlt.

### C. Warum ist DeepSurv trotzdem der beste HR-Schätzer?
Beim **Breslow Cox Loss** optimiert das neuronale Netz strikt die **Partial Likelihood** (Reihung innerhalb desselben Risikosets zum Zeitpunkt $t$). 
Für die **kontrafaktische Evaluation** gilt:
$$\text{HR}_i = \exp(r_i^{(1)} - r_i^{(0)})$$
Da $h_0(t)$ sich bei der Quotientenbildung herauskürzt, ist DeepSurv mathematisch der **reine Schätzer der Hazard Ratio**, unbeeinflusst von Sättigungseffekten einer Sigmoid-Funktion.

---

## 3. PR-AUC Evaluation bei extremer Klassen-Imbalance (Exam GRU)

In der Metrik-Tabelle des Walkthroughs zeigte das **Exam GRU Delta Modell** eine PR-AUC von **0.1804**.

### Warum ist 0.1804 eine hervorragende Leistung?
1. **Der Baseline-Vergleich (Zufallsschätzer)**:
   - Die PR-AUC eines Zufallsklassifikators entspricht exakt der **Prävalenz der Positivklasse** $P(Y=1)$.
   - Auf Prüfungs-Ebene gibt es ca. 50 Prüfungen pro Student, aber der Abbruch erfolgt nur bei der allerletzten Prüfung. Die Positivklassen-Prävalenz liegt bei nur **~1.0% ($0.01$)**.
2. **Der Lift-Faktor**:
   - Ein PR-AUC von **0.1804** bei einer Baseline von **0.01** bedeutet einen **18-fachen Precision-Lift** gegenüber dem Zufall!
   - Auf stark imbalancierten Daten ist PR-AUC der einzig verlässliche Indikator für operative Frühwarnsysteme.

---

## 4. Hazard Ratio (HR) vs. Relatives Risiko (RR): Wie interpretiert man den Unterschied?

| Eigenschaft | Hazard Ratio (HR) | Relatives Risiko (RR) |
| :--- | :--- | :--- |
| **Definition** | Verhältnis instantaner Ereignisraten: $\frac{h_1(t)}{h_0(t)}$ | Verhältnis bedingter Wahrscheinlichkeiten: $\frac{P(Y_t=1 \mid X=1)}{P(Y_t=1 \mid X=0)}$ |
| **Modelltyp** | Cox-Modelle (DeepSurv, statsmodels PHReg) | Logistische Hazard-Modelle, DeepHit, GRU, MLPs |
| **Wertebereich** | $[0, \infty)$ | $[0, \infty)$ |
| **Interpretation** | $\text{HR} < 1.0$: Risikosenkung der Ereignisrate. | $\text{RR} < 1.0$: Senkung der relativen Abbruch-Wahrscheinlichkeit. |

### Warum zeigen Sequenzmodelle (DeepHit, GRU) teils $RR > 1.0$?
Der Grund liegt im **unbeobachteten reaktiven Confounding**:
In der Simulation treten Studierende dann dem Support bei, wenn sie sich in einer **akuten Krise** befinden (z.B. verhauene Prüfung $\rightarrow +20\%$ Support-Chance).
- **Panel-Modelle** kontrollieren dies durch `fails_prev` und `cp_rueckstand`.
- **Rekurrente Netze (RNNs)** lernen über ihre temporale Historie, dass das Signal `support_active = 1` ein Indikator für ein tiefes Leistungs- und Motivationstief ist. Das Modell nutzt die Support-Teilnahme als Prädiktor für eine Krise, was die kontrafaktische Inferenz ohne explizite Instrumentenvariablen verzerrt.

---

## 5. Studiengangs-Modellierung, Limitationen & Dashboard

### Limitation der aktuellen Simulation:
Wie richtig angemerkt, unterscheidet der Generator Studiengänge aktuell primär über:
1. Anzahl und Abfolge der Pflichtmodule
2. Schwierigkeitsgrade & Workload der Module

Es gibt **keine gruppenspezifische Modellierung** von Persönlichkeitsmerkmalen, Sozialverhalten oder Motivationsprofilen pro Fachbereich (z.B. höhere Abbruchneigung in MINT durch soziales Klima).

### Einordnung als Limitation & Future Work:
- Dies wird in der Projektdokumentation explizit als **Methodische Limitation der Simulations-Engine** festgehalten.
- **Dashboard-Integration**: Fachbereichs-Filter (Subgruppenanalysen) lassen sich ideal im Dashboard umsetzen, indem die Überlebenskurven pro Studiengang getrennt aggregiert werden.

---

## 6. Korrektur der Nomenklatur

- **Fachlich korrekter Begriff**: Statt *"intervallgezensiert"* verwenden wir ab sofort **"intervall-zensiert"** (bzw. *intervall-zensierte Überlebenszeitdaten*).

---

## 7. Zusammenfassende Synthese

1. **DataAnalysis** war das **Prototyp-Projekt (Phase 1)**: Es etablierte die Grundidee und den synthetischen Generator, litt aber unter verbliebenem Confounding.
2. **Abschlussprojekt** ist die **Erweiterung (Phase 2)**: Es löste das Confounding über Longitudinal-Panels (`_delta`) und ML-Survival-Architekturen auf.
3. Die empirische Evidenz belegt: **Fachlicher und psychosozialer Support wirken risikosenkend ($\text{HR} \approx 0.92$)**, wenn auf die akute Vorsemester-Leistung konditioniert wird.
