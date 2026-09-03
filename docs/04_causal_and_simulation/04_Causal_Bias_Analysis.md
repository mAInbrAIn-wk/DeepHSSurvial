# Causal Bias Analysis: Naive Estimation vs. Counterfactual Ground Truth

## 1. Absolute Ground Truth (Universum A vs. Universum B)
Um den wahren kausalen Effekt des Support-Angebots zu evaluieren, betrachten wir zwei parallele Universen:
- **Universum A (Baseline)**: Support-Angebote sind verfügbar.
- **Universum B (Kontrafaktisch)**: Support-Angebote sind global blockiert.

**Ergebnisse:**
- Dropout-Rate Universum A: **23.92%**
- Dropout-Rate Universum B: **27.59%**
- **Wahrer Effekt (Absolute Risikoreduktion):** -3.67 Prozentpunkte
- **Wahre Relative Risikoreduktion (RR):** 0.239 / 0.276 ≈ **0.867**

Das globale Support-Programm reduziert die Abbruchquote in der Gesamtpopulation also tatsächlich um etwa 3.7 Prozentpunkte.

## 2. Der Naive Selektionsbias (innerhalb Universum A)
Wenn wir das Universum B nicht hätten und nur die Beobachtungsdaten aus Universum A naiv auswerten (Vergleich: Studierende *mit* vs. *ohne* Support-Nutzung):

- Dropout-Rate von Support-Nutzern: **20.65%**
- Dropout-Rate von Nicht-Nutzern: **36.03%**
- **Naiver Relativer Risiko-Quotient (RR):** **0.573**

> [!WARNING]
> **Immortal Time Bias & Confounding**
> Der naive Vergleich suggeriert, dass Support die Abbruchwahrscheinlichkeit beinahe halbiert (RR = 0.57). Dies ist eine massive Überschätzung des wahren Effekts (RR = 0.867). Dieses Artefakt entsteht größtenteils durch **Immortal Time Bias**: Studierende müssen lange genug im Studium "überleben", um überhaupt an Support-Maßnahmen teilnehmen zu können. Wer früh abbricht, hatte schlicht weniger Zeit und Gelegenheiten für Support-Maßnahmen, was zu einer artifiziellen Korrelation zwischen Überleben und Support-Nutzung führt.

*(Hinweis zur ursprünglichen Hypothese: Die Vermutung, dass Support-Nutzer naiverweise ein HÖHERES Risiko zeigen könnten - etwa weil nur stark gefährdete Studierende Support suchen (Confounding by Indication) -, wird hier vom gegenläufigen Immortal Time Bias drastisch überdeckt.)*

## 3. Scheitern klassischer Kontrollvariablen (Logistische Regression)
Ein Standardansatz in der Datenanalyse ist es, für beobachtbare Störfaktoren zu kontrollieren. Wir haben eine logistische Regression auf Universum A angewendet und den Faktor `hzb_note` (Hochschulzugangsberechtigungs-Note) statistisch kontrolliert:

| Variable | Koeffizient (Log-Odds) | p-Wert |
| :--- | :--- | :--- |
| `used_support` | -1.1857 | < 0.001 |
| `hzb_note` | +1.6012 | < 0.001 |

**Interpretation:**
Selbst nach statischer Kontrolle der HZB-Note bleibt der Effekt von `used_support` massiv überschätzt (-1.18 Log-Odds entsprechen einer Odds-Ratio von ca. 0.31). 
Das belegt empirisch die in "03_Uebersicht_Kausale_Ansaetze.md" diskutierte Problematik: Einfache Regressionsmodelle mit statischen Kontrollvariablen sind ungeeignet, um zeitliche Dynamiken (wie den Immortal Time Bias) aufzulösen, und scheitern folglich komplett daran, den wahren kausalen Effekt isoliert zu schätzen.
