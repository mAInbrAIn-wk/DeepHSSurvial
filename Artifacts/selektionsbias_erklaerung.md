# Die Architektur des Selektionsbias: Kausale Feedback-Schleifen in DeepSupport

Dieses Artefakt dokumentiert die mathematischen und strukturellen Mechanismen des Selektionsbias im DeepSupport-Simulator (`simulation_v3.py`). Es verdeutlicht, warum maschinelle Lernverfahren (ohne DGP-Orakel) die wahren Kausaleffekte der Support-Maßnahmen systematisch unterschätzen oder sogar ins Gegenteil verkehren (falsches Risiko).

## 1. Das Konzept der Zielgruppenerreichung vs. Bias
Aus Sicht der Hochschuldidaktik und des Support-Managements ist ein starker Selektionsbias "wünschenswert": Wenn Support-Programme vor allem von den Studierenden genutzt werden, die sie am dringendsten benötigen, spricht dies für eine hervorragende Zielgruppenerreichung.
Für das maschinelle Lernen (Prediction) führt exakt dieses erwünschte Verhalten jedoch zu einem fundamentalen Bias: Das Modell lernt die Korrelation **"Support-Nutzung geht mit Studienabbruch einher"**, da die Maßnahme als Proxy für die zugrundeliegende Krise fungiert.

## 2. Analyse der drei Support-Kategorien im Simulator

Die drei Support-Kategorien (Fachlich, Überfachlich, Psychosozial) sind im Code als unterschiedliche Kausalstrukturen angelegt. Dies erlaubt es, die Fähigkeiten der Modelle unter verschiedenen Bias-Szenarien zu evaluieren.

### A. Fachlicher Support (Feedback-Schleife auf Leistungs-Ebene)
Der fachliche Support enthält den direktesten und härtesten Feedback-Loop, gekoppelt an die akademische Leistung (Prüfungsversuche).

**Mechanismus im Code:**
```python
p = 0.05 + (studi.erwartete_note - 2.0) * 0.05
for m in geplante_relevante:
    if studi.modul_states[m].versuche > 0:
        p += 0.20
```
- **Der Loop:** Studierende, die bei einer Prüfung durchfallen (`versuche > 0`), weisen im Folgesemester eine um $20$ Prozentpunkte erhöhte Wahrscheinlichkeit auf, fachlichen Support aufzusuchen.
- **Kausale Kette:** Durchfall $\rightarrow$ Support-Nutzung $\rightarrow$ (leicht verbesserte Note) $\rightarrow$ Risiko-Reduktion.
- **Problem für ML:** Ein durchgefallener Versuch korreliert extrem stark mit endgültigem Dropout (z.B. Drittversuch-Exmatrikulation oder CP-Rückstand). Da das Modell nicht weiß, dass der Support *aufgrund* des Durchfallens aufgesucht wurde, "bestraft" das Modell den Support als Vorbote des nahenden Dropouts. Der wahre Schutzeffekt wird maskiert.

### B. Überfachlicher Support (Feedback-Schleife auf Motivations-Ebene)
Hier wirkt die Feedback-Schleife subtiler und verläuft über eine latente (verborgene) Variable: die Motivation.

**Mechanismus im Code:**
```python
# 1. Trigger
p = 0.05 + (0.5 - studi.motivation) * 0.15

# 2. Performance Feedback auf Motivation
studi.motivation = max(0.05, studi.motivation - 0.05 * durchgefallen_dieses_sem)
```
- **Der Loop:** Schlechte Leistungen reduzieren am Semesterende die Motivation. Niedrige Motivation erhöht unmittelbar das Risiko für Dropout **und** gleichzeitig die Wahrscheinlichkeit, überfachlichen Support (z.B. Zeitmanagement, Lerncoaching) aufzusuchen.
- **Problem für ML:** Da die Variable `motivation` im echten Leben (und im `realistic`/`standard`-Modus) nicht gemessen wird, ist sie ein *Unobserved Confounder*. Das Modell sieht nur: Studierende mit vielen Fehlversuchen nehmen Support und brechen dann ab. Es kann den positiven Effekt des Supports auf die Motivation nicht vom extremen Risiko der niedrigen Initial-Motivation trennen.

### C. Psychosozialer Support (Kontrollgruppe ohne Feedback-Schleife)
Der psychosoziale Support fungiert in der Simulations-Methodik als strukturelle Kontrollgruppe, um zu prüfen, ob die Modelle Support-Effekte isolieren können, wenn *kein* reaktives Leistungs-Feedback vorliegt.

**Mechanismus im Code:**
```python
# 1. Trigger
p = 0.01 + (0.5 - studi.soziale_integration) * 0.12

# 2. Update (Random Walk, KEIN Einfluss durch Klausuren)
studi.soziale_integration = float(np.clip(studi.soziale_integration + rng_social.normal(0, 0.05), 0.05, 1.0))
```
- **Der Loop:** Die `soziale_integration` entwickelt sich als Random Walk (Zufallsprozess). Sie wird *nicht* durch Fehlversuche oder CP-Rückstände gesteuert (im Gegensatz zur Motivation). Sie beeinflusst zwar das finale Dropout-Risiko, ist aber unabhängig von der akademischen Historie.
- **ML Konsequenz:** Hier fehlt die klassische Leistungs-Feedbackschleife. Wer psychosozialen Support nimmt, tut dies aus zufälligen (im Modell: extrinsischen) Schwankungen der sozialen Einbindung. Wenn Modelle hier dennoch den Kausaleffekt verfehlen, liegt dies an der geringen Basisrate und statistischem Rauschen, nicht aber an einer harten Verknüpfung mit "Durchfallern".

## Fazit zur Kausal-Inferenz
Die Simulation ist highly realistic:
1. **Zielgruppen-Paradox:** Support wird von den vulnerabelsten Gruppen genutzt.
2. **Kausale Blindheit von Standard-ML:** Ohne Instrumentvariablen (IV), Propensity Score Matching (PSM) oder das Oracle-Wissen (`hidden_motivation`) ist ein Machine-Learning-Modell mathematisch gezwungen, den Support teilweise als Risikomarker (Proxy) fehlzuinterpretieren.
3. Dies erklärt, warum in der Grid-Evaluation Modelle im `realistic` Modus zwar hochpräzise Prognosen liefern (hoher PR-AUC), aber die Kausalkraft (RR/HR) verfehlen oder invertieren.
