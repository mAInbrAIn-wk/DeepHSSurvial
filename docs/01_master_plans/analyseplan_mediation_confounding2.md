# Systematischer Analyseplan: Confounding & Kausale Mediation

Dieses Dokument definiert den Prüfplan, um das Kernproblem des Projekts – **Confounding by Indication (Auswahlverzerrung)** – quantitativ in den Griff zu bekommen. Da wir die Simulation (Ground Truth) zu 100 % kontrollieren, nutzen wir diesen Vorteil konsequent aus: Wir vergleichen die Ergebnisse einer realistischen (blinden) statistischen Analyse mit einer "Oracle"-Analyse, die Zugriff auf die versteckten Variablen hat.

---

## Teil 1: Systematische DGP-Hypothesen nach Support-Art

Für jede der drei Support-Arten definieren wir den wahren Mechanismus, die erwartete statistische Verzerrung und die daraus folgende Hypothese.

### A. Fachlicher Support (Fach_Supp)
*   **Wahrer Trigger:** Spezifische Fehlversuche im betroffenen Modul (`modul_states[m].versuche > 0`) und generelle Leistungserwartung (`erwartete_note`).
*   **Wahrer Mechanismus:** Erhöht dynamisch die latente Modul-Leistung (via `fachlicher_boost`), was im Schnitt zu besseren Noten führt und dadurch weitere Fehlversuche verhindert.
*   **Verzerrungs-Hypothese (Realistic):** Der Support wird gezielt an Studierende vergeben, die bereits in dem spezifischen Modul durchgefallen sind. Da diese Fehlversuche (`fails_prev`) in den Daten sichtbar sind, kann ein gutes Modell (wie Imai) dieses Confounding teilweise abfangen.
*   **Oracle-Hypothese:** Die Oracle-Mediationsanalyse (mit exaktem Wissen über den wahren Overload) wird die Schätzung nur bedingt verändern, da der Haupt-Confounder (Fehlversuche/Noten) messbar ist.

### B. Überfachlicher Support (Uebf_Supp)
*   **Wahrer Trigger:** Primär eine fallende Motivation (Wahrscheinlichkeit steigt proportional zu `0.5 - motivation`).
*   **Wahrer Mechanismus:** Direkter Boost auf `hidden_motivation`.
*   **Verzerrungs-Hypothese (Realistic):** Der Support wird getriggert, wenn die versteckte Motivation im Keller ist. Da realistische Modelle die Motivation nicht sehen, verwechseln sie den Trigger (Krise) mit der Ursache. *Resultat:* Die realistische Mediationsanalyse wird Überfachlichen Support fälschlicherweise als **schädlich** (positiver Average Direct Effect auf Dropout) deklarieren.
*   **Oracle-Hypothese:** Wenn wir `hidden_motivation` in das Oracle-Mediationsmodell aufnehmen, verschwindet der schädliche direkte Effekt.

### C. Psychosozialer Support (Psych_Supp)
*   **Wahrer Trigger:** Ein Random Walk Absturz der Integration (Wahrscheinlichkeit steigt proportional zu `0.5 - soziale_integration`).
*   **Wahrer Mechanismus:** Direkter Boost auf `hidden_integration`.
*   **Verzerrungs-Hypothese (Realistic):** Studierende mit sehr niedriger sozialer Integration (hohes Dropout-Risiko) erhalten diesen Support. Da `hidden_integration` im Modell fehlt, erscheint der Support massiv schädlich.
*   **Oracle-Hypothese:** Durch die Inklusion von `hidden_integration` kehrt sich der scheinbar schädliche Effekt um.

---

## Teil 2: Quantitativer 4-Stufen-Prüfplan

Um diese Hypothesen lückenlos zu beweisen, implementieren und evaluieren wir folgende vier Stufen für **alle drei Support-Typen**:

### Stufe 1: Der Ground-Truth-Abgleich (Macro Effects)
*   **Aktion:** Abgleich der Dropout-Raten zwischen Universum A (alle aktiv) und den Isolations-Universen (B, C, D, E, F, G, H).
*   **Ziel:** Der empirische Nachweis auf Basis unserer synthetischen Kohorte, dass jeder Support-Typ den Dropout in der echten DGP-Mechanik *tatsächlich senkt*.

### Stufe 2: Quantifizierung der Auswahlverzerrung (Data Mining)
*   **Aktion:** SQL-Auswertung des t0-Zustands. Wir vergleichen die Metriken (`fails_prev`, `hidden_motivation`, `hidden_integration`) von Support-Empfängern im Moment der *ersten Inanspruchnahme* mit dem Durchschnitt der Nicht-Empfänger.
*   **Ziel:** Wir weisen die statistische Ausgangslage (das Confounding) quantitativ nach. Wir werden sehen, dass Support-Empfänger eine stark negativ selektierte Kohorte sind.

### Stufe 3: Die Realistische Mediationsanalyse
*   **Aktion:** Auswertung der Metriken aus `structural_mediation_analysis.py` (die aktuell im Nachtlauf berechnet werden). Hier stehen den Modellen nur beobachtbare Variablen (`gpa`, `cp`) zur Verfügung.
*   **Ziel:** Nachweis, dass etablierte statistische Causal-Inference-Methoden an fehlenden Confoundern scheitern und falsche kausale Schlüsse ziehen.

### Stufe 4: Die Oracle Mediationsanalyse (NEU)
*   **Aktion:** Wir schreiben ein neues Analyse-Skript (`oracle_mediation_analysis.py`). Wir zwingen das Mediationsmodell, die Simulations-internen, versteckten Variablen (`hidden_motivation`, `hidden_integration`) zu berücksichtigen. Um zu isolieren, ob die versteckten Variablen primär als Selektions-Trigger oder als Wirkungs-Kanal fungieren, führen wir hierbei **drei separate Analysen** durch:
    1. **Nur als Confounder:** Die Hidden-Variablen werden nur als Kontrollvariablen für das Treatment verwendet.
    2. **Nur als Mediator:** Die Hidden-Variablen werden als reiner Wirkkanal genutzt.
    3. **Beides:** Die Hidden-Variablen fungieren gleichzeitig als Confounder und Mediator.
*   **Ziel:** Wir zeigen, wie sich der ADE und ACME verändern, wenn das Modell die versteckten Variablen auf verschiedene Arten einbezieht.

---

## Offene Punkte & Feedback
1. Bist Du mit der Erweiterung um ein dediziertes `oracle_mediation_analysis.py` Skript einverstanden?
2. Sollen wir in Stufe 4 (Oracle) die versteckten Variablen als **Confounder** (Kontrollvariablen für das Treatment) oder als **Mediatoren** (die Pipeline *durch* die der Support wirkt) modellieren? (Aus Sicht des DGP sind sie eigentlich beides: Trigger und gleichzeitig Wirkkanal!).
