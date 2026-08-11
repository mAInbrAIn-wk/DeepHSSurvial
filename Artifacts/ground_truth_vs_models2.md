# Methodologischer Abgleich: Wahre vs. Geschätzte Effekte

In diesem Dokument stellen wir die **wahren makroskopischen Effekte** (True Causal Macro Effects) aus unserem "Simulator V2" den geschätzten Effekten aus unseren Double Machine Learning (DML) Modellen gegenüber. 

Da wir im Simulator Zugriff auf die "Paralleluniversen" (Klon A: Support, Klon B: Kein Support) haben, können wir die exakte kontrafaktische Wirksamkeit auswerten und die Validität unserer ML-Verfahren überprüfen.

> [!NOTE]
> Wir verwenden in der Projektkommunikation den Begriff **"realistische Effektschätzungen"** für unsere ML-Modelle und sprechen von **"Entstörung"** der Konfunder, anstatt von "wahren Effekten", da in der Realität der "wahre" Kausal-Effekt nie mit absoluter Gewissheit messbar ist. Nur in unserer Simulation können wir das wahre Delta berechnen.

## 1. Das Konzept: Der Trajektorien-Klon (Simulator V2)

Um den tatsächlichen kausalen Effekt zu berechnen, ohne Verzerrungen durch Selektions-Bias (Studenten mit schlechteren Noten nutzen öfter Support), haben wir die komplette Population von 50.000 Studierenden *zweimal* simuliert:
- **Universum A**: Support-Angebote können ganz normal entsprechend der Präferenzen genutzt werden.
- **Universum B**: Support-Angebote sind strikt blockiert. Niemand nimmt teil.

> [!IMPORTANT]
> Um sicherzustellen, dass die Unterschiede *ausschließlich* auf den Support zurückzuführen sind, haben beide Klone eine **exakt synchrone Rausch-Quelle (Seeded RNG)** pro Person erhalten. Krankheiten, Notenabweichungen und Zufallsereignisse treffen Klon A und Klon B im exakt selben Moment – es sei denn, der Support verhindert ein solches negatives Ereignis bei Klon A!

## 2. Ergebnisse der Orakel-Modelle

Bevor wir die Makro-Effekte vergleichen, haben wir untersucht, wie stark die Vorhersagekraft steigt, wenn ein Modell Zugriff auf die normalerweise unsichtbaren mentalen Zustände (`Motivation`, `Soziale Integration`, `Erwartete Note`) hätte.

| Modell-Typ | Baseline AUC | Oracle AUC (mit Hidden-Variablen) | Lift |
| :--- | :--- | :--- | :--- |
| **Logistic Hazard Delta** | 0.7969 | 0.8065 | **+0.0096** |
| **DeepSurv Delta** | 0.5206 | 0.5269 | **+0.0063** |

Der Lift zeigt: Diese versteckten Zustände haben eine direkte kausale Relevanz für den Dropout, die dem Basismodell verborgen bleibt und erst verzögert durch schlechtere Prüfungsleistungen sichtbar wird.

## 3. Vergleich: Wahre vs. Geschätzte Effekte

> [!CAUTION]
> Dieser Abschnitt wird nach dem Simulation-Rerun (mit stärkerem Signal und 5 kontrafaktischen Welten) mit verifizierten, berechneten Zahlen neu befüllt. Die vorherige Version enthielt nicht-belegbare Zahlen.

*Ergebnisse ausstehend.*

## Fazit zur Methodik

*Wird nach dem Rerun geschrieben — ausschließlich auf Basis berechneter, nachprüfbarer Metriken.*
