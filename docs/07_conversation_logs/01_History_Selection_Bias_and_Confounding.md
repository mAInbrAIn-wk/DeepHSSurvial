# Historie: Selektionsbias, Immortal-Time Bias & Kontrafaktische Sanity-Checks

Dieses Dokument zeichnet die wohl wichtigste methodische Debatte des gesamten *DeepHSSurvival*-Projekts nach: Die schrittweise Auflösung des Dropout-Paradoxons und die Jagd nach dem wahren kausalen Effekt von Hochschulsupport. Es basiert auf dem historischen Gesprächsverlauf (vgl. `00_Historisches_Gesamtprotokoll.md`).

## 1. Die Legacy-Ära (DataAnalysis) und erste Alarmzeichen
Die Auseinandersetzung mit Confounding und Bias begann nicht erst mit den Deep-Learning-Modellen. Bereits im ursprünglichen **Projekt DataAnalysis (DA)** tauchte das Problem auf. In dieser ersten Version gab es noch kein Zeitkontomodell, keine Supportkosten und eine grundlegend andere Supportauswahl. 
Trotzdem produzierten die Modelle bereits damals extrem hohe Raten für die HR-Reduktion. Wir versuchten, dieses Confounding über **Mixed Effects** bzw. Kontrollvariablen in klassischen Cox-Analysen in den Griff zu bekommen (dokumentiert im DA-Dashboard). Dass die extremen HR-Werte durch fehlende Landmarks bzw. den **Immortal-Time Bias** getrieben waren, war uns zu diesem Zeitpunkt noch nicht bewusst.

## 2. Der trügerische Durchbruch im DL-Projekt: HR ~ 0.37
Als wir im DeepLearning-Projekt die dynamischen Confounder in Person-Semester-Panels einführten, schienen sich die extrem starken Effekte des Supports zu wiederholen (vgl. alte Präsentation: HR ~ 0.37 im Extended Cox Modell). Wir schöpften zunächst keinen Verdacht, da wir das Setting (dynamisches Confounding) ja eigentlich erschwert hatten. 
Die methodische Ernüchterung trat erst ein, als sich diese massiven Werte im weiteren Verlauf absolut nicht reproduzieren ließen (*"Das HR > 1 -> HR ~ 0.37-Paradox ist das stärkste methodische Argument im Portfolio. Leider ist gerade dieser Punkt inzwischen fraglich..."*).

Es stellte sich heraus, dass zwei Bias-Quellen nicht vollständig eliminiert waren:
1. **Future-Leakage:** Post-hoc aggregierte Werte (z.B. der finale CP-Rückstand) flossen unbemerkt in frühe Semester ein.
2. **Immortal-Time Bias:** Modelle, die ohne strikte zeitaufgelöste Risikosets auskamen, attestierten dem Support fälschlicherweise einen künstlichen Überlebens-Vorteil, da Support-Nutzer allein durch die Nutzung bewiesen, dass sie noch im System waren.

## 3. Die Entstehung der Paralleluniversen (Ab V3)
Da die "echten" Schätzungen methodisch "in der Luft hingen", brauchte das Projekt eine absolute *Ground Truth*. Daraus entstanden die **deterministischen Paralleluniversen**, deren Einführung bereits in Version 3 geschah (nicht erst im V4-Refactoring!).
Die Datengenerierung schickt denselben Studenten (mit exakt demselben Zufalls-Seed) durch verschiedene Universen. Anfangs waren es 5 Universen, später expandierte dies zu 8 (A-H), wobei die Synchronisierung der Zufallsstreams ein massiver Aufwand war.

* **Universum A:** Voller Support (Baseline)
* **Universum B:** Jede Supportart ist komplett geblockt.
* **Universen C-E:** Jeweils *eine* Supportart ist geblockt.
* **Universen F-H:** Jeweils *zwei* Supportarten sind geblockt (bzw. eine isoliert verfügbar).

## 4. Der "Wahre" Effekt vs. Pearl/Imai
Der direkte Vergleich dieser Universen offenbarte den "wahren" Effekt des Supports (die naive HR von 0.37 war massiv überschätzt). 
**Aber Achtung:** Dieser Effekt ist nur "wahr" im Sinne unserer eigenen kontrafaktischen Simulations-Regeln. Da wir in der realen Welt keine Paralleluniversen haben, bedienen wir uns für die eigentliche Kausalanalyse fortgeschrittener Frameworks nach **Pearl und Imai (Strukturelle Mediation)**. Wir nutzen separate Kontrafaktik-Skripte, um diese Schätzungen *aus unseren Modellen* (unter Nutzung nur eines Universums) zu ziehen, anstatt einfach die Welten A und B zu subtrahieren.

## 5. Der Oracle-Arc und Feedback-Schleifen
Ein weiterer entscheidender Baustein war der Einsatz von **Oracle-Modellen**. Die Simulation nutzt verborgene Zustände (Hidden Motivation, Soziale Integration etc.).
Durch Übergabe dieser latenten Variablen konnten wir mathematisch beweisen, wie das Vorzeichen für den überfachlichen Support im Modell von scheinbar schädlich (RR > 1) auf protektiv umsprang. Dies lieferte den formalen Beweis der fehlerhaften Confounder-Kontrolle in naiven Modellen.

Zusätzlich zeigte sich, dass der Selektionsbias stark von der Support-Art abhängt:
* **Fachlicher Support:** Von den Modellen relativ gut zu erfassen.
* **Überfachlicher Support:** Äußerst schwer zu isolieren, da hier eine direkte Feedback-Schleife auf die Motivation vorliegt.
* **Psychosozialer Support:** Wird eher randomisiert getriggert und daher von den Modellen am leichtesten als echter Schutzfaktor erkannt.

**Status:** Eine umfassende *quantitative* Analyse dieses Selektionsbias quer über alle Kausal-Ansätze (Universen vs. Pearl/Imai vs. Oracle) steht als finale ToDo noch aus.
