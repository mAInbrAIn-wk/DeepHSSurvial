# Warum synthetische Daten selbst generieren, statt auf bestehende Daten zurückzugreifen?

Lieber Axel,

falls Du Dich fragst, warum ich Deinem Rat nicht gefolgt bin, und die Erstellung eines Datensatzes der Arbeit mit öffentlich zugänglichen Daten vorgezogen habe, habe ich hier die Rechtfertigung (und die vorausgehende Recherche) dokumentiert.

## Allgemeine Überlegungen

- selbst generierte, synthetische Daten erlauben völlige *Kontrolle und Transparenz* der zugrundeliegenden Parameter (bzw. allgemeiner des Modells)
    - das gewählte lineare Modell mit Rauschen sollte sich ideal zur statistischen Analyse eignen
    - es kann erweitert und verkompliziert werden, um zu sehen, wie die am quasi Idealfall entwickelte Analyse damit zurecht kommt, bzw. wo sie erweitert werden soll
    - in gewissem Maße kann es also als Benchmark der Analyse dienen
- ich habe quasi dieselbe Fragestellung schon quasi schon im vorherigen Data Engineer Modul bearbeitet, da allerdings komplett "zu Fuß" in SQL 
    - es war sehr *lehrreich* in mehreren Hinsichten mir Python Code mit Hilfe der KI generieren zu lassen (und dann natürlich selbst nachzuvollziehen und anzupassen); ich habe das in Mammouth AI u.a. mittels Deines Systemprompts `Promptgenerator` und Claude Opus 4.7 erstellt
        - ich muss wirklich sagen, dass die KI das Programmieren enorm vereinfacht und beschleunigt, auch wenn es ein paar Punkte gibt, die schneller von Hand gehen, als im Wechselspiel mit der KI (key mismatch, wenn der Code in Blöcken kommt ;-)
        - auch auf konzepueller Ebene kann der Dialog mit dem Chatbot sehr nützlich sein
        - Python eignet sich wirklich sehr gut für flexible, quantitative Modellierung und Simulation
    - ich hoffe, dass ich mir so über Module hinweg ein größeres Projekt für ein Portfolio aufbauen kann


## Spezifische Überlegungen 

### ...zu den Alternativen (i.e. freie Datensätze)

In der Recherche haben sich vo allem folgende zwei Optionen als attraktiv herausgestellt:

1. „Die Studierendenbefragung in Deutschland 2021“ (DZHW, SID2021): [Metadaten & Datenzugang](https://metadata.fdz.dzhw.eu/de/data-packages/stu-sid2021)
2. MoSAiK-Studierendenbefragung (Längsschnitt, Uni Koblenz/Landau): [PsychArchives](https://www.psycharchives.org/handle/20.500.12034/8518)

Erstere ist eine *große* __Querschnittstudie__  mit zahlreichen Variablen (an der selben Stelle finden sich zwar auch einige Panele, aber diese haben meist weniger interessante Variablen), letztere eine kleinere, aber ebenfalls Variablenreiche Lägnsschnittstudie, die aber leider noch nicht öffentlich zugänglich ist (`embargoed` bis Ende *nächsten* Jahres); für erstere müsste man einen Zugang beantragen. Nun sind Punktaufnahmen für mein Analysevorhaben weniger interessant und würden andere Anaylseverfahren erfordern. Beide Datensätze stützen sich auf Fragebögen i.e. Selbstauskünfte, könnten also primär für *wahrgenommene* Effektivität genutzt werden (pace Selbstauskunft zu Notenentwicklung); bislang habe ich keinen Datenzugang für diese Daten beantragt. 

Die meisten Datensätze aus kaggle zu Suchgebriffen mit Stoßrichtung meines Projektes liefern Datensätze, die synthetisch sind und/oder nicht so viele relevante Parameter enthalten, vor allem aber meist zu higher education außerhalb Deutschlands.

Prinzipiell wären auch Panele aus anderen Ländern relevant, sofern sie das mit der Bologna Reform eingeführte ECTS nutzen, aber an dieser Stelle habe ich die Suche erstmal beendet.

### ...zu den syntethischen Daten

Die jetzt per script generierten Datensätze sollten sehr gut für eine Analyse geeignet sein und ihre Qualität ist steuerbar und automatisch prüfbar. (Ein paar basale Validierungstests werden gleich bei der Erzeugung ausgeführt, weitere sind freilich denkbar - s.u..)

Sie haben eine klare Struktur, die sich leicht etwa in einer SQL-Datenbank realisieren ließe.

Zudem sind es Daten, die im Realfall auf Rechnern (sicher nicht zentralisiert, Studierendensekretariat; Dekanat und Prüfungsamt haben andere Informationen, aber die in dem erzeugten Datensatz enthaltenen Daten wären meist im Prüfungsamt lokalisiert) der Hochschule in der ein oder anderen Form vorgehalten werden würden, darüber hinaus in einer Form, die mehr oder weniger gut zu der (gewissermaßen normalisierten) Datenstruktur, die hier verwendet wird, korrespondiert.

> *Wichtige Ausnahme*: Die Daten zum Veranstaltungsbesuch werden im Normalfall (gerade bei extracurriculären Veranstaltungen *ohne* Prüfung und CP-Erwerb) nicht so detailliert erfasst, oder ohne Aufwand einzelnen Studierenden zuzuordnen. Sofern die Supportangebote eine digitale Komponente enthalten, wäre das aber *technisch* durchaus realisierbar.

> ### *Warnung:* 
>Das heißt natürlich **nicht**, dass eine solche Analyse problemlos durchgeführt werden könnte! Der entscheidende Faktor ist hier der **Datenschutz**! Gerade Prüfungsdaten sind hochsensible Informationen (auf die leider auch etliche Begehrlichkeiten bestehen), diese zu verarbeiten und mit anderen Daten zu kombinieren ist *höchst* problematisch und darf in keinem Fall intransparent oder ohne explizite Zustimmung verlaufen. Eine echte Anonymisierung der Daten ist hier auch schwierig, es kann sehr leicht Profiling betrieben werden etc. 
>
> Wie bei vielen realen Anwendungsszenarien sollte man sich hier der Risiken bewusst sein, eine rechtliche Absicherung suchen und eine genaue Kosten-/Nutzen-Analyse durchführen!

## Möglicher Bezug auf reale Datensätze

Falls noch Zeit bleibt, könnte aus allgemein verfügbaren Datensätzen weitere Validierungen des generierten Datensatzes abgeleitet werden, oder manche der Parameter geschätzt werden. --> Nice2Have