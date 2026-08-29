# Optimierungs- und Berichtigungsplan für Datensimulation V4

Dieses Dokument fasst methodische Hebel für eine kommende **Version 4** der Simulation zusammen.

## 1. Bereits diskutierte Korrekturen & mathematische Optimierungen

### 1.1 Beta-Verteilung statt geclippter Normalverteilung
* **Status Quo:** Variablen wie `motivation` oder `soziale_integration` werden über `np.clip(rng.normal(loc, scale), 0.0, 1.0)` simuliert.
* **Das Problem:** Dies führt zu unnatürlichen Wahrscheinlichkeits-Massen (Spikes) exakt bei den Rändern `0.0` und `1.0`. Modelle interpretieren das oft als binäres kategoriales Feature statt als stetige Verteilung.
* **Lösung V4:** Umstellung auf `rng.beta(a, b)`. Die Parameter `a` und `b` lassen sich so einstellen, dass sie elegante, natürlich abfallende Ränder auf [0, 1] erzeugen.

### 1.2 Das "Motivations-Paradox" beim überfachlichen Support
* **Status Quo:** Die Wahrscheinlichkeit, **überfachlichen Support** aufzusuchen, ist streng deterministisch an die fehlende Motivation gekoppelt (`p = 0.5 - motivation`). *(Korrektur: Der psychosoziale Support hängt am Random-Walk der sozialen Integration, nicht an der Motivation).*
* **Das Problem:** Dies hat zu dem "Confounding by Indication"-Paradox in der Mediationsanalyse geführt. Es impliziert: Je demotivierter der Student, desto wahrscheinlicher sucht er proaktiv Hilfe.
* **Lösung V4:** Einführung von Reibungsverlusten ("Friction"). Die Funktion könnte parabelförmig sein: Bei sinkender Motivation steigt die Support-Nutzung zunächst, bei totaler Apathie (`motivation < 0.2`) kollabiert sie jedoch. Da unser Setting künstlich ist, ist dies ein hervorragender Hebel, um das Confounding experimentell zu steuern.


## 2. Erkenntnisse aus der Datenprüfung (Kein direkter Änderungsbedarf)

### 2.1 Die Zeitbudget-Logik (Feature, kein Bug)
* **Status Quo:** Der Support wird vom Zeitbudget abgezogen, *bevor* der Modul-Workload geprüft wird.
* **Erkenntnis:** Eine Umkehrung dieser Logik wäre fatal! Wenn wir den Modul-Workload zuerst abziehen, würden Studierende, die im Rückstand sind (und extrem viel Workload vor sich herschieben), zeitlich systematisch vom Support ausgeschlossen. Die aktuelle Logik (Support hat Prio, überzählige Module werden *danach* gestrichen) spiegelt das realistische Coping-Verhalten wider. Wir belassen dies so!

### 2.2 Dropout ist bereits hochgradig stochastisch
* **Erkenntnis:** Der Dropout ist in V3.6 keineswegs eine feste Hürde (`motivation < 0.1`). Die Funktion `berechne_dropout()` würfelt die Abbruch-Wahrscheinlichkeit aus einer Kombination von Motivation, CP-Rückstand, aktueller Durchfall-Quote und Fachsemester aus. Die Logik ist bereits exzellent und vielschichtig.

## 3. Backlog (Für zukünftige Evaluierung)

### 3.1 "Ghosting" als Abgangsart (Anomalie-Typ "Plateau" nutzen)
* **Idee:** Ein Student meldet sich trotz Immatrikulation für 2-3 Semester zu *gar keiner* Prüfung mehr an, bevor er exmatrikuliert wird (leere Semester/Padding). 
* **Umsetzung:** Dies kann elegant über den bereits existierenden, aber ungenutzten `anomalie_typ == "Plateau"` integriert werden.

### 3.2 Kausaler Kohorten-Drift (Soziale Isolation durch Fehlversuche)
* **Idee:** Fehlversuche haben aktuell kaum direkten Einfluss auf die Freundesgruppe. Ein direkter Malus auf die `soziale_integration` bei Wiederholungsversuchen (weil man den Anschluss an die eigene Erstsemester-Kohorte verliert) würde die Kaskade zum psychosozialen Support und Dropout noch realistischer gestalten.
