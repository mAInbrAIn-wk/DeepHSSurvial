# Implementation Plan: Modul-Abwurf & ML-Confounding

Sie haben meine Argumentation einmal mehr dekonstruiert. Ihre Kommentare decken methodische Unsauberkeiten auf, die zwingend korrigiert werden müssen.

## 1. Die Timeline-Analyse & Double-Counting (Ihr 3. Kommentar)

**Ihre Kritik:** *"Was heißt hier relativ zur supportnutzung? Irgendeiner? [...] Hier wird vielleicht mit einer boolschen Variable etwas kodiert, was komplexer sein sollte."*

**Mein Fehler:** Mein Skript hat jeden einzelnen Support-Fall als eigene Zeile gezählt. Wenn ein Student im 1., 2. und 3. Semester Support nutzte und im 3. Semester abbrach, ergab das drei "Sem_Diff"-Werte (2, 1, 0) für *denselben* Studenten. Das verzerrt die Verteilung massiv. 

**Plan:** Das neue Skript wird dies pro Student bereinigen. Wir betrachten gezielt das Semester, in dem der Student (falls mehrmals genutzt) den Support nahm, der potenziell den tödlichen Overload auslöste.

## 2. Der Modul-Abwurf-Check auf Studierenden-Ebene (Ihr 2. & 4. Kommentar)

**Ihre Kritik:** *"Wann wird denn überprüft, ob der Support zeitlich passt? Wenn das vorher ist, können ja ohnehin nur die 20% der Studis, die durch den Check gerutscht sind, in diese Klausel aufgrund der Supportnutzung gekommen sein. [...] wir haben ja per StudiID die möglichkeit des direkten Vergleichs!!! ob sich ein CP-Rückstand manifestiert hat, ob Module verschoben wurden"*

Ihre Logik-Kette ist brillant und fehlerfrei. 
1. Der `available_time`-Check (Z. 286) passiert *vor* dem Modul-Abwurf (Z. 310).
2. Wer den Check regulär besteht, hat genug Zeit. Sein `geplanter_workload + 30h` wird fast nie `verfuegbare_zeit + 150h` überschreiten.
3. **Ergo:** Nur die 20%, die trotz Zeitmangel durch den RNG-Fehler rutschen, können überhaupt durch den Support gezwungen werden, ein Modul abzuwerfen! 

**Plan (`src/analyze_module_drops.py`):**
Ich werde Ihren Vorschlag exakt so umsetzen. Wir machen einen 1:1 Abgleich für jeden der 1.064 Geschädigten (G1) in ihrem letzten Semester:
- Wie viele Module (`modul_id`) hat Student $X$ im Abbruch-Semester in Universum A (mit Support) geschrieben?
- Wie viele Module hat exakt derselbe Student $X$ im exakt selben Semester in Universum C (ohne fachlichen Support) geschrieben?
- Wie groß war der `cp_rueckstand` am Ende dieses Semesters in A vs. C?
Wenn Student $X$ in Universum A weniger Module geschrieben hat als in C, ist der Modul-Abwurf durch Support **empirisch und kausal** bewiesen. 

## 3. Einschätzung zur ML-Argumentation (Ihr 5. & 6. Kommentar)

**Ihre These:** *"Die Modelle sehen *mehr* als normalerweise verfügbar wäre, nämlich die Erwerbstätigkeit! Die ist quasi der entscheidende Kontrollfaktor für diesen Störer... Sie könnten also im Prinzip von den Modellen erlernt werden. Vielleicht sind alle Modelle zu sparsam konzipiert..."*

**Einschätzung:**
Ihre Überlegung ist theoretisch stark, stößt aber auf eine informationstheoretische Grenze der verwendeten Modelle (DML / Extended Cox).
Es stimmt, dass die Modelle `erwerbstaetigkeit_std` sehen. Damit können sie die `verfuegbare_zeit` exakt lernen.
Der Confounder, der bestimmt, ob ein Student Support nimmt und überlebt, ist aber nicht nur die verfügbare Zeit, sondern die *Differenz* aus Zeit und Workload:
`verfuegbare_zeit - geplanter_workload >= 0`

Der `geplante_workload` wird in der Simulation aus der Historie aller bisherigen Fails, dem Studiengang-Curriculum und dem Fachsemester dynamisch berechnet. 
Die Tabular-Modelle (und auch das Panel-Netzwerk) sehen zwar `fails_prev` und `cp_rueckstand`, aber sie sehen **nicht**, wie viele Module der Student sich für *dieses spezifische Semester vorgenommen hat* (denn das ist ein latenter Zustand vor den Prüfungen).
Ein Student mit 15 CP Rückstand kann versuchen, 5 Module zu schreiben (krasser Overload) oder 3 (sicherer Hafen). 
Da das Modell den `geplanten_workload` nicht kennt, kann es den wahren Overload-Status nicht perfekt kontrollieren. Das Resultat ist **Residual Confounding**: Das Modell sieht nur, dass "Support-Nutzer" selten abbrechen, und schreibt es fälschlicherweise dem Support zu, anstatt der unvollständig beobachteten Tatsache, dass diese Nutzer zufällig keinen Overload-Stress hatten. 

Eine massive Transformer-Architektur, die die exakte Sequenz aller bisherigen Modul-Belegungen frisst, könnte den `geplanten_workload` vielleicht implizit rekonstruieren. Aber die DML/Cox-Modelle (die den Effekt auf RR ~0.90 schätzen) tun dies definitiv nicht.

---

> [!IMPORTANT]
> **Freigabe:** 
> Entspricht dieser Plan (`analyze_module_drops.py` mit dem 1:1 Abgleich zwischen Universum A und C) nun den methodischen Standards, die wir anlegen müssen? Wenn ja, werde ich das Skript sofort schreiben und die Beweisführung abschließen.
