# Methodische Reflexionen & Real-World Kontext

*Dieses Dokument bündelt die methodischen Überlegungen aus den früheren Projektphasen (DataEngineering und DataAnalysis), die als konzeptionelles Fundament für die kausale Modellierung in DeepHSSurvival dienen.*

## 1. Herausforderungen in der Realität (aus DataEngineering)

In der Praxis einer Hochschule sind bei der Datenerhebung und -zusammenführung folgende Schwierigkeiten zu erwarten, die in unserer Simulation abstrahiert wurden:
- **Verteilte Systeme:** Daten sind vermutlich in verschiedenen operativen Systemen gespeichert (etwa Prüfungsamt, Studentensekretariat und Personalabteilung).
- **Fehlende Zuordnung:** Gerade bei optionalen Veranstaltungen gibt es oft keine verlässlichen Anmeldeinformationen (bei Vorkursen ist etwa noch gar keine Matrikelnummer vorhanden).
- **Digitale Spuren (LMS):** Bei digitalen Veranstaltungen (Veranstaltungen mit Online-Komponente, Materialdownload):
    - Im Prinzip sind Log- und Userinformationen verfügbar, wenn auch ggfs. anonymisiert &rarr; Konsolidierung/Verknüpfung schwierig im ETL.
    - Diese Informationen können statistisch aufschlussreich sein (manchmal Engagementsindikatoren enthalten, z.B. bei digitalen Aufgaben) &rarr; aber oft schwer zu aggregieren (Anonymisierung und verschiedene Tools LMS vs. Tools zum Aufgabenerstellen etc.).
- **Datenschutz & Profiling:** Datenschutz ist hier besonders relevant, da sensible Informationen (Prüfungsdaten) und Aggregierung eventuell Profiling möglich machen.

## 2. Parameter zur Evaluation der Support-Wirksamkeit (aus DataAnalysis/kpi.md)

Die Kernfrage lautet: Erhöht Support den Studienerfolg?

**Zielmetriken (Erwartungen):**
- Die Quote für späte Dropouts trotz früher/häufiger Supportnutzung sollte gering sein.
- Es sollte keine Verlängerung oder Verschlechterung trotz Support eintreten (Supportnutzung erhöht idealerweise nicht den Erwartungswert für Abschlussdauer oder Note).
- Anzahl der Fehlversuche sinkt.
- **Konzeptionelle Reibung:** Support beseitigt gezielt Engpässe und Reibungsstellen. *Aber ist Reibung nicht essentiell für echtes Lernen?* (Wichtig: Reibung heißt nicht zwingend Durchfallen!).

**Zielgruppen-Erreichung (Targeting):**
- Es sollte wenig Studierende mit Schwierigkeiten geben, die keinen Support nutzen.
- Umgekehrt: Studierende mit Schwierigkeiten suchen sich passenden Support.
- Ein Fehlversuch ohne Support erhöht die Wahrscheinlichkeit der passenden Supportnutzung im Folgeversuch.

### Konkrete KPIs (Deskriptiv vs. Kausal)
| KPI | Ebene | Interpretation |
|---|---:|---|
| Bestehensquote mit/ohne Supportexposition | Prüfung | unmittelbarer Prüfungserfolg |
| Durchschnittsnote mit/ohne Support | Prüfung | Leistungsqualität |
| Fehlversuchsquote | Prüfung/Studium | Reduktion von Reibungsverlusten |
| Abschlussquote nach Supportnutzung | Studium | langfristiger Erfolg |
| Abschlussnote bei Absolvent:innen | Studium | langfristige Leistungsqualität |

**Kausalitäts-Disclaimer:** Diese Unterschiede sind zunächst deskriptiv und *nicht* kausal interpretierbar, da Supportnutzung selektiv erfolgen kann (Selektionsbias). Genau hier setzt die Survival-Analyse (V4) von DeepHSSurvival an.
