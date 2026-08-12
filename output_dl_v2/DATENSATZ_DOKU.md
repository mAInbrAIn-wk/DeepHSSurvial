# Synthetische Studienverlaufsdaten – Design & Validierung (DL-Edition)

## Zweck
Synthetischer Datensatz zur Analyse der Wirkung von Studierenden-Support-Angeboten auf Studienerfolg, Abschlussnote und Abbruchverhalten. Speziell optimiert für **Deep Learning / Zeitreihenmodelle**, Kausalanalyse (Counterfactual Ground Truth) und Survival-Analysen.

## Umfang
- **Studierende**: 50,000
- **Prüfungen gesamt**: 839,648
- **Support-Teilnahmen gesamt**: 145,602
- Kohorten: 2015–2024 (WS-Start)
- 5 Studiengänge, 89 Module, 12 Support-Angebote

## Datenmodell (Tabellen)
| Tabelle | Inhalt |
|---|---|
| `studierende` | Stammdaten + latente Eigenschaften (Motivation, soziale Integration, Erwerb) |
| `studiengaenge` | 5 Studiengänge mit Regelstudienzeit und CP-Summe |
| `module` | Module mit CP, Schwierigkeit, Turnus (WS/SS/beides), Workload (h) |
| `modul_studiengang` | n:m-Zuordnung Modul ↔ Studiengang mit empfohlenem Fachsemester |
| `semester` | Chronologische Semester-Liste (SS/WS) |
| `einschreibungen` | Pro Studi × Semester (aktive Einschreibung & Fachsemester) |
| `pruefungen` | Prüfungsversuche mit Note, Bestehen und `note_counterfactual` (Ground Truth) |
| `support_angebote` | Tutorien, Beratung, Sprachkurse etc. (3 Kategorien) mit Zeitkosten |
| `support_modul_zuordnung` | Welches Angebot unterstützt welches Modul, mit Wirkungsstärke |
| `support_teilnahmen` | Studi × Angebot × Semester |
| `abschluesse` | Endstatus pro Studi (`abgeschlossen`, `abgebrochen`, `exmatrikuliert`, `zeitueberschreitung`) mit `bachelorarbeitsnote` |
| `agg_pruefungen` | **(Aggregiert)** Längsschnitt-Tabelle aller Prüfungen inkl. Support-Exposition (vorher/gleichzeitig) |
| `agg_abschluesse` | **(Aggregiert)** Querschnitt-Tabelle auf Studierenden-Ebene mit allen Kontrollvariablen für EDA/Dashboards |

## Statistische Übersicht der Abgangsarten

### Gesamtübersicht
- **Abschlussquote gesamt**: 70.0%
- **Insg. nicht erfolgreich** (Abbruch/Exmatrikulation/Zeitüberschreitung): 30.0%
- **Mittlere Abschlussnote**: 2.13
- **Mittlere Bachelorarbeitsnote**: 1.69
- **Mittlere Studiendauer (Erfolgreiche)**: 7.97 Semester

### Aufschlüsselung nach Studiengängen
| stg_name | abgeschlossen | abgebrochen | exmatrikuliert | zeitueberschreitung | Gesamt | Abschlussquote |
| --- | --- | --- | --- | --- | --- | --- |
| BWL | 9957 | 3545 | 467 | 32 | 14001 | 71.1% |
| Informatik | 8680 | 3431 | 371 | 59 | 12541 | 69.2% |
| Maschinenbau | 5861 | 2548 | 347 | 85 | 8841 | 66.3% |
| Psychologie | 6060 | 2122 | 326 | 31 | 8539 | 71.0% |
| Soziale Arbeit | 4462 | 1408 | 160 | 48 | 6078 | 73.4% |

## Modellarchitektur & Erweiterungen (DL-Edition)

### 1. Zeitkontenmodell (Time Budgeting)
- Studierende verfügen pro Semester über ein festes **Zeitbudget** (Standard: 900 Stunden für Vollzeit).
- Erwerbstätigkeit (z.B. 15h/Woche) zieht direkt Stunden vom Zeitkonto ab.
- Das Belegen von Modulen (Workload = CP × 30h) und die Teilnahme an Support-Angeboten zehren ebenfalls am Zeitkonto.
- Wenn der Gesamtaufwand das verfügbare Budget übersteigt, tritt eine **Overload-Penalty** ein (lineare Senkung der latenten Prüfungsleistung).

### 2. Turnus & Kalendarische Semester
- Module finden spezifisch im **Wintersemester (WS)**, **Sommersemester (SS)** oder **beiden** statt.
- Die Simulation unterscheidet strikt zwischen chronologischem Kalendersemester und absolviertem Fachsemester. 

### 3. Reaktive Support-Nutzung
- **Fachbezug**: Fachlicher Support (z.B. Mathe-Tutorium) wird nur gewählt, wenn in dem Semester auch ein passendes Modul belegt wird.
- **Reaktivität auf Fehlversuche**: Nach einem Nichtbestehen im Vorsemester steigt die Inanspruchnahme von Support-Angeboten für dieses Modul im Wiederholungsversuch um bis zu +20 Prozentpunkte.

### 4. Counterfactual Ground Truth
- In `pruefungen.csv` wird die **`note_counterfactual`** geloggt (die hypothetische Note ohne den Support-Boost).
- Dies ermöglicht die spätere kausale Evaluierung von Deep-Learning-Modellen gegen die wahre Ground Truth.

## Validierungsergebnisse (Konsistenz-Checks)

- ✅ Alle Prüfungen referenzieren existierende Studierende
- ✅ Alle Einschreibungen referenzieren existierende Studierende
- ✅ Alle Abschluss-Datensätze referenzieren existierende Studierende
- ✅ Jede/r Studierende hat genau einen Abschluss-Datensatz
- ✅ Alle Noten in gültigem Bereich (1.0–5.0)
- ✅ Max. 3 Versuche pro Modul
- ✅ Bestanden ⇔ Note ≤ 4.0
- ✅ Nur erfolgreiche Abschlüsse ('abgeschlossen') enthalten eine Abschlussnote
- ✅ Erfolgreiche Abschlüsse besitzen eine valide Bachelorarbeitsnote
- ✅ Gesamtabschlussquote ist größer als 50% (ist: 70.0%)
- ✅ Abschlussquote in allen Studiengängen einzeln > 50% (Min: 66.3%)
- ✅ Ground Truth Counterfactual Notes korrekt generiert

## Reproduzierbarkeit
Seed: `42` (in CONFIG). Bei gleichem Seed identische Daten.
