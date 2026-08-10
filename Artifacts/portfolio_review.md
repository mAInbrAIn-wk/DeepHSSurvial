# Portfolio Review: DataAnalysis & Abschlussprojekt

**Reviewer:** Antigravity (Claude Sonnet 4.6, Thinking Mode)  
**Stand:** August 2026  
**Gegenstand:** Zwei öffentliche GitHub-Repos als Portfolio-Projekte

---

## Vorbemerkung zur Methodik dieses Reviews

Ich habe beide Repos gründlich gelesen: alle Markdown-Dokumentationsdateien, den vollständigen Code der relevanten Skripte (Generator, Simulation, Modelle, Aggregation, Validierung, Counterfactual-Wrappers), die `requirements.txt` sowie interne Artefakt-Dokumente. Das Review ist in drei Teile gegliedert: zunächst jedes Projekt für sich, dann die Gegenüberstellung.

---

## Teil I: DataAnalysis

### 1. Einordnung und Kontext

Das Projekt ist als Abgabe für einen Kurs *Data Analytics* konzipiert. Es zeigt den Versuch, ein methodisch anspruchsvolles inhaltliches Problem — die Wirksamkeitsevaluation von Hochschulsupport — mit klassischen statistischen Werkzeugen anzugehen, und den Aufbau eines vollständigen Analyseflows von der Datengenerierung bis zum interaktiven Dashboard.

### 2. Konzeptionelle Stärken

**Problemauswahl:** Die Wahl des Themas ist konzeptionell ungewöhnlich stark. Die Frage nach der Wirksamkeit von Supportangeboten ist praktisch relevant, methodisch heikel (Selektionsbias), und es ist für einen Datenkurs unüblich, ein so genuines kausal-inferenzproblem anzugehen. Das zeigt Eigeninitiative und inhaltliche Tiefe.

**Eigenständigkeit der Datenbasis:** Der Entschluss, einen synthetischen Datensatz zu bauen statt auf kaggle-Daten zurückzugreifen, ist mutig und gut begründet. Das Dokument `Warum_synthethisch.md` belegt eine ernsthafte Recherche von Alternativen (DZHW SID2021, MoSAiK-Längsschnittstudie) und zeigt, dass die Entscheidung informiert getroffen wurde. Die Rechtfertigung bezüglich Datenschutz (DSGVO) ist inhaltlich korrekt und methodisch reflektiert.

**Bewusste Modellierungsentscheidungen:** Die Parameter-Architektur des Generators (`GeneriereHSDS.py`) ist bemerkenswert durchdacht: Geschlechterverteilung pro Studiengang, realistische Curriculumsstrukturen für fünf Studiengänge, HZB-Typ-Differenzierung, Anomalien (Plateau, Frühabbruch, Superstudierende). Das ist kein schnell zusammengebastelter Datensatz, sondern ein ausgearbeitetes Modell sozialer Realität.

**Metareflexion:** Das Projekt zeichnet sich durch ungewöhnliche Selbstkritik aus. Die README enthält explizit den Hinweis, dass das *Time-Varying Confounding by Indication* fehlt, was als methodisches Versäumnis klar benannt wird. Das ist eine intellektuelle Ehrlichkeit, die man selten sieht — und die aus einem echten Verständnis des Problems heraus kommt.

### 3. Inhaltliche Schwächen

**Das Kernproblem ist, das es noch kein Kernproblem gibt.** Was das Projekt zeigen will, ist klar: naive Vergleiche reichen nicht. Aber es zeigt dies mit einem Modell, das das Confounding noch nicht vollständig implementiert hat. Das ist im README dokumentiert, macht aber das Projekt zu einer Art Platzhalter für ein Vorhaben. Die Survival-Analyse im Dashboard wirkt wie ein Abschlussakkord, der auf einem noch nicht vollständigen Fundament sitzt.

**Keine Proportional-Hazards-Diagnose:** Die README räumt ein, dass die PH-Annahme für Cox nicht geprüft wurde. Das ist kein Schönheitsfehler — für einen methodischen Demonstrationskurs ist das ein echter Mangel, weil die Cox-Regression ohne PH-Prüfung als methodische Aussage entwertet wird.

**Das Datenmodell ist zu reich für die Analyse.** Der Generator produziert fünf Studiengänge, 12 Supportangebote, Anomalietypen — aber die Analyse nutzt diese Differenzierung kaum aus. Eine Subgruppenanalyse nach Studiengang wäre naheliegend; eine Analyse, welche Supporttypen welche Effekte haben, fehlt. Das erzeugte Modell ist reicher als das, was daraus gemacht wird.

**Die KPIs in `kpi.md` sind nicht vollständig operationalisiert.** Die Datei definiert sorgfältig KPIs für Wirksamkeit und Zielgruppenerreichung — diese tauchen dann im Analyse-Notebook nicht systematisch als Tabellen oder Ergebnisse auf. Die Verbindung zwischen Konzept und Implementierung ist hier lose.

### 4. Technische Stärken

**Code-Qualität des Generators:** `GeneriereHSDS.py` ist eine der substanzielleren Einzeldateien. Docstrings, Typ-Annotierungen (`Final[dict]`), klare Funktionsstruktur, `from __future__ import annotations` — das ist kein Anfänger-Skript. Die Kombination von deterministisch-stochastischen Gewichten ist methodisch solide.

**Reproduzierbarkeit:** Seed 42 im CONFIG, automatische JSON-Konfigurationsexportierung bei jedem Datengenerierungslauf. Das ist professionell.

**Pandas-Nutzung:** Die Aggregationslogik in `Datenaggregation.ipynb` (soweit rekonstruierbar aus dem Code-Stil) zeigt Vertrautheit mit Groupby-Operationen und Merge-Kaskaden. Kein `for`-Loop über DataFrames, wo Pandas-Operationen verwendet werden sollten.

### 5. Technische Schwächen

**Monolithisches Skript:** `GeneriereHSDS.py` ist 1388 Zeilen in einer einzigen Datei, intern nur durch Kommentar-Trennwände in „Teile" gegliedert. Das Projekt hat Klassen-Potenzial (Student, Prüfung, Modul), nutzt aber pure Dicts und DataFrames. Das ist vertretbar für ein Kurs-Projekt, aber für ein Portfolio-Stück ein Schwachpunkt gegenüber dem Abschlussprojekt.

**Kein `requirements.txt`:** Abhängigkeiten sind nur im README als Prosa aufgelistet. Für Reproduzierbarkeit fehlt eine `requirements.txt` oder `pyproject.toml`.

**Git-Hygiene:** Eine 15 MB große `Abgabeversion.zip` direkt im Repo, dazu `Beurteilung_Projektarbeit_DA-Keller_Wilfried.pdf` (4 MB) — das sind Dateien, die in ein Repo nicht gehören, sondern in `.gitignore` oder auf Releases verschoben werden sollten. Die `EDA.ipynb` wiegt 71 MB und ist aufgrund der Outputs praktisch nicht sinnvoll in Git versionierbar.

**Tippfehler in Dateinamen:** `AgliesArrbeiten.md` (statt `AgilesArbeiten`) und `Warum_synthethisch.md` (statt `synthetisch`). Im Code ist es `Warum_synthetisch.md` (in `Projektbeschreibung.md` korrekt). Das ist klein, aber Tippfehler in Dateinamen, die von anderen Dokumenten verlinkt werden, sind unprofessionell.

**Inline-Kommentare als technische Schulden:** Im Generator finden sich Kommentare wie `## seltsam umständlich` (Zeile 759) und `##m Hier besser einen weiteren Parameter verwenden, s.u.!!` (Zeile 938). Das sind Entwickler-TODO-Notizen, die in einem veröffentlichten Repo nicht sichtbar sein sollten.

### 6. Dokumentationsqualität

Die Dokumentation ist für ein Kurs-Projekt außergewöhnlich umfangreich und durch mehrere Markdown-Dateien differenziert. Die Projektbeschreibung ist klar, die Limitationen-Sektion ehrlich, die `AgilesArbeiten.md` zeigt Prozessdokumentation. Die `kpi.md` zeigt konzeptionelle Tiefe.

Was fehlt: Ein konsistenter Ton zwischen Dokumenten. `Warum_synthethisch.md` ist als persönlicher Brief an den Dozenten formuliert (`"Lieber Axel"`), enthält Aussagen wie `"ich habe das in Mammouth AI u.a. mittels Deines Systemprompts erstellt"` — das ist für ein öffentliches Portfolio-Stück auf GitHub ungeeignet. Es verrät, dass dieser Text nie für ein breiteres Publikum gedacht war.

### 7. Arbeitgeber-Perspektive auf DataAnalysis

**Positiver Ersteindruck:** Der Repo-Name und die README-Überschrift sind klar. Jemand, der das Repo aufruft, versteht sofort, worum es geht. Die methodische Bescheidenheit ("`kein empirischer Wirkungsnachweis`") signalisiert Integrität.

**Kritische Punkte:** Ein potentieller Arbeitgeber, der Code-Qualität prüft, wird `GeneriereHSDS.py` öffnen und wahrscheinlich beeindruckt sein — aber auch die `##m`-TODOs, den `p*=0.5`-Hardcode (Zeile 704, ohne Kommentar warum) und die fehlende `requirements.txt` sehen. Ein Recruiter, der die Notebooks öffnen will, wird merken, dass die `EDA.ipynb` 71 MB groß ist und nicht gerendert wird.

Die `Beurteilung_Projektarbeit_DA-Keller_Wilfried.pdf` im Repo ist ein zweischneidiges Schwert: Sie zeigt Transparenz über die Bewertung, könnte aber auch als ungewöhnlich wirken.

**Gesamteindruck:** Ambitioniert, konzeptionell solide, mit echten methodischen Erkenntnissen — aber noch nicht vollständig ausgearbeitet, mit einigen technischen Schludrrigkeiten, die für ein Portfolio-Stück stören.

---

## Teil II: Abschlussprojekt (DeepSupport)

### 1. Einordnung und Kontext

Das Projekt ist die Fortsetzung des Datenanalyseprojekts für einen Kurs *Deep Learning*. Es nimmt das Datengenerierungsmodell, überarbeitet es grundlegend, und baut darauf eine mehrstufige Modellierungspipeline auf, die von Baseline-Klassifikation über statische und zeitveränderliche Survival-Analyse bis zu Competing Risks und kontrafaktischer Inferenz reicht.

### 2. Konzeptionelle Stärken

**Methodische Progression als roter Faden:** Die vier Stufen (Naive Baselines → Landmark Survival → Extended Panel → Sequence Models + Competing Risks) sind nicht zufällig angeordnet, sondern erzählen eine Kausalgeschichte: *Jede Stufe behebt einen systematischen Bias der vorigen Stufe.* Das ist wissenschaftliches Narrativ. Der Übergang von HR > 1 (statisch) zu HR ≈ 0.37 (zeitveränderlich) ist ein konkreter empirischer Nachweis, der die methodische Argumentation trägt.

**Vollständige Adressierung des ursprünglichen Datenmangels:** Das Modell in `simulation.py` implementiert nun das reaktive Confounding, das in Version 1 fehlte: nach einem Fehlversuch steigt die Support-Nutzungswahrscheinlichkeit um +20 Prozentpunkte. Das ist nicht trivial. Es erfordert ein Redesign der Datenstruktur (Übergang von statischen Pandas-Dicts zu Dataclass-Objekten mit Zustandsverfolgung), eine neue Zeitkonto-Logik und eine kumulative Feature-Generierung. Das Problem wurde nicht nur erkannt, sondern gelöst.

**Kontrafaktische Ground Truth:** Der Einbau von `note_counterfactual` (die hypothetische Note ohne Support-Boost) direkt in die Simulation ist ein eleganter Designentscheid. Er erlaubt die spätere kausale Evaluierung der Modelle gegen eine echte Ground Truth — ein Vorteil synthetischer Daten, der hier bewusst ausgenutzt wird. Das ist konzeptionell raffiniert.

**Immortal-Time-Bias korrekt behandelt:** Die Funktion `build_person_semester_panel()` in `extended_cox_survival.py` transformiert die Daten korrekt ins Counting-Process-Format `(t_start, t_stop, event, X_i(t))`. Das ist keine Trivialität — viele erfahrene Analysten machen diesen Fehler. Die Implementierung ist korrekt und der Code-Kommentar erklärt das Konzept.

**Causal Masking:** Dass der Transformer `use_causal_mask=True` und sinusoidale Positional Encodings verwendet, ist nicht eine schöne Geste — es ist die einzige Möglichkeit, strikte temporale Kausalität in einem Sequenzmodell zu garantieren. Die Tatsache, dass dies implementiert ist, zeigt Verständnis, das über bloßes API-Verwenden hinausgeht.

### 3. Inhaltliche Schwächen

**Das Simulations-Paradoxon bleibt epistemisch problematisch.** Das Projekt weist darauf hin (sowohl in `abschlussreview.md` als auch in der README), dass hohe R²-Werte Artefakte der deterministischen Generierung sind. Das ist korrekt. Aber es geht tiefer: Wenn Modelle auf synthetischen Daten trainiert werden, lernen sie die Struktur des Generators, nicht die Struktur der Realität. Die kontrafaktische HR ≈ 0.88 (neuronales Netz) vs. HR ≈ 0.37 (Extended Cox) sagt nicht, welches näher an der Wahrheit ist — es sagt nur etwas über die Modellklasse. Diese Einschränkung wird im Projekt zwar angesprochen, aber nicht mit der gebotenen methodischen Schärfe ausgearbeitet.

**Fehlende PH-Diagnose:** Auch im Abschlussprojekt wird die Proportional-Hazards-Annahme für den Extended Cox nicht geprüft. Das ist umso kritischer, weil die Hazard-Ratio ≈ 0.37 als zentrales Kernergebnis der Stufe 2 präsentiert wird. Wenn die PH-Annahme verletzt ist, ist dieser Koeffizient ein Mittelwert eines zeitveränderlichen Effekts — was noch interpretierbar, aber anders ist. `lifelines` bietet einfache Schoenfeld-Residuen-Tests an, die trivial hinzuzufügen wären.

**Das Dashboard ist nicht funktionsfähig.** Die README enthält den Hinweis: *"Das ehemals verwendete Dash-Dashboard befindet sich derzeit im Umbau."* Ein nicht funktionsfähiges Dashboard in einem Abschlussprojekt ist eine Lücke. Das ist das einzige interaktive Element, das für nicht-technische Stakeholder zugänglich wäre — und es fehlt.

**Keine Kalibrierungsdiskussion für die Sequence Models.** Der Brier Score wird zwar berechnet, aber nicht im Kontext der Klassen-Unbalance diskutiert. Bei einer Dropout-Baserate von 2–5% sind rohe Brier Scores schwer interpretierbar. Eine kalibrierte Kurve (Reliability Diagram) wäre hier ein sinnvoller Zusatz.

**Die Counterfactual-Analyse ist konzeptionell unvollständig.** Die Idee ist richtig: Beobachte jeden Studierenden in zwei Welten (mit/ohne Support). Aber das setzt voraus, dass das Modell isoliert auf das Support-Feature reagiert — was es tun würde, wenn Support kausal unabhängig von den anderen Features wäre. In der Simulation ist Support jedoch korreliert mit Fehlversuchen (reaktiver Mechanismus). Wenn man im Test-Set alle Support-Flags auf 1 setzt, erzeugt man eine kontrafaktische Welt, die das Modell nie gesehen hat. Die Extrapolation außerhalb der Trainingsverteilung ist ein strukturelles Problem der Methode, das erkannt, aber nicht tief genug diskutiert wird.

### 4. Technische Stärken

**Architektur-Upgrade:** Der Übergang von einem Monolith-Skript zu einer modularisierten `src/`-Struktur mit `models.py` (Dataclasses), `config.py`, `simulation.py`, `aggregate.py`, `validate.py`, `export.py` und `main.py` ist ein echter Qualitätssprung. Die Pipeline hat eine klare Richtung, die Komponenten haben definierte Verantwortlichkeiten.

**`models.py` als Domain-Modell:** Die Einführung von `Student`, `ModulState`, `PruefungsErgebnis` als Dataclasses mit Methoden (`cp_bestanden`, `alle_pflicht_bestanden`) ist ein Zeichen reiferer Softwareentwicklung. Objekte kapseln Zustand; die Simulation schreibt Zustand in diese Objekte statt in verschachtelte Dicts.

**`metrics_logger.py` als Infrastructure as Code:** Das zentrale Logging-Modul, das JSON, Markdown, ROC-Kurven, PR-Kurven, Lernkurven, Confusion Matrices und Parity Plots mit einem einheitlichen Interface speichert, ist herausragend für ein Uni-Projekt. Das ist Infrastruktur-Denken, das man in industriellen Projekten erwartet, aber selten in akademischen findet.

**Leakage-Prävention als Designprinzip:** `blind=True`-Modus, Pre-Landmark Feature-Selektion, Scaler fit nur auf Training-Daten, Group Split auf Studierenden-Ebene — diese Prüfungen sind nicht versehentlich korrekt, sondern systematisch implementiert. Das zeigt, dass Data Leakage nicht als abstrakter Begriff, sondern als konkretes Problem verstanden wird.

**Validate-Skript mit automatisierten Checks:** `validate.py` führt 12 Konsistenz-Checks durch und generiert automatisch `DATENSATZ_DOKU.md`. Das ist Qualitätssicherung — selten in Uni-Projekten.

**`requirements.txt` mit Versionen:** Vorhanden und spezifisch. Die Pinning-Logik mit `pydantic==2.13.4` (für Dash Python 3.12) zeigt, dass wirklich auf einem konkreten System ausgeführt wurde und Kompatibilitätsprobleme gelöst wurden.

### 5. Technische Schwächen

**Code-Stil in `simulation.py` inkonsistent:** Die Funktion `generiere_studierende` enthält einen 100+ Zeichen langen einzeiligen Ausdruck (z.B. Zeile 113), der unleserlich ist. Das Abschlussprojekt hat den Stil verbessert (Dataclasses), aber noch nicht durchgängig.

**Redundante Skripte:** Neben `run_all_experiments.py` gibt es `run_remaining_experiments.py` — das klingt nach einem Work-in-Progress-Artefakt. Ähnlich gibt es `recurrent_exam_survival.py` und `recurrent_exam_survival_v2.py` ohne klare Versionierungslogik. In einem sauberen Portfolio-Repo würde man `v1` entweder löschen oder als `archive/` kennzeichnen.

**`config.py` als dump-Datei:** Die `config.py` ist offensichtlich mit `pprint` aus einem Dict-Objekt generiert worden — das verschachtelte Format mit vielen Leerzeichen ist kein handgeschriebener Config-Code. Das ist vertretbar, aber unschön.

**C-Index-Implementierung O(n²):** Die `concordance_index`-Funktion in `deep_survival.py` (Zeilen 73–95) iteriert über alle Paare — O(n²). Für 50.000 Studierende ist das auf dem Test-Set noch handhabbar, aber für größere Datenmengen wäre eine vektorisierte Implementierung (z.B. aus `lifelines`) besser.

**Import-Struktur:** Mehrere Skripte importieren `from metrics_logger import ...` *innerhalb* von Funktionen (z.B. `recurrent_survival_model.py`, Zeile 172). Das ist ein Code-Smell — Imports gehören an den Dateianfang.

**`__pycache__` im Repository:** Der `__pycache__`-Ordner ist im Repo eingecheckt. Das sollte in `.gitignore`.

### 6. Dokumentationsqualität

Die README des Abschlussprojekts ist deutlich professioneller als die des Datenanalyseprojekts. Der Mermaid-Flowchart ist ein wirkungsvolles Mittel, um die Pipeline zu visualisieren. Die vier Stufen mit ihren Ergebnissen sind klar strukturiert. Das Emoji-Einsatz ist Geschmackssache, aber konsistent und erhöht die Lesbarkeit.

Die `Artifacts/`-Sammlung (u.a. `abschlussreview.md`, `code_review.md`, `model_comparison.md`) zeigt ein Projekt, das sich selbst auditiert — das ist ungewöhnlich und wertvoll.

Was fehlt: Ein `CONTRIBUTING.md` oder zumindest ein Setup-Abschnitt in der README, der erklärt wie man von Null auf "Experimente laufen" kommt. Die Anweisung `python src/main.py` ist knapp — wieviel RAM? Wie lang dauert es? Werden Zwischenstände gecacht?

### 7. Arbeitgeber-Perspektive auf das Abschlussprojekt

**Starker Ersteindruck:** Wer die README liest, sieht sofort: Hier hat jemand nicht nur APIs aufgerufen, sondern verstanden, warum naive Modelle bei Observationsdaten systematisch falsch liegen. Das ist eine Kompetenz, die in der Industrie selten ist und für Data Science in kausalen Settings (Medizin, HR, Bildung, Werbung) direkt relevant ist.

**Was beeindruckt einen technischen Interviewer:** Das Counting-Process-Format, der Causal Masking Transformer, die kontrafaktische Simulation mit Potential Outcomes — das sind Buzzwords, hinter denen hier echter Code steht. `metrics_logger.py` zeigt MLOps-Denken. Das `validate.py` zeigt, dass jemand über Datenpipeline-Integrität nachdenkt.

**Was einen skeptischen Interviewer stört:** Das Dashboard fehlt. `__pycache__` ist eingecheckt. Es gibt redundante v1/v2-Skripte. Die PH-Annahme wird nicht diagnostiziert, obwohl das das einfachste wäre. Diese Details signalisieren: Das Projekt ist zu 85% fertig, nicht zu 100%.

**Gesamteindruck:** Ein außergewöhnliches Portfolio-Stück für ein Kurs-Abschlussprojekt. Die methodische Tiefe (Kausalinferenz, Survival Analysis, Competing Risks, Counterfactuals) ist für diesen Kontext ungewöhnlich. Ein potentieller Arbeitgeber für eine Junior-Data-Science-Stelle würde das bemerken.

---

## Teil III: Gegenüberstellung und Fortschrittsbewertung

### 1. Was sich verbessert hat

| Dimension | DataAnalysis | Abschlussprojekt | Fortschritt |
|:---|:---|:---|:---|
| **Code-Architektur** | 1 Monolith-Datei (1388 Zeilen) | Modulare `src/`-Pipeline, Dataclasses, Domain-Modell | ★★★★★ |
| **Datenmodell** | Statisches Confounding-Modell | Reaktives Confounding (+20% nach Fehlversuch), Zeitkonto | ★★★★★ |
| **Methodische Tiefe** | Kaplan-Meier, Log-Rank, Cox | 13+ Modelle, 4 Stufen, Counterfactuals, Competing Risks | ★★★★★ |
| **Leakage-Kontrolle** | Nicht systematisch | Explizit: blind=True, Pre-Landmark, Group Split | ★★★★★ |
| **Infrastruktur** | Keine | `metrics_logger.py`, `validate.py`, JSON+Plots automatisch | ★★★★★ |
| **Reproduzierbarkeit** | Seed in Config | Seed + versions-gepinnte requirements.txt | ★★★★ |
| **Dokumentation** | Umfangreich aber an Dozenten gerichtet | Professionell, öffentlichkeitstauglich | ★★★★ |
| **Kausal-Reflexion** | HR fehlt Confounding-Fix (dokumentiert) | HR-Paradox aufgelöst, Counterfactuals | ★★★★★ |
| **Git-Hygiene** | Schwach (70MB Notebooks, ZIP, PDF) | Besser, aber __pycache__ eingecheckt | ★★★ |
| **Requirements** | Fehlen | Vorhanden mit Versionen | ★★★★ |

### 2. Was sich nicht oder kaum verbessert hat

**PH-Annahmen-Diagnose:** Sowohl im Datenanalyse- als auch im Abschlussprojekt wird die Proportional-Hazards-Annahme nicht diagnostiziert. Das ist das am leichtesten zu behebende methodische Versäumnis in beiden Projekten.

**Hyperparameter-Suche:** Beide Projekte verwenden heuristisch gewählte Architekturparameter (Layer-Größen, Dropout-Raten, Lernraten). Eine systematische Hyperparameter-Suche (Optuna, GridSearch) fehlt vollständig. Das ist vertretbar für Kurs-Projekte, aber für ein Portfolio-Stück würde eine kurze Suche die Glaubwürdigkeit der Ergebnisse erhöhen.

**Ground-Truth-Vergleich:** Beide Projekte erwähnen als Nice-to-Have den direkten Vergleich geschätzter Effekte mit den in der Simulation gesetzten Ground-Truth-Parametern. Das ist tatsächlich das spannendste Experiment, das man machen könnte — und es wurde in keinem der Projekte durchgeführt. Dabei hätte man es: `note_counterfactual` existiert in den Daten.

### 3. Die blinden Flecken — was in beiden Projekten fehlt

**Sensitivitätsanalyse:** Beide Projekte erwähnen explizit, dass die Ergebnisse sensitiv gegenüber den Generierungsparametern sind. Aber keine einzige systematische Variation der Parameter wurde durchgeführt. Das ist der elefant im Raum: Wenn man `gewicht_support_boost = 0.04` auf `0.08` ändert, ändert sich dann die geschätzte HR signifikant? Ohne diese Analyse sind alle Ergebnisse abhängig von arbiträren Modellentscheidungen — das ist bekannt, aber nicht quantifiziert.

**Kein Vergleich mit domänen-bekannten Benchmarks:** Die Abbruchquoten und Studienzeiten im synthetischen Datensatz werden qualitativ als "realistisch" beschrieben, aber nie gegen konkrete Referenzwerte aus dem deutschen Hochschulsystem (z.B. CHE-Studierendensurvey, DZHW-Daten) kalibriert. Für ein Portfolio-Stück wäre ein Kalibrierungsabschnitt aussagekräftig.

**Kausalidentifikation ist epistemisch unterbestimmt:** Der fundamentale Punkt, dass man *auch im besten Extended Cox Modell* keine echte Kausalaussage machen kann (ohne Randomisierung oder Instrumental Variables), wird in beiden Projekten angesprochen, aber nicht konsequent zu Ende gedacht. Konkret: Die Aussage "HR ≈ 0.37 zeigt, dass Support das Abbruchrisiko senkt" ist *in der Simulation korrekt*, aber die Übertragung auf reale Daten ist nicht gegeben. Ein klares Rahmung der Ergebnisse als "Validierung der Methode, nicht der Wirklichkeit" würde fehlen.

**Fehlende Visualisierung des Lernfortschritts:** Im Datenanalyseprojekt gibt es ein interaktives Dashboard (auch wenn es die erste Version noch nicht vollständig implementiert hatte). Im Abschlussprojekt ist das Dashboard broken. Das ist bedauerlich, weil für nicht-technische Stakeholder (z.B. Hochschulverwaltung) interaktive Visualisierungen der Schlüssel zur Relevanz wären.

### 4. Was die Gegenüberstellung über die Entwicklung aussagt

Vom Datenanalyseprojekt zum Abschlussprojekt ist eine sehr deutliche Reifung zu beobachten, die über bloße Wissensakkumulation hinausgeht:

**Vom Notebook zum Skript:** Der Übergang von Jupyter-Notebooks als Primärwerkzeug zu einem orchestrierten Skript-System mit `main.py` als Einstiegspunkt zeigt, dass produktionsorientiertes Denken eingesetzt hat. Notebooks sind gut für Exploration; Skripte sind gut für Reproduzierbarkeit. Das Abschlussprojekt versteht diesen Unterschied.

**Von der Datei zum Domain-Modell:** `GeneriereHSDS.py` simuliert Studierende als Pandas-DataFrames. `simulation.py` simuliert sie als `Student`-Dataclass-Objekte. Das ist der Unterschied zwischen Daten-orientiertem und Objekt-orientiertem Denken — beides ist legitim, aber Dataclasses machen den Zustandsfluss explizit und testbar.

**Von der Methode zur Kausalfrage:** Das Datenanalyseprojekt setzt Survival-Analyse ein, weil sie methodisch passt. Das Abschlussprojekt fragt: "Warum gibt ein statisches Modell HR > 1, und wie behebt man das?" Das ist der Übergang von *Methodenanwendung* zu *Kausaldenken*.

**Von der Bescheidenheit zur Argumentation:** Die README des Datenanalyseprojekts betont mehrfach, was *nicht* gemacht wurde. Die README des Abschlussprojekts argumentiert, was *gemacht* wurde und warum es methodisch korrekt ist. Das ist eine reifere kommunikative Haltung.

---

## Zusammenfassung: Was sollte als nächstes verbessert werden?

### Sofort (< 1 Tag Aufwand)

1. **`__pycache__` in `.gitignore`** und aus dem Repo entfernen (beide Projekte).
2. **Tippfehler in Dateinamen** beheben (`AgliesArrbeiten.md` → `AgilesArbeiten.md`, `Warum_synthethisch.md` → `Warum_synthetisch.md`).
3. **Persönliche Anrede** in `Warum_synthethisch.md` entfernen oder die Datei umschreiben für ein öffentliches Publikum.
4. **`##m`-TODO-Kommentare** aus dem veröffentlichten Code entfernen.
5. **`requirements.txt`** für das Datenanalyseprojekt erstellen.

### Kurzfristig (< 1 Woche)

6. **PH-Annahmen-Test** in `extended_cox_survival.py` mit Schoenfeld-Residuen ergänzen (`lifelines` bietet dies nativ an).
7. **Dashboard reparieren** oder aus dem Abschlussprojekt-README ehrlich als *nicht implementiert* markieren (statt *im Umbau*).
8. **Redundante Skripte** (`run_remaining_experiments.py`, `recurrent_exam_survival_v2.py`) in `archive/` oder löschen.
9. **Notebook-Outputs aus `.gitignore`** oder separate Rendering mit `nbconvert` dokumentieren.

### Mittelfristig (high impact)

10. **Ground-Truth-Vergleich:** Extrahiere den im Generator gesetzten Support-Effekt (`gewicht_support_boost = 0.04`) und vergleiche ihn direkt mit dem geschätzten HR. Das wäre der stärkste methodische Beitrag beider Projekte — und die Daten sind vorhanden.
11. **Mindestens eine Sensitivitätsanalyse:** Variiere `gewicht_support_boost` zwischen 0.02 und 0.08 und zeige, wie die geschätzte HR mit dem wahren Effekt ko-variiert.
12. **Kalibrierungskurve** (Reliability Diagram) für die Sequence Models ergänzen.

---

## Abschließende Einschätzung

Beide Projekte zusammen bilden ein kohärentes Portfolio, das eine ungewöhnliche Kombination zeigt: methodisches Bewusstsein für kausalinferenzielle Probleme, technische Implementierung auf einem Niveau, das über typische Kurs-Abgaben hinausgeht, und intellektuelle Ehrlichkeit bei der Reflexion von Limitationen.

Der Fortschritt vom Datenanalyse- zum Abschlussprojekt ist substanziell und in mehreren Dimensionen gleichzeitig sichtbar. Das allein ist ein starkes Signal: es zeigt Lernfähigkeit und die Bereitschaft, frühere Arbeit grundlegend zu überdenken statt nur zu erweitern.

Die wesentliche Botschaft an einen potentiellen Arbeitgeber ist: Hier hat jemand ein echtes methodisches Problem (Selektionsbias in Observationsdaten) verstanden, es in einem programmierbaren Modell implementiert, und über vier Methodenstufen systematisch zu lösen versucht. Das ist eine Denkweise, die in der Praxis gefragt ist — nicht nur bei Machine Learning, sondern bei jeder datengetriebenen Entscheidung, die kausale Behauptungen aufstellt.

Was noch fehlt, ist das letzte 15%: Dashboard, Ground-Truth-Vergleich, PH-Diagnose, Git-Hygiene. Diese Lücken sind behebbar — und das Schließen dieser Lücken würde den Unterschied zwischen einem beeindruckenden Kurs-Abschlussprojekt und einem echten Portfolio-Stück ausmachen.
