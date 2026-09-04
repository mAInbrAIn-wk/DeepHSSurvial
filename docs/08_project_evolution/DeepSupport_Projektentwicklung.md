# DeepSupport: Projektentwicklung & Methodische Evolution

**Erstellt:** September 2026 | **Basis:** Gesamtprojekt inkl. Legacy Submodule
**Autor (Analyse):** Antigravity / Claude Sonnet 4.6 Thinking Mode

> Dieses Dokument zeichnet die intellektuelle und technische Reise des Projekts von seinen Ursprüngen im Data-Engineering-Kurs bis zum aktuellen Stand (V4.2) nach. Die Legacy-Phasen DE, DA und DL sind als Git-Submodule unter `legacy_projects/` vollständig erhalten und lesbar.

---

## Phase 1: Data Engineering (DE) — Das Fundament

### Zeitraum & Kontext
Die erste Phase entstand als Portfolio-/Modularbeit in einem Data-Engineering-Kurs, unter hohem Zeitdruck. Das zentrale Thema war von Beginn an die Frage, die das gesamte Projekt begleiten würde: *„Lassen sich die Wirkungen von Hochschul-Supportangeboten quantitativ nachweisen?"*

### Technologien & Architektur
Das Projekt realisierte einen vollständigen Data-Warehouse-Stack:
- **Operatives System (OLTP):** Relationales Datenmodell in 3. Normalform (3NF) mit T-SQL auf Microsoft SQL Server. Modellierte Entitäten: Studierende, Studiengänge, Module, Prüfungsleistungen, Veranstaltungen und fakultative Supportangebote.
- **Data Warehouse (ROLAP):** Star-Schema mit `dwh_Faktentabelle` (Grain: Modulabschluss pro Studierendem), drei Dimensionstabellen (`dwh_Student`, `dwh_Modul`, `dwh_Zeit`) und analytisch verdichteten Spalten wie `AnzSupp_01`, `AnzSupp_02`, `HasSupportParticipation`.
- **ETL-Pipeline:** Vier Stored Procedures (Full Load & Delta Load) mit Change-Tracking-Spalten.
- **DuckDB-Port:** Als nachträgliche Ergänzung entstand `run_duckdb_pipeline.py` — ein serverloser Python-Runner, der die gesamte Pipeline in-memory ausführt und die Architektur-Diskussion vorwegnimmt, die in V4 zentral wird.

### Erste analytische Ergebnisse
Aus den OLAP-Queries ergab sich ein erster Befund: Studierende mit Support-Teilnahme erzielten im Schnitt **ca. 0,45 Notenstufen** bessere Ergebnisse und hatten signifikant weniger Fehlversuche. Diese Zahl klang plausibel — aber schon die README des Submoduls enthält einen selbstkritischen Disclaimer: Der Entwurf hatte *„einen Foreign-Key-Fehler, uneinheitliche Zeitstempel und naive Joins"*.

### Charakteristik dieser Phase
Die DE-Phase ist ein Beweis solider handwerklicher Grundlagen (Datenbankdesign, ETL, OLAP), aber noch naiv bezüglich kausaler Interpretierbarkeit. Es fehlt jede Reflexion über den **Selektionsbias**: Ob Studierende mit besseren Ausgangsvoraussetzungen häufiger oder seltener Support nutzen, war noch keine Frage, die das System stellen konnte. Das Star-Schema mit `HasSupportParticipation` als aggregiertem Binär-Flag spiegelt eine konfundierte, nicht-zeitvariable Sicht auf das Problem wider.

---

## Phase 2: Data Analysis (DA) — Die Entdeckung des Paradoxons

### Zeitraum & Kontext
Die zweite Phase, entstanden im Rahmen eines Data-Analytics-Kurses, ist methodisch der erste wirkliche intellektuelle Sprung. Das Projekt greift das Datenbankmodell aus DE auf, fügt aber einen entscheidenden neuen Baustein hinzu: einen **dynamisch-stochastischen Studierendensimulator** (`GeneriereHSDS.py`).

### Der synthetische Weg — eine strategische Entscheidung
Das README des DA-Projekts enthält in der Datei `Warum_synthethisch.md` die Begründung, die die gesamte weitere Entwicklung prägt: Echte Hochschuldaten sind aus Datenschutzgründen (Stichwort DSGVO, Silos zwischen Prüfungsamt und LMS) schwer zugänglich. Die Entscheidung für synthetische Simulation war daher pragmatisch notwendig und methodisch vertretbar — aber sie erzeugt eine epistemische Grenze, die das Projekt bis heute begleitet.

### Methoden & Werkzeuge
- **Datengenerierung:** Kohortenweise Simulation (2015–2024), semesterbezogen, mit statischen Initialeigenschaften pro Studierendem (HZB-Note, Erstakademiker-Status, initiale Motivation, soziale Integration).
- **Analyse:** Deskriptive KPIs, **Kaplan-Meier-Überlebensschätzer**, **Log-Rank-Tests** und **Cox-Proportional-Hazards-Regression** zur Untersuchung der Studiendauer und des Abbruchrisikos.
- **Visualisierung:** Interaktive Dash-Dashboards.

### Die zentrale methodische Entdeckung
Das DA-Projekt dokumentiert in seiner eigenen README die bedeutsamste Entdeckung dieser Phase: **Das Fehlen des Time-Varying Confounding by Indication**. Zitat aus `README.md`:

> *„Was in dieser Version fehlt, ist das ursprünglich angedachte Time-Varying Confounding by Indication (ein reaktiver Mechanismus). In der Realität – und in späteren Überarbeitungen des Modells – führt beispielsweise ein nicht bestandenes Modul (Fehlversuch) im Semester zu einem Einbruch der Motivation und gleichzeitig zu einer stark erhöhten Wahrscheinlichkeit (+20 %), Supportangebote wahrzunehmen."*

Da dieser reaktive Mechanismus fehlt, ließ sich die positive Wirkung des Supports hier noch „künstlich leicht" nachweisen — ein Problem, das erst in der nächsten Phase vollständig sichtbar wird.

### Weitere methodische Grenzen
- Die **Proportional-Hazards-Annahme** der Cox-Regression wurde nicht geprüft (und ist vermutlich verletzt).
- Das Modell operiert noch mit einem **statischen Feature-Raum** (keine zeitvariablen Confounder).
- Die Analyse liefert demonstrative Ergebnisse, aber keinen empirischen Wirkungsnachweis für die Realität.

### Charakteristik dieser Phase
Die DA-Phase ist ehrlich, selbstkritisch und methodisch reflektiert — was die ausführliche Eigendiagnose in der README zeigt. Das Projekt hat gelernt, die richtigen Fragen zu stellen (Selektionsbias, Immortal-Time-Bias), kann sie aber mit dem damaligen Werkzeugkasten noch nicht vollständig beantworten. Die Basis für die nächste Entwicklungsstufe ist jedoch gelegt.

---

## Phase 3: Deep Learning (DL-Vorgänger) — Durchbruch und neue Fallen

### Zeitraum & Kontext
Die dritte Phase, das direkte Vorläuferprojekt im Kurs Deep Learning (Dozent: Dr. Bernd Ebenhoch), ist der radikale Methodensprung. Aus dem analytischen Projekt wird ein vollständig implementiertes Deep-Learning-Framework mit 13 Modellarchitekturen in vier methodischen Stufen.

### Die vier Modellstufen

**Stufe 0 — Statische Baselines:**  
Klassische ML-Modelle (Naive Bayes, Random Forest, SVM, Ridge Regression, Dense-MLPs). Diese zeigen erstmals das volle Ausmaß des **Dropout-Paradoxons**: Statische Modelle, die Support-Nutzung als Feature enthalten, schätzen fälschlicherweise eine Risikoerhöhung — Hazard Ratio > 1. Ein Studierender mit Support-Teilnahme *scheint* ein höheres Abbruchrisiko zu haben, weil er überwiegend dann Support sucht, wenn er ohnehin in Schwierigkeiten ist.

**Stufe 1 — Landmark Survival:**  
DeepSurv (Keras Cox-Partial-Likelihood) und Discrete-Time Logistic (DTL) Hazard — statische Survival-Modelle. Bestätigen das Paradoxon bei naiver Anwendung.

**Stufe 2 — Extended Panel Survival (Zeitveränderlich):**  
Der entscheidende methodische Durchbruch: Die Daten werden in **Person-Semester-Panels** (Counting-Process-Format) umgebaut. Jedes Semester wird separat als Beobachtungseinheit behandelt. Ergebnis: Das Dropout-Paradoxon ist aufgelöst. Das Extended Cox-Modell weist eine **Hazard Ratio von ≈ 0.37** aus — ein dramatischer Richtungswechsel.

**Stufe 3 — Sequenz-Survival (GRU, LSTM, Causal Masked Transformer):**  
Rekurrente Modelle, die die gesamte bisherige Geschichte eines Studierenden verarbeiten. Causal Masking mit Padding-Wert `-99.0` verhindert den Blick in die Zukunft. ROC-AUC erreicht bis zu **0.90**.

**Stufe 4 — Competing Risks (Dynamic DeepHit):**  
Multi-Task-Architektur mit Shared-GRU-Backbone und zwei Heads: Studienabbruch vs. erfolgreicher Abschluss als konkurrierende Risiken.

### Der kontrafaktische Ansatz als Vorstufe
Eine kontrafaktische Simulation trainiert jedes Modell zweimal: einmal mit und einmal ohne Support-Intervention. Ergebnis: eine mediane **Hazard Ratio von ≈ 0.88** (individuelle Risikosenkung ~12 %) — deutlich moderater als die 0.37 des parametrischen Extended Cox. Dies entlarvt das 0.37-Ergebnis als Artefakt, nicht als kausale Wahrheit.

### Neue Probleme, die entdeckt wurden
Das DL-Projekt enthält in seiner README einen klaren Disclaimer:
> *„Data Leakage: Es gab unbewusstes Future-Leakage in einigen Features (z. B. flossen post-hoc aggregierte Werte in frühe Semester). Uneinheitliche Features: Die Feature-Räume zwischen statischen Baselines und rekurrenten Modellen waren noch nicht strikt standardisiert."*

Außerdem: Die **drei DeepSurv-Varianten scheiterten** technisch — ROC-AUC-Werte von 0.46 bis 0.56 (kaum besser als Zufall) zeigen, dass der Breslow-Cox-Loss mit Keras-Netzwerken nicht zuverlässig konvergiert. Ein problematischer Befund, der im späteren Project-Review (August 2026) klar benannt wird.

### Charakteristik dieser Phase
Das DL-Vorgängerprojekt ist ambitioniert und produziert beeindruckende Ergebnisse (ROC-AUC 0.90, 13 Architekturen, kontrafaktische Simulation). Es ist aber auch das Projekt, in dem die tiefsten methodischen Fallen aufgedeckt werden: Leakage, uneinheitliche Feature-Räume, non-konvergierende Survival-Losses. Diese ehrliche Selbstkritik ist es, die den Übergang zur V4-Generation motiviert.

---

## Phase 4: DeepSupport V4.x — Methodische Reife & systematisches Benchmarking

### Zeitraum & Kontext
Das aktuelle Projekt (V4.0 bis V4.2, August–September 2026) ist keine Iteration — es ist eine vollständige Neuarchitektur auf Basis der gesammelten Erkenntnisse. Das zentrale Dokument [`docs/01_master_plans/02_Methodische_Evolution_und_Synthese.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/01_master_plans/02_Methodische_Evolution_und_Synthese.md) nennt es treffend eine „methodische Brücke" zwischen der Fragestellung der frühen Phasen und den High-End-Lösungen des aktuellen Systems.

### Das 8-Universen Counterfactual Framework
Der radikalste konzeptionelle Schritt ist die Einführung eines **deterministischen Paralleluniversen-Simulators**. $N=50.000$ Studierende werden mit identischen Charakteristika in acht parallelen Welten (A–H) simuliert:

| Universum | Konfiguration |
|:---|:---|
| **A (Baseline)** | Alle 3 Supportangebote aktiv (fachlich, überfachlich, psychosozial) |
| **B (Null-Support)** | Kein Support aktiv — reine kontrafaktische Vergleichswelt |
| **C–E** | Partieller Wegfall je eines Angebots (Ablation) |
| **F–H** | Isolierte Einzelwirkung je eines Angebots |

Dies erlaubt erstmals die **exakte Berechnung der kausalen Ground Truth**: Der Absolute Risk Reduction (ARR) lässt sich als Differenz zweier deterministisch konstruierter Welten ablesen, ohne Schätzfehler. In S01 (Baseline): Dropout A=29.2%, B=37.1% → **ARR = +7.95 pp, RR = 0.786, NNT = 12.6**.

### Systematische Sensitivitätsanalyse (S01–S15)
Das Master Sensitivity Grid variiert 15 Simulationsparameter in 15 Szenarien und liefert ein vollständiges Bild der Modell-Robustheit:
- **225 trainierte Modelle** (15 Szenarien × 3 Architekturen × 5 Feature-Modi)
- Untersuchte Dimensionen: Rauschen, Support-Wirkungsstärke, Notenboost, Zeitkosten, Überladungsstrafe, RCT-Selektionsmodus, Kombinations-Effekte

### Das modulare `deepsupport/` Package
Das V4-Refactoring überführt über 60 Monolith-Skripte in eine konsistente Package-Struktur:

```
src/deepsupport/
├── data_engine/      # aggregate.py, config.py, feature_builder.py
├── evaluation/       # metrics_logger.py, cross_scenario_engine.py,
│                     # completeness_auditor.py, mediation_analysis.py
│                     # causal/  (14 Counterfactual-Skripte)
├── models/           # 14 Modell-Implementierungen
├── runners/          # fast_suite.py, heavy_suite.py, grid_runner.py, master_suite.py
└── simulation/       # engine.py, ground_truth.py, sensitivity_grid.py
```

Die **Zero-Imputation-Policy** (`null` statt `0.0` für fehlende Metriken), konsequente I/O-Trennung (separate `data_root` und `output_root`), und DuckDB als Hochleistungs-Aggregations-Backend sind architektonische Grundprinzipien.

### Die Heavy Deep Suite (Exam-Level Transformer)
Parallel zum Grid-Run auf der Workstation läuft die Heavy Suite auf einem ausgelagerten Homeserver-Cluster:
- **Dual-Head Autoregressive GRU:** Simultane Next-Exam-Noten- und Bestehensvorhersage
- **Deep Autoregressive Transformer mit Sin/Cos Positional Encoding:** Überlegene Notenvorhersage ($R^2 = 0.70$ in S01, $R^2 = 0.86$ in S07)
- **Landmark Representation Learning:** Gefrorene Transformer-Embeddings nach 2 Semestern erklären bereits **76.5%** der Varianz der finalen Abschlussnote

### Wichtigste Kernergebnisse (empirisch verifiziert)
- Support reduziert das Dropout-Risiko kausal um **21.3%** (A vs. B, S01)
- **Selbstselektion schlägt RCT:** Bedarfsgesteuerte Zuweisung (S01) erzielt NNT=12.6 vs. zufällige RCT-Zuweisung (S11) NNT=22.5 — ein bildungsökonomisch zentraler Befund
- **Exam-GRU übertrifft Semester-Modelle dramatisch:** ROC-AUC bis 0.91 (vs. ~0.82 der Semester-Modelle)
- **Transformer > GRU** bei Notenvorhersage konsistent über alle Rausch-Niveaus (+0.08 bis +0.25 $R^2$)

---

## Übergänge & die roten methodischen Fäden

### Faden 1: Vom Nachweis zum Paradoxon zur Auflösung
```
DE:  0.45 Notenboost (naiv, konfundiert)
↓
DA:  Cox HR < 1.0 (aber PH verletzt, statischer Confounder)
↓
DL:  HR > 1 (Paradoxon) → HR 0.37 (Leakage) → HR 0.88 (kontrafaktisch)
↓
V4:  ARR = +7.95 pp (Ground Truth aus 8 Parallelwelten, zero Schätzfehler)
```

### Faden 2: Die Architektur-Evolution
```
DE:  T-SQL Stored Procedures → DuckDB-Port (Vorschau)
↓
DA:  Pandas-Notebooks, Matplotlib, Dash
↓
DL:  60+ monolithische Skripte, manuelle Feature-Konstruktion, CSV-Chaos
↓
V4:  deepsupport/ Package, DuckDB-Backend, modulare Evaluatoren, Git-Submodule
```

### Faden 3: Feature-Integrität und Leakage-Kontrolle
Eines der tiefsten Lernfelder war die schrittweise Erkenntnis, wann eine Variable *Leakage* verursacht:
- **DA:** Kein explizites Leakage-Bewusstsein (statische Features)
- **DL:** Unbewusstes Future-Leakage entdeckt (post-hoc aggregierte Werte in frühen Semestern)
- **V4:** Striktes Causal Masking (`-99.0` Padding), `cp_cum_prev` statt `cp_rueckstand`, 5-Modi-Feature-Builder, Oracle-Modi als kontrollierter Information-Lift-Test

### Faden 4: Competing Risks & Granularität
```
DA:  Binäres Dropout (ja/nein)
↓
DL:  Competing Risks (Dropout vs. Abschluss) via Dynamic DeepHit
↓
V4:  4-Klassen Landmark-Prognose (Absolviert / Abbruch Freiwillig / Exma Zwang / Zeitüberschreitung)
     + Exam-Level-Sequenzen (40 Prüfungsschritte)
     + Next-Exam-Note als kontinuierliches Ziel (MSE-Regression)
```

---

## Offene Fäden & Ausblick

### 1. Evaluierungs-Refactoring (Priorität: Hoch)
Das Refactoring-Dokument [`refactoring_plan_evaluation_pipeline1.md`](file:///C:/GitHub_public/Abschlussprojekt/docs/01_master_plans/refactoring_plan_evaluation_pipeline1.md) beschreibt die Ablösung manuellen Logging-Boilerplates durch **5 typisierte Evaluator-Klassen**:
- `SurvivalEvaluator`, `RegressionEvaluator`, `MulticlassEvaluator`, `CausalEvaluator`, `DualHeadEvaluator`
- Automatische Baseline-Linie $\pi_0$ und Brier-Skill-Score

Dieser Schritt würde die bestehenden Schwächen in der Metriken-Konsistenz beseitigen und den Code erheblich wartbarer machen.

### 2. MoE / Stacking Router (Priorität: Mittel, wissenschaftlich spannend)
Die kontrafaktischen Schätzungen zeigen: Support wirkt **hochgradig individuell**. Ein Mixture-of-Experts-Router, der basierend auf individuellem Profil und bisheriger Trajektorie entscheidet, welche Maßnahme am besten wirkt, wäre der logische nächste Schritt hin zu einem operativen Frühwarnsystem. Das ist auch der Punkt, wo das Projekt von einer Evaluation zum **Decision-Support-System** werden würde.

### 3. PyTorch / PyCox Portierung (Priorität: Mittel, Zukunftsfähigkeit)
Das Framework stützt sich vollständig auf TensorFlow 2.21 / Keras. Die Survival-Analysis-Community hat sich jedoch mehrheitlich in Richtung PyTorch (mit PyCox, lifelines, auton-survival) entwickelt. Eine Migration würde die Nutzbarkeit und Anschlussfähigkeit an aktuelle Forschung deutlich verbessern.

### 4. Dashboard & Interaktive Visualisierung (Priorität: Mittel, Präsentierbarkeit)
Das interaktive Dash-Dashboard wurde in der DL-Phase als funktionierend beschrieben, aber im August 2026 Review noch als „Work in Progress" markiert. Ein vollständig funktionsfähiges Dashboard, das die Sensitivitäts-Synopsen, die Kausal-Schätzer und die Transformer-Visualisierungen interaktiv zeigt, würde die Zugänglichkeit des Projekts für externe Betrachter enorm erhöhen.

### 5. Empirische Validierung (Priorität: Langfristig, transformativ)
Der fundamentalste offene Faden ist die Validierung an echten Hochschuldaten. Das Projekt hat demonstriert, dass sein Framework methodisch belastbar ist. Ob die gelernten Repräsentationen und Effektgrößen auf reale Hochschuldaten übertragen werden können, bleibt die offene Kernfrage — und das Kernargument für weitere Forschungsfinanzierung.

### 6. ER-Diagramm & Strukturdokumentation aktualisieren (Priorität: Niedrig)
Das ER-Diagramm aus der DE-Phase ist veraltet. Die aktuelle V4-Datenbankarchitektur (DuckDB-Backend, Parquet-Intermediate, Universum-Schema) verdient ein aktualisiertes Entity-Relationship-Diagram und Datenflussdokumentation.
