---
created: 2026-09-09
last_updated: 2026-09-09
status: abgeschlossen
tags: [data-architecture, eda, erd, curricula, simulation-dgp, causal-inference]
---

# Datenarchitektur, Relationales Schema & Explorative Datenanalyse (DeepSupport V4.1)

## 1. Relationale Datenbankarchitektur & Entity-Relationship-Modell

Die Simulations- und Analyseplattform DeepSupport basiert auf einem normalisierten, relationalen Datenmodell in Dritter Normalform (3NF) bzw. Boyce-Codd-Normalform (BCNF). Die Architektur gewährleistet strikte referenzielle Integrität (ACID-Eigenschaften) und trennt zeitinvariante institutionelle Stammdaten, soziodemografische Personenstammdaten, curriculare Regelwerke sowie dynamische Transaktions- und Verlaufsdaten voneinander.

### 1.1 Relationales Entity-Relationship-Diagramm (ERD)

Das Gesamtschema umfasst elf Kern-Entitäten. Das Curriculum wird als assoziative Relation zwischen Studiengängen und Modulen modelliert; Unterstützungsangebote bilden eine autonome Primärentität, die bei fachbezogenen Angeboten curricular an Module gekoppelt ist.

```mermaid
erDiagram
    studiengaenge ||--o{ modul_studiengang : definiert
    module ||--o{ modul_studiengang : zugeordnet
    studiengaenge ||--o{ studierende : immatrikuliert
    semester ||--o{ studierende : immatrikuliert_in
    semester ||--o{ einschreibungen : referenziert
    semester ||--o{ pruefungen : abgelegt_in
    semester ||--o{ support_teilnahmen : belegt_in
    semester ||--o{ abschluesse : beendet_in

    studierende ||--|| abschluesse : fuehrt_zu
    studierende ||--o{ einschreibungen : durchlaeuft
    studierende ||--o{ pruefungen : absolviert
    studierende ||--o{ support_teilnahmen : nutzt

    module ||--o{ pruefungen : prueft
    support_angebote ||--o{ support_modul_zuordnung : gekoppelt_mit
    module ||--o{ support_modul_zuordnung : profitiert_von
    support_angebote ||--o{ support_teilnahmen : realisiert

    studiengaenge {
        string studiengang_id PK
        string name
        string abschluss
        int regelstudienzeit
        int cp_gesamt
    }

    module {
        string modul_id PK
        string name
        int cp
        float schwierigkeit
        string turnus
        int workload_h
    }

    modul_studiengang {
        string modul_id PK, FK
        string studiengang_id PK, FK
        int empfohlenes_fachsemester
        boolean pflicht
    }

    semester {
        string semester_id PK
        int semester_nr
        string typ
        int jahr
    }

    studierende {
        string studierenden_id PK
        string studiengang_id FK
        string kohorten_semester_id FK
        string geschlecht
        int alter_immatrikulation
        float hzb_note
        string hzb_typ
        boolean migrationshintergrund
        boolean erstakademiker
        int erwerbstaetigkeit_std
        float motivation_initial
        float soziale_integration_initial
        float motivation_final
        float soziale_integration_final
        float hidden_erwartete_note_initial
        float hidden_erwartete_note_final
        float hidden_zeit_puffer
    }

    abschluesse {
        string studierenden_id PK, FK
        string status
        string abschluss_semester_id FK
        int studiendauer_semester
        float abschlussnote
        float bachelorarbeitsnote
        string anomalie_typ
    }

    einschreibungen {
        string studierenden_id PK, FK
        string semester_id PK, FK
        int fachsemester
        string status
    }

    pruefungen {
        string studierenden_id PK, FK
        string modul_id PK, FK
        int versuch PK
        string semester_id FK
        float note
        boolean bestanden
        float note_counterfactual
        boolean support_genutzt
        float hidden_motivation
        float hidden_soziale_integration
        float hidden_erwartete_note
        float hidden_overload
        float hidden_zeit_puffer
        boolean hidden_penalty_capped
        boolean hidden_support_capped
    }

    support_angebote {
        string angebot_id PK
        string name
        string typ
        int kosten_h
        boolean fachbezogen
    }

    support_modul_zuordnung {
        string angebot_id PK, FK
        string modul_id PK, FK
        float wirkungsstaerke
    }

    support_teilnahmen {
        string studierenden_id PK, FK
        string semester_id PK, FK
        string angebot_id PK, FK
    }
```

### 1.2 Normalisierungslogik und relationale Designentscheidungen

1. **Keine monolithische `curricula`-Tabelle:**  
   Die curriculare Zuordnung erfolgt über die assoziative Relation `modul_studiengang` mit dem zusammengesetzten Primärschlüssel `(modul_id, studiengang_id)`. Dadurch können Module (wie Höhere Mathematik I oder Programmierung I) von mehreren Studiengängen geteilt werden ($n:m$-Beziehung), wobei `empfohlenes_fachsemester` und `pflicht` studiengangspezifisch attribuiert werden.
2. **1:1-Isolation der administrativen Endpunkte (`abschluesse`):**  
   Die Tabelle `abschluesse` steht in einer echten 1:1-Relation zu `studierende`. Diese strikte Trennung separiert zeitlich invariante Ausgangsmerkmale (Geschlecht, HZB, Bildungsherkunft) von den erst am Studienende feststehenden Ergebniskriterien (Status, Gesamtnote, Dauer).
3. **Autonome Förderprogramm-Entität (`support_angebote`):**  
   Interventionen sind als eigenständige Primärentität modelliert. Während transversale Programme (Lerncoaching, psychologische Beratung) modulinvariant wirken, verknüpft `support_modul_zuordnung` fachspezifische Angebote (Mathe-Tutorium, Repetitorien) gezielt mit den jeweiligen Problemfächern unter Angabe einer modulspezifischen `wirkungsstaerke`.
4. **Prüfungsgranularität mit Kontrafaktum:**  
   In `pruefungen` wird neben der realisierten `note` der kontrafaktische Zwilling `note_counterfactual` (die Note bei identischem Zufallsrauschen ohne Förderboost) sowie der Satz latenter Systemzustände (`hidden_motivation`, `hidden_overload`) zeilenweise protokolliert. Dies ermöglicht eine rückwirkungsfreie Validierung kausaler Schätzer.

### 1.3 Schemadokumentation und Tabellenstatistik (Baseline $N = 50.000$)

| Tabelle | Datensätze | Schlüsselstruktur | Funktion im Gesamtsystem |
|:---|:---|:---|:---|
| `studiengaenge` | 5 | PK: `studiengang_id` | Definition der Studienprogramme, Regelstudienzeit und ECTS-Abschlussgrenzen. |
| `module` | 89 | PK: `modul_id` | Modulkatalog mit ECTS-Credits, Nenn-Workload, Grundschwierigkeit und Semesterturnus. |
| `modul_studiengang` | 89 | PK: `(modul_id, studiengang_id)` | Curriculares Regelwerk mit Pflicht-/Wahlstatus und Semesterempfehlung. |
| `semester` | 36 | PK: `semester_id` | Kalendarische Zeitachse (WS2015 bis SS2033) mit Typisierung (WS/SS). |
| `studierende` | 50.000 | PK: `studierenden_id` | Kohorten-Stammdaten, Soziodemografie, Vorbildung und latente Startparameter. |
| `abschluesse` | 50.000 | PK/FK: `studierenden_id` | Administrativer Studienausgang (Abschluss, Dropout, Zwangsexmatrikulation). |
| `einschreibungen` | 347.791 | PK: `(studierenden_id, semester_id)` | Semesterpanel mit Fachsemesterzählung und Immatrikulationsstatus. |
| `pruefungen` | 852.368 | PK: `(studierenden_id, modul_id, versuch)` | Granulare Prüfungsversuche, Noten, Kontrafakta und latente Belastung. |
| `support_angebote` | 12 | PK: `angebot_id` | Katalog der 12 Interventionsmaßnahmen mit Typ und Nenn-Zeitkosten. |
| `support_modul_zuordnung` | 44 | PK: `(angebot_id, modul_id)` | Spezifische Wirkungsstärken fachlicher Angebote auf Zielmodule. |
| `support_teilnahmen` | 134.549 | PK: `(studierenden_id, semester_id, angebot_id)` | Semesterweise Transaktionsdaten über realisierte Förderteilnahmen. |

---

## 2. Curriculares Fächer- und Modulgefüge

Das Curriculum steuert über Workload, fachliche Hürden und Prüfungsturnus die mechanische Selektion der Kohorte. Die folgenden Visualisierungen spiegeln die hierarchische Verflechtung der fünf Studiengänge, 89 Module und zwölf Hilfsangebote wider.

### 2.1 Fächerübergreifender Noten- und Schwierigkeitsvergleich (Sunburst I)

Das erste Sunburst-Diagramm visualisiert das Notengefüge der gesamten Modelluniversität unter Verwendung der originalen hierarchischen Aggregationslogik (`go.Sunburst` mit `branchvalues="total"` und der Farbskala `RdYlGn_r` von $1{,}0$ bis $5{,}0$). Der innere Ring repräsentiert die fünf Fachrichtungen, während der äußere Ring alle 89 akkreditierten Module mit ihren Klarnamen auffächert.

![Sunburst Noten-Vergleich aller Module](../images/sunburst_noten_vergleich.png)

> **Interaktives Diagramm:** Eine stufenlos zoombare Vektorversion mit Tooltips zu Prüfungsvolumina und exakten Notenschnitten steht unter [docs/interactive/sunburst_noten_vergleich.html](../interactive/sunburst_noten_vergleich.html) bereit.

#### Empirische Notenbefunde nach Fachbereichen ($N = 852.368$ Prüfungen)

1. **Maschinenbau (SG03) als curriculares Nadelöhr:**  
   Mit einer durchschnittlichen Modulnote von **$3{,}24$** und einer Erstversuchs-Durchfallquote von **$26{,}8\,\%$** stellt *Technische Mechanik I* (`MOD0035`, 9 ECTS) das am schärfsten selektierende Modul dar. Zusammen mit *Technische Mechanik II* (`MOD0038`, $\varnothing = 3{,}16$, Fail: $25{,}3\,\%$) und *Konstruktion I/II* (`MOD0036`/`MOD0041`) erzeugt dieser Strang eine kumulative Überlastung, die die Gesamtabbruchquote im Maschinenbau auf **$34{,}46\,\%$** anhebt (Studiendauer $\varnothing = 7{,}53$ Semester).
2. **Mathematische Grundlagen in der Informatik (SG01):**  
   *Programmierung I* (`MOD0001`, $\varnothing = 2{,}98$) und *Diskrete Strukturen* (`MOD0004`, $\varnothing = 3{,}02$) bündeln im 1. Fachsemester die Misserfolge. Der Informatik-Notenschnitt stabilisiert sich erst in den höheren Fachsemestern ($\varnothing = 2{,}48$ im 5. Semester; Gesamt-Dropout: $27{,}44\,\%$).
3. **Abschlussarbeiten als Leistungsplateau:**  
   Die Bachelorarbeiten (`MOD0015`, `MOD0032`, `MOD0051`, `MOD0069`, `MOD0088`) weisen über alle Fakultäten hinweg Notenmittelwerte zwischen **$1{,}45$ und $1{,}55$** bei Durchfallquoten von unter $2{,}0\,\%$ auf. Wer das Hauptstudium erreicht, schließt die Abschlussphase mit hoher Verlässlichkeit ab.

---

### 2.2 Curriculare Support-Verzahnung (Sunburst II)

Das komplementäre Sunburst-Diagramm bildet die dreistufige Hierarchie der Förderlandschaft ab: Programmname $\rightarrow$ Profitierender Studiengang $\rightarrow$ Zielmodul (`path=['sup_name', 'stg_name', 'mod_name']`).

![Sunburst Verteilung der Förderangebote](../images/sunburst_support_verteilung.png)

> **Interaktives Diagramm:** Die interaktive Version zur Exploration der Angebotsstrukturen ist abrufbar unter [docs/interactive/sunburst_support_verteilung.html](../interactive/sunburst_support_verteilung.html).

#### Struktur und Inanspruchnahme der zwölf Support-Programme ($N = 134.549$ Teilnahmen)

| Angebots-ID | Programmname | Domäne | Zeitkosten | Unique Nutzer | Teilnahmen | Hauptsächliche Zielmodule |
|:---|:---|:---|:---:|:---:|:---:|:---|
| `SUP01` | Mathe-Tutorium | fachlich | $30\,\text{h}$ | 13.636 | 17.446 | Höhere Mathematik, Diskrete Strukturen, Statistik |
| `SUP02` | Programmier-Lerngruppe | fachlich | $30\,\text{h}$ | 10.328 | 12.810 | Programmierung I & II, Algorithmen |
| `SUP03` | Statistik-Repetitorium | fachlich | $30\,\text{h}$ | 14.023 | 18.347 | Statistik I & II, Empirische Methoden |
| `SUP04` | Schreibwerkstatt | fachlich | $30\,\text{h}$ | 11.509 | 13.738 | Wissenschaftliches Arbeiten, Seminararbeiten |
| `SUP05` | Sprachkurs Englisch | fachlich | $30\,\text{h}$ | 10.370 | 12.360 | Fachsprache Wirtschaft/Technik |
| `SUP06` | Examenscoaching | fachlich | $30\,\text{h}$ | 8.472 | 9.993 | Bachelor-Kolloquium, Abschlussarbeiten |
| `SUP07` | Zeitmanagement-Workshop | überfachlich | $10\,\text{h}$ | 7.854 | 8.871 | Transversal (studiengangsunabhängig) |
| `SUP08` | Lerncoaching | überfachlich | $10\,\text{h}$ | 7.951 | 8.859 | Transversal (studiengangsunabhängig) |
| `SUP09` | Mentoring-Programm | überfachlich | $10\,\text{h}$ | 7.989 | 8.964 | Transversal (studiengangsunabhängig) |
| `SUP10` | Psychologische Beratung | psychosozial | $5\,\text{h}$ | 7.197 | 7.909 | Transversal (Krisenintervention) |
| `SUP11` | Studienberatung | psychosozial | $5\,\text{h}$ | 6.986 | 7.634 | Transversal (Status- und Fachberatung) |
| `SUP12` | Peer-Support-Gruppe | psychosozial | $15\,\text{h}$ | 6.951 | 7.618 | Transversal (Gemeinschaftsbildung) |

---

### 2.3 Prüfungsmechanik & Stochastische Notenfindung: Das Auswürfeln der Prüfungen

Die Generierung von Klausurversuchen und Noten folgt einem mehrstufigen, deterministisch reproduzierbaren stochastischen Prozess (`simuliere_pruefung` in `src/deepsupport/simulation/engine.py`).

#### 1. Semesterweise Modulbelegung
Zu Beginn jedes Semesters ermittelt der Simulator die zu belegenden Module:
- **Turnusprüfung:** Ein Modul wird nur zugelassen, wenn sein Prüfungsturnus (`WS`, `SS` oder `beides`) mit dem aktuellen Semestertyp übereinstimmt (`turnus in ("beides", akt_sem_typ)`).
- **Zulassungsvoraussetzung:** Das Modul muss offen sein und entweder im aktuellen Fachsemester curriculär empfohlen sein (`empfohlenes_fachsemester <= fachsem`) oder es handelt sich um eine Wiederholungsprüfung (`versuche > 0`).
- **Abschlussarbeits-Sperre:** Die Bachelorarbeit darf erst belegt werden, wenn nahezu alle Leistungspunkte erworben sind ($\text{CP}_{\text{bestanden}} \ge \text{CP}_{\text{gesamt}} - 18$).
- **Modulabwurf bei Zeitüberlastung:** Übersteigt der geplante Workload das Zeitbudget, werden die anspruchsvollsten Module probabilistisch zurückgestellt ($p_{\text{drop}}$, siehe Abschnitt 3.1).

#### 2. Kontinuierliche latente Leistungsfunktion ($L_{i,m}$)
Die Prüfungsleistung eines Studierenden $i$ im Modul $m$ beim Versuch $v$ wird als kontinuierliche latente Variable berechnet:

$$L_{i,m} = L_{\text{start}} + (2{,}5 - \text{erwartete\_note}_i) \cdot \gamma_{\text{hzb}} + (M_i - 0{,}5) \cdot \gamma_{\text{mot}} + (I_i - 0{,}5) \cdot \gamma_{\text{int}} - S_m \cdot \gamma_{\text{diff}} + (v - 1) \cdot \gamma_{\text{learn}} - P_{\text{overload}} + \epsilon_{\text{exam}} + B_{\text{supp}}$$

Die Modellparameter sind wie folgt kalibriert:
- $L_{\text{start}} = 0{,}50$: Basis-Leistungsniveau.
- $\text{erwartete\_note}_i \in [1{,}0; 4{,}0]$: Individuelle Leistungserwartung, initialisiert über die HZB-Abiturnote (Gewicht $\gamma_{\text{hzb}} = 0{,}15$).
- Motivation $M_i \in [0{,}05; 1{,}0]$ (Gewicht $\gamma_{\text{mot}} = 0{,}15$).
- Soziale Integration $I_i \in [0{,}05; 1{,}0]$ (Gewicht $\gamma_{\text{int}} = 0{,}05$).
- Modulschwierigkeit $S_m \in [0{,}20; 0{,}90]$ (Gewicht $\gamma_{\text{diff}} = 0{,}30$).
- **Lerneffekt bei Wiederholungsversuchen:** Für jeden Wiederholungsversuch ($v > 1$) wird ein Vorbereitungsgewinn addiert: $(v - 1) \cdot \gamma_{\text{learn}}$ mit $\gamma_{\text{learn}} = 0{,}05$.
- **Overload Penalty:** $P_{\text{overload}} = (\text{Overload-Stunden} / 100) \cdot 0{,}10$, dämpft die Klausurleistung bei Zeitmangel.
- **Deterministisches Prüfungsrauschen ($\epsilon_{\text{exam}}$):**  
  Um perfekte Kausalisolation zwischen Universen zu garantieren, wird das Rauschen nicht über den globalen Simulator-Zufall gezogen, sondern über einen isolierten Hash-Seed generiert:
  $$\text{seed}_{\text{exam}} = (\text{base\_seed} \oplus \text{CRC32}(m\_id \parallel v)) \pmod{2^{32}}$$
  $$\epsilon_{\text{exam}} \sim \mathcal{N}(0; \, \sigma_{\text{rauschen}}^2) \quad \text{mit} \quad \sigma_{\text{rauschen}} = 0{,}18$$
- **Fachlicher Förderboost ($B_{\text{supp}}$):**  
  Additiver Leistungsgewinn durch Tutorien inklusive $2/3$-Carry-over-Gedächtnis aus Vorsemestern (gedeckelt auf $\text{deckel} = 1{,}0$).

#### 3. Diskretisierung in das deutsche Hochschulnotensystem
Die latente Leistung $L_{i,m}$ wird über die Funktion `leistung_zu_note` in diskrete Notenstufen überführt:

$$\text{note}_{\text{raw}} = \text{clip}(5{,}0 - L_{i,m} \cdot 4{,}0; \, 1{,}0; \, 5{,}0)$$

Das deutsche Notenspektrum umfasst elf diskrete Stufen:
$$\mathcal{G} = \{1{,}0, \, 1{,}3, \, 1{,}7, \, 2{,}0, \, 2{,}3, \, 2{,}7, \, 3{,}0, \, 3{,}3, \, 3{,}7, \, 4{,}0, \, 5{,}0\}$$

Die Vergabe erfolgt mit einer **scharfen Durchfall-Schwelle**:
$$\text{note} = \begin{cases} 5{,}0 & \text{falls } \text{note}_{\text{raw}} \ge 4{,}0 \quad (\text{bestanden} = \text{False}) \\ \arg\min_{g \in \mathcal{G}} |g - \text{note}_{\text{raw}}| & \text{falls } \text{note}_{\text{raw}} < 4{,}0 \quad (\text{bestanden} = \text{True}) \end{cases}$$

#### 4. Kontrafaktischer Zwilling (`note_counterfactual`)
Simultan zur realisierten Note berechnet der Simulator die kontrafaktische Note:
$$\text{note}_{\text{counterfactual}} = \text{leistung\_zu\_note}(L_{i,m} - B_{\text{supp}})$$
Da $\epsilon_{\text{exam}}$ identisch ist, beziffert `note - note_counterfactual` zeilenweise den reinen Kausaleffekt des Support-Boosts auf Klausurebene.

#### 5. Drittversuchs-Regel & Zwangsexmatrikulation
Fällt ein Studierender durch (`note == 5.0`), steigt der Versuchsindex:
- Bei $v < 3$: Modul verbleibt im Status "offen" und wird im nächsten zulässigen Turnus erneut belegt.
- Bei $v \ge 3$: Das Modul erhält den Status "gescheitert". Sofern es sich nicht um die Bachelorarbeit handelt, greift die Prüfungsordnung: `studi.exmatrikuliert = True` (Endgültig nicht bestanden).

---

## 3. Aggregierte Kohorten-Dynamik: Dekonstruktion über den DGP

Die makroskopischen Verteilungsmuster und Verweildauern sind keine emergenten soziologischen Zufallsprodukte, sondern die mathematisch zwingende Konsequenz der Differential- und Differenzengleichungen des Data Generating Process (DGP in `src/deepsupport/simulation/engine.py`).

### 3.1 Das Zeitmangel-Dilemma: Skylla (Modulabwurf) vs. Charybdis (Overload)

Der Simulator modelliert Studierende mit einem rigiden Zeitbudget von $900\,\text{h}$ pro Semester (entsprechend $30\,\text{ECTS} \times 30\,\text{h}$). Die verfügbare Netto-Studienzeit $T_{\text{verf}}$ wird durch Erwerbstätigkeit gekürzt:

$$T_{\text{verf}} = \max\left(100; \, 900 - h_{\text{erwerb}} \times 20\right)$$

Treffen hoher Modul-Workload $W_{\text{plan}}$ und zeitintensive Support-Maßnahmen $C_{\text{supp}}$ auf ein geschrumpftes Zeitbudget, entsteht ein Überschuss über den individuellen Zeitpuffer $\tau_i \sim \text{Beta}(2{,}64; 5{,}36) \times 180\,\text{h}$ ($\varnothing \approx 60\,\text{h}$):

$$\Delta T_{\text{excess}} = W_{\text{plan}} + C_{\text{supp}} - T_{\text{verf}} - \tau_i$$

Studierende stehen daraufhin vor zwei schädlichen Alternativen:

```mermaid
flowchart TD
    A["Zeitbudget-Überschuss: ΔT > 0"] --> B{"Entscheidung: Modulabwurf?"}
    B -- "p_drop = ΔT / (ΔT + 50)" --> C["Skylla: Modulabwurf"]
    C --> D["Kein akuter Overload"]
    D --> E["Kumulation von CP-Rückstand"]
    E --> F["Dauerhafte Risikoerhöhung in berechne_dropout: + min(CP / 30, 1) * 0.075"]
    
    B -- "Module behalten" --> G["Charybdis: Overload"]
    G --> H["Overload Penalty = (Overload / 100) * 0.10"]
    H --> I["Leistungsabzug in Klausur (simuliere_pruefung)"]
    H --> J["Direkte Dropout-Strafe: + min(Penalty, 0.3) * 0.05"]
```

#### Empirische Beweisführung in der Gesamtkohorte

Von den $852.368$ abgelegten Prüfungen fanden **$274.718$ ($32{,}23\,\%$)** unter akutem Overload statt (mittlerer Overload bei Überlast: $161{,}0\,\text{h}$). Die resultierende Dropout-Verteilung nach Erwerbsstufen belegt die Zuspitzung ab $15$ bis $20$ Wochenstunden:

| Erwerbstätigkeit | Studierende ($N$) | Regulär Beendet | Dropout Gesamt | Abbruchquote | Hauptmechanismus im DGP |
|:---:|:---:|:---:|:---:|:---:|:---|
| $0\,\text{h}$ (Vollzeit) | 12.543 | 10.307 | 2.236 | $17{,}83\,\%$ | Kein Overload; Ausfälle rein leistungsmotiviert. |
| $5\,\text{h}$ | 7.598 | 6.071 | 1.526 | $20{,}10\,\%$ | Puffer kompensiert Zeitverlust meist vollständig. |
| $10\,\text{h}$ | 10.077 | 7.806 | 2.269 | $22{,}54\,\%$ | Gelegentlicher Abwurf von 1 Modul; moderater CP-Verzug. |
| $15\,\text{h}$ | 7.356 | 5.171 | 2.184 | $29{,}70\,\%$ | Kipp-Bereich: Puffer erschöpft, Beginn systematischer Strafen. |
| $20\,\text{h}$ (Werkstudent) | 6.474 | 3.721 | 2.747 | **$42{,}52\,\%$** | Massiver Workload-Konflikt ($400\,\text{h}$ Defizit); hohe Abwurfraten. |
| $25\,\text{h}$ | 3.519 | 1.572 | 1.944 | **$55{,}33\,\%$** | Chronischer CP-Rückstand (> 30 ECTS); permanente Hinge-Aktivierung. |
| $30\,\text{h}$ (Teilzeit) | 2.433 | 774 | 1.657 | **$68{,}19\,\%$** | Simultane Maximierung von Overload-Penalty und CP-Lag. |

---

### 3.2 Stochastische Abbruchdynamik & Timing der ersten Fachsemester

Ein scheinbares Paradoxon der deskriptiven Datenanalyse liegt im zeitlichen Profil freiwilliger Abbrüche (`status == 'abgebrochen'`):
- **Semester 1:** $2.027$ Abbrüche ($16{,}95\,\%$ aller freiwilligen Abbrüche)
- **Semester 2:** $1.705$ Abbrüche
- **Semester 3:** $1.926$ Abbrüche
- **Semester 4:** $2.104$ Abbrüche
- **Semester 5+:** Degressive Verläufe ($1.097 \rightarrow 1.168 \rightarrow 718 \dots$)

Warum brechen im 1. Semester bereits über $2.000$ Studierende ab, obwohl deren durchschnittliche Startmotivation mit **$0{,}555 \pm 0{,}124$** weit über der kritischen Wahrnehmungsschwelle von $0{,}40$ liegt?

#### Die mathematische Struktur von `berechne_dropout`

Die Dropout-Wahrscheinlichkeit $p_{\text{drop}}$ wird in jedem Semester nach Notenvergabe berechnet:

$$p_{\text{drop}} = \left[ 0{,}01 + 0{,}30 \cdot \max(0; 0{,}40 - M) + 0{,}20 \cdot \max(0; 0{,}40 - I) + 0{,}15 \cdot \min\left(\frac{\Delta_{\text{CP}}}{30}; 1\right) + 0{,}04 \cdot K_{\text{fail}} + 0{,}10 \cdot \min(P_{\text{overload}}; 0{,}3) \right] \cdot c_{\text{sem}} \cdot 0{,}5$$

mit $c_{\text{sem}} = 1{,}4$ für Fachsemester 1 und $c_{\text{sem}} = 0{,}6$ für Fachsemester $\ge 5$.

```mermaid
flowchart LR
    subgraph S1["Additive Terme in berechne_dropout"]
        T1["Basisterm: 0.01"]
        T2["Fehlversuche: + 0.04 * K_fail"]
        T3["CP-Rückstand: + 0.15 * min(CP/30, 1)"]
        T4["Motivation-Hinge: + 0.30 * max(0, 0.40 - M)"]
        T5["Integration-Hinge: + 0.20 * max(0, 0.40 - I)"]
        T6["Overload: + 0.10 * min(Penalty, 0.3)"]
    end
    T1 & T2 & T3 & T4 & T5 & T6 --> SUM["Summe p"]
    SUM --> COND{"Fachsemester == 1?"}
    COND -- "Ja" --> MULT["Faktor 1.4"]
    COND -- "Nein" --> NORM["Faktor 1.0 (Sem 2-4) bzw. 0.6 (Sem >=5)"]
    MULT --> HALF["Dämpfung: * 0.5 (Clip: [0.0, 0.45])"]
    NORM --> HALF
    HALF --> P_FINAL["Endgültiges p_drop"]
```

#### Auflösung des Mechanismus:
1. **Kein Schwellenwert, sondern additive Risikokomponenten:**  
   Die $0{,}40$-Grenze ist kein Ausschlusskriterium für Dropout, sondern lediglich der Aktivierungspunkt eines linearen Hinge-Zuschlags ($\max(0; 0{,}40 - M)$).
2. **Der Hebel der Erstsemester-Klausuren:**  
   Im 1. Semester gilt $\Delta_{\text{CP}} = 0$. Ein Student mit $M = 0{,}55$ hat einen Hinge-Wert von $0$. Scheitert er jedoch an zwei Hürdenklausuren ($K_{\text{fail}} = 2$), steigt sein Basiswert schlagartig um $+0{,}08$. Unter Einberechnung des Erstsemester-Multiplikators $1{,}4$ und der Dämpfung $0{,}5$ ergibt sich:
   $$p_{\text{drop}} = (0{,}01 + 0{,}08) \times 1{,}4 \times 0{,}5 = 0{,}063 \quad (6{,}3\,\%)$$
   Bei drei Fehlversuchen resultieren $9{,}1\,\%$.
3. **Empirische Bestätigung:**  
   Von den $2.027$ Abbrechern des 1. Semesters weisen **$1.500$ Studierende ($74{,}0\,\%$)** mindestens einen Fehlversuch auf ($531$ mit $1$ Fehlversuch, $579$ mit $2$ Fehlversuchen, $390$ mit $3$ Fehlversuchen).
4. **Motivations-Rückkopplung:**  
   Nach dem 1. Semester zieht jeder Fehlversuch die Motivation um $-0{,}05$ nach unten (`motivation = max(0.05, motivation - 0.05 * durchgefallen)`). Erst hierdurch rutschen Studierende in den Semestern 2 bis 4 unter die $0{,}40$-Schranke, was die anhaltend hohen Dropout-Zahlen bis Semester 4 erklärt.
5. **Motivationsgewinn:**  
   Ein Motivationsgewinn von $+0{,}02$ tritt ausschließlich in Semestern mit **$0$ Fehlversuchen** ein (`elif len(geplante_module) > 0`). Ein zusätzlicher Schub ("Super-Klausur") verlangt, dass die erzielte Note mindestens $0{,}5$ Notenstufen besser als die individuelle Erwartung ausfällt ($\Delta M = 0{,}005 + 0{,}01 \times (\text{Diff} - 0{,}5)$).

---

### 3.3 Zwangsexmatrikulationen: Semesterturnus-Taktung (WS/SS)

Während freiwillige Studienabbrüche unmittelbar ab dem 1. Semester auftreten, zeigt die administrative Zwangsexmatrikulation (`status == 'exmatrikuliert'`) eine scharfe Phasenverschiebung:

| Fachsemester | Zwangsexmatrikulationen ($N$) | Anteil | Kumulativ | Mechanischer Hintergrund |
|:---:|:---:|:---:|:---:|:---|
| Semester 1–4 | **0** | $0{,}0\,\%$ | $0{,}0\,\%$ | Drittversuch vor Semester 5 curricular mathematisch unmöglich. |
| Semester 5 | **998** | $39{,}89\,\%$ | $39{,}89\,\%$ | **Erster Hauptturnus:** Fehlversuch Sem 1 (WS) $\rightarrow$ Wiederholung Sem 3 (WS) $\rightarrow$ Drittversuch Sem 5 (WS). |
| Semester 6 | 364 | $14{,}55\,\%$ | $54{,}44\,\%$ | Zweiter Turnus für Module aus dem Sommersemester (Sem 2 $\rightarrow$ Sem 4 $\rightarrow$ Sem 6). |
| Semester 7 | 507 | $20{,}26\,\%$ | $74{,}70\,\%$ | Verzögerte Zweitversuche nach Modulabwurf oder Krankheitsabmeldung. |
| Semester 8–10 | 570 | $22{,}78\,\%$ | $97{,}48\,\%$ | Höhere Fachsemester / Vertiefungsmodule. |
| Semester 11–15 | 63 | $2{,}52\,\%$ | $100{,}0\,\%$ | Spätphasen-Verluste bei Langzeitstudierenden. |

Die Ursache für das Ausbleiben jeglicher Zwangsexmatrikulationen vor Semester 5 liegt in der rigiden Taktung des Modulkatalogs: Pflichtmodule werden überwiegend jährlich im Wintersemester (`turnus == 'WS'`) angeboten. Ein Studierender, der im 1. Fachsemester durchfällt, kann die Wiederholungsprüfung erst im 3. Fachsemester belegen. Scheitert auch dieser Zweitversuch, findet der finale Drittversuch zwingend im 5. Fachsemester statt. Erst der dortige Misserfolg triggert die Klausel `if m_state.versuche >= 3: studi.exmatrikuliert = True`.

---

### 3.4 Bildungsherkunft & Erstakademiker-Dynamik (V4 vs. V5)

In der Baseline-Simulation V4.1 verteilt sich der Studienausgang zwischen Studierenden aus akademischem und nicht-akademischem Elternhaus wie folgt:

| Gruppe | Studierende ($N$) | Erfolgreicher Abschluss | Gesamter Dropout | Dropout-Quote |
|:---|:---:|:---:|:---:|:---:|
| **Nicht-Erstakademiker** (`erstakademiker == False`) | 26.208 | 18.288 | 7.920 | $30{,}22\,\%$ |
| **Erstakademiker** (`erstakademiker == True`) | 23.792 | 17.134 | 6.658 | $27{,}98\,\%$ |

In Version 4 weisen Erstakademiker eine geringfügig *niedrigere* Abbruchquote auf ($-2{,}24\,\text{pp}$). Diese Eigenschaft rührt aus zwei gegensätzlichen Modellannahmen her:
1. **Sozioökonomische Belastung:** Erstakademiker erhielten im Generator einen Malus auf die soziale Startintegration (`gewicht_integration_erstakademiker = 0.08`) und weisen im Schnitt eine höhere Erwerbsneigung auf.
2. **Kompensatorische Inanspruchnahme:** In der Support-Entscheidung war ein additiver Term hinterlegt (`if studi.erstakademiker and typ in ("fachlich", "psychosozial"): p += 0.05`), der die Inanspruchnahmequote von Erstakademikern selektiv steigerte und deren Abbruchrisiko überproportional senkte.

Für die künftige Modellgeneration Version 5 ist daher ein empirisch rekalibriertes Schichtungsmodell vorgesehen (siehe [config_audit_und_v5_roadmap.md](config_audit_und_v5_roadmap.md)), das sich an den DZHW-Absolventenstudien orientiert und die strukturellen Nachteile über reale BAföG-Abhängigkeiten und finanzielle Notlagen präziser abbildet.

---

## 4. Biostatistische Methodik: Vom statischen Dashboard zur Kausalinferenz

Die methodische Evolution von DeepSupport dokumentiert den Übergang von deskriptiven, korrelativen Auswertungen zu rigorosen kausalen Identifikationsstrategien.

### 4.1 Die methodische Ausgangslage im historischen Dashboard (`Dashboard_Survival_beta.ipynb`)

Im ursprünglichen explorativen Dashboard wurden Überlebensanalysen mittels Cox Proportional Hazards und Kaplan-Meier-Schätzern gerechnet. Dem Modell lagen bereits umfangreiche Kontrollvariablen zugrunde:
- Hochschulzugangsberechtigung (HZB-Note)
- Alter bei Immatrikulation
- Geschlecht
- Erstakademiker-Status

Trotz der Inklusion dieser statischen Kontrollvariablen wiesen die Cox-Modelle für Förderangebote stark verzerrte Effektschätzer auf. Die Ursache lag in zwei methodischen Verzerrungsquellen: dem **Immortal Time Bias** und dem **Confounding by Indication**.

---

### 4.2 Dekonstruktion des Immortal Time Bias

Wird der Inanspruchnahmestatus auf aggregierter Personenebene definiert (z. B. `ever_used_support = True`, falls ein Studierender zu irgendeinem Zeitpunkt des Studiums Hilfe in Anspruch genommen hat), entsteht eine massive systematische Verzerrung:

```mermaid
flowchart TD
    subgraph S1["Gruppe A: Früher Dropout (Unbehandelt)"]
        direction LR
        A1["t = 0: Immatrikulation"] --> A2["t = 1: Fehlversuche in Klausuren"] --> A3["t = 1: Vorzeitiger Abbruch"]
        style A3 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
    end

    subgraph S2["Gruppe B: Später Absolvent (Support-Nutzer in Semester 4)"]
        direction LR
        B1["t = 0: Immatrikulation"] --> B2["t = 1 bis 3: Aktives Studium<br><b>IMMORTAL TIME WINDOW</b><br>(kann definitionsgemäß nicht abbrechen)"]
        B2 --> B3["t = 4: Erste Support-Nutzung"]
        B3 --> B4["t = 6: Erfolgreicher Abschluss"]
        style B2 fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,stroke-dasharray: 5 5
        style B4 fill:#dcfce7,stroke:#10b981,stroke-width:2px
    end

    subgraph S3["Systematischer Bias bei statischer Aggregation (ever_used_support)"]
        direction TB
        C1["Statisches Modell rechnet 'Immortal Time' (Sem 1-3) der Support-Gruppe zu"]
        C2["Frühe Dropouts (Sem 1-3) landen zu 100% in der Nicht-Nutzer-Gruppe"]
        C3["Resultat: Künstlich kollabierte Hazard Ratio (HR << 1.0) als Zensierungs-Artefakt!"]
        C1 --> C2 --> C3
        style C3 fill:#fee2e2,stroke:#b91c1c,stroke-width:2px
    end
```

1. **Die Asymmetrie der Expositionsgelegenheit:**  
   Um in einem höheren Fachsemester an einem Coaching oder Tutorium teilnehmen zu können, muss ein Studierender bis zu diesem Zeitpunkt immatrikuliert geblieben sein. Die Zeitspanne von der Einschreibung bis zur ersten Teilnahme ($t = 0$ bis $t_{\text{treat}}$) ist per Definition **"unsterblich"** (*Immortal Time*): Innerhalb dieses Intervalls kann kein Dropout registriert werden, da der Studierende andernfalls nicht als "späterer Support-Nutzer" klassifiziert worden wäre.
2. **Die Konsequenz:**  
   Frühe Misserfolge (wie die $2.027$ Abbrüche des 1. Semesters) fallen ausnahmslos der Gruppe der "Nicht-Nutzer" zu, da diese Personen gar keine Gelegenheit hatten, Förderangebote wahrzunehmen. Ein statisches Modell ordnet diese frühe Übersterblichkeit fälschlicherweise dem Fehlen von Support zu und überschätzt die Schutzwirkung drastisch ($HR \ll 1{,}0$ als reines Zensierungsartefakt).
3. **Die entgegengesetzte Verzerrung (Confounding by Indication):**  
   Wird umgekehrt ein naiver Querschnittsvergleich innerhalb eines einzelnen Semesters ohne Verlaufsdynamik gerechnet, kehrt sich die Verzerrung um: Da überwiegend Studierende in akuter Leistungsnot oder mit Durchfallrisiko Förderangebote aufsuchen, scheinen Support-Nutzer eine *höhere* Abbruchquote aufzuweisen ($HR = 1{,}20$, $p < 0{,}001$).

---

### 4.3 Die methodische Lösung: Counting Process Panel und Double Machine Learning

Um beide Verzerrungsformen simultan zu eliminieren, implementiert die DeepSupport-Pipeline zwei komplementäre Verfahren:

#### 1. Personen-Semester-Panel in Counting-Process-Notation
Der Studienverlauf wird in disjunkte Zeitintervalle $[t_{\text{start}}, t_{\text{stop}})$ zerlegt (Andersen-Gill-Erweiterung). Support-Teilnahmen werden als **streng zeitvariierende Kovariaten** modelliert:

$$Y_{i,t} = \mathbb{I}(\text{Dropout im Intervall } [t, t+1))$$

$$D_{i,t} = \mathbb{I}(\text{Support-Teilnahme im Intervall } [t, t+1))$$

Ein Studierender trägt bis zum Semester seiner ersten Fördermaßnahme ausschließlich Personen-Zeit zur unexposed Kontrollgruppe bei. Scheidet er vor einer Maßnahme aus, wird dieses Ereignis korrekt der unexposed Zeit zugeschlagen. Der Immortal Time Bias wird dadurch strukturell ausgeschlossen.

#### 2. Double Machine Learning (DML) nach Chernozhukov et al.
Um die Indikationsverzerrung (Selektion in Förderangebote auf Basis latenter Schwächen) aufzulösen, nutzt die Pipeline eine orthogonale Zweistufen-Residualisierung (Frisch-Waugh-Lovell-Theorem im Funktionsraum):

```mermaid
flowchart LR
    subgraph S1["Stufe 1: ML-Nuisance-Schätzung"]
        X["Kovariatenmatrix X (Noten, CP, Overload)"] --> M1["Propensity-Modell: g(X) = E[D|X]"]
        X --> M2["Outcome-Modell: m(X) = E[Y|X]"]
    end
    
    subgraph S2["Stufe 2: Residualisierung"]
        D["Treatment D"] --> RES_D["D_tilde = D - g(X)"]
        M1 --> RES_D
        Y["Outcome Y"] --> RES_Y["Y_tilde = Y - m(X)"]
        M2 --> RES_Y
    end
    
    subgraph S3["Stufe 3: Kausale Identifikation"]
        RES_D --> REG["Orthogonale Regression: Y_tilde = theta * D_tilde + epsilon"]
        RES_Y --> REG
        REG --> THETA["Kausaler Effekt: theta (ATE / RR)"]
    end
```

Durch die Kreuzvalidierungs-Orthogonalisierung ($K$-Fold Cross-Fitting) konvergiert der DML-Schätzer mit parametrischer Rate $\sqrt{N}$ gegen den wahren Behandlungseffekt, frei von Regularisierungs- und Selektionsverzerrungen.

---

### 4.4 Dynamische Feedbackschleifen: Teufelskreis vs. Protektionsspirale

Im Simulator existieren zirkuläre Rückkopplungen zwischen Leistung, Motivation, Zeitbelastung und Verbleib, die den Zeitverlauf prägen:

```mermaid
flowchart TD
    subgraph Negativ["Der Teufelskreis des Studienabbruchs"]
        direction TB
        N1["Prüfungsversagen (K_fail >= 1)"] --> N2["Demotivation (-0.05 pro Fehlversuch)"]
        N2 --> N3["Leistungsabfall in Folgeklausuren"]
        N2 --> N4["Steigender Hinge-Zuschlag in berechne_dropout"]
        N1 --> N5["Wiederholungszwang erzeugt Zeitdefizit"]
        N5 --> N6["Modulabwurf (p_drop) oder Overload-Penalty"]
        N6 --> N7["Kumulativer CP-Rückstand (> 15-30 CP)"]
        N7 --> N4
        N4 --> N8["Studienabbruch (Dropout)"]
        style N8 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
    end

    subgraph Positiv["Die Protektionsspirale durch Support"]
        direction TB
        P1["Bedarfsorientierte Support-Teilnahme"] --> P2["Fachlicher Notenboost (+ Boost)"]
        P2 --> P3["Prüfungserfolg (bestanden = True)"]
        P3 --> P4["Motivationserhalt / Super-Klausur-Boost"]
        P3 --> P5["Regulärer CP-Erwerb (kein CP-Rückstand)"]
        P4 & P5 --> P6["Minimierung von p_drop"]
        P6 --> P7["Erfolgreicher Studienabschluss"]
        style P7 fill:#dcfce7,stroke:#10b981,stroke-width:2px
    end
```

- **Der Teufelskreis:** Ein akutes Scheitern an einer Hürdenklausur im 1. Semester senkt die Motivation um $-0{,}05$. Im 2. Semester kollidiert die notwendige Wiederholung mit neuen Pflichtmodulen. Wählt der Student den Modulabwurf, baut er CP-Rückstand auf; behält er alle Module, droht Overload-Penalty. Beide Pfade erhöhen $p_{\text{drop}}$ in den Semestern 2 bis 4 nachhaltig.
- **Die Protektionsspirale:** Ein rechtzeitig in Anspruch genommenes Tutorium neutralisiert die Modulschwierigkeit, sichert das Bestehen und verhindert sowohl Demotivation als auch CP-Verzug.

---

### 4.5 Die methodische Reise zur Überwindung des Confounding by Indication

Das Auffinden des wahren kausalen Effekts erforderte eine mehrstufige methodische Evolution über sieben Stufen, die in den Forschungsdokumenten detailliert protokolliert ist:

| Stufe | Methodischer Ansatz | Punktschätzer | Kausale Gültigkeit & Limitation |
|:---:|:---|:---:|:---|
| **1** | **Naives Cox Proportional Hazards** | $HR = 1{,}199$ | **Fehlschlag:** Maskiert Schutzwirkung vollständig; weist Support eine Risikoerhöhung um $+20\,\%$ zu (*Confounding by Indication*). |
| **2** | **Statische Kovariaten-Adjustierung (Dashboard)** | $HR \ll 0{,}50$ | **Fehlschlag:** Verzerrt durch *Immortal Time Bias* (Nicht-Nutzer tragen frühe Dropouts allein; Spätteilnehmer müssen zwingend überleben). |
| **3** | **Subgruppen-Stratifikation** ($\text{Mot} < 0{,}40$) | $HR = 0{,}992$ | **Teilerfolg:** Schätzer kippt in die protektive Zone ($\Delta HR = -0{,}207$), eliminiert Indikation jedoch nur unvollständig. |
| **4** | **Counting-Process-Panel** ($[t_{\text{start}}, t_{\text{stop}})$) | $HR = 1{,}085$ | **Struktureller Fortschritt:** Beseitigt den Immortal Time Bias vollständig, lässt dynamische Selektion jedoch unkorrigiert. |
| **5** | **Inverse Probability Weighting (IPW / MSM)** | $RR = 0{,}982$ | **Fortschritt:** Balanciert beobachtbare Indikatoren, leidet jedoch unter Instabilitäten bei extremen Propensity-Gewichten. |
| **6** | **Double Machine Learning (DML, Frisch-Waugh-Lovell)** | $RR = 0{,}964$ | **Durchbruch:** Orthogonale Residualisierung entkoppelt Indikation vom Nettoeffekt ($p = 0{,}038$). |
| **7** | **Autoregressiver Transformer DML** | $RR = 0{,}887$ | **Beste Modellschätzung:** Erfasst sequentielle Feedbackschleifen und nähert sich der experimentellen Realität an. |
| **GT** | **Strukturelles Kausalmodell (Universum A vs. B)** | **$RR = 0{,}786$** | **Goldstandard:** Perfekt synchronisiertes kontrafaktisches Experiment ($ARR = 7{,}95\,\text{pp}$, $NNT = 12{,}6$). |

Die chronologische Entstehungsgeschichte, detaillierte mathematische Beweisführungen und Vergleichsanalysen sind in den folgenden Fachdokumenten dokumentiert:
- Zur historischen Entdeckung des Selektionsbias und dem Paradoxon: [`../07_conversation_logs/01_History_Selection_Bias_and_Confounding.md`](../07_conversation_logs/01_History_Selection_Bias_and_Confounding.md)
- Zum formalen Vergleich aller Kausalansätze (FWL vs. DML vs. Oracle): [`03_Uebersicht_Kausale_Ansaetze.md`](03_Uebersicht_Kausale_Ansaetze.md)
- Zur empirischen Gesamtauswertung der 8 Parallelwelten: [`04_Kausale_Vergleichsanalyse.md`](04_Kausale_Vergleichsanalyse.md)
- Zum systematischen Methodenreview kontrafaktischer Schätzer: [`counterfactual_methods_review.md`](counterfactual_methods_review.md)

---

## 5. Verwandte Dokumente

| Dokument | Pfad | Thematischer Bezug |
|:---|:---|:---|
| **Visuelle Datenexploration** | [visuelle_datenexploration_v4.md](visuelle_datenexploration_v4.md) | Publikationsfähige Plots (Kaplan-Meier, Treemap, KDE-Boxplots, Forest Plot). |
| **Kausale Vergleichsanalyse** | [04_Kausale_Vergleichsanalyse.md](04_Kausale_Vergleichsanalyse.md) | Ausführliche mathematische Herleitung von DML, IPW und SCM-Ground-Truth. |
| **Übersicht Kausale Ansätze** | [03_Uebersicht_Kausale_Ansaetze.md](03_Uebersicht_Kausale_Ansaetze.md) | Methodischer Vergleich von Naive vs. FWL-Partialling vs. DML vs. Oracle-Mediation. |
| **Review Kontrafaktischer Methoden** | [counterfactual_methods_review.md](counterfactual_methods_review.md) | Umfassende Evaluation kontrafaktischer Modellarchitekturen und Schätzmethoden. |
| **Historisches Selection-Bias-Protokoll** | [../07_conversation_logs/01_History_Selection_Bias_and_Confounding.md](../07_conversation_logs/01_History_Selection_Bias_and_Confounding.md) | Chronologischer Diskurs zur Genese der Parallelwelten und Aufdeckung des Indikations-Confoundings. |
| **Systematische Verteilungsanalyse** | [systematische_verteilungsanalyse_v36_vs_v41.md](systematische_verteilungsanalyse_v36_vs_v41.md) | Psychometrischer und demografischer Kovariatencheck (V3.6 vs. V4.1). |
| **Config-Audit & V5 Roadmap** | [config_audit_und_v5_roadmap.md](config_audit_und_v5_roadmap.md) | Parameter-Dokumentation des DGP und DZHW-Rekalibrierungsplan für Version 5. |
| **Master-Synopse V4 Gesamt** | [../03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md](../03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md) | Benchmarks aller 15 Sensitivitätsszenarien und 225 ML-/DL-Modelle. |
| **Datenprovenienz & Generatorzuordnung** | [datenprovenienz_und_generator_zuordnung_v36_v41.md](datenprovenienz_und_generator_zuordnung_v36_v41.md) | Herkunftsnachweis aller Dateien zwischen Legacy-V3- und Baukasten-V4-Struktur. |
