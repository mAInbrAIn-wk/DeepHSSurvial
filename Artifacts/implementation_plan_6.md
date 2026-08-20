# Implementation Plan: Simulation V3.3 & Modell-Portfolio Erweiterung (Rev. 2)

## Hintergrund & Motivation

Nach dem erfolgreichen Nachtlauf auf dem V3.2-Datensatz (mit Carry-over & verdoppeltem Support-Boost) wurden bei der Evaluation drei kritische Probleme identifiziert:

1. **Die RNG-Synchronisation zwischen den 5 Universen ist fehlerhaft** → Die kontrafaktische Validität der Makro-Effekte ist kompromittiert
2. **Die DML-Kausalschätzer zeigen instabile, teilweise widersprüchliche Ergebnisse** → Systematischer Vergleich und bessere Diagnostik nötig
3. **Exam-Level-Modelle schöpfen ihr Potenzial nicht aus** → Architektur-Anpassungen und neue Modellvarianten

---

## ARBEITSPAKET 1: RNG-Synchronisation fixen (Simulation V3.3)

> [!CAUTION]
> **Kritisches Problem:** Die RNG-Sequenz divergiert zwischen den Universen, weil der Dummy-Draw für blockierte Support-Teilnahmen die `support_zeit_kosten`-Divergenz nicht berücksichtigt. Dadurch erhalten identische Studierenden-Klone in verschiedenen Universen **verschiedene Würfelwürfe** für Prüfungsergebnisse, soziale Integration und Dropout-Entscheidungen.

### Lösung: Dedizierte RNG-Streams pro Funktionsbereich

#### [MODIFY] [`simulation_v3.py`](file:///c:/GitHub_public/Abschlussprojekt/src/simulation_v3.py)

Statt eines einzelnen sequenziellen `rng`-Streams pro Student werden **dedizierte Streams** erstellt:

```python
import zlib
base_seed = zlib.crc32(studi.studierenden_id.encode('utf-8'))  # Prozess-stabiler Hash

rng_init     = np.random.default_rng(base_seed)       # Einmalige Stage-Setting-Würfe
rng_support  = np.random.default_rng(base_seed + 1)   # Support-Entscheidungen
rng_exam     = np.random.default_rng(base_seed + 2)   # Prüfungsrauschen
rng_social   = np.random.default_rng(base_seed + 3)   # Soziale Integration Drift
rng_dropout  = np.random.default_rng(base_seed + 4)   # Dropout-Entscheidung
```

#### Stream-Zuordnung (detailliert nach Code-Analyse)

| Stream | Aufrufe | Zeilen | Synchronisations-Risiko |
| :--- | :--- | :--- | :--- |
| **`rng_init`** | Anomalie-Check (`rng.random() < anomalie_quote`), Anomalie-Typ (`rng.choice()`), "sehr_lang" Modul-Drop (`rng.random() < 0.4`) | 110, 112, 154 | **Keines.** Alle drei Draws passieren deterministisch am Anfang. Der Anomalie-Check (Z. 110) prüft, ob ein Student einer von 4 Sondertypen ist: `super_schnell` (20%), `sehr_lang` (40%), `fruehabbruch` (25%), `plateau` (15%). Der Anomalie-Typ wird einmal statisch gesetzt. Der "sehr_lang"-Draw (Z. 154) passiert zwar im Semester-Loop, aber da der Anomalie-Typ statisch ist, wird dieser Draw in allen Universen genau gleich oft aufgerufen (nur für "sehr_lang"-Studis, und dieser Typ ändert sich nie). → **`rng_init` kann alle drei zusammenfassen.** |
| **`rng_support`** | Support-Annahme (`rng.random() < p`), Zeitbudget-Override (`rng.random() < 0.2`) | 181, 187 | **Hoch, aber isoliert.** Divergiert absichtlich zwischen Universen (verschiedene Support-Pfade). Der entscheidende Punkt: Diese Divergenz betrifft NUR `rng_support` und infiziert keine anderen Streams mehr. |
| **`rng_exam`** | Prüfungsrauschen (`rng.normal(0, noise)`) in `simuliere_pruefung` | 146 (in simulation_v2.py) | **KRITISCH – hier liegt das Hauptproblem.** |
| **`rng_social`** | Soziale Integration Drift (`rng.normal(0, 0.05)`) | 285 | **Gering.** Wird genau 1× pro Semester aufgerufen. |
| **`rng_dropout`** | Dropout-Entscheidung (`rng.random() < p_drop`) | 298 | **Gering.** Wird genau 1× pro Semester aufgerufen, am Ende. |

#### Prüfungsreihenfolge: Das Kernproblem & die Lösung

**Ihre Frage:** *"Kann die Reihenfolge der Prüfungen durcheinander kommen, durch Verschiebung, Durchfallen etc.?"*

**Analyse des Codes (Z. 131-155, Z. 221):**
- `geplante_module` wird deterministisch aufgebaut: Zuerst alle Module des Studiengangs iterieren (Z. 132, Reihenfolge kommt aus `sg_module_dict`), dann ggf. Super-Schnell-Erweiterung (Z. 147-153), dann ggf. "sehr_lang"-Kürzung (Z. 154-155), dann Workload-Abwurf (Z. 209-212, deterministisch nach Schwierigkeit sortiert).
- **Aber:** Die Menge der Module in `geplante_module` divergiert zwischen Universen, weil Durchfallquoten unterschiedlich sind. Ein Student, der in Universum A eine Prüfung besteht (dank Support-Boost), hat diese Prüfung in Semester 3 nicht mehr auf der Liste. In Universum B (ohne Support) wiederholt er sie → andere Modulliste → **andere Anzahl `rng_exam`-Draws**.

**Lösung: Index-basiertes Prüfungsrauschen**

Statt sequenziell `rng_exam.normal()` zu ziehen, wird das Rauschen **positionsunabhängig** aus dem Modul-ID und Versuchsnummer berechnet:

```python
def get_exam_noise(base_seed: int, modul_id: str, versuch: int) -> float:
    """Deterministisches Prüfungsrauschen, unabhängig von der Reihenfolge."""
    exam_seed = base_seed ^ zlib.crc32(f"{modul_id}_{versuch}".encode())
    return np.random.default_rng(exam_seed).normal(0, CONFIG["gewicht_rauschen"])
```

Damit bekommt die Prüfung MOD0015 Versuch 2 für Student STUD00042 **immer denselben Rauschwert**, egal ob in Universum A oder B, egal ob vorher 3 oder 7 andere Prüfungen geschrieben wurden.

#### Dropout-Entscheidung: Synchronisation OK?

**Ihre Frage:** *"Wann wird die Dropout-Entscheidung getroffen? Einmal am Ende des Semesters?"*

**Antwort:** Ja, exakt einmal pro Semester am Ende (Z. 293-299). Der `rng_dropout.random()`-Aufruf passiert genau 1× pro aktivem Semester. Da `rng_dropout` ein eigener Stream ist, ist die Synchronisation gewährleistet – **solange** der Student in beiden Universen die gleiche Anzahl aktiver Semester hat. Wenn er in Universum B im Semester 3 abbricht, aber in A bis Semester 6 weiterlebt, dann hat A mehr `rng_dropout`-Draws, aber das ist die **tatsächliche kausale Divergenz** (ein Student der abbricht, hat keine weiteren Dropout-Entscheidungen mehr zu treffen). Das ist methodisch korrekt.

**Analoges gilt für `rng_social`:** Genau 1× pro Semester, am Ende.

---

## ARBEITSPAKET 2: Systematische DML-Evaluierung & Diagnostik

#### Befunde aus dem V2 ↔ V3.2 Vergleich:

| Support-Typ | V2 Standard-DML | V3.2 Standard-DML | Ground Truth V3.2 | Bewertung |
| :--- | :--- | :--- | :--- | :--- |
| **Fachlich** | RR = 0.8953 | RR = 0.7899 | RR = 0.9574 | Überschätzt massiv |
| **Überfachlich** | RR = 0.9708 | **RR = 1.0702** | RR = 0.9383 | **Richtung falsch!** |
| **Psychosozial** | RR = 0.8619 | RR = 0.9656 | RR = 0.9514 | Akzeptabel |

#### Geplante Maßnahmen:

1. **Deep Transformer-DML auf alle 3 Support-Typen erweitern** (aktuell: nur `fach_supp_active`)
2. **Erweiterte Metrik-Sammlung:** Konfidenzintervalle (Bootstrap), Propensity-Score-Balance, Residualdiagnostik
3. **Systematisches Evaluationsartefakt** mit Ground-Truth-Abgleich

#### [MODIFY] [`train_transformer_dml.py`](file:///c:/GitHub_public/Abschlussprojekt/src/train_transformer_dml.py)
- Treatment-Variable von nur `fach_supp_active` auf alle drei Typen (`fach`, `uebf`, `psych`) erweitern, Ergebnisse pro Typ separat in JSON speichern

---

## ARBEITSPAKET 3: Deep Transformer Regression & Survival

### A) Deep Exam-Transformer Regression (Noten-/GPA-Vorhersage)
- **Architektur:** `d_model=128`, `num_heads=8`, `3 Encoder-Blöcke`, **Attention-Weighted Pooling** statt GlobalAveragePooling1D
- **Epochen:** ≥ 50 mit EarlyStopping (patience=15, restore_best_weights=True)
- **Split:** 70/15/15 (Train/Val/Test)

### B) Deep Semester-Transformer Regression (Vergleichsmodell, identische Architektur, T=16)

### C) Deep Transformer Survival (Abbruchvorhersage mit Attention Pooling)
- Analog zur Regression, aber mit Sigmoid-Output und masked_binary_crossentropy
- Attention-Weighted Pooling lernt, welche Semester/Prüfungen für die Dropout-Vorhersage informativ sind

#### [NEW] `deep_transformer_regression.py` – Alle drei Modelle in einem Script

---

## ARBEITSPAKET 4: DeepSurv & Trainingsparameter-Audit

### DeepSurv-Modelle: Epochen verdoppeln & EarlyStopping

| Modell | Aktuell | Neu |
| :--- | :--- | :--- |
| **DeepSurv Landmark** | 150 Epochen, Full-Batch, kein ES | **300 Epochen**, Full-Batch, **EarlyStopping(patience=100, monitor='val_loss')** |
| **Extended DeepSurv Delta** | 150 Epochen, Full-Batch, kein ES | **300 Epochen**, Full-Batch, **EarlyStopping(patience=100)** |

> [!NOTE]
> Full-Batch-Training mit Cox-Partial-Likelihood erzeugt glatte, monotone Verlustkurven ohne Overfitting-Risiko durch Mini-Batch-Stochastik. 150 Epochen waren daher eher zu wenig als zu viel. Mit 300 Epochen und großzügigem EarlyStopping (patience=100) geben wir dem Modell maximale Konvergenzmöglichkeit.

### DeepSurv-Kapazität: Zu klein?
Die aktuelle Architektur ist 64→32→16→1 (≈ 3.800 Parameter). Für ein Person-Semester-Panel mit ~360.000 Zeilen und ~20 Features könnte das unterdimensioniert sein. **Vorschlag:** Eine zweite Variante mit 128→64→32→1 (≈ 15.000 Parameter) als Experiment hinzufügen.

### Three-Way-Split für alle Modelle vereinheitlichen

Aktuell fehlt bei folgenden Modellen ein Validation-Set:

| Modell | Aktuell | Neu |
| :--- | :--- | :--- |
| **DML Orthogonal Survival** | 80/20 (Train/Test) | **70/15/15 (Train/Val/Test)** + EarlyStopping |
| **Extended DeepSurv Delta** | 80/20 (Train/Test) | **70/15/15** + ES(patience=100) |
| **Extended Logistic Hazard Delta** | 80/20 (Train/Test) | **70/15/15** + EarlyStopping |

#### [MODIFY] [`extended_deep_survival_delta.py`](file:///c:/GitHub_public/Abschlussprojekt/src/extended_deep_survival_delta.py)
#### [MODIFY] [`dml_orthogonal_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/dml_orthogonal_survival.py)
#### [MODIFY] [`deep_survival.py`](file:///c:/GitHub_public/Abschlussprojekt/src/deep_survival.py) (DeepSurv Landmark)

---

## ARBEITSPAKET 5: Modell-Uniformitäts-Audit & Review-Artefakt

> [!IMPORTANT]
> Systematische Überprüfung aller 20+ Modelle auf Einheitlichkeit und Funktionalität.

### Audit-Checkliste:
1. **Feature-Konsistenz:** Nutzen alle Modelle dieselben Features? Sind die Feature-Spalten konsistent benannt? Werden `hidden_*`-Spalten korrekt ausgeschlossen?
2. **Split-Konsistenz:** Haben alle Modelle den 70/15/15 Three-Way-Split? (Werden Student-IDs oder Zeilen gesplittet – Group Split vs. Random Split?)
3. **Metrik-Konsistenz:** Speichern alle Modelle die gleichen Metriken im selben JSON-Format?
4. **Lernkurven-Diagnostik:** Welche Modelle zeigen Overfitting, welche sind untertrainiert?
5. **Funktionalität:** Laufen alle Modelle fehlerfrei durch? Gibt es Deprecation Warnings, die Ergebnisse beeinflussen?

#### [NEW] `model_uniformity_audit.md` – Review-Artefakt mit Ergebnissen

---

## ARBEITSPAKET 6: Hypothesen-Evolution & README-Update

✅ Bereits erledigt (Phase 10 & 11 hinzugefügt, README auf V3.2 aktualisiert).
Nach V3.3-Lauf: Erneute Aktualisierung der Zahlen.

---

## Priorisierung & Reihenfolge

| Prio | Arbeitspaket | Aufwand | Wann |
| :--- | :--- | :--- | :--- |
| 🔴 **1** | AP1: RNG-Synchronisation fixen (V3.3) | Mittel | **Vor dem Nachtlauf** |
| 🔴 **2** | AP4: DeepSurv Epochen + Three-Way-Split | Gering | **Vor dem Nachtlauf** |
| 🟡 **3** | AP3: Deep Transformer Regression & Survival | Mittel | **Vor dem Nachtlauf** |
| 🟡 **4** | AP2: DML erweitern (3 Support-Typen) | Gering | **Vor dem Nachtlauf** |
| 🟢 **5** | AP5: Uniformitäts-Audit | Gering | **Nach dem Nachtlauf** |
| 🟢 **6** | AP6: Doku-Updates | Gering | **Nach dem Nachtlauf** |

## Nachtlauf-Ablauf

1. **AP1-AP4 implementieren** (Code-Änderungen)
2. **`output_dl` umbenennen** (→ `output_dl_v3.2_carryover`)
3. **`run_overnight.py` starten** (V3.3 Simulation + 20+ Modelle + neue Deep Transformer + DML erweitert)
4. **AP5: Audit nach Abschluss**
5. **AP6: Doku-Update mit finalen V3.3-Zahlen**

Geschätzte Gesamtlaufzeit: **3-4 Stunden** (Nachtlauf).
