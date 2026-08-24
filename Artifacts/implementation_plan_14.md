# Implementation Plan V5: Nachtlauf V4, Diagnostik, Next-Exam & Mediation

## Hintergrund & Motivation

Dieses Update umfasst **7 Arbeitspakete**, die zusammen in einem konsolidierten Nachtlauf ausgeführt werden sollen. Grundlage ist eine vollständige Recherche des Seed-Managements, aller Clipping-Stellen, des Feature-Builder-Migrationsstands und der Orchestrierungsdateien.

---

## Vorab-Klärungen (Antworten auf Ihre Annotationen)

### A. Vollständiges Seed-Inventar

Es gibt **drei Seed-Schichten** im Projekt:

| Schicht | Seed-Wert | Dateien | Zweck |
| :--- | :---: | :--- | :--- |
| **1. Population-Seed** | `12345` | [`simulation_v3.py:347`](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L347), [`simulate_universes_fgh.py:55`](file:///C:/GitHub_public/Abschlussprojekt/src/simulate_universes_fgh.py#L55), [`run_overnight.py:90`](file:///C:/GitHub_public/Abschlussprojekt/src/run_overnight.py#L90) | Identische Studierenden-Population über alle 8 Universen |
| **2. Per-Student Trajectory Seeds** | `CRC32(studierenden_id)` | [`simulation_v3.py:107-111`](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L107-L111) | 4 isolierte RNG-Streams pro Student (Init, Support, Sozial, Dropout) |
| **3. ML-Train/Test-Split & TF-Seed** | `42` | **45+ Dateien** (alle `train_test_split`, `tf.random.set_seed`, `RandomForest`, etc.) | Reproduzierbare Datensplits und Gewichtsinitalisierung |
| *Legacy (V1)* | `CONFIG['seed'] = 42` | [`config.py:33`](file:///C:/GitHub_public/Abschlussprojekt/src/config.py#L33), [`main.py:14`](file:///C:/GitHub_public/Abschlussprojekt/src/main.py#L14) | Veraltet, wird nur noch von `main.py` (V1) genutzt |

> [!WARNING]
> **Kritischer Befund: Per-Student-Seeds sind NICHT mit dem Population-Seed gesalzen!**
> In [`simulation_v3.py:107`](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L107):
> ```python
> base_seed = zlib.crc32(studi.studierenden_id.encode('utf-8'))
> ```
> Da die Studierenden-IDs (`"STUD000001"` bis `"STUD050000"`) bei neuem Population-Seed identisch bleiben, würden die **Trajektorien-Zufallszahlen** (Prüfungsrauschen, Support-Entscheidungen, Dropout-Würfe) exakt **wiederholt** – nur die Stammdaten (HZB, Motivation, Integration) wären anders.
>
> **Lösung:** Salzen mit dem Population-Seed:
> ```python
> base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ POPULATION_SEED) & 0xFFFFFFFF
> ```

### B. Oracle-Modelle: Kein eigener Modelltyp, sondern Feature-Modus

Sie haben völlig Recht: Oracle ist wie `blind`, `gradeblind`, `realistic` ein **Feature-Modus**, keine eigenständige Modellklasse. Der Feature-Grid-Runner behandelt Oracle korrekt als 5. Modus. Die separaten `train_oracle_models.py` und `counterfactual_oracle_*.py` sind historische Überbleibsel aus der Zeit vor dem Grid-Runner. In einem vollständigen Lauf sollten **alle Modelle × alle 5 Modi** systematisch evaluiert werden.

### C. Warum fehlen Modelle im Grid-Runner?

**Kernbefund:** Von 27+ eigenständigen Trainings-Skripten nutzt aktuell **nur 1 Skript** den zentralen [`feature_builder.py`](file:///C:/GitHub_public/Abschlussprojekt/src/feature_builder.py):

| Status | Skripte |
| :--- | :--- |
| ✅ **Migriert** auf `feature_builder.py` | [`run_feature_grid_experiments.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_feature_grid_experiments.py) (4 Modelltypen × 5 Modi) |
| ❌ **Eigene Datenladung** (inline) | Alle 26+ anderen Trainings- und CF-Skripte |

Das ist der Grund, warum der Grid-Runner nur 4 Modelltypen abdeckt: Nur diese 4 wurden auf die neue, vereinheitlichte Feature Engine umgestellt. Die übrigen Skripte laden Daten weiterhin inline und unterstützen deshalb den Modi-Wechsel (`gradeblind`, `oracle` etc.) nicht.

### D. Orchestrierungs-Dateien: Status & Redundanzen

| Datei | Status | Funktion |
| :--- | :--- | :--- |
| [`main.py`](file:///C:/GitHub_public/Abschlussprojekt/src/main.py) | ❌ **Veraltet (V1)** | Importiert nur von `simulation.py` (V1). Erzeugt keine Universen. |
| [`run_all_experiments.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_all_experiments.py) | ⚠️ **Teils veraltet** | 10 Stufen, enthält Baselines (NB, SVM, RF), Timeseries-Regressoren und Calibration – fehlt aber die V3.3 Dual-Strand CF-Suite. |
| [`run_retrain_all.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_retrain_all.py) | ✅ **Aktuell** | 27 Schritte (V3.3 Edition), aber ohne Baselines und ohne Grid-Runner. |
| [`run_overnight.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_overnight.py) | ⚠️ **Ruft `run_all_experiments` statt `run_retrain_all` auf!** | Simulation → Validation → GT → ~~`run_all_experiments`~~ → Transformer-DML |
| [`run_feature_grid_experiments.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_feature_grid_experiments.py) | ✅ **Aktuell** | 4 Modelltypen × 5 Modi, aber standalone, nicht in Pipeline integriert. |

**→ Handlungsbedarf:** Konsolidierung in **eine** Master-Orchestrierung (`run_overnight.py` V4).

---

## Bestandsaufnahme: Simulation-Clipping (30+ Stellen)

### Vollständiges Clipping-Register in `simulation_v3.py`

| Kategorie | Variable | Bounds | Zeile | Diagn. Relevanz |
| :--- | :--- | :---: | :---: | :--- |
| **Stammdaten** | Alter | $[17, 45]$ | 33 | 🟡 Prüfenswert |
| | HZB-Note | $[1{,}0,\; 4{,}0]$ | 34 | 🟡 Prüfenswert |
| | Erwerbstätigkeit | $[0, 40]$ h/Woche | 51 | 🟢 Unkritisch |
| **Latente Zustände** | Motivation (Initial) | $[0{,}05,\; 1{,}0]$ | 53 | 🟡 Prüfenswert |
| | Soziale Integration (Init.) | $[0{,}05,\; 1{,}0]$ | 57 | 🟡 Prüfenswert |
| | Erwartete Note | $[1{,}0,\; 4{,}0]$ | 46 | 🟢 Unkritisch |
| | Zeitpuffer $B_i$ | $[0, 180]$ h | 60 | 🟡 Prüfenswert |
| **Dynamische Updates** | Motivation (Random Walk) | Floor $0{,}05$, Ceil $1{,}0$ | 295, 297 | 🔴 **Kritisch** |
| | Soziale Integration (Walk) | $[0{,}05,\; 1{,}0]$ | 299 | 🔴 **Kritisch** |
| **Support-Mechanik** | Support-Aufnahme-Wkt. | $[0{,}0,\; 0{,}9]$ | 190 | 🟡 Prüfenswert |
| | **Support-Boost (Noten)** | $[0{,}0,\; \text{support\_deckel}]$ = $[0, 1{,}0]$ | 242 | 🔴 **Kritisch** |
| | **Overload-Penalty** | $\le 0{,}15$ | 229 | 🔴 **Kritisch** |
| **Dropout** | **Dropout-Wkt. pro Semester** | $[0{,}0,\; 0{,}45]$ | V2:172 | 🔴 **Kritisch** |
| | CP-Rückstand-Faktor | $\le 1{,}0$ (normiert) | V2:169 | 🟡 Prüfenswert |

> [!IMPORTANT]
> **Gute Nachricht:** V3 loggt bereits die Booleans `support_capped` ([L243](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L243)) und `hidden_penalty_capped` ([L268](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L268)) pro Prüfung. Allerdings werden diese Flags **nicht aggregiert oder exportiert** – sie gehen beim CSV-Export verloren.

---

## ARBEITSPAKET 1: Verbose-Modus & Clipping-Diagnostik für die Simulation

### Ziel
Ein `--verbose`-Flag für die Simulation, das nach dem Lauf eine vollständige Statistik aller Clipping-Ereignisse als JSON + Markdown schreibt.

### Zu erfassende Metriken (pro Universum)

```python
clipping_stats = {
    "population": {
        "alter_clipped": {"lower": int, "upper": int, "total": int, "pct": float},
        "hzb_note_clipped": {"lower": int, "upper": int, "total": int, "pct": float},
        "motivation_initial_clipped": {"lower": int, "upper": int, "pct": float},
        "soziale_integration_initial_clipped": {"lower": int, "upper": int, "pct": float},
        "zeitpuffer_clipped": {"lower": int, "upper": int, "pct": float},
    },
    "trajectory_dynamics": {
        "motivation_floor_hits": int,    # wie oft fiel Motivation auf 0.05
        "motivation_ceil_hits": int,     # wie oft wurde 1.0 erreicht
        "soz_int_floor_hits": int,
        "soz_int_ceil_hits": int,
    },
    "support_mechanics": {
        "support_boost_capped": int,     # Anzahl Prüfungen mit gedeckeltem Boost
        "support_boost_capped_pct": float,
        "overload_penalty_capped": int,  # Anzahl Semester mit gedeckelter Penalty
        "overload_penalty_capped_pct": float,
        "support_uptake_prob_clipped": {"lower": int, "upper": int},
    },
    "dropout_mechanics": {
        "dropout_prob_capped_at_045": int,  # Wie oft hat der 0.45-Cap gegriffen?
        "dropout_prob_capped_pct": float,
    }
}
```

### Proposed Changes

#### [MODIFY] [`src/simulation_v3.py`](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py)
- Einführen einer `ClippingTracker`-Klasse, die bei jeder `np.clip()`/`min()`/`max()`-Stelle inkrementiert.
- Am Ende jedes Universums: Export als `clipping_diagnostics_{universe}.json`.
- Markdown-Summary: `clipping_diagnostics_summary.md` mit Tabellen und Prozentwerten.

#### [NEW] `src/output_dl/diagnostics/clipping_diagnostics_summary.md`
- Automatisch generierte Übersichtstabelle aller Caps über alle 8 Universen.

---

## ARBEITSPAKET 2: Pipeline-Benchmarks (Laufzeit, Speicher, CPU)

### Ziel
Jeder Schritt des Nachtlaufs liefert strukturierte Timing-Daten in einer zentralen JSON-Datei.

### Zu erfassende Metriken pro Schritt

```python
step_benchmark = {
    "step_name": str,
    "step_index": int,
    "wall_time_seconds": float,
    "peak_rss_mb": float,       # via psutil.Process().memory_info().rss
    "cpu_percent_avg": float,   # via psutil.cpu_percent(interval=None)
    "status": "OK" | "FEHLER",
    "error_message": str | None
}
```

### Proposed Changes

#### [MODIFY] [`src/run_overnight.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_overnight.py)
- `run_step()` wird erweitert um `psutil`-basiertes Memory/CPU-Tracking.
- Am Ende: Export als `pipeline_benchmark.json` + `pipeline_benchmark.md`.

#### [NEW] `src/output_dl/diagnostics/pipeline_benchmark.md`
- Automatisch generierte Tabelle mit Zeitverbrauch, Speicher-Peak und Status pro Schritt.

---

## ARBEITSPAKET 3: DuckDB vs. NumPy Backbone-Sanity-Check

### Befund
**DuckDB ist aktuell NICHT in `feature_builder.py` implementiert.** Es existiert nur ein Benchmark-Skript ([`test_duckdb_benchmark.py`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/scratch/test_duckdb_benchmark.py)) und ein Architektur-Designdokument. Die Feature Engine ist rein Pandas/NumPy.

### Maßnahme
Statt eines Dual-Backend-Vergleichs wird ein **Validierungs-Skript** erstellt, das:
1. Die Pandas-Pipeline (`feature_builder.py`) laufen lässt und das Ergebnis als Referenz speichert.
2. Die äquivalente DuckDB-SQL-Query auf denselben CSVs ausführt.
3. Beide Ergebnisse elementweise vergleicht (`np.allclose` / `pd.testing.assert_frame_equal`).
4. Timing und Speicherverbrauch beider Pfade erfasst.
5. Einen Abschlussbericht als Markdown generiert.

#### [NEW] [`src/benchmark_backbone_sanity_check.py`](file:///C:/GitHub_public/Abschlussprojekt/src/benchmark_backbone_sanity_check.py)
- Vergleicht Pandas vs. DuckDB für `build_exam_sequence_tensor` und `build_semester_panel_df`.
- Output: `backbone_sanity_check.md` + `backbone_sanity_check.json`.

---

## ARBEITSPAKET 4: V4-Replikation mit neuem Seed

### Seed-Strategie (zentralisiert & versioniert)

```python
# In config.py oder als CLI-Argument:
SIMULATION_CONFIG = {
    "population_seed": 12345,       # V3
    # "population_seed": 99999,     # V4
    "ml_seed": 42,                  # Train/Test-Splits & TF
    "output_dir": "output_dl",      # bzw. "output_dl_v4"
}
```

#### [MODIFY] [`src/config.py`](file:///C:/GitHub_public/Abschlussprojekt/src/config.py)
- Neues Feld `population_seed` (Default: `12345`).
- `output_dir` überschreibbar per Umgebungsvariable: `os.environ.get('DEEPSUPPORT_OUTPUT_DIR', 'output_dl')`.

#### [MODIFY] [`src/simulation_v3.py`](file:///C:/GitHub_public/Abschlussprojekt/src/simulation_v3.py#L107)
- Per-Student-Seed mit Population-Seed salzen:
  ```python
  base_seed = (zlib.crc32(studi.studierenden_id.encode('utf-8')) ^ POPULATION_SEED) & 0xFFFFFFFF
  ```
- `POPULATION_SEED` als Parameter der `main()`-Funktion statt Hardcode.

#### [MODIFY] [`src/simulate_universes_fgh.py`](file:///C:/GitHub_public/Abschlussprojekt/src/simulate_universes_fgh.py#L55)
- Gleiche Parametrisierung.

#### [NEW] `src/compare_v3_v4.py`
- Automatischer Vergleich der Makro-Effekte (Dropout-Raten, Relative Risks) zwischen `output_dl/` und `output_dl_v4/`.
- Berechnung der Monte-Carlo-Standardfehler: $\sigma \approx \sqrt{p(1-p)/N}$.
- Flagging, wenn Abweichungen > $2\sigma$.

---

## ARBEITSPAKET 5: Orchestrierungs-Konsolidierung

### Ziel
Eine einzige, vollständige Master-Pipeline, die **alle** Schritte umfasst.

### Neuer Ablauf [`run_overnight.py`](file:///C:/GitHub_public/Abschlussprojekt/src/run_overnight.py) (V4-Edition)

```
Phase 0: Konfiguration
  ├── Seed, Output-Dir, Verbose-Flag einlesen
  └── psutil Benchmark-Tracker initialisieren

Phase 1: Simulation V3 (8 Universen A–H) [~40 Min.]
  ├── Verbose Clipping-Diagnostik
  └── → output_dl/ oder output_dl_v4/

Phase 2: Validierung & Ground Truth [~2 Min.]
  ├── validate.py
  ├── calculate_true_effect.py
  └── extract_grade_duration_gt.py (aus scratch/ nach src/ migrieren)

Phase 3: Baselines (Klasse 1 & 2) [~5 Min.]
  ├── train_mlp_baseline.py (NB, RF, SVM, Keras MLP)
  └── train_mlp_regression.py (Ridge, SVR, RF, MLP Regressor)

Phase 4: Modell-Training (27 Schritte aus run_retrain_all) [~90 Min.]
  └── Alle Survival-, Transformer-, DML-, Oracle-Modelle

Phase 5: Feature-Grid Benchmark (4+ Modelle × 5 Modi) [~30 Min.]
  └── run_feature_grid_experiments.py

Phase 6: Next-Exam Regression [NEU, ~10 Min.]
  ├── next_exam_regression.py (GRU Dual-Head)
  └── next_exam_transformer.py (Sliding-Window)

Phase 7: Mediationsanalyse [NEU, ~5 Min.]
  └── mediation_analysis.py (Imai/Pearl Bootstrap)

Phase 8: Backbone-Sanity-Check [~2 Min.]
  └── benchmark_backbone_sanity_check.py

Phase 9: V3/V4-Vergleich (falls V4-Modus) [~1 Min.]
  └── compare_v3_v4.py

Phase 10: Analyse-Skripte & Reports [~5 Min.]
  ├── analyze_support_effects.py
  ├── analyze_v3_deep.py
  ├── summarize_final_results.py (aus scratch/ migrieren)
  └── Pipeline-Benchmark-Report generieren

Geschätzte Gesamtdauer: ~3–4 Stunden
```

### Geschätzte Laufzeiten pro Schritt (basierend auf bisherigen Logs)

| Phase | Schritte | Geschätzte Dauer | Basis |
| :--- | :--- | :---: | :--- |
| **1. Simulation** | 8 Universen × 50k | ~40 Min. | `overnight_run_v3.log` |
| **2. Validierung** | validate + GT | ~2 Min. | Erfahrungswert |
| **3. Baselines** | 4 Klassif. + 4 Regr. | ~5 Min. | sklearn = schnell |
| **4. Modell-Training** | 14 Trainings + 13 CF | ~90 Min. | `run_retrain_all` Logs |
| **5. Feature-Grid** | 4 × 5 = 20 Trainings | ~30 Min. | Grid-Runner Logs |
| **6. Next-Exam** | 2 neue Modelle | ~10 Min. | Schätzung (800k Samples) |
| **7. Mediation** | Bootstrap $B=1000$ | ~5 Min. | Schätzung |
| **8. Backbone-Check** | Pandas vs. DuckDB | ~2 Min. | Benchmark-Ergebnis |
| **9. V3/V4-Vergleich** | JSON-Diff | ~1 Min. | Trivial |
| **10. Analysen** | 3–5 Skripte | ~5 Min. | Erfahrungswert |
| **Gesamt** | | **~3–4 Stunden** | |

---

## ARBEITSPAKET 6: Next-Exam Autoregressive Regression (Klasse 10)

### Abgrenzung zu Survival-Modellen (Ihre Rückfrage)

Die Überlebensmodelle prognostizieren das binäre Dropout-Ereignis $Y_{i,t} \in \{0, 1\}$ an jedem Zeitschritt. Die Next-Exam Regression prognostiziert ein **kontinuierliches Outcome** (Note) **plus** ein binäres Outcome (Bestehen/Durchfallen) – beides auf **Prüfungsebene**, nicht Semesterebene.

Die Zensurmechanik ist hier nicht direkt anwendbar: Ein Student, der eine Prüfung nicht ablegt, ist nicht "zensiert" im Survival-Sinne, er hat sie einfach noch nicht geschrieben. Ein Competing-Risks-Ansatz auf Prüfungsebene (Event ∈ {Bestanden, Durchgefallen, Nicht angetreten}) wäre methodisch möglich, führt aber direkt zurück zu den bestehenden Exam-GRU Klassen 7. **Fazit:** Next-Exam Regression ist ein eigenständiges Modul, das die Prüfungsebene mit einer anderen Fragestellung bearbeitet.

### Zwei Varianten (wie besprochen)

**Variante A – Full-History Autoregressive:**
- Input: $(x_1, y_1, \dots, x_k, y_k)$ (variable Länge, gepaddet + Masking)
- Output: $(\hat{y}_{k+1}, P(\text{Fail}_{k+1}))$
- Datenmenge: $\sum_i (K_i - 1) \approx 800.000$ Trainingsbeispiele

**Variante B – Fixed-Window ($w = 8$):**
- Input: Die letzten $w$ Prüfungen + kumulierte Summary-Features (GPA, Fails, CP)
- Vorteil: Schneller, keine Masking-Problematik
- Ergänzt durch statische Features (HZB, Erwerb) und nächste-Prüfungs-Kontext (Schwierigkeit, Versuch, CP)

### Proposed Changes

#### [NEW] [`src/next_exam_regression.py`](file:///C:/GitHub_public/Abschlussprojekt/src/next_exam_regression.py)
- Variante A: Full-History GRU + Dual Head.
- 5-Mode Grid: `standard`, `gradeblind`, `blind`, `oracle`, `realistic`.
- Counterfactual Noten-Delta (ATE).

#### [NEW] [`src/next_exam_transformer.py`](file:///C:/GitHub_public/Abschlussprojekt/src/next_exam_transformer.py)
- Variante B: Sliding-Window Transformer ($w = 8$) + Summary-Features + Dual Head.

---

## ARBEITSPAKET 7: Strukturelle Mediationsanalyse (Imai/Pearl)

### Formales Framework

$$\text{Total Effect (TE)} = \underbrace{E[Y_i(1, M_i(1)) - Y_i(1, M_i(0))]}_{\text{ACME (indirekt über Noten)}} + \underbrace{E[Y_i(1, M_i(0)) - Y_i(0, M_i(0))]}_{\text{ADE (direkt: Motivationsboost)}}$$

- **ACME:** Der Anteil des Support-Effekts, der über Notenverbesserung ($\Delta\text{Note} = -0{,}09$) zum Dropout-Schutz führt.
- **ADE:** Der direkte Effekt (Motivation $\uparrow$, Soziale Integration $\uparrow$), der unabhängig von Noten wirkt.

### Proposed Changes

#### [NEW] [`src/mediation_analysis.py`](file:///C:/GitHub_public/Abschlussprojekt/src/mediation_analysis.py)
- Mediator-Modell: OLS Note auf Treatment + Kovariaten.
- Outcome-Modell: Logit Dropout auf Treatment + Mediator + Kovariaten.
- Bootstrap ACME/ADE ($B = 1000$).
- Validierung gegen DGP Ground Truth.

---

## Open Questions

> [!IMPORTANT]
> **Feature-Builder-Migration:** Sollen alle 26+ Standalone-Skripte auf `feature_builder.py` migriert werden, bevor der Nachtlauf läuft? Das wäre ein erheblicher Refactoring-Aufwand (geschätzt 2–3 Stunden Implementierungszeit), würde aber danach die vollständige 5-Modi-Evaluation für **alle** Modelle ermöglichen. Alternative: Nur die Grid-Runner-Abdeckung schrittweise erweitern (z.B. Dynamic DeepHit Delta und DML als nächstes).

> [!IMPORTANT]
> **Backlog-Projekte (explizit zurückgestellt):**
> - Dashboard-Reparatur (`dashboard_educational.py`)
> - PyTorch/PyCox Refaktor
> - Marginal Structural Models (MSM)
> - DGP-Sensitivitätsanalyse (kontinuierliches Parameter-Grid)
> - DuckDB-Produktiv-Migration (Backend-Swap in `feature_builder.py`)

---

## Verification Plan

### Automated Tests
```powershell
# Vollständiger V4-Nachtlauf
python -u src/run_overnight.py --seed 99999 --output-dir output_dl_v4 --verbose

# Oder sequenziell (zum Debugging):
python -u src/simulation_v3.py          # V4-Seed
python -u src/run_retrain_all.py        # 27 Schritte
python -u src/run_feature_grid_experiments.py
python -u src/next_exam_regression.py
python -u src/mediation_analysis.py
python -u src/benchmark_backbone_sanity_check.py
python -u src/compare_v3_v4.py
```

### Verifikationskriterien
1. **Clipping-Diagnostik:** Report zeigt, welche Caps häufig greifen (Erwartung: Support-Boost-Cap bei >5 % der behandelten Prüfungen).
2. **V4 Robustheit:** Makro-Dropout-Raten weichen um $< 2\sigma$ von V3 ab.
3. **Backbone-Check:** DuckDB und Pandas liefern identische Tensoren (`np.allclose` = True).
4. **Next-Exam:** $R^2 \in [0{,}50, 0{,}78]$ für Notenvorhersage, $\text{ROC-AUC} > 0{,}75$ für Fail-Prediction.
5. **Mediation:** ACME > 0 mit 95 %-Bootstrap-KI.
6. **Pipeline-Benchmark:** Alle 27+ Schritte terminieren fehlerfrei, Gesamtdauer < 5 Stunden.
