# Changelog

Alle nennenswerten Änderungen an diesem Projekt werden in dieser Datei dokumentiert.
Das Format basiert auf [Keep a Changelog](https://keepachangelog.com/de/1.0.0/).
## [Unreleased / V4 Master Refactoring] - 2026-09-02
### Added
- **`deepsupport/` Package:** Vollständig modulare Architektur. Alle Keras/Scikit-Modelle wurden isoliert (`src/deepsupport/models/`).
- **Annotation Tracking Pattern:** Einführung von `docs/07_conversation_logs/` zur transparenten Aufzeichnung von Design-Entscheidungen und User-Prompts.
- **Git LFS im Archiv:** Das `archive/` Subrepo ist jetzt vollständig auf Git LFS konfiguriert, um große CSV/JSONs effizient zu verwalten.

### Changed
- **Orchestrierung (`grid_runner.py`):** Massiv gehärtet. Der Runner iteriert nun fehlerfrei über die Cross-Szenarien (`S01` bis `S15`) und trennt I/O streng (`data_root` vs `output_root`), sodass Ground-Truth-Daten nie wieder überschrieben werden.
- **Metrics Logger:** Keine `0.0`-Imputation mehr! Fehlende Werte werden strikt als `null` getrackt, um fehlerhafte Aggregationen zu unterbinden.
- **Repository Struktur:** 129 irrelevante Skripte wurden in `legacy_code/` historisiert. Das Projekt-Root-Verzeichnis ist nun 100% sauber.

### Fixed
- Alle 16 Causal Inference-Skripte wurden an die neuen Modulpfade angepasst (Regex-Massenupsdate der `.keras` Imports).
- Beseitigung diverser harter Pfade (z.B. `src/output_dl` Fallbacks im `feature_builder.py`).



## [V4.1.1 Quality Fixes] — 2026-08-30

### 1. Sample Leakage Fix (5 Skripte)
- `autoregressive_next_exam.py`: Split geändert von Zeilenebene auf Studentenebene (group-consistent)
- `autoregressive_deep_transformer.py`: Selber Fix
- `eval_autoregressive_fail.py`: Selber Fix
- `run_transfer_learning.py`: Selber Fix  
- `counterfactual_deepsurv.py`: Selber Fix
- Alle verwenden `random_state=42`, 70/15/15 Split auf `studierenden_id.unique()`

### 2. Future Leakage Fix in Feature Builder
- `feature_builder.py`: `cp_rueckstand` in Exam Tensor und Exam Panel verwendet nun `cp_cum_prev` (shifted) statt `cp_cum` (inklusive aktueller Prüfung)
- `temporal='cum'` Modus nutzt inklusiven cumsum als dokumentierte Design-Entscheidung

### 3. Oracle Feature Erweiterung
- `feature_builder.py`: `hidden_overload` und `hidden_zeit_puffer` als Oracle-Features in ALLEN 5 Build-Funktionen hinzugefügt
- `build_exam_panel_df()`: Oracle-Modus fehlte komplett, jetzt implementiert
- Oracle-Modus hat nun 5 Hidden-Features statt 3:
  - hidden_motivation, hidden_soziale_integration, hidden_erwartete_note (bestehend)
  - hidden_overload (NEU: Workload in Stunden pro Prüfung)
  - hidden_zeit_puffer (NEU: statisch pro Student, individueller Zeitpuffer)

## [V4.1 Sensitivity Grid] — 2026-08-30 — Vollständiger Sensitivitäts-Gitterlauf

### Abgeschlossen
- **Vollständiges Sensitivitäts-Grid:** 15 Szenarien × 8 Universen = 120 Simulationsläufe bei N=50.000
- **Gesamtlaufzeit:** 14,6 Stunden (52.722 Sekunden) mit `ProcessPoolExecutor` und 5 Workern
- **Szenarien** umfassen 6 Parameterdimensionen: Support-Wirkung (S02/S03), Notenboost (S04/S05/S06), Rauschen (S07/S08), Zeitkosten (S09/S10), RCT-Selektion (S11), Overload-Penalty (S12/S13/S14), plus Kombi-Szenario (S15)

### Kernergebnisse
- **Baseline (S01):** Dropout A=29,2%, B=37,1%, ARR=7,9pp, NNT=12,6
- **Perfekte RNG-Synchronisierung:** B=37,1% über alle nicht-globalen Szenarien hinweg
- **Sensitivitäts-Ranking:** Overload-Penalty (8,5pp Spanne) > Support-Wirkung (7,3pp) > Rauschen (6,4pp) > Selektion (3,5pp) > Notenboost (3,0pp) > Zeitkosten (1,2pp)
- **ARR robust** gegenüber Overload-Kalibrierung (7,3–8,4pp)
- **S15 (Kombi):** Kostenverdopplung kostet nur 0,5pp bei gleichzeitiger Wirkungsverdopplung

### Erstellte Analyseskripte
- `analyze_nachtlauf_v41.py` — Synoptische Übersicht und szenarienübergreifender Vergleich
- `paradox_analysis_v41.py` — Paradoxe Statuswechsel-Analyse
- `paradox_s02s03_detail.py` — Detaillierte S02/S03-Divergenzanalyse


## [V4.1] — 2026-08-29/30 — RNG-Synchronisierung & Feature-Restaurierung

### Hintergrund
Beim V4-Refactoring (Performance-Optimierung) gingen mehrere V3-Features
verloren, die für die kausale Interpretierbarkeit der Multiuniversen-Simulation
kritisch waren. V4.1 restauriert alle verlorenen Features.

### Restauriert
- **Per-Student-Seeds (4+1 separate RNG-Streams):** `rng_support`, `rng_social`,
  `rng_dropout`, `rng_anomalie` via `zlib.crc32(studierenden_id) ^ population_seed`.
  Vorher: ein einziger globaler `rng`-Stream → 26% RNG-Artefakt-Statusdifferenzen.
  Nachher: 9,2% Statusdifferenzen (rein kausaler Support-Effekt, 90,8% identisch).
- **Deterministisches Prüfungsrauschen:** `get_exam_noise(base_seed, modul_id, versuch)`
  als reine Funktion statt sequentiellem `rng.normal()`.
- **Pad-Draws für blockierte Angebote:** Iteration über ALLE 12 Angebote in jedem
  Universum; `rng_support.random()` wird immer gezogen, blockierte Angebote nach
  dem Draw ignoriert.
- **Carry-over fachlicher Support-Boost:** 2/3 Nachwirkung aus Vorsemestern.
- **Soziale Integration Drift:** `rng_social.normal(0, 0.05)` statt `rng.beta()`.

### Geändert
- **Overload-Penalty Cap:** Nicht mehr Default, sondern konfigurierbar via
  `overload_penalty_cap` (z.B. `0.15` in Szenario S14). Default: kein Cap.
- **`population_seed` als expliziter Parameter:** `simuliere_verlaeufe()` akzeptiert
  jetzt `population_seed` als Funktionsparameter (wie V3), statt ihn aus der Config
  zu lesen. Runner aktualisiert.
- **Support-Kosten als Faktor:** `support_kosten_faktor` (Default: 1.0) multipliziert
  die individuellen Angebotskosten, statt alle auf denselben Wert zu setzen
  (`support_kosten_override` entfernt).
- **Probabilistischer Modulabwurf:** Statt deterministischem Schwellwert steigt die
  Abwurf-Wahrscheinlichkeit sigmoid mit dem Überschuss über dem individuellen
  Zeitpuffer: `p = ueberschuss / (ueberschuss + 50)`. Eigener RNG-Stream
  `rng_workload` (Stream +5) → keine Beeinflussung anderer Streams.
- **Individueller Zeitpuffer restauriert:** `hidden_zeit_puffer` wieder als
  `Beta(μ=0.33, κ=8) × 180h` pro Student generiert (mean≈60h, std≈26h).
  Ersetzt den fixen +150h-Schwellwert.
- **PruefungsErgebnis Hidden Fields restauriert:** Alle 7 Hidden Fields wieder
  gefüllt (`hidden_overload`, `hidden_zeit_puffer`, `hidden_penalty_capped`,
  `hidden_support_capped`).
- **Butterfly-Effekt Designnotiz:** Kommentar an `get_exam_noise` dokumentiert die
  bewusste Entscheidung zur RNG-Synchronisierung zwischen Universen.

### Hinzugefügt
- **Kalibrierter RCT-Modus:** `rct_support_uptake=True` verwendet jetzt per-Typ
  kalibrierte Raten (fachlich: 0.042, überfachlich: 0.025, psychosozial: 0.023)
  statt pauschal p=0.20, um das Baseline-Teilnahmevolumen beizubehalten.
- **Konfigurierbarer Overload-Penalty-Faktor:** `cfg["overload_penalty_factor"]`
  (Default: 0.1).
- **S14 Overload-Cap-Szenario:** `overload_penalty_cap: 0.15` als Grid-Variante.
- **S15 Kombi-Szenario:** Kosten UND Wirkung verdoppelt (faktor 2 + mult 10.0).

### Nicht geändert
- V4-Features (Precompute, Zeitkonto, Super-Klausur-Boost, dynamische
  erwartete Note, Modulabwurf) bleiben unverändert.
- `simuliere_pruefung`- und `berechne_dropout`-Formeln bleiben identisch.


## [3.6.0] - 2026-08-24

### Hinzugefügt / Erledigt
- **AP0 (3-Way-Backbone & Feature-Factory):** `src/aggregate.py` unterstützt 3 austauschbare Backends (`duckdb`, `numpy`, `pandas`), `cp_attempted`-Spalte und optimierten Multi-Column Student-Join. `src/feature_builder.py` mit Vektorisierung in NumPy (34x Speedup), temporalem Switch (`temporal='prev'|'cum'`), `build_exam_panel_df`, Competing-Risks Dual-Target und flexiblen Landmark-Targets implementiert.
- **AP1 (Feature-Builder-Migration & Skript-Konsolidierung):** Alle 8 Modellklassen auf `src/feature_builder.py` umgestellt. Redundante Skripte (`*_delta.py`, `*_v2.py`) durch transparente, abwärtskompatible Wrapper ersetzt. Automatisierte Smoke-Test-Suite `src/verify_feature_migration.py` erstellt (10/10 Tests PASSED).
- **AP2 & AP4 (Master-Orchestrierung & psutil-Benchmarks):** `src/run_overnight.py` als einheitlicher V3.6 Master-Runner mit `PipelineBenchmarkTracker` (RAM-Delta & CPU-Messung pro Schritt) und automatischer HTML/Markdown-Berichterstellung implementiert.
- **AP3 (Verbose-Modus & Clipping-Diagnostik):** `ClippingTracker` in `src/simulation_v3.py` integriert; protokolliert Capping von Motivation, Integration, Overload-Penalty (Deckelung bei 0.15) und Support-Boost in `output_dl/diagnostics/clipping_report.json`.
- **AP5 (Backbone Sanity Check & Benchmark):** `src/benchmark_backbone_sanity_check.py` ausgeführt. Bit-identische Äquivalenz (0.0 Diff) aller 7 Support-/CP-Merkmale über 812.143 Zeilen bewiesen; DuckDB liefert 1.92x Speedup.
- **AP7 (Autoregressive Next-Exam-Vorhersage):** `src/autoregressive_next_exam.py` mit Dual-Head Multi-Task Architektur (GRU-Encoder + Late-Fusion) implementiert. Erreicht ROC-AUC = 0.9202 für Prüfungsbestehen und $R^2 = 0.4618$ für Noten-Regression auf 114k Test-Prüfungen.
- **AP8 (Strukturelle Mediationsanalyse):** `src/structural_mediation_analysis.py` implementiert (Imai/Pearl Framework). Zerlegt Support-Effekte in direkte (ADE) und vermittelte Leistungs-Pfade (ACME).
- **AP9 (Dokumentation & Changelog):** Vollständige Aktualisierung von `CHANGELOG.md`, `walkthrough.md` und `variablen_kausalitaet_und_temporalitaet.md`.

---

## [3.5.0] - 2026-08-23

### Hinzugefügt
- `src/feature_builder.py`: Zentrale Feature-Factory mit 5 Modi (`standard`, `gradeblind`, `blind`, `oracle`, `realistic`).
- `src/run_feature_grid_experiments.py`: Grid-Runner für 4 Modellklassen über alle 5 Modi.
- Theoretische Vorhersagbarkeits-Schranke: $R^2_{\max} = 0.7816$, Bayes-Risiko $= 0.0348$, $\text{AUC}^* = 0.8974$.
- `Artifacts/projekt_evolution_und_methodenvergleich.md`: Dokumentation der Projektgeschichte über alle Phasen.
- `Artifacts/project_index.md` & `Artifacts/dokumentation_der_dokumentation.md`.

### Behoben
- Oracle-Feature-Bug: Hardcodierte `0.5`-Werte in `aggregate.py`, `feature_builder.py` und `extended_cox_delta.py` korrigiert; dynamische $t-1$ Latenzen wiederhergestellt.
- Mermaid-Diagramm-Rendering in `projekt_evolution_und_methodenvergleich.md` repariert.

---

## [3.3.0] - 2026-08-22

### Hinzugefügt
- Universen F, G, H zur Isolation von Confounder-Strukturen (`simulate_universes_fgh.py`).
- Cross-Modal Causal Transformer-DML Pipeline (`train_transformer_dml.py`).
- Erweiterte Survival-Delta-Modelle (`extended_cox_delta.py`, `extended_deep_survival_delta.py`, `recurrent_exam_survival_delta.py`).
- 27-stufige Orchestrierung in `run_retrain_all.py`.

---

## [3.0.0] - 2026-08-20

### Hinzugefügt
- Simulation V3 mit stochastischem Zeitbudget, Überlastungsmechanismus und 5 Paralleluniversen A–E (`simulation_v3.py`).
- Counterfactual Ground Truth Berechnung (`oracle_lift.py`, `compute_macro_effects.py`).

### 2026-09-02 (Submodules, Synthesis & Grid Run)
- **Portfolio Architecture**: Eingliederung der Legacy-Projekte (\DataAnalysis\, \DataEngineering\, \DeepLearning\) als Read-Only Git Submodules im Hauptprojekt.
- **LFS-Archiv**: Migration historischer Modell- und Simulationsausgaben in ein separates LFS-Archiv (\rchive/\).
- **Methodische Synthese**: Erstellung einer finalen Übersicht über die intellektuelle Projektentwicklung (Dropout-Paradoxon $\rightarrow$ Causal Panels $\rightarrow$ Masked Sequence Models).
- **Grid Run**: Automatisierter Cross-Scenario Grid Run (S02-S15) inkl. automatischer DuckDB-Aggregation angestoßen.
