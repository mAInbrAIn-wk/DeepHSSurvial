# Versionsvergleich: V3.6 → V4.1

## RNG-Sync Validierung (N=5.000, seed=99999)

| Metrik | Vorher (V4.0, kaputt) | **Nachher (V4.1)** |
| :--- | :---: | :---: |
| Gleicher Endstatus A↔B | 68,5% | **90,8%** ✅ |
| Prüfungsnoten 1. Semester Match | ❌ (ab Prüfung 2) | **100% Match** ✅ |
| Dropout A (Full Support) | 27,8% | **27,3%** |
| Dropout B (Kein Support) | 33,9% | **35,5%** |
| Netto-Schutzeffekt | Kontaminiert | **409 Studis (8,2pp)** |

---

## 1. Studentengenerierung

| Attribut | V3.6 | V4.1 | Auswirkung |
| :--- | :--- | :--- | :--- |
| **Alter** | `Normal(20.5, 2.8)` clipped $[17, 45]$ | `Beta(μ=0.125, κ=12.8)` skaliert $[17, 45]$ | Keine Clip-Artefakte an Rändern |
| **HZB-Note** | `Normal(2.4, 0.55)` clipped $[1.0, 4.0]$ | `Beta(μ=0.466, κ=6.5)` skaliert $[1.0, 4.0]$ | Natürlich begrenzt, keine Häufung an 1.0/4.0 |
| **Motivation** | `mean + Normal(0, rauschen)` clipped $[0.05, 1.0]$ | `Beta(mean·20, (1-mean)·20)` | Kein `gewicht_motivation_rauschen` nötig |
| **Soz. Integration** | `mean + Normal(0, rauschen)` clipped $[0.05, 1.0]$ | `Beta(mean·20, (1-mean)·20)` | Kein `gewicht_integration_rauschen` nötig |
| **hidden_zeit_puffer** | `Normal(60, 30)` clipped $[0, 180]$, pro Student | **Nicht vorhanden** (fester Schwellwert +150h) | ⚠️ Individueller Puffer fehlt |
| Rest (Geschlecht, Erwerb, Migration, ...) | identisch | identisch | — |

> [!NOTE]
> Der fehlende `hidden_zeit_puffer` ist eine bewusste V4-Designentscheidung: Statt individuellem
> Puffer wird ein fixer Schwellwert von 150h verwendet. Der Effekt ist ähnlich (Median des V3-Puffers
> war ~60h, aber mit dem fixen 150h-Wert sind Modulabwürfe seltener).

---

## 2. Support-Nutzungslogik

| Mechanismus | V3.6 | V4.1 | Auswirkung |
| :--- | :--- | :--- | :--- |
| **Überfachlich P(Nutzung)** | Linear: $0.05 + (0.5 - \text{mot}) \cdot 0.15$ | **+Dampening**: wenn $\text{mot} < 0.2$: $p \cdot \frac{\text{mot}}{0.2}$ | V4.1: Stark demotivierte suchen weniger Support |
| **Psychosozial P(Nutzung)** | Linear: $0.01 + (0.5 - \text{soz}) \cdot 0.12$ | **+Dampening**: wenn $\text{soz} < 0.2$: $p \cdot \frac{\text{soz}}{0.2}$ | V4.1: Stark isolierte suchen weniger Support |
| **RCT-Modus** | Nicht vorhanden | Kalibrierte Raten (f:0.042, ü:0.025, p:0.023) | V4.1: Experimenteller Uptake-Modus |
| **Kostenüberschreibung** | Nicht vorhanden | `support_kosten_override` in cfg | V4.1: Zeitkosten-Szenarien möglich |
| **Pad-Draws** | ✅ vorhanden | ✅ restauriert | Identisch |
| **Stochast. Zeitcheck** | `rng_support.random() < 0.2` | identisch | Identisch |
| **Carry-over** | 2/3 Nachwirkung | ✅ restauriert | Identisch |

> [!IMPORTANT]
> Das **Low-Motivation-Dampening** ist ein neues V4-Feature (nicht in V3). Es verhindert, dass
> stark demotivierte/isolierte Studis Support suchen — realistischer, aber potenziell Performance-relevant.

---

## 3. Modulauswahl und Workload

| Mechanismus | V3.6 | V4.1 | Auswirkung |
| :--- | :--- | :--- | :--- |
| **Drop-Bedingung** | `workload > zeit + studi.hidden_zeit_puffer` | `workload + support_zeit > zeit + 150` | V4.1 zählt Support-Zeit mit! |
| **Puffer** | Pro-Student: `Normal(60, 30)` | Fix: 150h | V4.1 hat höheren, uniformen Puffer |
| **Drop-Zähler** | Keine | `stat_modules_dropped` pro Student + Tracker | V4.1 zählt Abwürfe |

---

## 4. Overload-Penalty

| Mechanismus | V3.6 | V4.1 (jetzt) |
| :--- | :--- | :--- |
| **Formel** | `min(0.15, (overload/100) × 0.1)` | `(overload/100) × factor` |
| **Cap** | 0.15 (hart) | **Kein Cap** (per Nutzer-Entscheidung entfernt) |
| **Factor** | Fest 0.1 | Konfigurierbar: `overload_penalty_factor` |

---

## 5. Prüfungsrauschen und Noten

| Mechanismus | V3.6 | V4.1 |
| :--- | :--- | :--- |
| **Prüfungsrauschen** | `get_exam_noise(seed, modul, versuch)` deterministisch | ✅ identisch restauriert |
| **Pruefung-Funktion** | Import aus `simulation_v2.py` | Lokal definiert (identische Formel) |
| **Hidden Fields in PruefungsErgebnis** | 7 (inkl. overload, zeit_puffer, penalty_capped, support_capped) | 3 (nur motivation, soz_int, erwartete_note) |
| **Carry-over Boost** | 2/3 aus Vorsemestern | ✅ identisch restauriert |

---

## 6. RNG-Architektur

| Stream | V3.6 | V4.1 |
| :--- | :--- | :--- |
| `base_seed` | `crc32(id) ^ population_seed` | ✅ identisch |
| `rng_support` | Stream +1 | ✅ identisch |
| `rng_social` | Stream +2 | ✅ identisch |
| `rng_dropout` | Stream +3 | ✅ identisch |
| `rng_anomalie` | **`rng_init` (Stream 0)** — geteilt mit Anomalie | **Stream +4** (eigener, isolierter Stream) |

> V4.1 hat eine **sauberere** RNG-Isolation: Anomalie-Entscheidungen haben einen eigenen Stream
> statt sich den Base-Stream mit `rng_init` zu teilen.

---

## 7. Nur in V4.1 (neue Features)

| Feature | Beschreibung |
| :--- | :--- |
| **Config Dependency Injection** | Alle Funktionen akzeptieren `cfg: Dict` |
| **Beta-Verteilungen** | Natürlich begrenzte Verteilungen statt Clip-Normalverteilungen |
| **Apathie-Dampening** | Support-Suche sinkt bei sehr niedriger Motivation/Integration |
| **RCT-Modus** | Experimentelle Support-Zuweisung |
| **Kostenüberschreibung** | Zeitkosten pro Szenario variierbar |
| **Overload-Factor** | Konfigurierbare Penalty-Steigung |
| **Tracking-Metriken** | Modulabwürfe, Overload-Hits pro Student und global |
| **Precompute-Caches** | Pflicht-/BA-Module vorberechnet |
| **Vorbereitete Demotivation** | Kommentierter Code für Notenentäuschungs-Penalty |

---

## 8. Noch fehlend in V4.1 (aus V3.6)

| Feature | Status | Empfehlung |
| :--- | :--- | :--- |
| `hidden_zeit_puffer` (pro Student) | ❌ nicht übertragen | Optional: Könnte als Szenario-Variante getestet werden |
| 7 Hidden Fields in PruefungsErgebnis | ❌ nur 3 übertragen | Optional: `hidden_overload`, `hidden_support_capped` nachrüsten |
| Built-in 8-Uni-Orchestrator | ❌ extern in `run_v4_universes.py` | Bewusste Trennung, ok |
