# Musteranalyse V4.1 — Reproduzierbare Analyseanleitung

> [!IMPORTANT]
> Dieses Dokument beschreibt die vollständige Analyse-Pipeline für den V4.1-Simulationsdatensatz.
> Alle Skripte sind als kopierbare Python-Snippets formuliert und direkt ausführbar.
> **Datensatz:** 15 Szenarien × 8 Universen = 120 Runs, N=50.000 pro Universum, seed=99999.

---

## 1. Datensatzübersicht

### 1.1 Verzeichnisstruktur

```
src/output_v4_grid_v41/
├── S01_baseline/
│   ├── universe_A/          # Alle Support-Typen erlaubt
│   │   ├── studierende.csv
│   │   ├── einschreibungen.csv
│   │   ├── pruefungen.csv
│   │   ├── support_teilnahmen.csv
│   │   ├── abschluesse.csv
│   │   ├── semester.csv         # statisch, identisch über alle Universen
│   │   ├── module.csv           # statisch
│   │   ├── studiengaenge.csv    # statisch
│   │   ├── modul_studiengang.csv
│   │   ├── support_angebote.csv # variiert je nach Universum (blockierte Typen)
│   │   └── support_modul_zuordnung.csv
│   ├── universe_B/          # Kein Support (komplett blockiert)
│   ├── universe_C/          # Kein fachlicher Support
│   ├── universe_D/          # Kein überfachlicher Support
│   ├── universe_E/          # Kein psychosozialer Support
│   ├── universe_F/          # Nur fachlicher Support
│   ├── universe_G/          # Nur überfachlicher Support
│   ├── universe_H/          # Nur psychosozialer Support
│   └── metrics/
│       └── true_macro_effects.json
├── S02_supp_half/
│   ├── universe_A/ ... universe_H/
│   └── metrics/
├── ...
├── S15_cost_effect_double/
│   ├── universe_A/ ... universe_H/
│   └── metrics/
└── metrics/
    └── full_sensitivity_grid_results.json   # Aggregierte Gesamtergebnisse
```

### 1.2 Universum-Design

| Key | Label | fachlich | überfachlich | psychosozial |
| :---: | :--- | :---: | :---: | :---: |
| **A** | Alle Support-Typen erlaubt | ✅ | ✅ | ✅ |
| **B** | Kein Support (komplett blockiert) | ❌ | ❌ | ❌ |
| **C** | Kein fachlicher Support | ❌ | ✅ | ✅ |
| **D** | Kein überfachlicher Support | ✅ | ❌ | ✅ |
| **E** | Kein psychosozialer Support | ✅ | ✅ | ❌ |
| **F** | Nur fachlicher Support | ✅ | ❌ | ❌ |
| **G** | Nur überfachlicher Support | ❌ | ✅ | ❌ |
| **H** | Nur psychosozialer Support | ❌ | ❌ | ✅ |

### 1.3 CSV-Dateien pro Universum

| Datei | Beschreibung | Schlüsselspalten | Ungefähre Größe |
| :--- | :--- | :--- | ---: |
| `studierende.csv` | Studierendenprofile (N=50.000) | `studierenden_id`, `studiengang_id`, `hzb_note`, `erwerbstaetigkeit_std`, `motivation`, `soziale_integration` | ~5 MB |
| `einschreibungen.csv` | Semestereinschreibungen | `studierenden_id`, `semester_id`, `fachsemester`, `status` | ~9 MB |
| `pruefungen.csv` | Prüfungsergebnisse | `studierenden_id`, `semester_id`, `modul_id`, `versuch`, `note`, `bestanden`, `note_counterfactual`, `support_genutzt` | ~70 MB |
| `support_teilnahmen.csv` | Support-Nutzungen pro Semester | `studierenden_id`, `semester_id`, `angebot_id` | ~3 MB |
| `abschluesse.csv` | Endstatus pro Student | `studierenden_id`, `status` | ~2 MB |
| `semester.csv` | Semesterliste (statisch) | `semester_id`, `semester_nr`, `typ`, `jahr` | <1 KB |
| `module.csv` | Modulkatalog (statisch) | `modul_id`, `name`, `cp`, `schwierigkeit`, `workload_h` | ~4 KB |
| `studiengaenge.csv` | Studiengänge (statisch) | `studiengang_id`, `name`, `regelstudienzeit`, `cp_gesamt` | <1 KB |
| `support_angebote.csv` | Support-Angebote (universumsabhängig) | `angebot_id`, `typ`, `kosten_h` | <1 KB |
| `modul_studiengang.csv` | Modul-Studiengang-Zuordnung | `modul_id`, `studiengang_id`, `empfohlenes_fachsemester`, `pflicht` | ~2 KB |
| `support_modul_zuordnung.csv` | Support-Modul-Wirkungen | `angebot_id`, `modul_id`, `wirkungsstaerke` | <1 KB |

### 1.4 Statuswerte in `abschluesse.csv`

| Status | Bedeutung | Mechanismus |
| :--- | :--- | :--- |
| `abgeschlossen` | Studium erfolgreich abgeschlossen | Alle Pflichtmodule bestanden + BA |
| `abgebrochen` | Freiwilliger Abbruch | Dropout-Wahrscheinlichkeit überschritten |
| `exmatrikuliert` | Zwangsexmatrikulation | 3× durchgefallen in einem Pflichtmodul |
| `zeitueberschreitung` | Max. Semester erreicht (16) | Zeitlimit ohne Abschluss |

### 1.5 Simulationsparameter

- **Population Seed:** `99999`
- **N pro Universum:** `50.000`
- **Simulationsengine:** `simulation_v4.py` (V4.1 mit deterministischem Prüfungsrauschen)
- **Grid-Runner:** `run_v4_simulation_grid.py`

---

### 1.6 Feature Builder & Datensätze (V4.1.1 Update)

Die Extraktion der Rohdaten in analysefertige Matrizen erfolgt über `src/feature_builder.py`. Seit dem V4.1.1 Update sind alle **Sample Leakages** (Student-Level Split) und **Future Leakages** (`cp_rueckstand` Fix) behoben. 
Die Oracle-Features umfassen nun 5 statt 3 Felder (neu: `hidden_overload`, `hidden_zeit_puffer`).

Feature-Anzahlen nach Format:

| Format | standard | oracle |
|:---|:---:|:---:|
| Semester Tensor | 18 | 23 |
| Exam Tensor | 24 | 29 |
| Semester Panel | 16 | 21 |
| Exam Panel | 23 | 28 |
| Landmark | 16 | 21 |

---

## 2. Szenario-Beschreibungen

### 2.1 Baseline-Parameter (S01)

| Parameter | Baseline-Wert | Beschreibung |
| :--- | ---: | :--- |
| `support_effect_multiplier` | 5.0 | Multiplikator für Motivation/Integration-Boost durch Support |
| `gewicht_support_boost` | 0.08 | Gewicht des fachlichen Notenboosts |
| `gewicht_rauschen` | 0.18 | Standardabweichung des Prüfungsrauschens |
| `overload_penalty_factor` | 0.1 | Strafterm für Workload-Überbelastung |
| `support_kosten_faktor` | 1.0 | Multiplikator für Support-Zeitkosten |
| `rct_support_uptake` | `False` | Reaktive (bedarfsgesteuerte) Support-Nutzung |
| `overload_penalty_cap` | `None` | Kein Cap auf den Overload-Strafterm |

### 2.2 Alle 15 Szenarien

| ID | Verzeichnis | Dimension | Beschreibung | Override-Parameter |
| :--- | :--- | :--- | :--- | :--- |
| **S01** | `S01_baseline` | Baseline | Referenzszenario mit Standardparametern | *keine — alle Defaults* |
| **S02** | `S02_supp_half` | Support-Wirkung | Support-Wirkung halbiert (0,5× Baseline) | `support_effect_multiplier=2.5` |
| **S03** | `S03_supp_double` | Support-Wirkung | Support-Wirkung verdoppelt (2× Baseline) | `support_effect_multiplier=10.0` |
| **S04** | `S04_grade_half` | Notenboost | Notenboost halbiert | `gewicht_support_boost=0.04` |
| **S05** | `S05_grade_double` | Notenboost | Notenboost verdoppelt | `gewicht_support_boost=0.16` |
| **S06** | `S06_grade_quad` | Notenboost | Notenboost vervierfacht | `gewicht_support_boost=0.32` |
| **S07** | `S07_noise_half` | Rauschen | Prüfungsrauschen halbiert | `gewicht_rauschen=0.09` |
| **S08** | `S08_noise_double` | Rauschen | Prüfungsrauschen verdoppelt | `gewicht_rauschen=0.36` |
| **S09** | `S09_cost_zero` | Zeitkosten | Support kostenlos (keine Zeitkosten) | `support_kosten_faktor=0.0` |
| **S10** | `S10_cost_double` | Zeitkosten | Support-Kosten verdoppelt | `support_kosten_faktor=2.0` |
| **S11** | `S11_rct_calibrated` | Selektion | RCT kalibriert (zufällige Zuordnung) | `rct_support_uptake=True` |
| **S12** | `S12_overload_half` | Overload-Penalty | Overload-Penalty halbiert | `overload_penalty_factor=0.05` |
| **S13** | `S13_overload_double` | Overload-Penalty | Overload-Penalty verdoppelt | `overload_penalty_factor=0.2` |
| **S14** | `S14_overload_cap` | Overload-Penalty | Overload mit Cap (wie V3.6) | `overload_penalty_cap=0.15` |
| **S15** | `S15_cost_effect_double` | Kombi | Kosten UND Wirkung verdoppelt | `support_kosten_faktor=2.0`, `support_effect_multiplier=10.0` |

> [!NOTE]
> **S01 (Baseline)** wird im Grid-Runner separat behandelt — die Parameter entsprechen
> den Standardwerten in `config.py`. S02–S15 überschreiben jeweils nur die genannten Parameter.

---

## 3. Reproduzierbare Analyse-Skripte

### 3.1 Voraussetzungen

```bash
# Alle Skripte setzen voraus:
pip install pandas numpy
# Arbeitsverzeichnis: Projekt-Root (c:\GitHub_public\Abschlussprojekt)
```

```python
# Gemeinsame Imports und Pfade
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")

SCENARIOS = [
    "S01_baseline", "S02_supp_half", "S03_supp_double",
    "S04_grade_half", "S05_grade_double", "S06_grade_quad",
    "S07_noise_half", "S08_noise_double",
    "S09_cost_zero", "S10_cost_double",
    "S11_rct_calibrated",
    "S12_overload_half", "S13_overload_double", "S14_overload_cap",
    "S15_cost_effect_double"
]

UNIVERSES = ["A", "B", "C", "D", "E", "F", "G", "H"]
```

---

### 3.2 Analyse (a): Synoptische Übersicht — Dropout-Raten, ARR, NNT

**Ziel:** Berechnet die Dropout-Raten für alle 15×8 Kombinationen, die Absolute Risk
Reduction (ARR = B−A) und die Number Needed to Treat (NNT = 100/ARR).

**Skript:** `src/analyse_synoptische_uebersicht_v41.py` (inline — kopieren und ausführen)

```python
"""
Synoptische Übersicht V4.1 — Dropout-Raten, ARR, NNT
=====================================================
Liest abschluesse.csv aus jedem Szenario/Universum und berechnet Dropout-Raten.

Ausführung:
    cd c:\GitHub_public\Abschlussprojekt
    python src/analyse_synoptische_uebersicht_v41.py
"""
import pandas as pd
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")

SCENARIOS = [
    "S01_baseline", "S02_supp_half", "S03_supp_double",
    "S04_grade_half", "S05_grade_double", "S06_grade_quad",
    "S07_noise_half", "S08_noise_double",
    "S09_cost_zero", "S10_cost_double",
    "S11_rct_calibrated",
    "S12_overload_half", "S13_overload_double", "S14_overload_cap",
    "S15_cost_effect_double"
]
UNIVERSES = ["A", "B", "C", "D", "E", "F", "G", "H"]
DROPOUT_STATI = {"abgebrochen", "exmatrikuliert", "zeitueberschreitung"}

rows = []
for sc in SCENARIOS:
    row = {"Szenario": sc}
    for uni in UNIVERSES:
        csv_path = BASE / sc / f"universe_{uni}" / "abschluesse.csv"
        if not csv_path.exists():
            row[uni] = None
            continue
        df = pd.read_csv(csv_path)
        n = len(df)
        n_drop = df["status"].isin(DROPOUT_STATI).sum()
        row[uni] = round(n_drop / n * 100, 1) if n > 0 else None
    
    # ARR = B − A, NNT = 100 / ARR
    if row.get("A") is not None and row.get("B") is not None:
        arr = round(row["B"] - row["A"], 1)
        row["ARR_pp"] = arr
        row["NNT"] = round(100 / arr, 1) if arr > 0 else float("inf")
    
    rows.append(row)

result = pd.DataFrame(rows)
print("\n=== SYNOPTISCHE ÜBERSICHT V4.1 ===\n")
print(result.to_string(index=False))

# Optional: Als CSV speichern
result.to_csv(BASE / "metrics" / "synoptische_uebersicht.csv", index=False)
print(f"\nGespeichert: {BASE / 'metrics' / 'synoptische_uebersicht.csv'}")
```

**Erwartete Ausgabe:**

| Szenario | A | B | ARR | NNT |
| :--- | ---: | ---: | ---: | ---: |
| S01_baseline | 29,2% | 37,1% | 7,9pp | 12,6 |
| S02_supp_half | 32,7% | 37,1% | 4,4pp | 22,7 |
| ... | ... | ... | ... | ... |

---

### 3.3 Analyse (b): Cross-Szenario-Differenzen (vs. Baseline)

**Ziel:** Berechnet für jedes Szenario die Dropout-Differenz zu S01 (Baseline) in Prozentpunkten.

```python
"""
Cross-Szenario-Differenzen (vs. Baseline S01)
==============================================
Zeigt, wie sich jedes Szenario von der Baseline unterscheidet.

Ausführung:
    cd c:\GitHub_public\Abschlussprojekt
    python src/analyse_cross_szenario_diff_v41.py
"""
import pandas as pd
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")
SCENARIOS = [
    "S01_baseline", "S02_supp_half", "S03_supp_double",
    "S04_grade_half", "S05_grade_double", "S06_grade_quad",
    "S07_noise_half", "S08_noise_double",
    "S09_cost_zero", "S10_cost_double",
    "S11_rct_calibrated",
    "S12_overload_half", "S13_overload_double", "S14_overload_cap",
    "S15_cost_effect_double"
]
UNIVERSES = ["A", "B", "C", "D", "E", "F", "G", "H"]
DROPOUT_STATI = {"abgebrochen", "exmatrikuliert", "zeitueberschreitung"}


def dropout_rate(sc, uni):
    csv_path = BASE / sc / f"universe_{uni}" / "abschluesse.csv"
    df = pd.read_csv(csv_path)
    return df["status"].isin(DROPOUT_STATI).sum() / len(df) * 100


# Baseline-Raten berechnen
baseline = {uni: dropout_rate("S01_baseline", uni) for uni in UNIVERSES}

# Differenzen berechnen
rows = []
for sc in SCENARIOS[1:]:  # S02–S15
    row = {"Szenario": sc}
    for uni in UNIVERSES:
        rate = dropout_rate(sc, uni)
        delta = round(rate - baseline[uni], 1)
        row[f"Δ{uni}"] = f"+{delta}" if delta > 0 else str(delta)
    rows.append(row)

result = pd.DataFrame(rows)
print("\n=== CROSS-SZENARIO-DIFFERENZEN (vs. S01 Baseline, in pp) ===\n")
print(result.to_string(index=False))
```

**Interpretation:**
- Spalten mit `0,0` bei Universum B bestätigen korrekte RNG-Synchronisation
- Notenboost-Szenarien (S04–S06) zeigen `0,0` bei C, G, H (kein fachlicher Support dort)
- Rauschen (S07/S08) verändert alle Universen gleichmäßig

---

### 3.4 Analyse (c): Cross-Szenario Migrationsanalyse

**Ziel:** Trackt individuelle Statuswechsel zwischen Szenarien — wer wird „gerettet",
wer geht „verloren" im Vergleich zur Baseline.

```python
"""
Cross-Szenario Migrationsanalyse (Universum A)
===============================================
Verfolgt individuelle Studierenden-Schicksale zwischen Szenarien.

Ausführung:
    cd c:\GitHub_public\Abschlussprojekt
    python src/analyse_migration_v41.py
"""
import pandas as pd
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")
DROPOUT_STATI = {"abgebrochen", "exmatrikuliert", "zeitueberschreitung"}
UNIVERSUM = "A"  # Vergleich immer in Universum A

SCENARIOS_CMP = [
    "S02_supp_half", "S03_supp_double",
    "S04_grade_half", "S05_grade_double", "S06_grade_quad",
    "S07_noise_half", "S08_noise_double",
    "S09_cost_zero", "S10_cost_double",
    "S11_rct_calibrated",
    "S12_overload_half", "S13_overload_double", "S14_overload_cap",
    "S15_cost_effect_double"
]

# Baseline laden
df_base = pd.read_csv(BASE / "S01_baseline" / f"universe_{UNIVERSUM}" / "abschluesse.csv")
df_base["is_dropout_base"] = df_base["status"].isin(DROPOUT_STATI)

rows = []
for sc in SCENARIOS_CMP:
    df_sc = pd.read_csv(BASE / sc / f"universe_{UNIVERSUM}" / "abschluesse.csv")
    df_sc["is_dropout_sc"] = df_sc["status"].isin(DROPOUT_STATI)
    
    merged = df_base.merge(df_sc, on="studierenden_id", suffixes=("_base", "_sc"))
    
    n = len(merged)
    same = (merged["status_base"] == merged["status_sc"]).sum()
    
    # Gerettet: Dropout in Baseline → Abschluss im Szenario
    saved = ((merged["is_dropout_base"]) & (~merged["is_dropout_sc"])).sum()
    
    # Verloren: Abschluss in Baseline → Dropout im Szenario
    lost = ((~merged["is_dropout_base"]) & (merged["is_dropout_sc"])).sum()
    
    netto = saved - lost
    ratio_str = f"{saved}:{lost}" if lost > 0 else f"{saved}:0"
    
    rows.append({
        "Szenario": sc,
        "Gleich": f"{same} ({same/n*100:.1f}%)",
        "Gerettet": saved,
        "Verloren": lost,
        "Netto": f"+{netto}" if netto >= 0 else str(netto),
        "Ratio": ratio_str
    })

result = pd.DataFrame(rows)
print("\n=== CROSS-SZENARIO MIGRATIONSANALYSE (Universum A vs. S01 Baseline) ===\n")
print(result.to_string(index=False))
```

**Schlüsselmetriken:**
- **Gerettet:** Studierender bricht in S01 ab, schließt im Vergleichsszenario ab
- **Verloren:** Studierender schließt in S01 ab, bricht im Vergleichsszenario ab
- **Netto:** Gerettet − Verloren (positiv = Verbesserung gegenüber Baseline)
- **Ratio:** Asymmetrie-Verhältnis als Qualitätsindikator

---

### 3.5 Analyse (d): Paradoxe Statuswechsel — Detailuntersuchung

**Ziel:** Identifiziert Studierende mit kontraintuitiven Ergebnissen (z.B. stärkerer
Support → trotzdem Abbruch) und analysiert den Divergenzpunkt.

```python
"""
Paradoxe Statuswechsel — Semester-für-Semester-Vergleich
========================================================
Vergleicht Prüfungsverläufe von paradoxen Fällen zwischen S01 und S03.

Ausführung:
    cd c:\GitHub_public\Abschlussprojekt
    python src/analyse_paradoxe_faelle_v41.py
"""
import pandas as pd
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")
DROPOUT_STATI = {"abgebrochen", "exmatrikuliert", "zeitueberschreitung"}
UNIVERSUM = "A"

# --- Schritt 1: Paradoxe Fälle identifizieren ---
df_s01 = pd.read_csv(BASE / "S01_baseline" / f"universe_{UNIVERSUM}" / "abschluesse.csv")
df_s03 = pd.read_csv(BASE / "S03_supp_double" / f"universe_{UNIVERSUM}" / "abschluesse.csv")

df_s01["drop_base"] = df_s01["status"].isin(DROPOUT_STATI)
df_s03["drop_s03"] = df_s03["status"].isin(DROPOUT_STATI)

merged = df_s01.merge(df_s03, on="studierenden_id", suffixes=("_s01", "_s03"))

# Paradox: Abschluss in S01 (schwächerer Support) ABER Dropout in S03 (stärkerer Support)
paradox_lost = merged[(~merged["drop_base"]) & (merged["drop_s03"])]
print(f"\n=== PARADOXE FÄLLE S03 vs S01 ===")
print(f"Verloren (Abschluss→Dropout trotz stärkerem Support): {len(paradox_lost)}")

# --- Schritt 2: Prüfungsverläufe vergleichen ---
pru_s01 = pd.read_csv(BASE / "S01_baseline" / f"universe_{UNIVERSUM}" / "pruefungen.csv")
pru_s03 = pd.read_csv(BASE / "S03_supp_double" / f"universe_{UNIVERSUM}" / "pruefungen.csv")

# Beispiel: Erste 5 paradoxe Fälle im Detail
for _, row in paradox_lost.head(5).iterrows():
    sid = row["studierenden_id"]
    print(f"\n--- {sid} (S01: {row['status_s01']}, S03: {row['status_s03']}) ---")
    
    p1 = pru_s01[pru_s01["studierenden_id"] == sid].sort_values("semester_id")
    p3 = pru_s03[pru_s03["studierenden_id"] == sid].sort_values("semester_id")
    
    # Divergenzpunkt finden
    merged_p = p1.merge(p3, on=["semester_id", "modul_id", "versuch"],
                        suffixes=("_s01", "_s03"), how="outer", indicator=True)
    
    # Notendifferenzen
    both = merged_p[merged_p["_merge"] == "both"]
    if len(both) > 0:
        both = both.copy()
        both["note_diff"] = both["note_s03"] - both["note_s01"]
        divergent = both[both["note_diff"].abs() > 0.01]
        if len(divergent) > 0:
            first_div = divergent.iloc[0]
            print(f"  Erste Divergenz: {first_div['semester_id']} {first_div['modul_id']}")
            print(f"  Note S01={first_div['note_s01']}, Note S03={first_div['note_s03']}")
            print(f"  Bestanden S01={first_div.get('bestanden_s01')}, S03={first_div.get('bestanden_s03')}")
    
    # Modul-Set-Differenzen (Module in S03 aber nicht in S01, oder umgekehrt)
    only_s01 = merged_p[merged_p["_merge"] == "left_only"]
    only_s03 = merged_p[merged_p["_merge"] == "right_only"]
    if len(only_s01) > 0 or len(only_s03) > 0:
        print(f"  Module nur in S01: {len(only_s01)}, nur in S03: {len(only_s03)}")
```

**Erwartetes Ergebnis:**
- ~30–50 paradoxe Fälle (< 0,1% der Kohorte)
- Mechanismus: **Curricular-Pfad-Schmetterlingseffekt** — bessere Note → anderes
  Modulportfolio → andere Workload → in seltenen Fällen ungünstigerer Gesamtverlauf

---

### 3.6 Analyse (e): Erwerbstätigkeit × Dropout

**Ziel:** Korrelation zwischen Erwerbstätigkeit (Std/Woche) und Dropout-Rate,
Modulleistung und Support-Nutzung.

```python
"""
Erwerbstätigkeit × Dropout — Subgruppen-Analyse
================================================
Gruppiert Studierende nach Erwerbstätigkeit und berechnet Dropout-Raten.

Ausführung:
    cd c:\GitHub_public\Abschlussprojekt
    python src/analyse_erwerbstaetigkeit_v41.py
"""
import pandas as pd
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")
DROPOUT_STATI = {"abgebrochen", "exmatrikuliert", "zeitueberschreitung"}

# Baseline S01, Universum A
stud = pd.read_csv(BASE / "S01_baseline" / "universe_A" / "studierende.csv")
abschl = pd.read_csv(BASE / "S01_baseline" / "universe_A" / "abschluesse.csv")
pru = pd.read_csv(BASE / "S01_baseline" / "universe_A" / "pruefungen.csv")
einschr = pd.read_csv(BASE / "S01_baseline" / "universe_A" / "einschreibungen.csv")

# Merge Studierende mit Endstatus
df = stud.merge(abschl, on="studierenden_id")
df["is_dropout"] = df["status"].isin(DROPOUT_STATI)

# Prüfungsstatistiken pro Student
pru_stats = pru.groupby("studierenden_id").agg(
    n_pruefungen=("note", "count"),
    n_bestanden=("bestanden", "sum"),
    schnitt=("note", "mean")
).reset_index()

# Semester-Anzahl pro Student
sem_count = einschr.groupby("studierenden_id")["semester_id"].nunique().reset_index()
sem_count.columns = ["studierenden_id", "n_semester"]

df = df.merge(pru_stats, on="studierenden_id", how="left")
df = df.merge(sem_count, on="studierenden_id", how="left")
df["module_pro_sem"] = df["n_pruefungen"] / df["n_semester"]

# --- Gruppierung nach Erwerbstätigkeit ---
result = df.groupby("erwerbstaetigkeit_std").agg(
    n=("studierenden_id", "count"),
    dropout_rate=("is_dropout", "mean"),
    mean_note=("schnitt", "mean"),
    module_pro_sem=("module_pro_sem", "mean"),
    n_pruefungen=("n_pruefungen", "mean")
).round(3)

print("\n=== ERWERBSTÄTIGKEIT × DROPOUT (S01 Baseline, Universum A) ===\n")
print(result.to_string())

# --- Cross-Universum Vergleich ---
print("\n=== ERWERBSTÄTIGKEIT × DROPOUT: UNIVERSUM A vs B ===\n")
for uni in ["A", "B"]:
    s = pd.read_csv(BASE / "S01_baseline" / f"universe_{uni}" / "studierende.csv")
    a = pd.read_csv(BASE / "S01_baseline" / f"universe_{uni}" / "abschluesse.csv")
    m = s.merge(a, on="studierenden_id")
    m["is_dropout"] = m["status"].isin(DROPOUT_STATI)
    grouped = m.groupby("erwerbstaetigkeit_std")["is_dropout"].mean().round(3)
    print(f"Universum {uni}:")
    print(grouped.to_string())
    print()
```

**Erwartete Beobachtungen:**
- Monoton steigende Dropout-Rate mit zunehmender Erwerbstätigkeit
- Stärkerer Effekt in Universum B (kein Support) als in A (alle Supports)
- Support kompensiert teilweise den Erwerbstätigkeits-Nachteil

---

### 3.7 Analyse (f): Zeitkosten × Modulabwurf

**Ziel:** Vergleicht Prüfungen pro Semester über S09 (kostenlos), S01 (Baseline),
S10 (Kosten verdoppelt) und korreliert mit dem Modulabwurf-Mechanismus.

```python
"""
Zeitkosten × Modulabwurf — Analyse
===================================
Vergleicht die Auswirkung von Support-Zeitkosten auf Modulabwürfe und
Prüfungen pro Semester.

Ausführung:
    cd c:\GitHub_public\Abschlussprojekt
    python src/analyse_zeitkosten_modulabwurf_v41.py
"""
import pandas as pd
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")

COST_SCENARIOS = {
    "S09_cost_zero": "Kosten 0",
    "S01_baseline":  "Baseline (1×)",
    "S10_cost_double": "Kosten 2×"
}

print("\n=== ZEITKOSTEN × MODULABWURF ===\n")

for sc_id, label in COST_SCENARIOS.items():
    print(f"--- {label} ({sc_id}) ---")
    
    for uni in ["A", "B"]:
        pru = pd.read_csv(BASE / sc_id / f"universe_{uni}" / "pruefungen.csv")
        einschr = pd.read_csv(BASE / sc_id / f"universe_{uni}" / "einschreibungen.csv")
        
        # Prüfungen pro Semester pro Student
        pru_per_sem = pru.groupby(["studierenden_id", "semester_id"]).size().reset_index(name="n_pru")
        
        # Semesteranzahl pro Student
        n_semester = einschr.groupby("studierenden_id")["semester_id"].nunique()
        n_pru_total = pru.groupby("studierenden_id").size()
        
        mean_pru_per_sem = (n_pru_total / n_semester).mean()
        
        print(f"  Uni {uni}: Ø Prüfungen/Semester = {mean_pru_per_sem:.2f}")
    
    print()

# --- Detailvergleich: Support-Teilnahmen vs Modulabwurf ---
print("=== SUPPORT-TEILNAHMEN PRO STUDENT (Uni A) ===\n")
for sc_id, label in COST_SCENARIOS.items():
    sup = pd.read_csv(BASE / sc_id / "universe_A" / "support_teilnahmen.csv")
    n_teilnahmen = sup.groupby("studierenden_id").size()
    print(f"  {label}: Ø {n_teilnahmen.mean():.2f} Teilnahmen/Student "
          f"(Median: {n_teilnahmen.median():.0f}, Max: {n_teilnahmen.max()})")
```

**Erwartete Beobachtungen:**
- Universum B zeigt identische Werte über alle drei Szenarien (kein Support → keine Zeitkosten)
- In Universum A: leicht höhere Prüfungen/Semester bei S09 (kostenlos), da mehr Zeitbudget
- Der Unterschied ist marginal (~±0,5pp Dropout), Zeitkosten sind der schwächste Parameter

---

### 3.8 Analyse (g): Aggregierte Metriken aus JSON

**Ziel:** Liest die vorberechneten Makro-Effekte aus dem Grid-Runner und erstellt
eine kompakte Übersicht inklusive Synergie-Analyse und Equalizer-Effekt.

```python
"""
Aggregierte Metriken aus full_sensitivity_grid_results.json
===========================================================
Liest die vorberechneten Ergebnisse aus dem Grid-Runner.

Ausführung:
    cd c:\GitHub_public\Abschlussprojekt
    python src/analyse_aggregierte_metriken_v41.py
"""
import json
from pathlib import Path

BASE = Path("src/output_v4_grid_v41")

with open(BASE / "metrics" / "full_sensitivity_grid_results.json") as f:
    grid = json.load(f)

print("\n=== AGGREGIERTE MAKRO-EFFEKTE ===\n")
print(f"{'Szenario':<30s} {'ARR':>8s} {'NNT':>8s} {'Synergie':>10s} {'Equalizer':>12s}")
print("-" * 72)

for sc in grid:
    drop_a = sc["dropout_A"]
    drop_b = sc["dropout_B"]
    arr = (drop_b - drop_a) * 100
    nnt = 100 / arr if arr > 0 else float("inf")
    synergy = sc.get("synergy_gap_pct_pts", 0)
    equalizer = sc.get("equalizer_gain_pct_pts", 0)
    
    print(f"{sc['scenario_id']:<30s} {arr:>7.1f}pp {nnt:>7.1f} {synergy:>+9.1f}pp {equalizer:>+11.1f}pp")

print("\n=== ISOLIERTE SUPPORT-TYP-PROTEKTION ===\n")
print(f"{'Szenario':<30s} {'Fachlich':>10s} {'Überfachl.':>12s} {'Psychosoz.':>12s} {'Alle':>8s}")
print("-" * 76)

for sc in grid:
    print(f"{sc['scenario_id']:<30s} "
          f"{sc['protection_fach_pct']:>+9.1f}% "
          f"{sc['protection_uebf_pct']:>+11.1f}% "
          f"{sc['protection_psych_pct']:>+11.1f}% "
          f"{sc['protection_all_pct']:>+7.1f}%")
```

---

## 4. Ergebnisübersicht

### 4.1 Vollständige Synoptische Tabelle

| # | Szenario | A | B | C | D | E | F | G | H | ARR | NNT |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **S01** | **Baseline** | **29,2%** | **37,1%** | 32,1% | 31,7% | 30,8% | 33,6% | 34,0% | 34,8% | **7,9pp** | **12,6** |
| | *Support-Wirkung* | | | | | | | | | | |
| S02 | Support ½ | 32,7% | 37,1% | 34,4% | 34,0% | 33,8% | 35,2% | 35,6% | 35,8% | 4,4pp | 22,7 |
| S03 | Support 2× | 25,3% | 37,1% | 29,0% | 28,8% | 27,2% | 31,5% | 31,5% | 33,6% | 11,8pp | 8,5 |
| | *Notenboost* | | | | | | | | | | |
| S04 | Boost ½ | 30,6% | 37,1% | 32,1% | 33,1% | 32,4% | 35,2% | 34,0% | 34,8% | 6,5pp | 15,4 |
| S05 | Boost 2× | 27,6% | 37,1% | 32,1% | 29,7% | 29,0% | 31,5% | 34,0% | 34,8% | 9,5pp | 10,5 |
| S06 | Boost 4× | 27,1% | 37,1% | 32,1% | 29,2% | 28,5% | 30,8% | 34,0% | 34,8% | 10,0pp | 10,0 |
| | *Rauschen* | | | | | | | | | | |
| S07 | Rauschen ½ | 26,7% | 33,0% | 29,1% | 28,7% | 28,1% | 30,4% | 30,6% | 31,3% | 6,3pp | 15,8 |
| S08 | Rauschen 2× | 33,2% | 41,0% | 35,7% | 35,7% | 34,8% | 37,8% | 37,6% | 38,7% | 7,9pp | 12,7 |
| | *Zeitkosten* | | | | | | | | | | |
| S09 | Kosten 0 | 28,6% | 37,1% | 32,0% | 31,1% | 30,3% | 33,1% | 33,9% | 34,8% | 8,5pp | 11,8 |
| S10 | Kosten 2× | 29,7% | 37,1% | 32,2% | 32,1% | 31,3% | 34,0% | 34,1% | 34,9% | 7,4pp | 13,5 |
| | *Selektion* | | | | | | | | | | |
| S11 | RCT | 32,6% | 37,1% | 33,7% | 34,1% | 34,4% | 35,9% | 35,5% | 35,1% | 4,5pp | 22,5 |
| | *Overload-Penalty* | | | | | | | | | | |
| S12 | Overload ½ | 26,0% | 34,1% | 29,0% | 28,6% | 27,6% | 30,4% | 30,9% | 31,9% | 8,1pp | 12,3 |
| S13 | Overload 2× | 34,6% | 41,8% | 37,1% | 36,9% | 36,3% | 38,8% | 39,0% | 39,7% | 7,3pp | 13,8 |
| S14 | Overload Cap | 26,7% | 35,0% | 29,9% | 29,3% | 28,3% | 31,3% | 31,8% | 32,8% | 8,4pp | 12,0 |
| | *Kombi* | | | | | | | | | | |
| **S15** | **Kosten+Wirkung 2×** | **25,8%** | **37,1%** | 29,0% | 29,3% | 27,7% | 31,9% | 31,5% | 33,6% | **11,3pp** | **8,9** |

> **Legende:** A = Alle Supports, B = Kein Support, C = −Fachlich, D = −Überfachlich, E = −Psychosozial,
> F = Nur Fachlich, G = Nur Überfachlich, H = Nur Psychosozial.
> ARR = Absolute Risk Reduction (B−A), NNT = Number Needed to Treat (100/ARR).

### 4.2 Sensitivitätsranking

| Rang | Parameter | Δ bei Verdopplung | Δ bei Halbierung | Spannweite | ARR-Stabilität |
| :---: | :--- | ---: | ---: | ---: | :---: |
| 🥇 | **Overload-Penalty** | +5,4pp | −3,1pp | **8,5pp** | stabil (7,3–8,4) |
| 🥈 | **Support-Wirkung** | −3,8pp | +3,5pp | **7,3pp** | variabel (4,4–11,8) |
| 🥉 | **Rauschen** | +4,0pp | −2,4pp | **6,4pp** | stabil (6,3–7,9) |
| 4 | **Selektion (RCT)** | +3,5pp | — | **3,5pp** | — |
| 5 | **Notenboost** | −1,6pp | +1,4pp | **3,0pp** | variabel (6,5–10,0) |
| 6 | **Zeitkosten** | +0,6pp | −0,6pp | **1,2pp** | stabil (7,4–8,5) |

### 4.3 Migrationsanalyse (Universum A vs. S01 Baseline)

| Szenario | Gleich | Gerettet | Verloren | Netto | Ratio |
| :--- | ---: | ---: | ---: | ---: | :---: |
| S03 Support 2× | 95,3% | **1.964** | 41 | **+1.923** | 48:1 |
| S15 Kombi | 94,9% | **1.898** | 229 | **+1.669** | 8:1 |
| S12 Overload ½ | 95,8% | 1.634 | 64 | +1.570 | 26:1 |
| S14 Overload Cap | 96,5% | 1.281 | 48 | +1.233 | 27:1 |
| S07 Rauschen ½ | 93,8% | 1.846 | 624 | +1.222 | 3:1 |
| S06 Boost 4× | 97,4% | 1.061 | 23 | +1.038 | 46:1 |
| S05 Boost 2× | 98,0% | 821 | 29 | +792 | 28:1 |
| S09 Kosten 0 | 98,1% | 482 | 207 | +275 | 2:1 |
| S10 Kosten 2× | 98,0% | 227 | 506 | −279 | 1:2 |
| S04 Boost ½ | 98,0% | 30 | 751 | −721 | 1:25 |
| S11 RCT | 93,9% | 359 | 2.106 | −1.747 | 1:6 |
| S02 Support ½ | 95,4% | 31 | 1.800 | −1.769 | 1:58 |
| S08 Rauschen 2× | 89,4% | 1.265 | 3.265 | −2.000 | 1:3 |
| S13 Overload 2× | 92,7% | 89 | 2.801 | −2.712 | 1:31 |

### 4.4 Validierungsmerkmale

> [!IMPORTANT]
> **RNG-Synchronisation validiert:**
> - Universum B = 37,1% in allen 10 Szenarien ohne globale Parameteränderung
> - Nur Rauschen (S07/08) und Overload (S12/13/14) verändern B — korrekt
> - Notenboost (S04–S06) verändert nur Universen mit fachlichem Support (A, D, E, F)

---

## 5. Referenzen zu weiteren Analysen

| Dokument | Pfad | Inhalt |
| :--- | :--- | :--- |
| **Sensitivitätsbericht (vollständig)** | `Artifacts/sensitivitaetsanalyse_v41_nachtlauf_fnal.md` | Alle 15 Szenarien mit Interpretation, Sensitivitätsranking, S15-Kombianalyse |
| **Sensitivitätsbericht (Zwischenbericht)** | `Artifacts/sensitivitaetsanalyse_v41_nachtlauf.md` | Erste 11 Szenarien (historisch) |
| **Paradoxe Fälle** | `Artifacts/detailanalyse_paradoxe_faelle_v41.md` | Mechanismus des Curricular-Pfad-Schmetterlingseffekts, Fallbeispiele |
| **Versionsvergleich V3.6 → V4.1** | `Artifacts/versionsvergleich_v36_v41.md` | RNG-Sync-Validierung, Studentengenerierung, Support-Logik, Modulabwurf |
| **Grid-Sensitivitätsanalyse (V4.0)** | `Artifacts/sensitivitaetsanalyse_v4_grid.md` | Älterer V4.0-Bericht (vor RNG-Fix) |
| **Simulationsengine** | `src/simulation_v4.py` | V4.1-Simulationscode mit deterministischem Prüfungsrauschen |
| **Grid-Runner** | `src/run_v4_simulation_grid.py` | Multiprocessing-Runner für alle 14 Grid-Szenarien (S02–S15) |
| **Konfiguration** | `src/config.py` | Baseline-Parameter (`CONFIG`), Curricula, Support-Angebote |
| **Aggregierte JSON-Ergebnisse** | `src/output_v4_grid_v41/metrics/full_sensitivity_grid_results.json` | Vorberechnete Makro-Effekte für alle Szenarien |

---

## 6. Reproduktionsanleitung

### 6.1 Datensatz neu generieren

> [!CAUTION]
> Die Simulation läuft ca. 14–15 Stunden auf einem 5-Core-System.
> Ergebnisse sind deterministisch bei identischem Seed.

```bash
cd c:\GitHub_public\Abschlussprojekt

# Baseline S01 (über run_v4_universes.py)
python src/run_v4_universes.py --n 50000 --seed 99999 --out src/output_v4_grid_v41/S01_baseline

# Grid S02–S15 (über run_v4_simulation_grid.py)
python src/run_v4_simulation_grid.py
# → Standardwerte: n=25000, seed=99999, workers=5
# → Für N=50.000: Parameterwerte im Skript anpassen
```

### 6.2 Analysen ausführen

```bash
cd c:\GitHub_public\Abschlussprojekt

# (a) Synoptische Übersicht
python src/analyse_synoptische_uebersicht_v41.py

# (b) Cross-Szenario-Differenzen
python src/analyse_cross_szenario_diff_v41.py

# (c) Migrationsanalyse
python src/analyse_migration_v41.py

# (d) Paradoxe Fälle
python src/analyse_paradoxe_faelle_v41.py

# (e) Erwerbstätigkeit × Dropout
python src/analyse_erwerbstaetigkeit_v41.py

# (f) Zeitkosten × Modulabwurf
python src/analyse_zeitkosten_modulabwurf_v41.py

# (g) Aggregierte Metriken (aus JSON)
python src/analyse_aggregierte_metriken_v41.py
```

> [!TIP]
> Alle Analyse-Skripte lesen nur CSV/JSON-Dateien und benötigen keine GPU oder
> Simulation. Laufzeit: jeweils 10–60 Sekunden je nach Dateigröße der pruefungen.csv.

---

*Erstellt: 2026-08-30 | Simulationsversion: V4.1 | Seed: 99999 | N: 50.000/Universum*
