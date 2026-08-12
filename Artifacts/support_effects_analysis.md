# Analyse der kontrafaktischen Support-Effekte

**Stand:** 11. August 2026  
**Datenquelle:** [analyze_support_effects.py](file:///c:/GitHub_public/Abschlussprojekt/src/analyze_support_effects.py) → [full_output.txt](file:///c:/GitHub_public/Abschlussprojekt/output_dl/analysis/full_output.txt)  
**Datenbasis:** 5 Universen mit je **50.000 Studierenden** (aus [config.py](file:///c:/GitHub_public/Abschlussprojekt/src/config.py), Zeile 31: `n_studierende: 50000`)

> [!IMPORTANT]
> Alle Zahlen in diesem Dokument stammen ausschließlich aus dem verifizierten Skript-Output ([full_output.txt](file:///c:/GitHub_public/Abschlussprojekt/output_dl/analysis/full_output.txt)). Keine Zahl ist geschätzt oder extrapoliert.

---

## 1. Dropout-Raten und Status-Verteilung

### 1.1 Status-Verteilung pro Universum

Quelle: `abschluesse.csv` pro Universum.

| Universum | Label | abgeschlossen | abgebrochen | exmatrikuliert | zeitüberschreitung |
|:---------:|:------|:-------------:|:-----------:|:--------------:|:------------------:|
| A | Baseline (alle Support-Typen) | 35.030 | 12.993 | 1.732 | 245 |
| B | Kein Support (komplett blockiert) | 31.669 | 14.893 | 3.195 | 243 |
| C | Kein fachlicher Support | 34.754 | 13.029 | 2.024 | 193 |
| D | Kein überfachlicher Support | 33.450 | 13.935 | 2.329 | 286 |
| E | Kein psychosozialer Support | 33.857 | 13.821 | 2.042 | 280 |

### 1.2 Dropout-Raten (breite Definition: abgebrochen + exmatrikuliert + zeitüberschreitung)

| Universum | Label | Dropout | Rate | Diff. vs. A | RR |
|:---------:|:------|--------:|-----:|:-----------:|:--:|
| A | Baseline | 14.970 | 29.94% | — | 1.0000 |
| B | Kein Support | 18.331 | 36.66% | +6.72 pp | 0.8166 |
| C | Kein fachlicher Support | 15.246 | 30.49% | +0.55 pp | 0.9819 |
| D | Kein überfachlicher Support | 16.550 | 33.10% | +3.16 pp | 0.9045 |
| E | Kein psychosozialer Support | 16.143 | 32.29% | +2.35 pp | 0.9273 |

### 1.3 Zum Vergleich: Enge Definition (nur `s.abgebrochen`, aus `true_macro_effects_v2.json`)

| Universum | Rate (eng) | Rate (breit) | Differenz |
|:---------:|:----------:|:------------:|:---------:|
| A | 25.986% | 29.94% | +3.95 pp |
| B | 29.786% | 36.66% | +6.87 pp |
| C | 26.058% | 30.49% | +4.43 pp |
| D | 27.870% | 33.10% | +5.23 pp |
| E | 27.642% | 32.29% | +4.65 pp |

> [!NOTE]
> Die breite Definition fängt auch diejenigen ein, die durch endgültiges Nichtbestehen exmatrikuliert werden oder die maximale Studienzeit überschreiten. Beide Definitionen zeigen dasselbe Muster: Fachlicher Support hat den schwächsten, überfachlicher den stärksten Effekt auf Dropout.

---

## 2. Individuelle Studierenden-Migration zwischen Universen

Dropout-Definition: abgebrochen + exmatrikuliert + zeitüberschreitung.

### A vs. B (Kein Support → komplett blockiert)

|  | Nicht-Dropout in B | Dropout in B |
|:-|:------------------:|:------------:|
| **Nicht-Dropout in A** | 30.816 | 4.214 |
| **Dropout in A** | 853 | 14.117 |

### A vs. C (Kein fachlicher Support)

|  | Nicht-Dropout in C | Dropout in C |
|:-|:------------------:|:------------:|
| **Nicht-Dropout in A** | 33.690 | 1.340 |
| **Dropout in A** | 1.064 | 13.906 |

### A vs. D (Kein überfachlicher Support)

|  | Nicht-Dropout in D | Dropout in D |
|:-|:------------------:|:------------:|
| **Nicht-Dropout in A** | 33.170 | 1.860 |
| **Dropout in A** | 280 | 14.690 |

### A vs. E (Kein psychosozialer Support)

|  | Nicht-Dropout in E | Dropout in E |
|:-|:------------------:|:------------:|
| **Nicht-Dropout in A** | 33.634 | 1.396 |
| **Dropout in A** | 223 | 14.747 |

### Zusammenfassung Migration

| Vergleich | Zum Dropout getrieben | Dafür gerettet | **Netto-Effekt** |
|:---------:|:---------------------:|:--------------:|:----------------:|
| A vs. B (kein Support) | 4.214 | 853 | **3.361** |
| A vs. C (kein fachlich) | 1.340 | 1.064 | **276** |
| A vs. D (kein überfachlich) | 1.860 | 280 | **1.580** |
| A vs. E (kein psychosozial) | 1.396 | 223 | **1.173** |

> [!IMPORTANT]
> **Beim fachlichen Support (C) ist der "Rettungseffekt" fast so groß wie der "Dropout-Effekt"!** 1.064 Studierende profitieren vom Wegfall des fachlichen Supports – wahrscheinlich durch die eingesparten 30h Zeitkosten pro Angebot. Der Netto-Effekt beträgt nur 276 von 50.000 Studierenden.

---

## 3. Noteneffekte des fachlichen Supports

### 3.1 Innerhalb von Universum A: `note` vs. `note_counterfactual`

Quelle: `pruefungen.csv` (Universum A), Spalten `support_genutzt` und `note_counterfactual`.

| Metrik | Wert |
|:-------|:-----|
| Prüfungen mit Support | 37.373 |
| Prüfungen ohne Support | 800.558 |
| **Mittlere Notenverbesserung (ATT)** | **+0.7835 Notenpunkte** |
| Median Notenverbesserung | +1.0000 Notenpunkte |
| Prüfungen mit Verbesserung | 30.409 (81.4%) |
| Prüfungen ohne Veränderung | 6.964 (18.6%) |
| **Vor Durchfallen gerettet** | **5.059 (13.54%)** |
| Bestehensquote MIT Support | 92.77% |
| Bestehensquote OHNE Support (kontrafaktisch) | 79.23% |
| **Bestehensquoten-Differenz** | **+13.54 pp** |

### 3.2 Verteilung der Notenverbesserungen

| Verbesserungsgrad | Anzahl | Anteil |
|:------------------|-------:|-------:|
| 0 (kein Effekt) | 6.964 | 18.6% |
| 0.01–0.3 (klein) | 788 | 2.1% |
| 0.3–0.5 (mittel) | 3.133 | 8.4% |
| 0.5–1.0 (groß) | 8.952 | 24.0% |
| 1.0–2.0 (sehr groß) | 16.508 | 44.2% |
| 2.0+ (extrem) | 1.028 | 2.8% |

### 3.3 Direkter Notenvergleich A vs. C (gleiche Studis, gleiche Module)

| Metrik | Wert |
|:-------|:-----|
| Gematchte Prüfungen (merge auf Studi+Semester+Modul) | 805.737 |
| Davon mit Support in A | 33.837 |
| Mittlerer Notenunterschied (C − A) | +0.7928 (A besser) |
| Median | +1.0000 |
| Bestanden in A | 31.424 |
| Bestanden in C | 26.868 |
| **Nur durch fachlichen Support bestanden** | **4.642** |

### 3.4 Finaler Fähigkeitswert (`hidden_erwartete_note_final`)

| Metrik | Wert |
|:-------|:-----|
| Mittlerer Unterschied (C − A) | +0.0394 (A besser) |
| Median | +0.0000 |
| Standardabweichung | 0.1456 |
| A besser (Diff > 0.01) | 12.921 (25.8%) |
| C besser (Diff < −0.01) | 3.339 (6.7%) |
| Gleich (\|Diff\| ≤ 0.01) | 33.740 (67.5%) |

> [!NOTE]
> Fachlicher Support hat einen **starken Noteneffekt** (+0.78 Notenpunkte, +13.5pp Bestehensquote), aber einen **nahezu null Dropout-Effekt**. Das Paradox erklärt sich durch die Simulationsarchitektur (siehe Abschnitt 5).

---

## 4. Synergie-Effekte

Quelle: `true_macro_effects_v2.json` (enge Definition: nur `abgebrochen`).

| Support-Typ | Dropout-Reduktion (pp) |
|:------------|:----------------------:|
| Fachlicher Support | +0.072 pp |
| Überfachlicher Support | +1.884 pp |
| Psychosozialer Support | +1.656 pp |
| **Summe der Einzeleffekte** | **+3.612 pp** |
| **Tatsächlicher Gesamteffekt (B vs. A)** | **+3.800 pp** |
| **Synergie (Rest)** | **+0.188 pp** |

**Synergie-Anteil:** 4.9% des Gesamteffekts. Die Support-Typen verstärken sich leicht gegenseitig, die Effekte sind aber nahezu additiv.

---

## 5. Kausal-Diagnose: Warum fachlicher Support kaum auf Dropout wirkt

### 5.1 Architektur der Dropout-Funktion

Quelle: [simulation.py](file:///c:/GitHub_public/Abschlussprojekt/src/simulation.py), Zeile 160–166.

```python
p = 0.01 
  + max(0.0, (0.4 - motivation)) * 0.30       # ← kein direkter Pfad von fachlichem Support
  + max(0.0, (0.4 - soz_int)) * 0.20           # ← kein direkter Pfad von fachlichem Support
  + min(cp_rueckstand / 30.0, 1.0) * 0.15      # ← indirekter Pfad (mehr bestanden → weniger Rückstand)
  + durchgefallen_aktuell * 0.04               # ← indirekter Pfad (weniger Failures)
  + min(overload_penalty, 0.3) * 0.10          # ← NEGATIVER Pfad (30h Zeitkosten!)
```

### 5.2 Vier Ursachen

**Ursache 1: Kein direkter Pfad.** Fachlicher Support beeinflusst `motivation` und `soz_integration` **nicht direkt** ([simulation.py](file:///c:/GitHub_public/Abschlussprojekt/src/simulation.py), Zeile 287–294: nur überfachlich und psychosozial haben Motivation/Integrations-Boosts).

**Ursache 2: Schwacher indirekter Pfad.** `durchgefallen_aktuell` hat nur Gewicht 0.04 pro Fail. Ein vermiedenes Durchfallen senkt die Dropout-Wahrscheinlichkeit minimal.

**Ursache 3: Zeitkosten konterkarieren.** Fachliche Angebote kosten **30h** ([config.py](file:///c:/GitHub_public/Abschlussprojekt/src/config.py), Zeile 689–718: `kosten_h: 30`), überfachliche 10h, psychosoziale 5–15h. Die 30h erhöhen den `overload_penalty` und können Module verdrängen (Zeile 299–303).

**Ursache 4: Notenverbesserung ≠ Dropout-Vermeidung.** `berechne_dropout()` zählt nur binäre Failures (Note > 4.0). Von 37.373 Support-Prüfungen liegen nur **8.926 (23.9%)** im Grenzbereich (kontrafaktische Note ≥ 4.0), und davon werden **5.059 (13.54%)** tatsächlich gerettet.

### 5.3 Das Kernproblem: Fast symmetrischer Netto-Effekt

Der fachliche Support hat gleichzeitig:
- **Positiv:** 1.340 Studierende werden vor dem Dropout bewahrt
- **Negativ:** 1.064 Studierende droppen *wegen* des Supports (Zeitkosten!)
- **Netto: nur 276 Studierende** von 50.000 (0.55 pp)

---

## 6. Vergleich: DML-Schätzungen vs. kontrafaktische Ground Truth

Quelle: [dml_orthogonal_survival_metrics.json](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/dml_orthogonal_survival_metrics.json) und [true_macro_effects_v2.json](file:///c:/GitHub_public/Abschlussprojekt/output_dl/metrics/true_macro_effects_v2.json).

| Support-Typ | Ground Truth RR | DML Mean RR | Differenz | Bewertung |
|:------------|:---------------:|:-----------:|:---------:|:---------:|
| Fachlich | 0.9972 | 0.8953 | −0.1020 | Abweichung |
| Überfachlich | 0.9324 | 0.9708 | +0.0384 | Gut |
| Psychosozial | 0.9401 | 0.8619 | −0.0782 | Akzeptabel |

**Gesamteffekt Ground Truth RR (B vs. A):** 0.8724

> [!WARNING]
> Das DML-Modell **überschätzt** den Effekt des fachlichen Supports deutlich (RR 0.90 statt 1.00). Es kann den Nicht-Effekt nicht korrekt identifizieren, vermutlich weil fachlicher Support in den Observationsdaten mit schlechteren Ausgangsnoten korreliert (Confounding durch Selektionseffekt: schwächere Studierende nehmen häufiger teil).

---

## Exportierte Dateien

- [universe_comparison_summary.csv](file:///c:/GitHub_public/Abschlussprojekt/output_dl/analysis/universe_comparison_summary.csv) — Zusammenfassung pro Universum
- [student_migration_matrix.csv](file:///c:/GitHub_public/Abschlussprojekt/output_dl/analysis/student_migration_matrix.csv) — Dropout-Status pro Student pro Universum
- [full_output.txt](file:///c:/GitHub_public/Abschlussprojekt/output_dl/analysis/full_output.txt) — Vollständiger, reproduzierbarer Skript-Output
