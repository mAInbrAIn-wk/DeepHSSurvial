# Kontrafaktische Ground-Truth: V3.6 vs. V4 (korrigiert)

## Drei-Wege-Vergleich der Dropout-Raten

| Universum | V3.6 | V4 (κ=20, alt) | V4 (κ korr.) | Beschreibung |
|:---|:---|:---|:---|:---|
| **A** (Alles) | 31.18 % | 21.05 % | **27.54 %** | Alle Support-Typen erlaubt |
| **B** (Nichts) | 39.08 % | 27.46 % | **34.75 %** | Kein Support |
| **C** (ohne Fach) | 33.52 % | 22.14 % | **29.46 %** | Kein fachlicher Support |
| **D** (ohne Übf) | 33.92 % | 23.58 % | **30.46 %** | Kein überfachlicher Support |
| **E** (ohne Psych) | 33.00 % | 22.93 % | **29.24 %** | Kein psychosozialer Support |
| **F** (nur Fach) | 36.10 % | 25.93 % | **32.79 %** | Nur fachlicher Support |
| **G** (nur Übf) | 35.62 % | 24.28 % | **31.59 %** | Nur überfachlicher Support |
| **H** (nur Psych) | 36.59 % | 25.14 % | **32.70 %** | Nur psychosozialer Support |

> [!NOTE]
> Die korrigierten κ-Werte bringen die absoluten Dropout-Raten deutlich näher an V3.6 (Δ ≈ 3-4 pp statt 10 pp). Die verbleibende Differenz kommt von der parabolischen Support-Friction und davon, dass Beta-Verteilungen naturgemäß etwas weniger extreme Ränder produzieren als geclippte Normalverteilungen.

---

## Relative Risiken (Partielle Welten vs. Welt A)

| Universum | V3.6 RR | V4 (κ=20) RR | V4 (κ korr.) RR | Interpretation |
|:---|:---|:---|:---|:---|
| **C** vs A (Fach weg) | 1.075 | 1.052 | **1.069** | Fachlicher Support-Effekt |
| **D** vs A (Übf weg) | 1.088 | 1.120 | **1.106** | Überfachlicher Support-Effekt |
| **E** vs A (Psych weg) | 1.058 | 1.089 | **1.061** | Psychosozialer Support-Effekt |
| **B** vs A (Alles weg) | 1.253 | 1.305 | **1.262** | Gesamteffekt Support |

> [!IMPORTANT]
> Der **Gesamteffekt** (RR B vs A) liegt jetzt bei **1.262** – fast identisch mit V3.6 (1.253). Die κ-Korrektur war erfolgreich.

### Support-Typ Ranking

| Rang | V3.6 | V4 (κ korr.) |
|:---|:---|:---|
| 1. | Überfachlich (RR 1.088) | **Überfachlich (RR 1.106)** |
| 2. | Fachlich (RR 1.075) | **Fachlich (RR 1.069)** |
| 3. | Psychosozial (RR 1.058) | **Psychosozial (RR 1.061)** |

Das Ranking ist identisch geblieben. Der überfachliche Support hat in V4 leicht zugelegt (+1.8 pp), was konsistent mit dem neuen Zeitbudget-Tracker ist: Modulabwürfe bei Überlast werden jetzt sauberer getrackt und der überfachliche Support (der bei Studienplanung und Zeitmanagement hilft) greift dort stärker.

---

## Isolierte Welten (F, G, H vs. Null-Support B)

| Universum | V3.6 RR vs B | V4 (κ korr.) RR vs B | Δ |
|:---|:---|:---|:---|
| **F** (nur Fach) | 0.924 (-7.6 %) | **0.944 (-5.6 %)** | Fachlich alleine schwächer |
| **G** (nur Übf) | 0.912 (-8.8 %) | **0.909 (-9.1 %)** | Überfachlich alleine stabil |
| **H** (nur Psych) | 0.936 (-6.4 %) | **0.941 (-5.9 %)** | Psychosozial alleine stabil |

---

## κ-Parameter Dokumentation

| Variable | V3 (clipped Normal) | V4 alt (κ) | V4 korr. (κ) | σ-Ziel |
|:---|:---|:---|:---|:---|
| `alter` | `clip(N(20.5, 2.8), 17, 45)` | 20.0 | **12.8** | ≈ 2.49 |
| `hzb_note` | `clip(N(2.4, 0.55), 1.0, 4.0)` | 20.0 | **6.5** | ≈ 0.55 |
| `motivation` | `clip(mean + N(0, 0.1), 0.05, 1.0)` | 20.0 | **20.0** | ≈ 0.10 |
| `soz_integration` (Walk) | `clip(x + N(0, 0.05), 0.05, 1.0)` | 40.0 | **95.0** | ≈ 0.05 |
