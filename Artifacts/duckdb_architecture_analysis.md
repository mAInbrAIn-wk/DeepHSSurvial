# Architektur-Analyse: DuckDB & Apache Arrow als DWH-Backend

Dieses Artefakt untersucht das Potenzial, die aktuelle flache CSV- und Pandas-Architektur von DeepSupport durch ein eingebettetes relationales Data Warehouse (DuckDB) mit Zero-Copy-Speicherverwaltung (Apache Arrow) zu ersetzen.

## 1. Aktuelle Architektur-Engpässe (Pandas & CSV)
Derzeit simuliert `simulation_v3.py` Daten für 5 Universen und speichert diese als `.csv`-Dateien unter `output_dl/`. 
Die `feature_builder.py` Pipeline lädt diese CSVs in den RAM und berechnet Features (wie rollierende GPAs oder kummulierte Fehlversuche) in Pandas.

**Kritische Flaschenhälse:**
1. **Speicher-Duplikation (RAM):** Pandas lädt den gesamten CSV-Text, konvertiert ihn und dupliziert DataFrames bei `merge()` oder `.copy()`. Bei 5 Universen mit je 10.000 Studierenden explodiert der Speicherbedarf.
2. **Schleifen und Windowing:** Die Berechnung von "CP-Rückstand im Fachsemester $t$" oder "Bisherige Fach-Supports bis Semester $t$" erzwingt in Pandas `groupby().apply()`-Konstrukte, die langsam iterieren.
3. **Stille Typisierungsfehler:** Pandas konvertiert Integers automatisch zu Floats, sobald ein `NaN` auftaucht (z. B. fehlende Noten bei Abbrechern). Dies zwingt uns zu Workarounds (`fillna(-99.0)`), bevor TensorFlow die Tensoren akzeptiert.

## 2. Lösungsansatz: DuckDB + Parquet + Arrow

Durch den Einsatz von DuckDB (einer spaltenorientierten, in-process SQL-Datenbank) als zwischengeschalteten Feature-Store ließen sich diese Probleme auflösen.

### A. Feature Store mit SQL Window Functions
Die zeitvariablen Merkmale der Ebene B (Semester) und Ebene A (Exam) sind klassische relationale Zeitreihen.
Statt komplexer Pandas-Merges können Features extrem effizient über SQL-Windowing berechnet werden:

```sql
-- Beispiel: Berechnung der kumulierten Fehlversuche vor der aktuellen Prüfung
SELECT 
    studierenden_id,
    semester_id,
    modul_id,
    versuch,
    SUM(CASE WHEN bestand = 0 THEN 1 ELSE 0 END) OVER (
        PARTITION BY studierenden_id 
        ORDER BY semester_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) as cum_fails_vorher
FROM exam_events
```
*Vorteil:* DuckDB ist massiv parallelisiert. Solche Window-Abfragen werden auf C++-Ebene in Millisekunden über Millionen von Zeilen ausgeführt.

### B. Multi-Universen-Partitionierung in Parquet
Statt für jedes der 5 Universen eigene Ordner mit redundanten Demographie-CSVs zu pflegen, speichert der Simulator die Daten als partitioniertes Parquet-Verzeichnis:
`output_dl/data/universe=A/exams.parquet`

DuckDB kann diese Partitionen nativ abfragen:
```sql
SELECT * FROM parquet_scan('output_dl/data/universe=*/exams.parquet') WHERE universe IN ('A', 'E')
```
*Vorteil:* Kontrafaktische Analysen (Gegenüberstellung von Universum A und E) lassen sich ohne Speicheroverhead direkt im Query abhandeln.

### C. Zero-Copy Data Streaming für TensorFlow
Der vielleicht größte Hebel für Deep Learning: DuckDB kann Abfrageergebnisse nativ als Apache Arrow Tabellen ausgeben.

```python
# DuckDB Query liefert ein Arrow Table
arrow_table = con.execute("SELECT * FROM feature_view").arrow()

# TensorFlow Data Dataset liest Arrow direkt ohne Pandas-Overhead
import tensorflow_io as tfio
dataset = tfio.IODataset.from_arrow(arrow_table)
```
*Vorteil:* Es findet **kein Kopieren im RAM** statt (Zero-Copy). Die Daten fließen im binären Arrow-Spaltenformat direkt in die TensorFlow-Tensoren.

## 3. Implementierungs-Strategie (Wo passt es am besten?)

Die Umstellung sollte nicht im Kern des Simulators passieren (dieser generiert die Grunddaten effizient als Python-Objekte), sondern an der **Schnittstelle zwischen Simulator und Feature-Engine**.

**Empfohlener Refactoring-Pfad:**
1. *Export:* `simulation_v3.py` schreibt seine Listen von Dicts nicht mehr per `csv.DictWriter`, sondern via `pyarrow` direkt in `.parquet`-Dateien.
2. *Feature-Engine:* `feature_builder.py` wird von Pandas auf DuckDB-SQL-Strings umgestellt. Die Funktion `build_semester_sequence_tensor()` feuert eine SQL-Query ab und konvertiert das Arrow-Result direkt in das `(N, T, F)` Numpy-Array bzw. den Tensor.
3. *Typ-Sicherheit:* Die Datenbank-Schema erzwingt feste Typen (z.B. `INTEGER` für CPs), sodass keine stillen `NaN`-zu-`Float`-Castings das Netz destabilisieren.

**Synergien mit dem aktuellen Plan:**
Da wir das Feature-Engineering gerade in `feature_builder.py` zentralisiert haben, müsste die Datenbank-Anbindung nur in genau *einer* Datei geändert werden, anstatt alle Modell-Skripte anzufassen. Dies bestätigt die Weitsicht des letzten Refactorings.
