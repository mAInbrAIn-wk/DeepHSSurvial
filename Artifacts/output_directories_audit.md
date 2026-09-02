# Output Directories Audit & Versioning History

Ein rekursiver Tiefen-Scan über Deinen gesamten Workspace hat unglaubliche **20 verschiedene Output-Verzeichnisse** mit insgesamt fast **25 Gigabyte** an Daten zutage gefördert! 

Dieser historische "Sediment-Boden" zeigt perfekt die Evolution des Projekts. Hier ist die exakte Zuordnung und Versionierung, basierend auf Datei-Inhalten, Größen und Meta-Infos:

## 1. Die V1 & V2 Ära (Frühe Prototypen)
Diese Ordner stammen aus der Frühphase des Projekts. Sie enthalten oft noch die alte `DATENSATZ_DOKU.md`.
* `output_dl_v1` (210 MB, 27 Modelle)
* `output_dl_v2` (945 MB, 25 Modelle)
* `output_dl` (Root-Verzeichnis, ein altes Default-Verzeichnis)

## 2. Die V3 Ära (Regel-Experimente & Carryover)
Hier wurde intensiv an den Support-Regeln (Carryover, Caps) und der Seed-Stabilität geschraubt.
* `output_dl_v3.1` (1 GB)
* `output_dl_v3.2_carryover` (1.1 GB)
* `output_dl_v3.2_nocap` (1.1 GB)
* `output_dl_v3halfbacked` (1 GB)
* `src/output_dl_seed99999` (Seed-Variationstest)
* `src/output_dl_v36_clean` (Die letzte stabile V3-Iteration)
* `src/output_v36_clean_rerun` (Unser gestriger Kontroll-Lauf ohne Daten)
* `src/output_dl` (Das alte Standard-Verzeichnis, völlig überfüllt mit 1.7 GB)
* `src/src/output_dl` (Doppelt verschachtelte Fehlerläufe)

## 3. Die V4 Ära (Das Cross-Szenario Grid)
Hier wurde die Datenbasis massiv erweitert, um Sensitivitäten (Noise, Cost, Overload) zu testen.
* `src/output_v4_test` & `src/output_v4_universes` (Lokale V4-Tests)
* `src/output_v4_grid` (Erster Grid-Versuch, 2 GB, 48 Universen)
* **`src/output_v4_grid_v41` (10.6 GB, 120 Universen) $\rightarrow$ DAS IST DER AKTUELLE GOLD-STANDARD!**

---

### Aufklärung: Das Missverständnis zum V4.1 Grid-Lauf

Du hast völlig recht gefragt: *"Haben wir uns mißverstanden, als Du sagtest, auf den v4.1 Daten sei gar kein gridlauf passiert?"*

Ich habe mir den 10.6 GB großen Ordner `src/output_v4_grid_v41` jetzt exakt auf Dateiebene angesehen. Die Daten für S01 bis S15 existieren! Die Simulation lief fehlerfrei durch und hat alle 120 Universen (A-H für 15 Szenarien) erzeugt. 
**ABER:** Wenn man in die Ordner `S02_supp_half` bis `S15_cost_effect_double` schaut, liegt dort jeweils exakt *eine* Datei im `metrics/`-Ordner: die `true_macro_effects.json`. 
Die 72 trainierten Deep-Learning Modelle (und ihre 108 JSON-Metriken) liegen *ausschließlich* im Ordner `S01_baseline`. 

**Fazit:** Der Grid-Lauf für die *Datengenerierung* (Simulation) lief für alle Szenarien durch. Der Grid-Lauf für das *Modell-Training* (Neural Networks) lief jedoch nur auf der Baseline (S01). Das heißt, die Daten sind da – wir müssen die Modelle nur noch darauf trainieren!

---

## Mein Vorschlag für den "Snapshot & Safe"

Da wir jetzt den perfekten historischen Überblick haben:

1. **Einfrieren (Archive):** Alle Ordner der **V1, V2 und V3 Ära** sowie die abgebrochenen V4-Tests werden physisch nach `archive/legacy_outputs/` verschoben. Sie bleiben zu 100% erhalten, müllen aber den Workspace nicht mehr zu.
2. **Den Gold-Standard bewahren:** Den 10.6 GB großen Ordner `src/output_v4_grid_v41` behalten wir! Er enthält unsere neuesten Daten. Um seine Rolle klarzumachen, benenne ich ihn um in `data_v4_grid` (denn er ist eine Datenquelle, kein reiner Output-Ordner).
3. **Clean Slate für Modelle:** Die alten V4-Modelle und Metriken in `S01_baseline` verschiebe ich ebenfalls ins Archiv. So haben wir eine völlig saubere Datenquelle, über die unser neuer, refactorierter Grid-Runner laufen kann.
