---
## 📝 Entwicklungs-Historie & User-Annotation
**Datum:** 02. September 2026
**Kontext:** Abschluss des gigantischen Code-Refactorings (V3.6 auf V4)
**Auslösende Annotationen (Zusammenfassung):** 
> *"Bitte auf keinen fall fehlende Werte 0-imputieren, sonst fällt da u.U. gar nicht mehr auf, wenn eine Metrik nicht berechnet wird..."*
> *"Kannst Du die gegenwärtigen Dateien, ebenfalls in einen Teil des Archivs kopieren, bevor Du sie umstrukturierst?"*
> *"Tatsächlich hätte ich gerne ein weit vollständigeres Gesprächsprotokoll, dass eben auch meine vielen Annotationen trackt. Quasi zur vollständig transparenten Entwicklungsgeschichte."*
---

# Protokoll: Master-Refactoring & Repository-Bereinigung

## 1. Ausgangslage & Zielsetzung
Das Projekt hatte über 60 monolithische Skripte und ca. 25 GB an Output-Daten (CSV, Keras-Gewichte, JSONs) im Root- und `src/`-Verzeichnis akkumuliert. 
Ziel dieses Pair-Programming-Sprints war:
1. Modulare, objektorientierte Python-Architektur (`deepsupport/` Package).
2. Strikte Trennung von I/O im `grid_runner.py` (Vermeidung des Überschreibens von Ground-Truth-Daten).
3. Bereinigung des Haupt-Repositories für GitHub-Kompatibilität, OHNE wertvolle Code-Historie oder Design-Überlegungen (Walkthroughs) zu verlieren.

## 2. Architektonische Entscheidungen (ADRs)
* **Zero-Imputation Policy:** Im `metrics_logger.py` werden fehlende Metriken strikt als `null` gespeichert, niemals als `0.0`. Dies verhindert verdeckte Fehler in aggregierten Auswertungen.
* **One-Model-One-Script:** Keras-Netzwerk-Definitionen (z. B. `build_gru_model`) wurden aus den Runnern entfernt und in separierte Architektur-Dateien unter `src/deepsupport/models/` isoliert. 
* **Annotation Tracking Pattern:** Um die iterativen Denkprozesse bei der Arbeit mit der KI transparent zu machen, enthält zukünftig jedes Design-Dokument einen YAML-ähnlichen Header, der den initialen User-Prompt zitiert.

## 3. Die große Code- & Daten-Rettung
1. **Das Archiv-Subrepo:** Alle generierten Output-Ordner (insb. `output_dl_v36_clean`, `data_v4_grid` etc.) wurden lokal in ein Subrepo `archive/` verschoben und im Haupt-Repo untracked (`git rm --cached`).
2. **Die Code-Rettung:** 129 alte Python-Skripte (aus `archive/` und Root) wurden *im Haupt-Repo* in den Ordner `legacy_code/` verschoben. Git's Rename-Heuristik hat die vollständige Historie dieser Code-Dateien fehlerfrei erkannt und bewahrt.
3. **Dokumentations-Chaos beseitigt:** 129 Markdown-Dokumente aus dem versteckten `Artifacts/`-Ordner wurden kategorisiert und in `docs/` mit einem auto-generierten Index (`README.md`) strukturiert.

## 4. Status Quo
Das Haupt-Repository ist zu 100 % bereit für einen Master-Push auf GitHub und den Start des Cross-Szenario Cluster-Runs (`grid_runner.py` auf S01-S15).
