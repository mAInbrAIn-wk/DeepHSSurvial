# ToDo / Erledigt
- [x] Leakage-Fix in den 5 ML-Skripten (Student-Level Split implementiert)
- [x] Future Leakage Fix: `cp_rueckstand` nutzt nun `cp_cum_prev` in Exam-Formaten
- [x] Oracle Feature Extension: `hidden_overload` und `hidden_zeit_puffer` in Feature Builder ergänzt
- [x] Feature Counts in README.md & Co. aktualisieren
- [ ] PyTorch / PyCox Portierung 
- [ ] Dropout-Regularisierung überprüfen, vielleicht besser L2-Regulierung?? Wie sehen die Lernkurven aus, gibt es da findetuning-Potential?
- [ ] SQL Backend implementieren? Was lässt sich am besten wo umsetzen? DuckDB und Arrow könnten ja vielleicht sogar Performance bringen?

## KI-Input

2. Wie Data Warehousing / eine relationale DB DeepSupport massiv verbessert
Hier schließt sich der Kreis zu Deinem ersten Projekt (Projekt_DE): Aktuell speichert DeepSupport Millionen von Datenzeilen in vielen flachen CSV-Dateien (output_dl/) und fügt sie über langwierige Pandas-Merges zusammen.

Ein relationaler, eingebetteter DWH-Layer (z. B. mit DuckDB und Parquet) würde folgende Quantensprünge bringen:

Feature Store mit SQL Window Functions: Zeitvariable Merkmale (rollierende GPAs, kumulierte Fehlversuche, CP-Rückstände) lassen sich in SQL-Fensterfunktionen in Millisekunden berechnen – statt in minutenlangen Pandas-Schleifen.
Multi-Universen-Partitionierung: Alle 5 Universen liegen in einer einzigen, partitionierten DWH-Faktentabelle. Kontrafaktische Abfragen werden zu simplen, performanten SQL-Queries (WHERE UniverseID IN ('A', 'E')).
Zero-Copy Data Streaming für TensorFlow: DuckDB kann Abfrageergebnisse via Apache Arrow speichereffizient und ohne RAM-Duplikation direkt an tf.data.Dataset übergeben.
Reproduzierbarkeit & ACID: Feste Typen, keine stillen NaN/float-Konvertierungsfallen.

review_abschlussprojekt_deepsupport.md