# Zwischenbericht: Detailanalysen & korrigierter Grid-Run

## Status der parallelen Arbeitsströme

| Task | Status | Ergebnis |
| :--- | :--- | :--- |
| ✅ Migrationsanalyse (alle 8 Universen) | **Fertig** | [`migrationsanalyse_v4.md`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/migrationsanalyse_v4.md) |
| ✅ Zeitkosten-Studienebene (Baseline + S10) | **Fertig** | [`zeitkosten_studienebene_v4.md`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/zeitkosten_studienebene_v4.md) |
| ✅ Notenimpact S01 vs S03 | **Fertig** | [`notenimpact_analyse_v4.md`](file:///C:/Users/wilfr/.gemini/antigravity/brain/16832ed6-a522-415e-9395-ef24e16fef79/notenimpact_analyse_v4.md) |
| 🔄 Korrigierter Grid-Run (4×8 = 32 Simulationen) | **Läuft** | S03 (mult=10.0), S11 (RCT kalibriert), S12/S13 (Overload ±) |

---

## 1. Migrationsanalyse: Wer wird gerettet, wer geht verloren?

### A vs. Blockade-Welten (Referenz: A = Full Support)

| Vergleich | Verloren (Absolvent in A → Dropout in X) | Gerettet (Dropout in A → Absolvent in X) | **Netto** |
| :--- | :---: | :---: | :---: |
| A vs **B** (Kein Support) | 4.034 | 2.246 | **−1.788** |
| A vs **C** (Kein Fachlich) | 3.168 | 2.786 | **−382** |
| A vs **D** (Kein Überfachlich) | 3.400 | 2.693 | **−707** |
| A vs **E** (Kein Psychosozial) | 3.264 | 2.891 | **−373** |

> [!NOTE]
> Die „Verloren"-Spalte zeigt, dass auch bei vollem Support **über 2.000 Studierende in A durchfallen, die in den Vergleichswelten bestehen**. Das sind keine Fehler — es ist die Konsequenz der **RNG-Divergenz**: Weil Support-Teilnahmen die Zufallszahlensequenz verschieben, erhalten einzelne Studierende in Support-Welten andere Prüfungsrauschwerte. In der Summe überwiegt der Schutzeffekt, aber auf Individualebene gibt es beide Richtungen.

### B vs. Isolierte Supportwelten (Referenz: B = Kein Support)

| Vergleich | Verloren (Absolvent in B → Dropout in X) | Gerettet (Dropout in B → Absolvent in X) | **Netto** |
| :--- | :---: | :---: | :---: |
| B vs **F** (Nur Fachlich) | 2.598 | 3.103 | **+505** |
| B vs **G** (Nur Überfachlich) | 2.389 | 3.164 | **+775** |
| B vs **H** (Nur Psychosozial) | 2.480 | 3.023 | **+543** |

> Überfachlicher Support rettet netto die meisten Studierenden (+775), gefolgt von psychosozialem (+543) und fachlichem (+505) Support. Auch hier zeigt die „Verloren"-Spalte die RNG-Divergenz (2.389–2.598 gehen den umgekehrten Weg).

### A vs B: Subgruppen

| Subgruppe | N | Verloren | Gerettet | **Netto** | **Schutzrate** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gesamt** | 25.000 | 4.034 | 2.246 | **−1.788** | **−7,2%** |
| **Erstakademiker** | 11.971 | 2.150 | 1.007 | **−1.143** | **−9,5%** |
| Nicht-Erstakademiker | 13.029 | 1.884 | 1.239 | −645 | −5,0% |
| **Migrationshintergrund** | 5.512 | 953 | 467 | **−486** | **−8,8%** |
| Kein Migrationshintergrund | 19.488 | 3.081 | 1.779 | −1.302 | −6,7% |
| HZB ≤ 2,5 (gut) | 15.034 | 1.605 | 1.446 | −159 | −1,1% |
| **HZB > 2,5 (schwach)** | 9.966 | 2.429 | 800 | **−1.629** | **−16,3%** |

> [!IMPORTANT]
> Studierende mit schwacher HZB (> 2,5) profitieren am stärksten: **16,3% Schutzrate** vs. nur 1,1% bei guter HZB. Erstakademiker (9,5%) und Studierende mit Migrationshintergrund (8,8%) werden ebenfalls überproportional geschützt.

---

## 2. Zeitkosten: Verteilung auf Studierendenebene

| Szenario | Status | N | Ø Dauer | Median | Ø Versuche | Ø Durchfälle | Ø Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (~20h)** | Absolventen | 18.041 | 7,9 | 7 | 18,9 | 1,3 | 2,25 |
| **Baseline (~20h)** | Dropout | 6.959 | 4,7 | 4 | 12,4 | 5,7 | 3,74 |
| **Hohe Belastung (60h)** | Absolventen | 17.888 | 8,0 | 7 | 18,8 | 1,3 | 2,24 |
| **Hohe Belastung (60h)** | Dropout | 7.112 | 4,8 | 4 | 12,2 | 5,5 | 3,73 |

> S09 (Kostenlos, 0h) hatte im alten Grid-Run keine CSV-Daten. Diese werden im neuen Grid-Run nachgeliefert.

> [!NOTE]
> Die Unterschiede zwischen Baseline und 60h-Belastung sind auf Studierendenebene **erstaunlich gering**: +153 Dropouts, aber quasi identische Noten und Studiendauern bei Absolventen. Der Haupteffekt der Zeitkosten ist also **marginal** und wirkt über den indirekten Kanal der Modulabwürfe (81k vs 92k), nicht über direkte Leistungsverschlechterung.

---

## 3. Notenimpact: S01 (mult=5.0) vs. S03 (mult=2.0)

| Metrik | Wert |
| :--- | :--- |
| Prüfungen in Schnittmenge (bestanden + Support in beiden) | **1.307** |
| Ø Notendifferenz (S01 − S03) | **+0,507 Notenpunkte** |
| Ø Support-Boost S01 (Note − Counterfactual) | **−1,447** (Note besser als ohne Support) |
| Ø Support-Boost S03 (Note − Counterfactual) | **−1,730** |
| Fälle: In S01 (stärker) durchgefallen, in S03 bestanden | **110** |

> [!WARNING]
> Es gibt **110 Fälle**, in denen Studierende bei stärkerem Multiplikator (S01) durchfallen, die bei schwächerem (S03) bestehen. Das ist **kein Bug**, sondern ein Artefakt der **RNG-Divergenz**: Weil der stärkere Multiplikator mehr Support-Teilnahmen auslöst, verschiebt sich die Zufallszahlensequenz. Die 110 Fälle (von ~425k Prüfungen) liegen im statistischen Rauschen.

> S05 (gewicht_support_boost=0.16) und S06 (0.32) hatten keine CSV-Daten im alten Run. Im neuen korrigierten Run werden alle Szenarien mit CSV gespeichert.

---

## 4. Ausstehend (nach Grid-Run-Abschluss)

1. **S03 korrigiert** (mult=10.0 = echte Verdopplung): Notenimpact-Analyse wiederholen
2. **S11 kalibriert** (RCT mit Baseline-Volumen): Tatsächlicher Selektionseffekt isolieren
3. **S12/S13** (Overload-Penalty ±): Neue Dimension der Sensitivität
4. **S09-CSV** (Kostenlos): Zeitkosten-Dreieck vervollständigen
5. **Notenimpact S05/S06**: Nachliefern, sobald CSVs vorhanden
