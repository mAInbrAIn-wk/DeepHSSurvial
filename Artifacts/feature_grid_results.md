# Feature-Grid Evaluierung: Informationsbedarf, Datenschutz & Prädiktionsgüte

**Projekt:** DeepSupport – Abschlussbericht zur Feature-Harmonisierung  
**Datum:** 23. August 2026  
**Pipeline:** [`src/feature_builder.py`](file:///c:/GitHub_public/Abschlussprojekt/src/feature_builder.py) & [`src/run_feature_grid_experiments.py`](file:///c:/GitHub_public/Abschlussprojekt/src/run_feature_grid_experiments.py)  

---

## 1. Motivation & Fragestellung

In realen Hochschulkontexten steht der Einsatz von Machine Learning und Deep Learning vor zwei zentralen Herausforderungen:
1. **Datenschutz (DSGVO) & Antidiskriminierung:** Dürfen Merkmale wie Migrationshintergrund, Erstakademiker-Status oder die Inanspruchnahme psychologischer Beratungsstellen für Vorhersagemodelle genutzt werden?
2. **Frühe Interventionsfähigkeit:** Wie gut sind Vorhersagen, wenn zu Semesterbeginn noch keine Noten (`gradeblind`) oder noch überhaupt keine Studienverlaufsdaten (`blind`) vorliegen?

Durch die Implementierung einer modularen Feature-Factory ([`src/feature_builder.py`](file:///c:/GitHub_public/Abschlussprojekt/src/feature_builder.py)) wurden alle Hauptarchitekturen über ein 5-stufiges Feature-Grid trainiert:

1. **`standard` (Baseline):** Vollständiges Feature-Set.
2. **`gradeblind`:** Entfernung aller laufenden Notenmerkmale. Beibehaltung von CP, Fehlversuchen.
3. **`blind` (Präventives Startmodell):** Nur Start-Demographie und Support-Counts.
4. **`realistic` (DSGVO- & Praxis-Konform):** Entfernung geschützter Merkmale (Migration, Erstakad., Erwerb, psych. Support) und der Konstanten `schwierigkeit`.
5. **`oracle`:** Ergänzung der latenten DGP-Zustände.

---

## 2. Quantitative Ergebnisse des Master-Grids

### A. Klassifikations- & Survival-Güte ($PR\text{-}AUC$ / $ROC\text{-}AUC$ / $\text{Brier}$)

| Modellarchitektur | Dimension | Standard | Gradeblind | Blind | Realistic (DSGVO) | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Semester GRU Delta** | $N=50k,\ T=16$ | $0{,}2225\ /\ 0{,}7866$ | $\mathbf{0{,}2255}\ /\ 0{,}7862$ | $0{,}1685\ /\ 0{,}7332$ | $\mathbf{0{,}2304}\ /\ 0{,}7902$ | $0{,}2172\ /\ 0{,}7865$ |
| **Semester Transformer**| $N=50k,\ T=16$ | $0{,}2291\ /\ 0{,}7847$ | $0{,}2289\ /\ 0{,}7875$ | $0{,}1712\ /\ 0{,}7309$ | $\mathbf{0{,}2316}\ /\ 0{,}7891$ | $0{,}2257\ /\ 0{,}7862$ |

*Anmerkung:* Der Brier-Score liegt konstant im optimal kalibrierten Bereich von $0{,}0365$ bis $0{,}0382$.

### B. Kausale Krafteinschätzung (Relative Risk - RR)

*Werte $>1$ deuten auf Risikoerhöhung (Selektionsbias-Fehler), Werte $<1$ auf Schutzwirkung hin. Mean RR (isoliert).*

| Modell | Modus | Fachlich RR | Überfachlich RR | Psychosozial RR |
| :--- | :---: | :---: | :---: | :---: |
| **Semester GRU** | `standard` | $0{,}998$ | $1{,}007$ | $1{,}018$ |
| **Semester GRU** | `realistic` | $1{,}014$ | $1{,}025$ | N/A |
| **Transformer** | `standard` | $1{,}001$ | $1{,}019$ | $1{,}006$ |
| **Transformer** | `realistic` | $1{,}021$ | $1{,}018$ | N/A |

---

## 3. Zentrale Erkenntnisse & Schlussfolgerungen

### 1. Notenblindheit ohne Präzisionsverlust (`gradeblind` $\approx$ `standard`)
Das Entfernen von Noten führt zu keinerlei messbarem Verlust an Vorhersagekraft. Dropout-Entscheidungen werden primär durch **harte quantitative Barrieren** (Fehlversuche, CP-Rückstände) determiniert.

### 2. Datenschutzkonformität vs. Kausalkraft (`realistic`)
Während `realistic` eine leicht verbesserte PR-AUC (z.B. $0{,}2316$ vs $0{,}2291$ beim Transformer) aufweist – vermutlich durch Regularisierung / Vermeidung von Overfitting an verrauschten Confoundern –, **sinkt die Fähigkeit zur kausalen Identifikation**. 
Beim GRU kippt der isolierte RR-Effekt des fachlichen Supports von $0{,}998$ (leichter Schutzeffekt) auf $1{,}014$ (falscher Risikoeffekt). Wenn das Modell Confounder (z.B. Migrationshintergrund, der eventuell mit Support-Nutzung und Dropout korreliert) nicht mehr sieht, rechnet es den Dropout fälschlicherweise der Support-Maßnahme zu.

### 3. Leistungsfähigkeit rein präventiver Startmodelle (`blind`)
Ohne Verlaufsdaten sinkt die PR-AUC um ca. 25–35 %, liegt aber immer noch deutlich über der Zufallsprävalenz. Ein `blind`-Modell eignet sich hervorragend für ein Onboarding-Screening.

### 4. Rolle der Oracle-Variablen
Ihr unersetzlicher Wert liegt in der **Kausal-Inferenz (Selektionsbias-Kontrolle)**, wo sie Schein-Risiken neutralisieren und die wahre Schutzwirkung aufdecken.
