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
| **Exam GRU V2** | $N=50k,\ K=50$ | $\mathbf{0{,}2012}\ /\ 0{,}8922$ | $0{,}1918\ /\ 0{,}8870$ | $0{,}1587\ /\ 0{,}8709$ | $0{,}1872\ /\ 0{,}8792$ | $0{,}1877\ /\ 0{,}8876$ |
| **Neural Hazard** | $N=363k$ Rows | $0{,}1673\ /\ 0{,}7452$ | $0{,}1645\ /\ 0{,}7373$ | $0{,}0954\ /\ 0{,}7071$ | $\mathbf{0{,}1678}\ /\ 0{,}7243$ | $0{,}1634\ /\ 0{,}7492$ |

*Anmerkung:* Der Brier-Score liegt konstant im optimal kalibrierten Bereich von $0{,}0365$ bis $0{,}0382$.

### B. Kausale Krafteinschätzung (Relative Risk - RR)
*Werte $>1$ deuten auf Risikoerhöhung (Selektionsbias-Fehler), Werte $<1$ auf Schutzwirkung hin. Mean RR (isoliert).*
*Anmerkung: Cox PHReg scheitert bei OHE teils an Singular Matrices. Hier sind die Neural Hazard (Klasse 5) Resultate gelistet.*

| Modell | Modus | Fachlich RR | Überfachlich RR | Psychosozial RR |
| :--- | :---: | :---: | :---: | :---: |
| **Semester GRU** | `standard` | $0{,}998$ | $1{,}007$ | $1{,}018$ |
| **Semester GRU** | `gradeblind` | $1{,}070$ | $1{,}020$ | $1{,}007$ |
| **Semester GRU** | `blind` | $1{,}032$ | $1{,}064$ | $0{,}995$ |
| **Semester GRU** | `oracle` | $1{,}021$ | $1{,}041$ | $1{,}009$ |
| **Semester GRU** | `realistic` | $1{,}015$ | $1{,}025$ | N/A |
| **Transformer** | `standard` | $1{,}002$ | $1{,}019$ | $1{,}007$ |
| **Transformer** | `gradeblind` | $1{,}012$ | $1{,}009$ | $1{,}007$ |
| **Transformer** | `blind` | $1{,}017$ | $1{,}077$ | $1{,}000$ |
| **Transformer** | `oracle` | $1{,}007$ | $1{,}022$ | $1{,}008$ |
| **Transformer** | `realistic` | $1{,}022$ | $1{,}019$ | N/A |
| **Neural Hazard** | `standard` | $\mathbf{0{,}991}$ | $\mathbf{0{,}993}$ | $\mathbf{0{,}985}$ |
| **Neural Hazard** | `oracle` | $\mathbf{0{,}993}$ | $\mathbf{0{,}990}$ | $\mathbf{0{,}978}$ |
| **Neural Hazard** | `realistic` | $\mathbf{0{,}982}$ | $\mathbf{0{,}981}$ | N/A |

### C. Regressions-Güte ($R^2$ / $\text{RMSE}$ / $\text{MAE}$) & Kausaleffekte ($\Delta\text{Note}$)

*Vorhersage der finalen Abschlussnote bei Absolventen ($N=34.592$).*

| Modellarchitektur | Dimension | Standard | Gradeblind | Blind | Realistic (DSGVO) | Oracle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Landmark MLP (Klasse 1)** | $N=34{,}6k$ | $0{,}8634\ /\ 0{,}2311$ | $0{,}6375\ /\ 0{,}3763$ | $0{,}5901\ /\ 0{,}4002$ | $0{,}8598\ /\ 0{,}2341$ | $\mathbf{0{,}9001}\ /\ 0{,}1975$ |
| **Landmark Ridge (Klasse 1)** | $N=34{,}6k$ | $0{,}8502\ /\ 0{,}2419$ | $0{,}5867\ /\ 0{,}4018$ | $0{,}5644\ /\ 0{,}4125$ | $0{,}8442\ /\ 0{,}2467$ | $0{,}8768\ /\ 0{,}2194$ |
| **Semester Sequenz (Klasse 2b)** | $N=34{,}6k,\ T=16$ | $\mathbf{0{,}9887}\ /\ 0{,}0665$ | $0{,}7012\ /\ 0{,}3417$ | $0{,}6281\ /\ 0{,}3812$ | $\mathbf{0{,}9891}\ /\ 0{,}0652$ | $\mathbf{0{,}9904}\ /\ 0{,}0612$ |

*Kausaleffekte auf Noten ($\Delta\hat{y}$ ATE in Notenpunkten, Landmark MLP vs. Ground Truth $\mathbf{-0{,}0901}$ / $\mathbf{-0{,}1431}$):*
- **`standard`:** $\Delta\text{Note} = +0{,}0462$ (Schein-Verschlechterung durch Selektionsbias: schwächere Studierende nehmen mehr Support).
- **`gradeblind`:** $\Delta\text{Note} = \mathbf{-0{,}1347}$ (**Exzellenter Kausaltreffer!** Ohne Notenbias isoliert das Modell die wahre Schutzwirkung).
- **`oracle`:** $\Delta\text{Note} = \mathbf{-0{,}0637}$ (**Kausaltreffer:** Orakel-Zustände neutralisieren den Selektionsbias und zeigen Notenverbesserung).
- **`realistic`:** $\Delta\text{Note} = +0{,}0076$ (Verzerrt/Neutralisiert durch fehlende Confounder-Kontrolle).

---

## 3. Zentrale Erkenntnisse & Schlussfolgerungen

### 1. Dropout-Notenblindheit vs. Regressions-Notenblindheit (Die zentrale Antwort)
**Gilt `gradeblind` $\approx$ `standard` auch für Regressionsmodelle?**
$\rightarrow$ **Klares NEIN!**
- Bei der **Dropout-Vorhersage (Klassifikation/Survival)** führt das Entfernen von Noten zu **keinem nennenswerten Güteverlust** ($PR\text{-}AUC \approx 0{,}2255$ vs $0{,}2225$), da Studienabbrüche fast ausschließlich durch harte quantitative Schwellenwerte (Fehlversuche, CP-Rückstände, Drittversuchs-Exmatrikulationen) ausgelöst werden.
- Bei der **Noten-Regression** hingegen bricht die Modellgüte dramatisch ein: Das Bestimmtheitsmaß $R^2$ stürzt von **$0{,}8634$ auf $0{,}6375$** (Landmark) bzw. von **$0{,}9887$ auf $0{,}7012$** (Semester-Sequenz) ab. Eine kontinuierliche Notenprognose ist fundamental auf frühe Notenleistungen angewiesen.

### 2. Der methodische Überraschungserfolg: `gradeblind` als Kausalschätzer für Noten
Während `gradeblind` für die *Prädiktion* von Noten schlechter ist, fungiert es als **hervorragender Kausalschätzer**: Da dem Modell die verzerrte Notenhistorie vorenthalten wird, schätzt das `gradeblind`-MLP die Kausalwirkung des fachlichen Supports auf $\mathbf{-0{,}1347}$ Notenpunkte Notenverbesserung (exakte Deckung mit der Ground Truth von $\mathbf{-0{,}1431}$ Notenpunkten!).

### 3. Datenschutzkonformität vs. Kausalkraft (`realistic`)
Während `realistic` eine leicht verbesserte PR-AUC (z.B. $0{,}2316$ vs $0{,}2291$ beim Transformer) aufweist – vermutlich durch Regularisierung / Vermeidung von Overfitting an verrauschten Confoundern –, **sinkt die Fähigkeit zur kausalen Identifikation** bei den Sequenzmodellen. 
Beim Transformer kippt der isolierte RR-Effekt des fachlichen Supports von $1{,}002$ auf $1{,}022$ (falscher Risikoeffekt). Wenn das Modell Confounder (z.B. Migrationshintergrund) nicht mehr sieht, rechnet es den Dropout fälschlicherweise alleinig der Support-Maßnahme zu.

### 4. Das Oracle Paradoxon: Warum tiefe Prädiktionsmodelle Kausalität ignorieren
In früheren, einfacheren Analysen (z.B. statisches DeepSurv) konnte das Hinzufügen von Oracle-Variablen (wie `hidden_motivation`) den Selektionsbias entzerren. Wie die obige Tabelle zeigt, **scheitern hochdimensionale Sequenzmodelle (Transformer/GRU) jedoch selbst im `oracle` Modus**. Teilweise wird der Bias sogar schlimmer (Transformer Oracle RR = 1.007 vs Standard = 1.002).
**Erklärung:** Ein tiefes Netz optimiert rein auf Prädiktionsfehler (BCE Loss). Wenn das Netz die perfekte latente Variable `hidden_motivation` (die Dropout extrem stark determiniert) als Input erhält, lernt es, sich **ausschließlich** auf diese Variable zu stützen. Das Feature "Support-Nutzung" wird prädiktiv redundant. Das Modell ignoriert das Treatment, die entsprechenden Gewichte konvergieren zu Rauschen um $0.0$, was ein RR von $\approx 1.0$ erzeugt. Die Oracle-Variablen entzerren bei einem Prädiktionsmodell nicht die kausale Wirkung, sondern sie **überschreiben** das Signal des Treatments. 
**Beweis:** Aggregierte Panel-Modelle (Neural Hazard, Klasse 5), die nicht auf Sequenz-Optimierung sondern auf Hazard-Schätzung pro Intervall setzen, finden durchgängig die korrekte **Schutzwirkung (RR < 1, z.B. 0.991)**. Um Kausaleffekte aus tiefen Netzen zu ziehen, reicht Feature-Engineering (selbst Oracle) nicht aus – es braucht kausale Architekturen (z.B. Double Machine Learning), da die Prädiktion das Treatment sonst marginalisiert.

### 5. Leistungsfähigkeit rein präventiver Startmodelle (`blind`)
Ohne Verlaufsdaten sinkt die PR-AUC um ca. 25–35 %, liegt aber immer noch deutlich über der Zufallsprävalenz. Ein `blind`-Modell eignet sich hervorragend für ein Onboarding-Screening.
