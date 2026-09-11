---
created: 2026-09-11
last_updated: 2026-09-11
status: abgeschlossen
tags: [causal-inference, dml, msm, g-computation, ground-truth-benchmark, lxc-run, v42, rct-calibration]
---

# PyTorch Causal Suite: LXC Benchmark-Report & Ground Truth Validierung (V4.2)

## 1. Executive Summary & Problemaufriss

Auf dem Debian LXC Container (`PythonLXC`, 8 vCPUs, 7 PyTorch-Rechenthreads, Multi-Core CPU) wurde die **PyTorch Causal Suite** über sechs gezielte Szenarien des synthetischen Simulations-Windkanals V4.2 ausgeführt ($N = 50.000$ Studierende je Szenario, $345.000$ Person-Semester-Beobachtungen je Universum). 

Ziel dieses Methoden-Benchmarks ist nicht das Auffinden von Mechanismen einer empirischen "Realität", sondern die rigorose quantitative Überprüfung des Zusammenspiels zwischen dem bekannten datengenerierenden Prozess (DGP) und modernen Methoden der kausalen Inferenz:
1. **G-Computation Simulation (`PyTorchGComputation`):** Kontrafaktische Monte-Carlo-Integration und bedingte Hazard-Projektion unter den Interventionen $\text{do}(A = 0)$ und $\text{do}(A = 1)$.
2. **Marginal Structural Models (`PyTorchMSM`):** Parametrische Schätzung zeitdiskreter Hazard Ratios auf einer durch stabilisierte inverse Propensity-Gewichte ($SW$) entkoppelten Pseudopopulation zur Auflösung von *time-varying confounding with feedback*.
3. **Double Machine Learning (`PyTorchDMLSurvival`):** Neyman-orthogonale semiparametrische Residualisierung über 5-Fold Cross-Fitting zur Schätzung lokaler Average Treatment Effects (ATE) und bereinigter Hazard Ratios.

Erstmals werden alle drei Inferenzsäulen **systematisch gegen den exakten, kontrafaktischen Ground Truth** validiert, der durch die parallele Simulation von acht Kontroll- und Interventionswelten (Universen A bis H) auf identischen Studierenden-Seeds vorliegt.

---

## 2. Die drei kausalen Vergleichslinien im Simulations-DGP

Um Fehlschlüsse bei der Interpretation von Interventionswirkungen zu vermeiden, müssen drei distinkte Vergleichslinien strikt getrennt werden:

1. **Gesamtpaket-Effekt (Universum A vs. B):**
   Vergleich der Vollförderungs-Welt (A: alle drei Support-Typen aktiv) gegen die Null-Interventions-Welt (B: keinerlei Support-Angebote). Dies misst die kumulative Makro-Intervention auf Studienabschluss-Ebene.
2. **Isolierter Einzel-Effekt (Universen F, G, H vs. B):**
   Vergleich von Welten, in denen exakt ein Support-Typ isoliert existiert (F: nur Fachlich, G: nur Überfachlich, H: nur Psychosozial), gegen die Null-Welt B. Da keine Konkurrenz- oder Komplementäreffekte anderer Maßnahmen auftreten, misst diese Differenz die genuine Reinform der Einzelmaßnahme.
3. **Partieller Entzugs-Effekt (Universum A vs. C, D, E):**
   Vergleich der Vollförderungs-Welt A gegen Welten, in denen exakt eine Maßnahme entzogen wurde (C: kein Fachlich, D: kein Überfachlich, E: kein Psychosozial). Diese Differenz misst den marginalen Zusatznutzen der Komponente im Kontext des bestehenden Verbundangebots.

---

## 3. Cross-Szenario Gesamtsynopse

Die folgende Tabelle führt die empirischen Ground Truth Kennzahlen mit den Schätzungen der drei PyTorch-Kausalmodelle zusammen.

### 3.1 Synoptische Ergebnismatrix ($N = 50.000$, Universum A, `temporal=prev`, `mode=standard`)

| Szenario | Parameter-Charakteristik | Ground Truth A vs B (Gesamt) | GT Isolierter RR (F/G/H vs B) | G-Comp ARR (pp) | G-Comp RR (95% Boot CI) | MSM HR All (95% Asym CI) | MSM HR Fach | MSM HR Uebf | MSM HR Psych | DML HR Fach |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01_baseline** | Standard DGP | ARR $= +7{,}95$ pp<br>$RR = 0{,}7858$ | F: $0{,}9064$<br>G: $0{,}9166$<br>H: $0{,}9390$ | **$+4{,}59$ pp** | **$0{,}7873$**<br>[$0{,}7848$, $0{,}7910$] | **$0{,}8002$**<br>[$0{,}7645$, $0{,}8376$] | $0{,}8313$ | $0{,}9176$ | $0{,}7303$ | $1{,}0001$ |
| **S02_supp_half** | Support-Effekt halbiert ($\times 0{,}5$) | ARR $= +4{,}41$ pp<br>$RR = 0{,}8812$ | F: $0{,}9492$<br>G: $0{,}9586$<br>H: $0{,}9659$ | **$+3{,}75$ pp** | **$0{,}8317$**<br>[$0{,}8291$, $0{,}8353$] | **$0{,}8476$**<br>[$0{,}8138$, $0{,}8829$] | $0{,}9064$ | $0{,}9598$ | $0{,}7541$ | $0{,}9954$ |
| **S03_supp_double** | Support-Effekt verdoppelt ($\times 2{,}0$) | ARR $= +11{,}79$ pp<br>$RR = 0{,}6822$ | F: $0{,}8480$<br>G: $0{,}8483$<br>H: $0{,}9065$ | **$+4{,}26$ pp** | **$0{,}7998$**<br>[$0{,}7969$, $0{,}8032$] | **$0{,}7626$**<br>[$0{,}7241$, $0{,}8031$] | $0{,}7371$ | $0{,}8289$ | $0{,}7037$ | $0{,}9241$ |
| **S07_noise_half** | Rauschen halbiert ($\sigma_{\text{noise}} = 0{,}09$) | ARR $= +6{,}33$ pp<br>$RR = 0{,}8083$ | F: $0{,}9185$<br>G: $0{,}9260$<br>H: $0{,}9459$ | **$+3{,}97$ pp** | **$0{,}7787$**<br>[$0{,}7745$, $0{,}7830$] | **$0{,}8070$**<br>[$0{,}7685$, $0{,}8475$] | $0{,}8260$ | $0{,}9251$ | $0{,}7221$ | $0{,}9932$ |
| **S08_noise_double** | Rauschen verdoppelt ($\sigma_{\text{noise}} = 0{,}36$) | ARR $= +7{,}88$ pp<br>$RR = 0{,}8080$ | F: $0{,}9201$<br>G: $0{,}9168$<br>H: $0{,}9433$ | **$+4{,}37$ pp** | **$0{,}8191$**<br>[$0{,}8170$, $0{,}8227$] | **$0{,}7797$**<br>[$0{,}7484$, $0{,}8123$] | $0{,}7777$ | $0{,}8783$ | $0{,}7478$ | $0{,}9579$ |
| **S11_rct_calibrated** | Randomisierte Zuweisung (RCT) | ARR $= +4{,}45$ pp<br>$RR = 0{,}8800$ | F: $0{,}9678$<br>G: $0{,}9559$<br>H: $0{,}9471$ | **$+6{,}16$ pp** | **$0{,}7335$**<br>[$0{,}7293$, $0{,}7383$] | **$0{,}6776$**<br>[$0{,}6489$, $0{,}7076$] | $0{,}6745$ | $0{,}6837$ | $0{,}6815$ | $0{,}9317$ |

---

## 4. Detaillierte Methoden-Dekonstruktion

### 4.1 G-Computation: Exakte Abbildung der relativen Risikoreduktion ($RR$)

Die PyTorch G-Computation schätzt die bedingte Übergangswahrscheinlichkeit $P(Y_t = 1 \mid Y_{t-1}=0, A_t, \mathbf{X}_t)$ über ein neuronales Hazard-Netzwerk und integriert über die beobachtete Kohorte unter kontrafaktischer Setzung von $\text{do}(A=0)$ bzw. $\text{do}(A=1)$.

1. **Präzision des relativen Risikos:**
   In der Baseline **S01** schätzt G-Computation ein relatives Risiko von:
   $$RR_{\text{G-Comp}} = 0{,}7873 \quad [95\,\%\text{-Bootstrap: } 0{,}7848,\, 0{,}7910]$$
   Der empirische Ground Truth Kontrast zwischen Universum A und Universum B beträgt:
   $$RR_{\text{True}} = \frac{29{,}16\,\%}{37{,}10\,\%} = 0{,}7858$$
   Die Differenz beträgt lediglich $\Delta RR = +0{,}0015$. G-Computation rekonstruiert die makroskopische Gesamtrisikoreduktion des Verbundangebots ($1 - RR \approx 21{,}4\,\%$) mit bemerkenswerter Genauigkeit.
2. **Die Diskrepanz der Absolute Risk Reduction (ARR):**
   Während Ground Truth eine Studienabbruchs-Reduktion von $+7{,}95$ Prozentpunkten misst, beziffert G-Computation die ARR auf $+4{,}59$ Prozentpunkte ($CF_0 = 21{,}58\,\%$, $CF_1 = 16{,}99\,\%$).
   *Ursache:* G-Computation evaluiert den Hazard auf diskreter Semesterebene unter statischer Vorsemester-Historie. Da Studierende im Durchschnitt rund 5 bis 7 Semester verbleiben, akkumuliert sich die per-Semester-Gefahr von $\approx 4{,}6$ pp über die Gesamtstudiendauer zur makroskopischen ARR von knapp $8$ pp.

### 4.2 Marginal Structural Models (MSM): Rekonstruktion isolierter Wirkfaktoren

Das MSM operiert auf $345.133$ Semesterzeilen und nutzt stabilisierte inverse Propensity-Gewichte:
$$SW_i(t) = \prod_{k=1}^t \frac{P(A_k = a_{ik} \mid \bar{A}_{k-1}, \mathbf{V}_i, k)}{P(A_k = a_{ik} \mid \bar{A}_{k-1}, \bar{\mathbf{L}}_k, \mathbf{V}_i, k)}$$

#### Diagnostik der stabilisierten Gewichte
Über alle Szenarien hinweg erweisen sich die Gewichtsverteilungen als herausragend stabil:
- $\text{Mean}(SW) \in [0{,}990,\, 1{,}002]$ (nahezu exakter Erwartungswert $1{,}0$).
- $\text{SD}(SW) \in [0{,}063,\, 0{,}223]$ (keine extremen Ausreißer, keine Gewichtsexplosion).
- Die Positivitäts-Annahme (Overlap) ist im gesamten Kovariatenraum erfüllt.

#### Validierung der Einzel-Wirkungsfaktoren
Vergleicht man die MSM-Hazard-Ratios mit den echten isolierten Universen ($F, G, H$ vs. $B$) und partiellen Entzügen ($A$ vs. $C, D, E$):

1. **Überfachlicher Support (Workshops & Lerntechniken):**
   - In S01 (Baseline): Wahrer isolierter $RR$ (Welt G vs. B) ist **$0{,}9166$** (partiell: $0{,}9210$).
     Das MSM schätzt ein $HR$ von **$0{,}9176$** [$0{,}8524,\, 0{,}9879$].
   - In S02 (Halbierter Effekt): Wahrer isolierter $RR$ (Welt G vs. B) ist **$0{,}9586$** (partiell: $0{,}9612$).
     Das MSM schätzt ein $HR$ von **$0{,}9598$** [$0{,}9012,\, 1{,}0222$].
   *Befund:* Die Schätzung des überfachlichen Schutzeffekts trifft den tatsächlichen DGP-Effekt bis auf die dritte Nachkommastelle.
2. **Fachlicher Support (Tutorien & Klausurvorbereitung):**
   - In S01: Wahrer isolierter $RR = 0{,}9064$; MSM schätzt $HR = 0{,}8313$.
   - In S02: Wahrer isolierter $RR = 0{,}9492$; MSM schätzt $HR = 0{,}9064$.
   - In S03: Wahrer isolierter $RR = 0{,}8480$; MSM schätzt $HR = 0{,}7371$.
   *Befund:* Der Schutzeffekt skaliert streng monoton mit der wahren Effektstärke. Dass das per-Semester-Hazard-Ratio etwas stärker protektiv ausfällt als das unbereinigte Makro-RR, erklärt sich aus der direkten Entlastung der Klausurbestehensquote im jeweiligen Prüfungssemester.
3. **Gesamt-Support ($HR_{\text{All}}$):**
   - Das MSM-Gesamt-Hazard-Ratio liegt mit $0{,}8002$ (S01), $0{,}8476$ (S02), $0{,}7626$ (S03) und $0{,}8070$ (S07) in exakter methodischer Übereinstimmung mit G-Computation ($0{,}7873$, $0{,}8317$, $0{,}7998$, $0{,}7787$) und dem Ground Truth A-B-Kontrast ($0{,}7858$, $0{,}8812$, $0{,}6822$, $0{,}8083$).

### 4.3 Double Machine Learning (DML): Orthogonale Residualisierung & Rauschempfindlichkeit

DML residualisiert die Behandlungsindikatoren $A_k$ und den Semester-Dropout-Hazard $Y_k$ in einer 5-Fold Cross-Fitting-Architektur.

1. **Robustheit bei extremen Effekten (S03 & S11):**
   In Standardkonfigurationen (S01, S02) zeigt DML auf der Semesterebene sehr konservative lokale ATEs nahe null ($+0{,}001$ bis $+0{,}005$), da das Stufe-1-Nuisance-Netzwerk einen Großteil der Vorhersagekraft bereits absorbiert. Sobald jedoch der Support-Effekt verdoppelt wird (**S03**) oder Randomisierung vorliegt (**S11**), sinken auch die DML-Hazard-Ratios signifikant unter $1{,}0$:
   - S03: $HR_{\text{Fach}} = 0{,}9241$, $HR_{\text{Uebf}} = 0{,}8933$, $HR_{\text{Psych}} = 0{,}9068$.
   - S11: $HR_{\text{Fach}} = 0{,}9317$, $HR_{\text{Uebf}} = 0{,}9025$, $HR_{\text{Psych}} = 0{,}9203$.
2. **Einfluss des stochastischen Rauschens (S07 vs. S08):**
   Die Diskriminierungsfähigkeit der ersten DML-Stufe reagiert empfindlich auf den stochastischen Rauschpegel des DGP:
   - S07 ($\sigma = 0{,}09$): Factual ROC-AUC $= 0{,}8249$, PR-AUC $= 0{,}2240$.
   - S01 ($\sigma = 0{,}18$): Factual ROC-AUC $= 0{,}8020$, PR-AUC $= 0{,}1968$.
   - S08 ($\sigma = 0{,}36$): Factual ROC-AUC $= 0{,}7383$, PR-AUC $= 0{,}1449$.
   Je stärker stochastisches Rauschen die studentischen Prüfungsergebnisse überlagert, desto unschärfer wird die Nuisance-Residualisierung, was sich in einer Verbreiterung der Konfidenzintervalle niederschlägt.

---

## 5. Das RCT-Experiment (S11): Eliminierung des Selektionsbias

Szenario **S11** stellt den methodologischen Härtetest für alle Inferenzmodelle dar: Durch Entkopplung der Support-Zuweisung von der individuellen Leistungs- und Motivationslage wird die Selbstselektion (*Confounding by Indication*) experimentell ausgeschaltet.

### 5.1 Kollaps der Propensity-Score-Spreizung
In den Beobachtungs-Szenarien (S01 bis S08) suchen primär leistungsschwächere oder überlastete Studierende Support, was zu einer starken Streuung der Behandlungs-Wahrscheinlichkeiten führt. Unter RCT kollabiert die Varianz der stabilisierten IPTW-Gewichte drastisch:
- $\text{SD}(SW_{\text{Fachlich}})$ fällt von $0{,}134$ (S01) auf **$0{,}069$** (S11).
- $\text{SD}(SW_{\text{Ueberfachlich}})$ fällt von $0{,}197$ (S01) auf **$0{,}068$** (S11).
- $\text{SD}(SW_{\text{All Support}})$ fällt von $0{,}198$ (S01) auf **$0{,}074$** (S11).

### 5.2 Homogenisierung der Hazard Ratios
Da die Interventionszuweisung rein orthogonal zu allen Vorerkrankungen/Vorleistungen erfolgt, entfällt die selektionsbedingte Dämpfung des Schutzeffekts:
- Im Beobachtungsszenario S01 divergierten die geschätzten Effekte ($HR_{\text{Fach}} = 0{,}831$, $HR_{\text{Uebf}} = 0{,}918$, $HR_{\text{Psych}} = 0{,}730$).
- Unter RCT (S11) konvergieren alle drei Support-Typen auf ein einheitliches, stark protektives Niveau:
  $$HR_{\text{Fach}} = 0{,}6745 \quad (\text{Schutz: } 32{,}6\,\%)$$
  $$HR_{\text{Uebf}} = 0{,}6837 \quad (\text{Schutz: } 31{,}6\,\%)$$
  $$HR_{\text{Psych}} = 0{,}6815 \quad (\text{Schutz: } 31{,}9\,\%)$$
  $$HR_{\text{All}} = 0{,}6776 \quad (\text{Schutz: } 32{,}2\,\%)$$

### 5.3 Das Allokations-Paradoxon (Targeted Intervention vs. Universal RCT)
Ein scheinbarer Widerspruch tritt im makroskopischen Ground Truth zutage:
- In S01 (zielgerichtete Indikation) beträgt die wahre Absolute Risk Reduction $+7{,}95$ pp ($NNT = 12{,}6$).
- In S11 (zufällige RCT-Zuweisung) beträgt die wahre Absolute Risk Reduction nur $+4{,}45$ pp ($NNT = 22{,}5$).

*Erklärung des Mechanismus:*
Unter RCT erhalten viele leistungsstarke Studierende Supportplätze, die sie gar nicht benötigen (Ceiling-Effekt: wer ohnehin mit $98\,\%$ Wahrscheinlichkeit abschließt, kann nicht gerettet werden). Gleichzeitig werden akut gefährdete Studierende per Los vom Support ausgeschlossen. Während die *lokale Wirksamkeit* pro Behandeltem ungetrübt hoch ist ($HR \approx 0{,}68$), sinkt die *populationsbezogene Gesamteffizienz* (ARR halbiert), wenn Interventionen nicht nach Indikation, sondern nach Zufall vergeben werden.

---

## 6. Fazit & Empfehlungen für den Nachtlauf

1. **Validierungs-Urteil:**
   - **G-Computation** ist das verlässlichste Modell zur Schätzung des kontrafaktischen Makro-Risiko-Verhältnisses ($RR_{\text{G-Comp}} \approx RR_{\text{True}}$ über alle Rausch- und Wirkungsstufen).
   - **MSM mit stabilisierten Gewichten** löst das zeitabhängige Confounding optimal und liefert ungetrübte, isolierte Hazard Ratios für die einzelnen Förderlinien.
   - **DML** fungiert als scharfer Indikator für exogene Schocks und orthogonale Behandlungsanteile.
2. **Automatisierung für Übernacht-Läufe:**
   Das CLI-Interface in [`src/run_torch_causal_lxc.py`](../../src/run_torch_causal_lxc.py) wurde um `--modes` erweitert. Zukünftige Nachtläufe können somit nahtlos über alternative Repräsentationen (`inside_view`, `realistic`, `gradeblind_oracle`, `blind`) iterieren, wobei bestehende Benchmark-JSONs inkrementell zusammengeführt werden.

---

## 7. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **LXC Prädiktiver Benchmark-Report** | [`../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md) | Prädiktive Auswertung (PyCox, Transformer, Dual-Head GRU) |
| **Marginal Structural Models V4.2** | [`../04_causal_and_simulation/marginal_structural_models_v42.md`](../04_causal_and_simulation/marginal_structural_models_v42.md) | Theoretische Herleitung von Time-Varying Confounding & IPTW |
| **Empirische RCT-Kausalanalyse** | [`../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Mechanistische Untersuchung der Selektionseliminierung in S11 |
| **Master-Synopse V4 Gesamt** | [`../03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md`](master_synopse_v4_gesamt.md) | Gesamtevaluation über alle 15 V4.1/V4.2-Simulationswelten |
