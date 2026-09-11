---
created: 2026-09-11
last_updated: 2026-09-11
status: abgeschlossen
tags: [evaluation-metrics, model-comparison, comparability, survival-analysis, deep-learning, cross-model-benchmarking, pr-auc, roc-auc, brier-score, harrell-c]
---

# Vergleichbarkeit von Evaluationsmetriken über heterogene Modellklassen

## 1. Executive Summary & Problemstellung

Im DeepSupport-Projekt werden unterschiedlichste Machine-Learning- und Deep-Learning-Architekturen nebeneinander evaluiert. Die Palette reicht von klassischen biostatistischen Panel-Survival-Modellen über sequentielle Recurrent- und Attention-Netzwerke bis hin zu autoregressiven Dual-Head-Modellen und Landmark-Mehrklassenklassifikatoren.

In Benchmark-Übersichten (wie in [`pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md)) werden diese Modelle häufig in zusammenfassenden Tabellen gegenübergestellt. 

> [!WARNING]
> **Das "Äpfel-und-Birnen"-Dilemma:**
> Ein naiver Vergleich standardisierter Metriken wie **ROC-AUC**, **PR-AUC**, **Brier Score** oder **$R^2$** über unterschiedliche Modellklassen hinweg ist methodisch hochgradig fehleranfällig. Ein ROC-AUC-Wert von $0{,}80$ in einem Semester-Panel misst etwas fundamental anderes als ein ROC-AUC-Wert von $0{,}80$ in einem sequentiellen Prüfungsmodell oder einem Landmark-Klassifikator.

Dieses Dokument liefert die theoretische und methodische Grundlage zur **fairen Kontextualisierung und verlässlichen Interpretation** aller Evaluationsmetriken in DeepSupport. Es definiert fünf fundamentale Dimensionen der Nicht-Vergleichbarkeit und etabliert eine verbindliche **Harmonisierungsmatrix**.

---

## 2. Die fünf Dimensionen der metrischen Nicht-Vergleichbarkeit

### Dimension 1: Aggregationsebene & Evaluierungsstichprobe ($N_{\text{eval}}$)

Jede Modellfamilie operiert auf einer spezifischen Granularitätsstufe der Daten:

| Modellfamilie | Beobachtungseinheit | Typische Stichprobengröße ($N_{\text{eval}}$) | Zielereignis ($Y$) |
| :--- | :--- | :---: | :--- |
| **Landmark-Modelle ($L=2$)** | Student ($i$) | $\approx 50.000$ | Tritt bis Semester 6 ein Abbruch ein? |
| **Panel-Survival (Keras & PyCox)** | Person-Semester ($i, t$) | $\approx 345.000$ | Bricht Student $i$ in genau Semester $t$ ab? |
| **Autoregressoren (Next-Exam)** | Prüfungsschritt ($i, k$) | $\approx 802.000$ | Fällt Student $i$ bei Klausur $k+1$ durch / Note? |
| **Causal Exam Survival** | Prüfungsschritt ($i, k$) | $\approx 802.000$ | Führt Klausur $k$ zum Studienabbruch? |

#### Warum dieser Unterschied Vergleiche verzerrt:
1. **Intra-Klassen-Korrelation:** In Prüfungs- und Semesterdaten stammen mehrere Zeilen vom selben Studierenden. Zeitschritt-Modelle müssen die dynamische Veränderung im Studienverlauf vorhersagen. 
2. **Konditionale Überlebenden-Population:** Ein Modell auf Semesterebene evaluiert in Semester 6 nur diejenigen Studierenden, die Semester 1 bis 5 überlebt haben ($N_{t=6} \approx 28.000$). Ein Landmark-Modell zu Studienbeginn bewertet die unkonditionierte Gesamtkohorte ($N = 50.000$).

---

### Dimension 2: Basisprävalenz $\pi_0$ und das PR-AUC-Dilemma

Während die **ROC-AUC** unempfindlich gegenüber der Klassenverteilung ist, hängt die **Precision-Recall AUC (PR-AUC)** mathematisch zwingend von der **Basisprävalenz $\pi_0$** ab:

$$\text{PR-AUC}_{\text{Zufall}} = \pi_0 = \frac{P}{P + N}$$

Vergleicht man die PR-AUC-Werte verschiedener Modellklassen, schwankt die Basisprävalenz im DeepSupport-Ökosystem um mehr als den **Faktor 50**:

```
Landmark-Dropout (L=2)          : [██████████████] pi0 = 0.2916 (29.2 %)
Semester-Abbruch (Panel)        : [██]            pi0 = 0.0420 (4.2 %)
Prüfungs-Abbruch (Exam Survival): [█]             pi0 = 0.0171 (1.7 %)
Prüfungs-Nichtbestehen (Fail)   : [████████]       pi0 = 0.1640 (16.4 %)
Prüfungs-Bestehen (Pass)        : [████████████████████████████████████████] pi0 = 0.8360 (83.6 %)
```

#### Der relative Lift als unverzichtbare Vergleichsmetrik:
Um Modelle über unterschiedliche Prävalenzen hinweg vergleichbar zu machen, muss der **relative Lift** herangezogen werden:
$$\text{Lift} = \frac{\text{PR-AUC}_{\text{Modell}}}{\pi_0}$$

| Modell | Zielereignis | Modell PR-AUC | Basisprävalenz $\pi_0$ | Relativer Lift | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Causal Exam Survival** | Dropout an Prüfung $k$ | **$0{,}1776$** | $0{,}0171$ | **$10{,}39\times$** | **Herausragende Trennschärfe:** Modell findet $10\times$ mehr Abbrüche als Zufall |
| **Semester GRU** | Dropout in Semester $t$ | **$0{,}3003$** | $0{,}0420$ | **$7{,}15\times$** | Sehr starkes sequentielles Frühwarnsignal |
| **Landmark MLP ($L=2$)** | Kumulativ bis Sem. 6 | **$0{,}5120$** | $0{,}2916$ | **$1{,}76\times$** | Scheinbar hoher PR-AUC-Wert, aber nur geringer relativer Zuwachs |
| **Autoregressor Pass** | Klausur bestanden ($y=1$) | **$0{,}9882$** | $0{,}8360$ | **$1{,}18\times$** | Trivialer Majority-Baseline-Effekt |
| **Autoregressor Fail** | Klausur durchgefallen ($y=0$)| **$0{,}7844$** | $0{,}1640$ | **$4{,}78\times$** | **Wahre Frühwarnleistung:** Fast $5\times$ über Zufallsniveau |

> [!IMPORTANT]
> Ein scheinbar niedriger PR-AUC-Wert von $0{,}18$ beim Prüfungs-Transformer ist ein **weitaus stärkeres Signal** ($10{,}4\times$ Lift) als ein PR-AUC-Wert von $0{,}51$ bei einem Landmark-Klassifikator ($1{,}8\times$ Lift). Tabellen, die PR-AUC ohne Ausweisung von $\pi_0$ oder Lift anführen, verleiten zu fatalen Fehlinterpretationen.

---

### Dimension 3: Momentaner Hazard $h_t$ vs. Kumulatives Risiko $F(t)$ vs. Ranking

Wie im Methodenvergleich ([`methodenvergleich_logistic_hazard_keras_vs_pycox.md`](methodenvergleich_logistic_hazard_keras_vs_pycox.md)) bewiesen, entscheidet die Zielgröße über die Interpretierbarkeit:

1. **Momentaner Hazard $h_t(x)$ (Keras extended_logistic_hazard):**
   - Gibt $P(Y_{it}=1 \mid \text{Überleben bis } t)$ an.
   - Passt exakt zum Zeilentarget $Y_{it}$ im Panel.
   - **Metrik:** ROC-AUC und PR-AUC bewerten die Klassifikation *in diesem spezifischen Zeitschritt*.
2. **Kumulatives Risiko $F(t \mid x) = 1 - S(t \mid x)$ (PyCox predict_risk):**
   - Gibt die Gesamtwahrscheinlichkeit an, bis zum Zeitpunkt $t$ ausgefallen zu sein.
   - Bestraft das Modell mit einem **Phase-Lead Bias**, wenn es gegen zeilenweise Statuswechsel $Y_{it}$ evaluiert wird (da gefährdete Studierende frühzeitig hohes $F(t)$ akkumulieren).
   - **Adäquate Metrik:** **Harrell's C-Index** (Konkordanz-Index) über die gesamte Überlebenszeit oder **Integrated Brier Score (IBS)** über alle Zeithorizonte.
3. **Koninuierlicher Hazard-Index (CoxPH & CoxTime):**
   - Der Cox-Score $\exp(g(x, t))$ ist kein Wahrscheinlichkeitswert, sondern ein relativer Risikomultiplikator gegenüber der Baseline-Hazard.
   - Er kann nicht direkt über binäre Cross-Entropy oder klassische Brier Scores bewertet werden, sondern erfordert entweder Breslow-Integrierung oder zensierungsbereinigte inverse Wahrscheinlichkeitsgewichtung (IPCW Brier Score nach Graf et al., 1999).

---

### Dimension 4: Multi-Task Trade-Offs bei Dual-Head Architekturen

Hybride Autoregressoren (wie `PyTorchAutoregressiveNextExamTransformer`) lösen simultan zwei Aufgaben über einen gemeinsamen Repräsentations-Backbone:
1. Kontinuierliche Examensnote $Y_{k+1} \in [1{,}0; 5{,}0]$ via **MSE-Loss** ($R^2$, RMSE).
2. Binäres Bestehen $P(\text{pass}_{k+1} = 1)$ via **BCEWithLogits-Loss** (ROC-AUC, PR-AUC, Brier).

#### Das Regularisierungs- und Kapazitäts-Phänomen:
- Ein reiner Single-Task-Klassifikator (der nur Bestehen/Nichtbestehen lernt) kann sich ausschließlich auf die Trennschärfe an der Schwelle von $4{,}0$ zu $5{,}0$ konzentrieren. Er erzielt unter Umständen eine minimal höhere ROC-AUC ($+0{,}002$).
- Der Dual-Head-Ansatz zwingt das Netzwerk jedoch, die **gesamte Notenmetrik** ($1{,}0$ bis $4{,}0$) linear und monoton abzubilden.
- **Konsequenz:** Der Dual-Head-Klassifikator ist durch die Noten-Hilfsaufgabe physikalisch regularisiert. Er verliert minimale Trennschärfe an extremen Rändern, generalisiert jedoch deutlich robuster auf Out-of-Distribution-Daten und liefert überlegene Wahrscheinlichkeitskalibrierungen (Brier Score sinkt von $0{,}0766$ auf $0{,}0668$).

---

### Dimension 5: Kausale Validität vs. Prädiktive Genauigkeit (Das Confounding-Paradoxon)

In der empirischen RCT-Kausalanalyse ([`../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md)) wurde nachgewiesen:

> [!CAUTION]
> **Hohe prädiktive Metriken garantieren KEINE kausale Korrektheit!**
> In Beobachtungsdaten (S01) nutzen rein prädiktive Modelle (wie MLP oder XGBoost) Confounder als hochgradig informative Prädiktoren: Ein Student, der Fördermaßnahmen besucht, wird vom Modell als krisenhaft eingestuft. Das Modell sagt den Studienabbruch mit exzellenter ROC-AUC ($> 0{,}85$) voraus – schätzt aber den kausalen Effekt der Förderung völlig falsch (möglicherweise sogar schädlich!).

- **Prädiktive Modelle (Survival / Regression):** Werden an ihrer Fähigkeit gemessen, Beobachtungen im Testset zu trennen (ROC-AUC, PR-AUC, RMSE).
- **Kausalschätzer (DML, MSM, G-Computation):** Werden an ihrer Fähigkeit gemessen, den **wahren kontrafaktischen Interventionseffekt** ($\text{ARR}$, $HR$, $RR$) unverzerrt zu schätzen.
- Eine Modellmatrix darf niemals einen DML-Schätzer gegen ein unkorrigiertes Survival-Modell anhand der ROC-AUC vergleichen.

---

## 3. Die Fair Comparison Matrix für DeepSupport

Die folgende Matrix regelt verbindlich, welche Modellklassen anhand welcher Metriken direkt miteinander verglichen werden dürfen:

| Vergleichspaar | Erlaubte direkte Metriken | Unzulässige / Irreführende Vergleiche | Notwendige Harmonisierung |
| :--- | :--- | :--- | :--- |
| **Keras LogisticHazard vs. PyTorch LogisticHazard** | Momentaner Zeitschritt ROC-AUC, Momentaner Zeitschritt PR-AUC, Brier Score | Vergleich von PyTorch `predict_risk` (kumulativ) gegen Keras `predict` (Zeilenhazard) | PyTorch muss Zeitschritt-Logit $\sigma(z_t)$ ausgeben (siehe Methodenvergleich) |
| **PyTorch LogisticHazard vs. PyTorch CoxTime / DeepHit** | Harrell C-Index, Integrierter Brier Score (IBS über $t$) | Direkte Zeilen-ROC-AUC gegen Panel-Event | Zeitabhängiges Konkordanz-Ranking über $S(t)$ |
| **Semester GRU vs. Semester Transformer** | Zeitschritt ROC-AUC, Zeitschritt PR-AUC, Brier Skill Score | Keine Einschränkung (vollständig paritätische Architektur) | Identische Sequenzmaskierung und Zeitschritt-Aggregation |
| **Next-Exam GRU vs. Next-Exam Transformer** | Noten-$R^2$, Noten-RMSE, Pass-ROC, Pass-PR, Fail-PR, Brier Score | Vergleich ausschließlich auf Pass-PR ($y=1$) ohne Fail-PR ($y=0$) | Beide Klassen müssen mit Baseline-Lift ausgewiesen werden |
| **Next-Exam Modelle vs. Semester-Survival Modelle** | Keine direkten Metrikvergleiche erlaubt | Gesamter direkter Tabellenvergleich | Vollständig unterschiedliche Zeitebenen (Klausur vs. Semester); nur qualitativer Vergleich des Frühwarn-Lifts |
| **Landmark-Modelle vs. Sequentielle Modelle** | Relativer PR-AUC-Lift über $\pi_0$ | Absoluter PR-AUC-Wert (z. B. $0{,}51$ vs. $0{,}18$) | Absoluter PR-AUC muss zwingend durch Basisprävalenz $\pi_0$ geteilt werden |
| **DML / MSM / G-Comp vs. Standard Survival** | Kausales Hazard Ratio ($HR$), Relative Risk ($RR$), Absolute Risk Reduction ($ARR$) | ROC-AUC oder PR-AUC auf Factual Data | Kausale Schätzer müssen gegen den Ground Truth aus Universum B validiert werden |

---

## 4. Leitfaden für Berichterstattung und wissenschaftliche Publikationen

Für alle künftigen Benchmarks und Berichte in DeepSupport gelten folgende Richtlinien:

1. **Keine PR-AUC ohne Baseline $\pi_0$ und Lift:**
   Jede Nennung von PR-AUC muss die Klassenprävalenz $\pi_0$ und den relativen Lift enthalten:
   $$\text{Format:} \quad \text{PR-AUC} = 0{,}1776 \quad (\pi_0 = 0{,}0171, \text{ Lift } 10{,}4\times)$$
2. **Mehrheits- und Minderheitsklasse ausweisen:**
   Bei binären Klassifikatoren mit Schiefe (z. B. Klausurprüfung mit $84\,\%$ Bestehensquote) müssen stets beide Klassen bilanziert werden:
   - Positive Klasse (Mehrheit, Bestehen): $\pi_0 = 0{,}836$
   - Negative Klasse (Minderheit, Durchfallen / Frühwarnung): $\pi_0 = 0{,}164$
3. **Kennzeichnung methodischer Modellunterschiede in Tabellen:**
   Wenn Modelle mit unterschiedlichen Zielgrößen (z. B. gepoolter Hazard vs. kumulative Intervall-PMF) in einer Gesamttabelle aufgeführt werden, müssen diese zwingend mit Fußnoten oder Sektionsüberschriften versehen werden.
4. **Verwendung von Brier Skill Scores (BSS):**
   Da der rohe Brier Score mit sinkender Prävalenz automatisch gegen Null konvergiert, sollte stets der Brier Skill Score ausgewiesen werden:
   $$\text{BSS} = 1 - \frac{\text{BS}}{\pi_0 (1 - \pi_0)}$$
   Er misst den prozentualen Kalibrierungsgewinn gegenüber einer naiven Vorhersage der Basisprävalenz.

---

## 5. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **Methodenvergleich LogisticHazard** | [`methodenvergleich_logistic_hazard_keras_vs_pycox.md`](methodenvergleich_logistic_hazard_keras_vs_pycox.md) | Mathematische Herleitung der Diskrepanz Keras vs. PyCox |
| **LXC Benchmark-Evaluation V4.2** | [`pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md) | Head-to-Head-Benchmark über 6 Szenarien |
| **Empirische RCT-Kausalanalyse** | [`../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Aufhebung des Selektionsbias und Auswirkung auf Signalreinheit |
| **Grundlagen der Survival-Analyse** | [`../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md`](../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md) | Greenwood-Formel, Zensierung und Competing Risks |
| **Kausale Mediationsanalyse V4.2** | [`../04_causal_and_simulation/kausale_mediationsanalyse_v42.md`](../04_causal_and_simulation/kausale_mediationsanalyse_v42.md) | Imai/Pearl Mediationsprüfung auf V4-Daten |
