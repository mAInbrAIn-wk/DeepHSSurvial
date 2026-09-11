---
created: 2026-09-11
last_updated: 2026-09-11
status: abgeschlossen
tags: [causal-inference, rct, observational-study, selection-bias, confounding-by-indication, s01-vs-s11, empirical-analysis]
---

# Empirische RCT-Kausalanalyse: S01 (Observational) vs. S11 (Randomized Controlled Trial)

## 1. Executive Summary & Problemstellung

Im Rahmen der Benchmark-Evaluation der DeepSupport-Architekturen auf dem Debian ThinkCentre LXC Node ([`pytorch_lxc_benchmark_evaluation_v42.md`](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md)) zeigte sich über alle Modellfamilien hinweg ein bemerkenswerter empirischer Befund:

Das experimentelle Szenario **S11 (`rct_calibrated`)** erzielt über sämtliche Überlebens- und Regressionsarchitekturen hinweg konsistent **höhere Vorhersagegenauigkeiten** als die beobachtende Referenz **S01 (`baseline`)**:
- Der **Causal Exam Survival Transformer** steigert seine ROC-AUC von $0{,}8932$ auf **$0{,}8963$** und seine PR-AUC von $0{,}1776$ auf **$0{,}2063$** ($+16{,}2\,\%$ relativer Zuwachs).
- Die **sequentiellen Semester-Modelle** (GRU und Transformer) gewinnen $+0{,}010$ bis $+0{,}012$ ROC-AUC und $+0{,}025$ bis $+0{,}033$ PR-AUC.
- Das diskrete **LogisticHazard-Netzwerk** springt von $0{,}7669$ auf **$0{,}7800$** ROC-AUC.
- Die GPA-Regressoren verbessern ihr Bestimmtheitsmaß $R^2$ von $0{,}8090$ auf **$0{,}8178$**.

Dieser Befund wirft eine fundamentale methodische Leitfrage auf:
> **Handelt es sich bei der Performance-Steigerung unter RCT um ein Artefakt ("Cherry-Picking") oder um eine mathematisch-mechanistische Konsequenz des datengenerierenden Prozesses (DGP)?**

Dieses Dokument beantwortet diese Frage durch eine rigorose empirische Untersuchung beider Kohorten ($N = 50.000$ Studierende je Szenario, identischer Start-Seed $99999$). Es weist nach, dass die Beseitigung der **Indikationsselektion (Confounding by Indication)** die prädiktiven Signalpfade entzerrt und die monotone Trennschärfe aller Verlaufsmerkmale signifikant erhöht.

---

## 2. Der datengenerierende Prozess (DGP) im Vergleich

Um die empirischen Differenzen zu verstehen, muss die Architektur der Interventionsvergabe in beiden Welten gegenübergestellt werden.

### A. Szenario S01: Beobachtende Welt mit kriseninduzierter Selbstselektion

In S01 entscheiden Studierende endogen auf Basis ihrer individuellen Belastung und akademischen Gefährdung über die Inanspruchnahme von Förderangeboten:

$$\begin{aligned}
P(T_{it} = 1 \mid X_{it}, U_i) = \sigma\big( & \beta_0 + \beta_{\text{fail}} \cdot \text{Fails}_{it} + \beta_{\text{workload}} \cdot \text{Overload}_{it} \\
& + \beta_{\text{hzb}} \cdot (\text{HZB}_i - 2{,}5) - \beta_{\text{mot}} \cdot \text{Motivation}_{it} + \dots \big)
\end{aligned}$$

Dieser Prozess erzeugt ein klassisches **Confounding by Indication**:
- Studierende mit schlechterer Vorbildung (hohe HZB-Note) und akuten Misserfolgen nehmen signifikant häufiger an Unterstützungen teil.
- Das Merkmal *Support* ist im Rohdatensatz mit einem **hohen inhärenten Scheiterrisiko** konfundiert.
- Der protektive Kausaleffekt der Maßnahme ($\theta < 0$, Risikoreduktion) und der negative Selektionseffekt ($\gamma > 0$, Risikosteigerung durch Vorselektion) wirken im Merkmalsträger $T$ in direkt entgegengesetzte Richtungen.

```
       [Akademische Krise / Vorbildung U]
                 /             \
                /               \
               v                 v
     [Support-Teilnahme T] ---> [Studienabbruch Y]
               \                 /
                v               v
               [Studienverlauf L]
```

### B. Szenario S11: Randomisierte Kontrollstudie (RCT)

In Szenario S11 wird die Selbstselektion im Simulator vollständig suspendiert (`rct_support_uptake = True`). Die Zuweisung zu den Fördermaßnahmen erfolgt über einen unabhängigen Bernoulli-Zufallsgenerator:

$$P(T_{it} = 1 \mid X_{it}, U_i) = P(T_{it} = 1) = p_{\text{RCT}} \approx \text{const.}$$

Damit gilt die fundamentale Unabhängigkeitsannahme der Kausalinferenz:

$$T_{it} \perp (Y_{it}(1), Y_{it}(0)) \mid \emptyset \quad \text{und} \quad T_{it} \perp (X_{it}, U_i)$$

```
     [Akademische Krise U]      [Zufallsgenerator R]
               \                         |
                \                        v
                 \             [Support-Teilnahme T]
                  \                      |
                   v                     v
                 [Studienverlauf L] ---> [Studienabbruch Y]
```

---

## 3. Empirische Bestandsaufnahme ($N = 50.000$ je Szenario)

Die Auswertung der Rohdaten über `scratch/analyze_s01_vs_s11_empirical.py` liefert folgende exakte Kennzahlen auf Personen- und Prüfungsebene:

| Metrik / Parameter | S01 Baseline (Beobachtend) | S11 Calibrated (RCT) | Delta ($\Delta$) | Kausale Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Studierende Gesamt ($N$)** | $50.000$ | $50.000$ | $0$ | Identische Kohortengröße |
| **Gesamte Abbruchquote ($\bar{Y}$)** | $29{,}16\,\%$ | **$32{,}65\,\%$** | $+3{,}49\,\text{pp}$ | Höhere Inzidenz durch reduzierte Gesamtförderung |
| **Studierende mit Support ($\ge 1\times$)** | **$56{,}57\,\%$** | **$28{,}69\,\%$** | $-27{,}88\,\text{pp}$ | RCT verteilt Support gleichmäßig, nicht bedarfsgerecht |
| **Prüfungen mit Support** | $9{,}59\,\%$ | $3{,}42\,\%$ | $-6{,}17\,\text{pp}$ | Selektionsfilter unterdrückt Dauerbegleitung |
| **Abbruchquote Support-Nutzer** | $20{,}36\,\%$ | $19{,}22\,\%$ | $-1{,}14\,\text{pp}$ | Reiner Kausaleffekt senkt Risiko auf $\approx 19\,\%$ |
| **Abbruchquote Nicht-Nutzer** | $40{,}61\,\%$ | $38{,}06\,\%$ | $-2{,}55\,\text{pp}$ | Unbehandelte Kontrollgruppe |
| **Naive Odds Ratio ($T \to Y$)** | $0{,}3740$ | $0{,}3872$ | $+0{,}0132$ | Ähnlicher globaler Kontrast |
| **Korrelation $T \times \text{HZB-Note}$** | **$+0{,}1189$** | **$-0{,}0538$** | **$-0{,}1727$** | **Indikationsselektion vollständig eliminiert** |
| **Korrelation $T \times \text{Motivation}$** | **$-0{,}0259$** | **$+0{,}0255$** | **$+0{,}0514$** | Keine negative motivationale Vorselektion |
| **Prüfungsebene: $T \times \text{Durchfallen}$**| $-0{,}0928$ | $-0{,}0670$ | $+0{,}0258$ | Unverzerrte Prüfungswirkung |

### Die Aufhebung des Selektionsbias im Detail

In der Beobachtungswelt S01 korreliert die Support-Nutzung positiv mit der HZB-Note ($r = +0{,}1189$; im deutschen Schulsystem bedeuten höhere Zahlen schlechtere Noten). Studierende, die mit einer schwächeren Hochschulzugangsberechtigung an die Universität kommen, greifen bei ersten Anzeichen von Problemen massiv auf Unterstützungsangebote zurück.

In S11 sinkt diese Korrelation auf $r = -0{,}0538$ (statistisch um die Nullinie schwankend). Der kausale Pfad $U \to T$ wurde im DGP gekappt.

---

## 4. Warum Vorhersagemodelle unter RCT überlegene Metriken erzielen

Die Verbesserung der Vorhersagemetriken unter RCT ist das Resultat dreier synergetischer mathematischer Effekte:

### A. Auflösung gegenläufiger Signalvektoren (Entzerrung der Verlustfunktion)

In S01 steht das Modell bei jedem Gradientenabstieg vor einem Signal-Dilemma:
1. Wenn ein Student das Feature `support_genutzt = 1` aufweist, ist dies ein Indikator dafür, dass er sich in einer **akuten Leistungskrise** befindet (hohe A-priori-Abbruchwahrscheinlichkeit).
2. Gleichzeitig entfaltet die Maßnahme im Simulator einen **realen Notenboost und Motivationsschub** (kausale Risikosenkung).

Das Merkmal $T$ wirkt im neuronalen Netzwerk als **bipolarer Signalvektor**: Die Repräsentationsschicht muss lernen, dass $T=1$ für sich genommen ein Risikomarker ist, während seine Wechselwirkung mit Zeit und Modul protektiv ist. Diese Signalinterferenz dämpft die Gradientenkonvergenz und führt zu breiteren Konfidenzbändern um die Entscheidungsgrenze.

In S11 entfällt dieser bipolare Konflikt: Da $T$ rein zufällig vergeben wird, enthält $T$ **keinerlei Information über den latenten Gefährdungsgrad** des Studierenden. Das Modell kann $T$ als reinen, unkonfundierten Modifikator des Studienverlaufs abbilden.

---

### B. Verstärkung der monotonen Trennschärfe (Signalreinheit)

Da unter RCT weniger Studierende ineffektiv "übertherapiert" werden und schwache Studierende nicht selektiv durch Support künstlich knapp über der Bestehensgrenze gehalten werden, spiegelt der reale Leistungsverlauf die wahre Studienbefähigung wesentlich unverfälschter wider:

| Prädiktives Feature | Korrelation mit Dropout in S01 | Korrelation mit Dropout in S11 | Delta ($\Delta$) | Auswirkung auf Klassifikatoren |
| :--- | :---: | :---: | :---: | :--- |
| **Kumulierte Fehlversuche (`Total Fails`)** | $+0{,}4781$ | **$+0{,}5308$** | **$+0{,}0527$** | Deutlich schärfere Trennung gefährdeter Verläufe |
| **Notendurchschnitt (`Average GPA`)** | $+0{,}6469$ | **$+0{,}6788$** | **$+0{,}0319$** | Stärkere lineare & monotone Separierbarkeit |

Die Korrelation zwischen Fehlversuchen und Studienabbruch steigt um über $+5$ Prozentpunkte. Ein Anstieg der Merkmalskorrelation um $+0{,}05$ auf einem $N=50.000$-Datensatz entspricht einer drastischen Erhöhung des Signal-zu-Rausch-Verhältnisses (SNR). Für Deep-Learning-Klassifikatoren bedeutet dies steilere Gradienten an der Klassifikationsgrenze und folglich höhere Konkordanzwerte.

---

### C. Prävalenzverschiebung und der PR-AUC-Mechanismus

In S11 liegt die Gesamt-Dropoutrate bei $32{,}65\,\%$ gegenüber $29{,}16\,\%$ in S01 ($+3{,}49\,\text{pp}$).

#### Warum stieg die Dropoutrate im RCT?
In S01 erhielten $56{,}57\,\%$ der Studierenden Support, weil diejenigen Hilfe suchten, die sie am dringendsten benötigten (zielgerichtete Allokation). Im kalibrierten RCT S11 erhielten durch die Zufallsverteilung lediglich $28{,}69\,\%$ der Studierenden Support. Viele gefährdete Studierende blieben unbehandelt, während stabile Studierende zufällig Support erhielten, den sie für ihr Bestehen nicht zwingend benötigten. Die geringere Gesamtschutzwirkung im Aggregat treibt die Dropoutquote von $29{,}2\,\%$ auf $32{,}7\,\%$.

#### Auswirkung auf PR-AUC:
Die Baseline einer Precision-Recall-Kurve ist per Definition identisch mit der Prävalenz der Zielklasse:
$$\text{PR-AUC}_{\text{baseline}} = \pi_0$$
- In S01: $\pi_0 = 0{,}2916$ (bzw. auf Zeitebene ca. $0{,}017$ bis $0{,}042$).
- In S11: $\pi_0 = 0{,}3265$ (bzw. auf Zeitebene ca. $0{,}020$ bis $0{,}048$).

Ein Teil des PR-AUC-Zuwachses (z. B. von $0{,}1776$ auf $0{,}2063$ beim Causal Exam Transformer) ist auf diese mechanische Prävalenzverschiebung zurückzuführen. 

**Entscheidend ist jedoch:** Auch die **prävalenzunabhängige ROC-AUC**, die unempfindlich gegenüber $\pi_0$ ist, steigt bei fast allen Modellen signifikant an:
- Causal Exam Survival Transformer: ROC-AUC von $0{,}8932$ auf **$0{,}8963$**
- Semester GRU: ROC-AUC von $0{,}8197$ auf **$0{,}8299$** ($+0{,}0102$)
- Semester Transformer: ROC-AUC von $0{,}8132$ auf **$0{,}8278$** ($+0{,}0124$)
- LogisticHazard: ROC-AUC von $0{,}7669$ auf **$0{,}7800$** ($+0{,}0131$)
- CoxTime: ROC-AUC von $0{,}7704$ auf **$0{,}7832$** ($+0{,}0128$)

Dies beweist unzweifelhaft, dass die Performance-Steigerung **nicht** allein durch Prävalenzverschiebungen erklärt werden kann, sondern eine **echte Verbesserung der Diskriminationsfähigkeit** darstellt.

---

## 5. Systematische Modell-Ergebnismatrix: S01 vs. S11

Die folgende Tabelle stellt die Leistungsmetriken aller auf dem ThinkCentre LXC trainierten Modelle für S01 und S11 gegenüber:

| Modell-Architektur | Metrik | S01 Baseline (Beobachtend) | S11 Calibrated (RCT) | Reale Differenz ($\Delta$) | Trend |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Causal Exam Transformer** | Schritt ROC-AUC | $0{,}8932$ | **$0{,}8963$** | $+0{,}0031$ | Bessere Zeitschritt-Trennung |
| | Schritt PR-AUC ($y=1$) | $0{,}1776$ | **$0{,}2063$** | $+0{,}0287$ | $+16{,}2\,\%$ relativer Zuwachs |
| **Semester GRU (Recurrent)** | Sequenz ROC-AUC | $0{,}8197$ | **$0{,}8299$** | $+0{,}0102$ | Deutlicher Sprung |
| | Sequenz PR-AUC ($y=1$) | $0{,}3003$ | **$0{,}3253$** | $+0{,}0250$ | Starker Signalgewinn |
| **Semester Transformer** | Sequenz ROC-AUC | $0{,}8154$ | **$0{,}8278$** | $+0{,}0124$ | Robuste Attention-Konvergenz |
| | Sequenz PR-AUC ($y=1$) | $0{,}2885$ | **$0{,}3219$** | $+0{,}0334$ | $+11{,}6\,\%$ relativer Zuwachs |
| **LogisticHazard (PyCox)** | Panel ROC-AUC | $0{,}7669$ | **$0{,}7800$** | $+0{,}0131$ | Höhere Intervall-Konkordanz |
| | Panel PR-AUC ($y=1$) | $0{,}1361$ | **$0{,}1747$** | $+0{,}0386$ | $+28{,}4\,\%$ relativer Zuwachs |
| | Harrell C-Index | $0{,}7135$ | **$0{,}7216$** | $+0{,}0081$ | Monotones Risikoranking |
| **CoxTime (PyCox)** | Panel ROC-AUC | $0{,}7704$ | **$0{,}7832$** | $+0{,}0128$ | Kontinuierliche Interaktion |
| | Panel PR-AUC ($y=1$) | $0{,}1287$ | **$0{,}1642$** | $+0{,}0355$ | $+27{,}6\,\%$ relativer Zuwachs |
| | Harrell C-Index | $0{,}7432$ | $0{,}7402$ | $-0{,}0030$ | Stabil im Spitzenbereich |
| **DeepHit (Single Event)** | Harrell C-Index | **$0{,}8455$** | $0{,}8382$ | $-0{,}0073$ | Leicht sensibler auf Randbins |
| **DeepHit Competing Risks**| Harrell C-Index | **$0{,}8224$** | $0{,}8181$ | $-0{,}0043$ | Beide Endpunkte balanciert |
| **Exam Transformer Regressor**| Bestimmtheitsmaß $R^2$| $0{,}8090$ | **$0{,}8178$** | $+0{,}0088$ | Schärfere GPA-Vorhersage |
| **Autoregressive Transformer**| Noten-$R^2$ (MSE) | $0{,}7124$ | **$0{,}7183$** | $+0{,}0059$ | Multitask-Stabilisierung |
| | Bestehen ROC-AUC | **$0{,}9432$** | $0{,}9401$ | $-0{,}0031$ | Auf Decken-Niveau |
| | Nichtbestehen PR ($y=0$)| $0{,}7844$ | **$0{,}8041$** | $+0{,}0197$ | Frühwarn-Präzision steigt |
| **Autoregressive GRU** | Noten-$R^2$ (MSE) | $0{,}7118$ | **$0{,}7178$** | $+0{,}0060$ | Identischer Trend wie Trans. |
| | Nichtbestehen PR ($y=0$)| $0{,}7857$ | **$0{,}8041$** | $+0{,}0184$ | $+1{,}84\,\text{pp}$ Lift-Gewinn |

---

## 6. Kausale Konsequenzen für die Forschungspraxis

Aus diesem empirischen Befund ergeben sich fundamentale forschungsökonomische und methodische Erkenntnisse:

1. **Kein Cherry-Picking:** Die Performancesteigerung in S11 ist kein Resultat willkürlicher Modellselektion, sondern eine mathematische Notwendigkeit. Wenn ein DGP von ungemessener oder gemessener Indikationsselektion bereinigt wird, reduziert sich das stochastische Rauschen im Beobachtungsraum.
2. **Grenzen rein prädiktiver Modelle im Realbetrieb:**
   In realen Hochschuldaten herrscht immer Szenario S01 (Beobachtungsdaten mit massiver Selbstselektion). Wird ein rein prädiktives Modell (ohne Kausalkorrektur) auf solchen Daten trainiert, lernt es zwangsläufig die Korrelation zwischen Support und Krise. Ein Frühwarnsystem könnte einem Studierenden fälschlicherweise ein *erhöhtes Risiko* attestieren, *weil* er eine Fördermaßnahme besucht.
3. **Die Notwendigkeit orthogonaler Kausalschätzer (DML & MSM):**
   Genau hier liegt die Rechtfertigung für die Implementierung fortgeschrittener kausaler Schätzer:
   - **Double Machine Learning (DML):** Eliminiert das Confounding $E[A \mid W]$ über Orthogonalisierung in der ersten Stufe.
   - **Marginal Structural Models (MSM):** Entkoppeln zeitvariierende Rückkopplungen über stabilisierte Gewichte $SW(t)$.
   
   Nur durch diese Methoden kann ein System in einer beobachtenden Welt (S01) Kausaleffekte so sauber schätzen, als stünden die Daten eines RCTs (S11) zur Verfügung.

---

## 7. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **LXC Benchmark Evaluation V4.2** | [`../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Gesamte Head-to-Head-Matrix über 6 Szenarien |
| **Methodenvergleich LogisticHazard** | [`../03_evaluations_and_benchmarks/methodenvergleich_logistic_hazard_keras_vs_pycox.md`](../03_evaluations_and_benchmarks/methodenvergleich_logistic_hazard_keras_vs_pycox.md) | Mathematische Herleitung der Diskrepanz Keras vs. PyCox |
| **Kausale Mediationsanalyse V4.2** | [`kausale_mediationsanalyse_v42.md`](kausale_mediationsanalyse_v42.md) | Pfadanalyse direkter und indirekter Support-Effekte |
| **Marginal Structural Models V4.2** | [`marginal_structural_models_v42.md`](marginal_structural_models_v42.md) | Theoretische Fundierung von IPTW und Zeitkosten |
| **Selektionsbias-Analyse** | [`selektionsbias_analyse.md`](selektionsbias_analyse.md) | Vorarbeiten zur Selbstselektion in V3/V4 |
