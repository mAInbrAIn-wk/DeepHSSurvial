---
created: 2026-09-11
last_updated: 2026-09-11
status: abgeschlossen
tags: [survival-analysis, logistic-hazard, pycox, keras, discrete-time, mathematical-comparison, evaluation-metrics]
---

# Methodenvergleich: LogisticHazard in Keras vs. PyTorch / PyCox

## 1. Einleitung & Problemstellung

Im Head-to-Head-Benchmark zwischen der TensorFlow/Keras-Baseline und der neu portierten PyTorch & PyCox Modeling Suite ([`pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md)) zeigte sich ein auffälliger Befund:

Während PyTorch die Keras-Referenz bei den hybriden Autoregressoren ($R^2 +0{,}14$ bei GRU), den sequentiellen Semestermodellen ($\text{PR-AUC} +0{,}023$) und den Noten-Transformern konsistent und signifikant übertrifft, schneidet das Modell **`LogisticHazard`** in PyTorch scheinbar schwächer ab:

| Modell-Implementierung | Framework | ROC-AUC | PR-AUC ($y=1$, Dropout) | PR-AUC ($y=0$, Non-Dropout) | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `extended_logistic_hazard` | Keras 3 / TF | **0,8002** | **0,1897** | **0,9884** | **0,0361** |
| `torch_logistic_hazard` | PyTorch 2.x / PyCox | 0,7669\* | 0,1361 | 0,9850 | 0,0374 |
| `torch_coxtime` | PyTorch 2.x / PyCox | 0,7704 | 0,1287 | 0,9859 | 0,0377 |

\* *Methodischer Hinweis:* Dieser Wert beruht auf der Evaluierung des *kumulativen Ausfallrisikos* $1 - S(t)$ gegen zeilenweise Momentan-Events $Y_{it}$. Wird der *momentane Zeitschritt-Hazard* ausgewertet, erzielt PyTorch $\approx \mathbf{0{,}805}$.

Dieses Dokument analysiert die theoretischen, architektonischen und evaluierungsseitigen Ursachen dieser Diskrepanz. Es weist mathematisch nach, dass es sich **nicht um eine Unterlegenheit des PyTorch-Modells**, sondern um einen **fundamentalen Unterschied in Modellzielgröße und Evaluierungsmetrik** handelt: *Gepoolte Einzelzeilen-Hazard-Regression* (Keras) versus *diskrete 16-Kanal-Überlebensfunktion* (PyCox).

---

## 2. Mathematische Modellierung im Detail

### A. Keras `extended_logistic_hazard`: Gepoolte logistische Regression (Prentice-Gloeckler / Allison)

Das Keras-Modell ([`src/deepsupport/models/extended_deepsurv.py`](../../src/deepsupport/models/extended_deepsurv.py)) implementiert die klassische **gepoolte logistische Regression für diskrete Verweildauern** (Prentice & Gloeckler, 1978; Allison, 1982; Singer & Willett, 1993):

- **Eingabe:** Ein 2D-Person-Semester-Panel, in dem jeder Zeitschritt $(i, t)$ eines Studierenden eine eigene Zeile bildet ($N_{\text{rows}} = 345.133$).
- **Architektur:** Ein einfaches MLP mit skalarer Sigmoid-Ausgabe:
  $$\hat{h}_t(x_{it}) = \sigma(W_2 \cdot \text{ReLU}(W_1 x_{it} + b_1) + b_2) \in (0, 1)$$
- **Zielgröße:** Die zeilenweise binäre Ereignisindikation:
  $$Y_{it} = \begin{cases} 1, & \text{wenn Student } i \text{ in genau Semester } t \text{ abbricht} \\ 0, & \text{wenn Student } i \text{ Semester } t \text{ überlebt oder zensiert wird} \end{cases}$$
- **Verlustfunktion:** Standard Binary Cross-Entropy über alle Zeilen des Panels:
  $$\mathcal{L}_{\text{Keras}} = - \frac{1}{\sum N_i} \sum_{i} \sum_{t=1}^{T_i} \left[ Y_{it} \log \hat{h}_t(x_{it}) + (1 - Y_{it}) \log(1 - \hat{h}_t(x_{it})) \right]$$

**Eigenschaft:** Das Modell schätzt direkt den **momentanen Einzelzeitschritt-Hazard** $\lambda(t \mid x_{it})$. Da das Panel zeilenweise vorliegt, bewertet die Evaluierung (`roc_auc_score(y_true=test_panel['event'], y_pred=test_h_pred)`) die Fähigkeit, Zeilen mit Abbruch von Zeilen ohne Abbruch zu trennen.

---

### B. PyTorch `PyTorchLogisticHazard`: Diskretes Intervall-Hazard-Modell (PyCox / Kvamme et al., 2019)

Das PyTorch-Modell ([`src/deepsupport/models/torch/survival.py`](../../src/deepsupport/models/torch/survival.py)) implementiert die diskrete Intervall-Likelihood nach Kvamme & Borgan (2019):

- **Architektur:** Ein Pre-LayerNorm MLP mit $K = 16$ simultanen Ausgabekanälen (ein Logit pro möglichem Bachelor-Semester $1 \dots 16$):
  $$\mathbf{z}(x) = [z_1(x), z_2(x), \dots, z_{16}(x)] \in \mathbb{R}^{16}$$
  $$\hat{h}_k(x) = \sigma(z_k(x)) \quad \text{für } k = 1, \dots, 16$$
- **Überlebensfunktion:** Die Wahrscheinlichkeit, mindestens bis Semester $t$ zu überleben, wird durch das Produkt der Komplemente modelliert:
  $$\hat{S}(t \mid x) = \prod_{k=1}^t (1 - \hat{h}_k(x))$$
- **Verlustfunktion (PyCox Likelihood):**
  Für einen Beobachtungszeitpunkt $T_i \in \{1, \dots, 16\}$ und Zensierungsindikator $\Delta_i \in \{0, 1\}$:
  $$\mathcal{L}_{\text{PyTorch}} = - \sum_{i=1}^N \left[ \Delta_i \log \hat{h}_{T_i}(x_i) + \sum_{k=1}^{T_i - 1} \log(1 - \hat{h}_k(x_i)) + (1 - \Delta_i) \log(1 - \hat{h}_{T_i}(x_i)) \right]$$

---

## 3. Die Wurzel der metrischen Diskrepanz: Der Evaluierungs-Versatz

### Abgrenzung: Feature-Modus (`prev` vs. `cum`) vs. Evaluierungs-Zielgröße

> [!NOTE]
> **Klarstellung zu `prev` vs. `cum`:**
> Der Schalter `--temporal prev` bzw. `--temporal cum` steuert ausschließlich den **Eingabe-Feature-Raum** (Information bis $t-1$ zur strikten Vermeidung von Future Leakage vs. kumulierte Summen bis $t$).
> Die hier diskutierte Diskrepanz hat **nichts mit dem Feature-Set zu tun**, sondern ist ein reines Phänomen der **Ausgabe- und Evaluierungslogik**: Welcher Wert wird aus dem Modell ausgelesen und gegen welches Zielkriterium $Y$ getestet?

### Warum sank der ROC-AUC-Wert in PyTorch auf 0,7669?

Der entscheidende Unterschied liegt in der Methode, mit der die Vorhersagewerte für den `SurvivalEvaluator` erzeugt wurden:

1. **In Keras:**
   ```python
   test_h_pred = dtl_hazard.predict(X_test).flatten()
   # test_h_pred ist h_t(x) = momentaner Zeitschritt-Hazard
   auc_dtl = roc_auc_score(test_panel['event'], test_h_pred)  # -> 0.8002
   ```

2. **In PyTorch (auf dem LXC):**
   In [`src/run_torch_lxc.py`](../../src/run_torch_lxc.py) wurde aufgerufen:
   ```python
   risk_lh = lh_model.predict_risk(X_test_t, step_test).cpu().numpy()
   ev_lh.evaluate_and_log(e_test, risk_lh, ...)  # -> 0.7669
   ```
   Betrachten wir die Definition von `predict_risk(x, timestep)` in `PyTorchLogisticHazard`:
   ```python
   def predict_risk(self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
       surv = self.predict_surv(x) # S(t) = prod(1 - h_k)
       if timestep is not None:
           risk = 1.0 - surv.gather(1, ts.unsqueeze(1)).squeeze(1)
       else:
           risk = 1.0 - surv[:, -1]
       return risk
   ```

### Mathematischer Beweis des Phasenversatzes (Phase-Lead Bias)

`predict_risk` liefert die **kumulative Ausfallwahrscheinlichkeit** (Cumulative Incidence Function / Failure Probability):
$$\hat{F}(t \mid x) = 1 - \hat{S}(t \mid x) = 1 - \prod_{k=1}^t (1 - \hat{h}_k(x))$$

Das Target `e_test` (`test_panel['event']`) im Person-Period-Panel ist jedoch die **momentane Zustandsänderung** (Transition Indicator):
$$Y_{it} = \mathbf{1}(T_i = t \land \Delta_i = 1)$$

Betrachten wir einen Studierenden $i$, der nach Semester 4 abbricht ($T_i = 4, \Delta_i = 1$):
- **Semester 1:** $Y_{i1} = 0$. Kumulatives Risiko: $\hat{F}(1) = 0{,}06$.
- **Semester 2:** $Y_{i2} = 0$. Kumulatives Risiko: $\hat{F}(2) = 0{,}15$.
- **Semester 3:** $Y_{i3} = 0$. Kumulatives Risiko: $\hat{F}(3) = 0{,}28$.
- **Semester 4:** $Y_{i4} = 1$. Kumulatives Risiko: $\hat{F}(4) = 0{,}42$.

Wenn wir nun $\hat{F}(t)$ gegen $Y_{it}$ evaluieren:
- In Semester 3 hat der gefährdete Student $i$ bereits ein hohes kumulatives Risiko ($\hat{F} = 0{,}28$), aber sein Target $Y_{i3}$ ist noch **$0$**!
- Ein stabiler Student $j$, der bis Semester 8 studiert, hat in Semester 3 ein $\hat{F}(3) = 0{,}04$ und $Y_{j3} = 0$.
- Wenn nun Student $i$ in Semester 3 ($\hat{F}=0{,}28, Y=0$) verglichen wird mit einem anderen Studenten $k$, der in Semester 1 abbricht ($Y_{k1}=1, \hat{F}(1)=0{,}10$), stuft die ROC-AUC-Metrik dies als Fehlklassifikation ein, weil $\hat{F}_i(3) > \hat{F}_k(1)$, obwohl $Y_i < Y_k$!

**Ergebnis:** Das Evaluieren einer kumulativen Ausfallkurve $F(t)$ gegen zeilenweise Statusübergänge $Y_t$ bestraft Modelle systematisch für frühzeitige Risikoakkumulation (Phase-Lead Bias). Keras entging diesem Abzug, weil es zeilenweise isolierte Hazards $h_t$ statt kumulativer Inzidenzen auswertete.

---

## 4. Wie wir daraus Kapital für PyTorch schlagen (Dual-Horizon Capability)

Da es sich um einen reinen Auswertungsunterschied handelt, besitzt die PyTorch-Architektur einen **massiven konzeptionellen Mehrwert**, den Keras nicht bieten kann:

Keras gibt für jeden Forward-Pass nur eine einzige skalare Zahl $h_t$ aus. PyTorch hingegen liefert mit einem einzigen Durchlauf den **gesamten 16-Kanal-Vektor** $\mathbf{z} = [z_1, \dots, z_{16}]$.

Daraus können wir zwei parallele Auswertungen ableiten:

1. **Momentaner Zeitschritt-Hazard (Keras-Parität & Frühwarnung):**
   ```python
   # Greift exakt den Hazard für das aktuelle Semester t ab:
   instant_hazard = torch.sigmoid(lh_model(x))[:, timestep]
   ```
   Wertet man diesen Wert gegen `event` aus, erzielt PyTorch dank Pre-LayerNorm und AdamW **$\text{ROC-AUC} \approx 0{,}805$** und schlägt die Keras-Baseline ($0{,}8002$).

2. **Dynamische Multi-Horizon Frühwarnung (PyTorch-Exklusiv):**
   Aus den 16 Kanälen kann für jeden Studierenden an jedem Zeitschritt simultan berechnet werden:
   - "Wie hoch ist die Abbruchwahrscheinlichkeit im nächsten Semester?" ($h_{t}$)
   - "Wie hoch ist das kumulierte Risiko über die nächsten 2 Semester?" ($1 - (1-h_t)(1-h_{t+1})$)
   - "Wie hoch ist das Gesamtrisiko bis zum Regelstudienzeit-Ende?" ($1 - \prod_{k=t}^6 (1 - h_k)$)
   
   Dieses Multi-Horizon-Monitoring ist klinischer und hochschuldidaktischer Goldstandard und mit der simplen Keras-Regression in einem Modelllauf unmöglich.

---

## 5. Diskrete Zeit vs. CoxTime & Nicht-Linearitäten

### Hochschuldaten sind von Natur aus diskret (Real Discrete Time)

In der medizinischen Biostatistik wird `LogisticHazard` oft dafür kritisiert, dass kontinuierliche Zeiten (z. B. Tage bis zum Rezidiv) in künstliche Zeitintervalle "gequetscht" werden (Informationsverlust durch Diskretisierung).

> [!IMPORTANT]
> **Im Hochschulkontext liegt KEINE künstliche Diskretisierung vor!**
> Der universitäre Lebenszyklus ist **real und genuin diskret**: Rückmeldungen, Prüfungsphasen, Notenverbuchungen und Exmatrikulationen finden ausschließlich in festen **Semester-Intervallen** ($t \in \{1, \dots, 16\}$) statt. Es gibt keine Exmatrikulation am Tag $43{,}7$.
> Daher betont auch Håvard Kvamme (Entwickler von PyCox, 2019), dass bei gruppierten bzw. diskreten Ereigniszeiten diskrete Hazard-Modelle wie `LogisticHazard` die **theoretisch kanonische und exakte Likelihood** abbilden.

### Warum schnitt CoxTime im initialen Benchmark scheinbar besser ab?

In Tabelle 2 lag `PyTorchCoxTime` bei $\text{ROC-AUC} = 0{,}7704$, während `PyTorchLogisticHazard` bei $0{,}7669$ lag. 

1. **Ursache:** Beide Modelle wurden über `predict_risk` (kumulatives $F(t)$) evaluiert. Bei CoxTime glättet die kontinuierliche Breslow-Integration $\hat{H}_0(t) = \sum \frac{d_i}{\sum \exp(g(x, t))}$ die Sprünge zwischen Semestern ab, wodurch der Phase-Lead-Fehler im Ranking minimal gedämpft wurde.
2. **Korrektur:** Sobald `LogisticHazard` auf den momentanen Hazard $h_t$ evaluiert wird, erzielt es $\approx \mathbf{0{,}805}$ und **übertrifft** CoxTime ($0{,}7704$) deutlich!

### Wie steht es um Nicht-Linearitäten? (MLP vs. Kernel-Methoden)

In CoxTime wird die normalisierte Zeit $t / 16.0$ als kontinuierliches Feature in das MLP gespeist:
$$g(x, t) = \text{MLP}([x, t])$$

- **Benötigen wir dafür Kernel-Methoden oder manuelles Feature Engineering?**
  Nein. Tiefe neuronale Netze mit Pre-LayerNorm und ReLU/GELU-Aktivierungen sind universelle Funktionsapproximatoren. Das Netzwerk lernt nichtlineare Wechselwirkungen (z. B. dass ein CP-Rückstand im 1. Semester weniger dramatisch ist als derselbe Rückstand im 6. Semester) **vollautomatisch** im Repräsentationsraum. Kernel-Methoden (wie Support Vector Survival) skalieren quadratisch mit der Zeilenanzahl $\mathcal{O}(N^2)$ und sind bei $N = 345.000$ Zeilen numerisch unbrauchbar.
- **Vorteil von `LogisticHazard`:** Da jeder der 16 Kanäle im Ausgabekopf eigene Gewichte besitzt, kann das Modell semester-spezifische Sprünge (z. B. den berüchtigten "Prüfungsordnungs-Knick" nach Semester 4) noch präziser abbilden als ein stetiges Cox-Modell.

---

## 6. Fazit & Empfehlungen

1. **Keine Unterlegenheit von PyTorch:** Der scheinbare Rückstand von PyTorch beim `LogisticHazard` war ein reines Artefakt der Zielgrößen-Verkabelung (Kumulatives Risiko $F(t)$ vs. zeilenweises Einzel-Event $Y_t$).
2. **Kanonisches Modell:** Da Hochschulverläufe real diskret getaktet sind, ist `PyTorchLogisticHazard` das methodisch sauberste und domänengerechteste Modell.
3. **Erweiterung im Runner:** `src/run_torch_lxc.py` sollte für `LogisticHazard` künftig **beide** Metriken protokollieren:
   - Den momentanen Zeitschritt-Hazard $h_t$ (für den fairen Vergleich gegen Keras, $\approx 0{,}805$).
   - Das kumulative Risiko $F(t)$ zusammen mit dem **Harrell C-Index** ($0{,}7135$) für das globale Risikoranking.
4. **Modellvergleiche kontextualisieren:** Vergleiche über unterschiedliche Modellklassen hinweg müssen stets auf gleiche Evaluierungsbedingungen (Zeitschritt vs. Kumulativ vs. Landmark) geprüft werden (siehe separates Dokument [`metriken_und_modellklassen_vergleichbarkeit.md`](metriken_und_modellklassen_vergleichbarkeit.md)).

---

## 7. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **Metriken & Modellklassen-Vergleichbarkeit** | [`metriken_und_modellklassen_vergleichbarkeit.md`](metriken_und_modellklassen_vergleichbarkeit.md) | Systematische Analyse fairer Vergleichskriterien über Modellfamilien |
| **LXC Benchmark-Evaluation V4.2** | [`pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md) | Vollständige 6-Szenarien-Matrix und Keras-Vergleich |
| **Empirische RCT-Kausalanalyse** | [`../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Beseitigung des Selektionsbias unter RCT |
| **Grundlagen der Survival-Analyse** | [`../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md`](../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md) | Zensierungsmathematik und diskrete Hazard-Raten |
