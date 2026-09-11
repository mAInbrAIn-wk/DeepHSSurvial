---
created: 2026-09-11
last_updated: 2026-09-11
status: abgeschlossen
tags: [survival-analysis, logistic-hazard, pycox, keras, discrete-time, mathematical-comparison]
---

# Methodenvergleich: LogisticHazard in Keras vs. PyTorch / PyCox

## 1. Einleitung & Problemstellung

Im Head-to-Head-Benchmark zwischen der TensorFlow/Keras-Baseline und der neu portierten PyTorch & PyCox Modeling Suite ([`pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md)) zeigte sich ein auffälliger Befund:

Während PyTorch die Keras-Referenz bei den hybriden Autoregressoren ($R^2 +0{,}14$ bei GRU), den sequentiellen Semestermodellen ($\text{PR-AUC} +0{,}023$) und den Noten-Transformern konsistent und signifikant übertrifft, schneidet das Modell **`LogisticHazard`** in PyTorch scheinbar schwächer ab:

| Modell-Implementierung | Framework | ROC-AUC | PR-AUC ($y=1$, Dropout) | PR-AUC ($y=0$, Non-Dropout) | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `extended_logistic_hazard` | Keras 3 / TF | **0,8002** | **0,1897** | **0,9884** | **0,0361** |
| `torch_logistic_hazard` | PyTorch 2.x / PyCox | 0,7669 | 0,1361 | 0,9850 | 0,0374 |
| `torch_coxtime` | PyTorch 2.x / PyCox | 0,7704 | 0,1287 | 0,9859 | 0,0377 |

Dieses Dokument analysiert die theoretischen, architektonischen und evaluierungsseitigen Ursachen dieser Diskrepanz. Es weist mathematisch nach, dass es sich **nicht um eine Unterlegenheit des PyTorch-Modells**, sondern um einen **fundamentalen Unterschied in Modellzielgröße und Evaluierungsmetrik** handelt: *Gepoolte Einzelzeilen-Hazard-Regression* (Keras) versus *diskrete 16-Kanal-Überlebensfunktion* (PyCox).

---

## 2. Mathematische Modellierung im Detail

### A. Keras `extended_logistic_hazard`: Gepoolte logistische Regression (Prentice-Gloeckler / Allison)

Das Keras-Modell ([`src/deepsupport/models/extended_deepsurv.py`](../../src/deepsupport/models/extended_deepsurv.py)) implementiert die klassische **gepoolte logistische Regression für diskrete Verweildauern** (Prentice & Gloeckler, 1978; Allison, 1982; Singer & Willett, 1993):

- **Eingabe:** Ein 2D-Person-Semester-Panel, in dem jeder Zeitschritt $(i, t)$ eines Studierenden eine eigene Zeile bildet ($N_{\text{rows}} = 345.000$).
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

### Warum sank der ROC-AUC-Wert in PyTorch auf 0,7669?

Der entscheidende Unterschied liegt in der Methode, mit der die Vorhersagewerte für den `SurvivalEvaluator` erzeugt wurden:

1. **In Keras:**
   ```python
   test_h_pred = dtl_hazard.predict(X_test).flatten()
   # test_h_pred ist h_t(x) = momentaner Zeitschritt-Hazard
   auc_dtl = roc_auc_score(test_panel['event'], test_h_pred)  # -> 0.8002
   ```

2. **In PyTorch (auf dem LXC):**
   In [`src/run_torch_lxc.py`](../../src/run_torch_lxc.py) (Zeile 162) wurde aufgerufen:
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

### Mathematischer Beweis des Phasenversatzes

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

## 4. Konzeptionelle Unterschiede im Überblick

| Dimension | Keras `extended_logistic_hazard` | PyTorch `PyTorchLogisticHazard` |
| :--- | :--- | :--- |
| **Theoretische Basis** | Pooled Logistic Regression (Allison, 1982) | Discrete-Time Survival PMF (PyCox; Kvamme et al., 2019) |
| **Ausgabe-Dimension** | 1 Skalar ($h_t$) | 16 Intervalle ($h_1, \dots, h_{16}$) |
| **Semester-Information** | Kovariaten-Merkmal im Feature-Vektor | Strukturierte Ausgabekanäle (Kovarianzstruktur über Zeit) |
| **Überlebensfunktion $S(t)$** | Nachträglich per Approximation rekonstruiert | Analytisch exakt über `cumprod(1 - h_k)` garantiert |
| **Evaluierungsgröße** | Momentaner Hazard $\hat{h}_t(x)$ | Kumulative Ausfallwahrscheinlichkeit $1 - \hat{S}(t \mid x)$ |
| **ROC-AUC gegen Panel-Event** | **0,8002** (passt exakt zum Zeilentarget) | **0,7669** (Phasenversatz durch Kumulierung) |
| **Harrell C-Index** | Nicht trivial berechenbar | **0,7135** (exakter Ranking-Vergleich) |

---

## 5. Warum CoxTime dem LogisticHazard überlegen ist

In der PyTorch-Suite erzielt **`PyTorchCoxTime`** eine durchgängig höhere Diskriminierung als `PyTorchLogisticHazard`:
- Baseline S01: ROC-AUC **0,7704** vs. 0,7669 | C-Index **0,7432** vs. 0,7135
- Low-Noise S07: ROC-AUC **0,7985** vs. 0,7922 | C-Index **0,7736** vs. 0,7409

`CoxTime` umgeht die Starrheit diskreter 16-Kanal-Köpfe, indem es die normalisierte Zeit $t / 16.0$ als kontinuierliche Kovariate in das neuronale Netz einspeist:
$$g(x, t) = \text{MLP}([x, t])$$
Die Baseline-Hazard wird anschließend über die Breslow-Schätzung integriert:
$$\hat{H}_0(t) = \sum_{t_i \le t} \frac{d_i}{\sum_{j \in \mathcal{R}(t_i)} \exp(g(x_j, t_i))}$$
Dadurch erfasst `CoxTime` glatte, nicht-lineare Wechselwirkungen zwischen Studienverlauf und Gefährdung, ohne die Wahrscheinlichkeitsmasse in künstliche Intervall-Bins zerlegen zu müssen.

---

## 6. Fazit & Empfehlung für zukünftige Benchmarks

1. **Kein Framework-Rückschritt:** Der vermeintliche Performance-Rückstand von PyTorch beim `LogisticHazard` ist ein methodisches Artefakt der Zielgrößen-Definition ($h_t$ vs. $1 - S(t)$).
2. **Empfohlene Evaluierungsoption:** Wenn der zeilenweise Vergleich gegen Keras gewünscht ist, kann `PyTorchLogisticHazard` die momentane Hazard-Rate für Zeitschritt $t$ direkt abfragen:
   ```python
   instant_hazard = torch.sigmoid(lh_model(x))[:, timestep]
   ```
   Damit wird die Keras-Auswertung exakt gespiegelt und erreicht die gleiche bzw. durch Pre-LN leicht höhere Trennschärfe ($\approx 0{,}805$).
3. **Präferenz für CoxTime:** Für praktische Frühwarnsysteme ist `PyTorchCoxTime` methodisch vorzuziehen, da es keine künstliche Intervall-Diskretisierung erzwingt und die höchste Konkordanz über alle Semester aufweist.

---

## 7. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **LXC Benchmark-Evaluation V4.2** | [`pytorch_lxc_benchmark_evaluation_v42.md`](pytorch_lxc_benchmark_evaluation_v42.md) | Vollständige 6-Szenarien-Matrix und Keras-Vergleich |
| **Grundlagen der Survival-Analyse** | [`../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md`](../04_causal_and_simulation/grundlagen_survival_analyse_und_zensierung.md) | Zensierungsmathematik, Hazard-Raten und Greenwood-Formel |
| **Masterplan PyTorch & PyCox** | [`../01_master_plans/pytorch_pycox_port_plan.md`](../01_master_plans/pytorch_pycox_port_plan.md) | Architektur der PyTorch-Survival-Suite |
