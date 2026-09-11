---
created: 2026-09-11
last_updated: 2026-09-11
status: in_bearbeitung
tags: [ablation-study, architecture-hypotheses, keras-vs-pytorch, state-decay, pre-ln, logits-loss, adamw]
---

# Experimenteller Ablationsplan: Empirische Prüfung der Keras-vs-PyTorch Architekturhypothesen

## 1. Ausgangslage & Zielsetzung

Im Head-to-Head-Benchmark zwischen Keras 3 / TensorFlow und der neu portierten PyTorch & PyCox Modeling Suite ([`../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md)) zeigten sich signifikante Performanzvorteile für PyTorch:
- **Autoregressiver Dual-Head GRU:** Noten-$R^2$ steigt von **$0{,}5706$** (Keras) auf **$0{,}7118$** (PyTorch) ($+0{,}1412$).
- **Autoregressiver Dual-Head Transformer:** Noten-$R^2$ steigt von $0{,}7021$ auf $0{,}7124$; Brier Score sinkt von $0{,}0766$ auf $0{,}0668$.
- **Sequentielle Semester-Modelle:** PR-AUC auf Zeitschrittebene gewinnt $+0{,}023$ (GRU) bzw. $+0{,}017$ (Transformer).

Zur Erklärung dieser Diskrepanzen wurden vier plausible Software- und Modellierungs-Hypothesen formuliert:
1. **$H_1$ (State Gathering vs. Padding Diffusion):** Keras propagiert Nullen durch ungefilterte Padding-Schritte, während PyTorch den realen Endzustand abgreift.
2. **$H_2$ (Pre-LayerNorm vs. Post-LayerNorm):** Pre-LN schützt den Residualpfad im Transformer und stabilisiert frühe Epochen.
3. **$H_3$ (Logits-Loss mit Log-Sum-Exp vs. Sigmoid + BCE):** Direkte Logits verhindern Gradientensättigung an den Verteilungsenden.
4. **$H_4$ (AdamW mit Cosine Annealing vs. Standard Adam):** Entkoppeltes Weight Decay begünstigt breitere, generalisierungsfähige Minima.

Dieser Plan definiert das experimentelle Design, um diese vier Hypothesen isoliert und kausal durch **faktorielle Ablationen** empirisch zu falsifizieren oder zu verifizieren.

---

## 2. Detaillierte Hypothesen & Falsifikationskriterien

### Hypothese 1: Recurrent State Decay durch Zero-Padding-Diffusion ($H_1$)

#### Mechanismus:
- **Keras:** `Masking(mask_value=-99.0) -> GRU(return_sequences=False)` überspringt zwar die Loss-Berechnung, aber in Standard-TensorFlow-GRUs führt `return_sequences=False` bei festen Sequenzlängen ($S=35$) dazu, dass nach $k$ validen Prüfungen $35-k$ Nullschritte $h_t = \text{GRU}(h_{t-1}, \mathbf{0})$ berechnet werden. Der finale Vektor $h_{35}$ leidet unter exponentiellem Informationszerfall.
- **PyTorch:** `out_gru.gather(1, (lengths - 1))` greift exakt $h_k$ ab.

#### Falsifizierbare Aussage:
> "Wird in PyTorch künstlich auf $h_{35}$ (Endzustand nach Padding) zurückgegriffen, bricht das Noten-$R^2$ von $0{,}71$ auf unter $0{,}60$ ein. Wird umgekehrt in Keras der reale Zustand $h_k$ via Lambda-Layer abgegriffen, springt das Keras-$R^2$ auf $> 0{,}68$."

#### Falsifikationskriterium:
- Falls $\Delta R^2 (\text{Gathering} - \text{Diffusion}) < 0{,}05$, ist $H_1$ falsifiziert.

---

### Hypothese 2: Pre-LayerNorm vs. Post-LayerNorm ($H_2$)

#### Mechanismus:
- **Post-LN (Keras):** Normalisierung *nach* der Addition: $x_{l+1} = \text{LN}(x_l + f(x_l))$. Der Gradient auf $x_l$ wird mit $1 / \sigma$ der Normalisierung skaliert.
- **Pre-LN (PyTorch):** Normalisierung *vor* der Operation: $x_{l+1} = x_l + f(\text{LN}(x_l))$. Der Gradient fließt unverfälscht durch den Residualpfad ($\frac{\partial x_{l+1}}{\partial x_l} = \mathbf{I} + \dots$).

#### Falsifizierbare Aussage:
> "Bei 2-4 Transformer-Schichten führt Pre-LN zu schnellerer Konvergenz in den ersten 5 Epochen und einem um mindestens $0{,}01$ höheren finalen $R^2$-Score im Vergleich zu Post-LN unter identischem Optimizer."

#### Falsifikationskriterium:
- Falls $|R^2_{\text{Pre-LN}} - R^2_{\text{Post-LN}}| < 0{,}005$, ist die Normalisierungsposition für die vorliegende Modellgröße vernachlässigbar.

---

### Hypothese 3: Numerische Stabilität des Logits-Loss ($H_3$)

#### Mechanismus:
- **Keras:** `Dense(1, sigmoid) -> binary_crossentropy`. Für Vorhersagen $p > 0{,}999$ clippt Keras intern auf $1 - 10^{-7}$, was die zweiten Ableitungen abflacht.
- **PyTorch:** `Dense(1) -> binary_cross_entropy_with_logits` (Log-Sum-Exp Trick).

#### Falsifizierbare Aussage:
> "Modelle mit direktem Logits-Loss weisen eine signifikant bessere Kalibrierung auf extremen Wahrscheinlichkeiten auf (gemessen am Brier Skill Score auf den $5\,\%$-Extremquantilen)."

#### Falsifikationskriterium:
- Falls der Brier Score zwischen Sigmoid+BCE und BCEWithLogits um weniger als $2\,\%$ divergiert, ist $H_3$ als Hauptursache falsifiziert.

---

### Hypothese 4: Optimizer-Regime AdamW vs. Standard Adam ($H_4$)

#### Mechanismus:
- Standard-Adam modifiziert die Gradienten mit $L_2$-Regularisierung vor der Momentumbildung ($g_t \leftarrow g_t + \lambda \theta_t$). Bei Gewichten mit großen Gradienten wird das Weight Decay unverhältnismäßig stark gedämpft.
- AdamW entkoppelt den Zerfall: $\theta_{t+1} \leftarrow \theta_t (1 - \eta \lambda) - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$.

#### Falsifizierbare Aussage:
> "AdamW mit Cosine Annealing verhindert Overfitting auf kleinen Subpopulationen (z. B. Spätsemester $t > 8$) und liefert gegenüber Adam mit fixer Lernrate einen konsistenten PR-AUC-Vorteil von mindestens $+0{,}01$."

---

## 3. Experimentelles Ablations-Design (Faktorieller $2^4$-Plan)

Um Interaktionseffekte zwischen den Faktoren aufzudecken, wird ein strukturierter Versuchsplan definiert. Getestet wird primär auf **S01 Baseline, Universe A** ($N = 50.000$):

| Run-ID | State Aggregation | Normalisierung | Loss-Interface | Optimizer | Framework | Erwartetes Noten-$R^2$ |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| **R0 (Keras Original)** | Padding-Diffusion | Post-LN | Sigmoid + BCE | Adam (fix) | Keras | $\approx 0{,}571$ |
| **R1** | Index-Gathering | Post-LN | Sigmoid + BCE | Adam (fix) | Keras | $\approx 0{,}690$ |
| **R2** | Index-Gathering | Pre-LN | Sigmoid + BCE | Adam (fix) | Keras | $\approx 0{,}702$ |
| **R3** | Index-Gathering | Pre-LN | Logits Loss | Adam (fix) | Keras | $\approx 0{,}708$ |
| **R4 (Keras Full-Parity)**| Index-Gathering | Pre-LN | Logits Loss | AdamW + Cosine | Keras | $\approx 0{,}712$ |
| **R5 (PyTorch Original)** | Index-Gathering | Pre-LN | Logits Loss | AdamW + Cosine | PyTorch | **$0{,}7118$** |
| **R6 (PyTorch Downgrade)**| Padding-Diffusion | Pre-LN | Logits Loss | AdamW + Cosine | PyTorch | $\approx 0{,}585$ |
| **R7 (PyTorch Post-LN)** | Index-Gathering | Post-LN | Logits Loss | AdamW + Cosine | PyTorch | $\approx 0{,}705$ |
| **R8 (PyTorch Adam)** | Index-Gathering | Pre-LN | Logits Loss | Adam (fix) | PyTorch | $\approx 0{,}708$ |

---

## 4. Implementierungsskizzen für die Ablations-Varianten

### A. Keras Index Gathering (Lösung für $H_1$)

In Keras kann der exakte Zustand über ein custom Lambda-Layer mit `tf.gather_nd` abgegriffen werden:

```python
def extract_last_valid_step(x, lengths):
    batch_size = tf.shape(x)[0]
    indices = tf.stack([tf.range(batch_size), tf.maximum(0, lengths - 1)], axis=1)
    return tf.gather_nd(x, indices)
```

### B. PyTorch Padding-Diffusion (Downgrade-Test für $H_1$)

Um das Keras-Fehlverhalten in PyTorch gezielt zu replizieren:

```python
# Statt out_gru.gather(1, idx):
seq_rep_decayed = out_gru[:, -1, :] # Greift Zeitschritt 35 nach 30 Padding-Nullen ab
```

---

## 5. Priorisierung & Ressourcenaufwand

- **Aufwand:** 9 Trainingsläufe à ca. 8 Minuten auf CPU/LXC $\approx 72$ Minuten Gesamtlaufzeit.
- **Strategische Einordnung:** Die Prüfung dieser Hypothesen ist forschungsanalytisch von hohem Wert, blockiert jedoch nicht die primäre Weiterentwicklung des Gesamtsystems.
- **Empfehlung:** Da die kausale Inferenz (DML, MSM, G-Computation) für die Forschungsfragen des Abschlussprojekts höchste Priorität besitzt, wird der Ablations-Lauf nach der Kausal-Suite eingeplant.

---

## 6. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **LXC Benchmark Evaluation V4.2** | [`../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](../03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Ausgangsbefund des Keras-vs-PyTorch-Vergleichs |
| **Methodenvergleich LogisticHazard** | [`../03_evaluations_and_benchmarks/methodenvergleich_logistic_hazard_keras_vs_pycox.md`](../03_evaluations_and_benchmarks/methodenvergleich_logistic_hazard_keras_vs_pycox.md) | Mathematische Klärung der Hazard-Diskrepanz |
| **Empirische RCT-Kausalanalyse** | [`../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](../04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Erklärung der Metrikverbesserung unter RCT |
