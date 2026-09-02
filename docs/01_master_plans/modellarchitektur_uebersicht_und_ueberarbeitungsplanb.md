# Synopse der Modellarchitekturen & Masterplan zur Modellüberarbeitung (V4.1)

> [!IMPORTANT]
> **Status:** Die `Deep Transformer Suite` ($d=128$) ist temporär deaktiviert (`Under Revision`).  
> Dieses Dokument bietet eine strukturierte, lesbare Übersicht aller 8 Modellfamilien, klärt die architektonischen Ursachen des Positional-Encoding-Gaps auf und definiert den Fahrplan für gezieltes Feintuning (GRU vs. LSTM, `gradeblind`-Standard, Focal Loss, L1/L2-Regularisierung).

---

## 1. Systematische Synopse aller 8 Modellfamilien im Projekt

```mermaid
flowchart TD
    subgraph Querschnitt ["A. Querschnitt & Baseline"]
        K1["1. Statische Landmark-Modelle (S1–S2)<br>• Naive Bayes, Random Forest, SVM (RBF)<br>• Ridge Regressor, Keras 3-Layer MLP"]
        K8["8. Repräsentationslernen<br>• Geköpfter Causal Transformer -> Embeddings (d=64)"]
    end

    subgraph PanelSurv ["B. Panel-Survival & Diskrete Hazard"]
        K2["2. Semiparametrisches & Diskretes Panel-Survival<br>• Extended Cox (PHReg, Breslow Ties)<br>• Extended Logistic Hazard (DTL, Bernoulli BCE)"]
        K5["5. Competing Risks<br>• Dynamic DeepHit (Diskretes Multi-Hazard für Abbruch & Abschluss)"]
    end

    subgraph Sequenz ["C. Rekurrente & Transformer-Sequenzmodelle"]
        K3["3. Rekurrente Sequenzmodelle (RNNs)<br>• Semester GRU / LSTM (16 Steps, Hidden 64/32)<br>• Exam GRU (40 Steps, Masked)"]
        K4["4. Transformer-Sequenzmodelle<br>• Semester Transformer (d=64, Learnable PosEnc)<br>• Exam Transformer (d=64, Causal Masked Attention)<br>• Deep Transformer Suite (d=128, Under Revision)"]
    end

    subgraph KausalAutoreg ["D. Kausale Inferenz & Autoregression"]
        K6["6. Autoregressive Next-Exam Modelle<br>• Dual-Head GRU/Dense (Note & Pass für t_k+1)<br>• Deep Transformer Autoregressor (SinCos PosEnc)"]
        K7["7. Kausal-Inferenz (DML)<br>• DML Orthogonal Survival (Ridge + MLP)<br>• Transformer DML (Pretrained Encoder + DML)"]
    end

    Querschnitt --> PanelSurv
    PanelSurv --> Sequenz
    Sequenz --> KausalAutoreg
```

---

### Detaillierte Architektur-Spezifikation

| Modell-ID / Skript | Granularität | Eingabe-Shape | Backbone / Schichten | Loss-Funktion | Positional Encoding | Hauptmetrik |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **`train_mlp_baseline.py`** | Landmark (S1-S2) | $(N, 21)$ | 3-Layer Dense ($128 \rightarrow 64 \rightarrow 32$), Dropout 0.2 | Categorical / Binary CE | Keine (Statisch) | ROC-AUC: 0,847 |
| **`train_mlp_regression.py`** | Landmark (S1-S2) | $(N, 21)$ | 3-Layer Dense ($128 \rightarrow 64 \rightarrow 32$), L2-Reg | MSE / MAE | Keine (Statisch) | **Gradeblind $R^2$: 0,682** |
| **`extended_cox_survival.py`** | Semester-Panel | $(N_{\text{panel}}, 18)$ | Semiparametrisches PHReg (Statsmodels) | Cox Partial Likelihood | `t_stop` (Time-Varying) | C-Index: 0,742 |
| **`extended_deep_survival.py` (LH)**| Semester-Panel | $(N_{\text{panel}}, 18)$ | 2-Layer Dense ($64 \rightarrow 32$) + LayerNorm | Binary Crossentropy | `t_stop` als Feature | ROC-AUC: 0,769 |
| **`recurrent_survival_model.py`** | Semester-Sequenz | $(N, 16, 20)$ | Masking $\rightarrow$ GRU(64) $\rightarrow$ GRU(32) $\rightarrow$ Dense(1) | Time-Distributed BCE | Implicit (Recurrent) | ROC-AUC: 0,787 |
| **`recurrent_exam_survival.py`** | Prüfungs-Sequenz | $(N, 40, 20)$ | Masking $\rightarrow$ GRU(64) $\rightarrow$ GRU(32) $\rightarrow$ Dense(1) | Time-Distributed BCE | Implicit (Recurrent) | **ROC-AUC: 0,893** |
| **`transformer_survival_model.py`** | Semester-Sequenz | $(N, 16, 20)$ | Dense(64) $\rightarrow$ PosEnc $\rightarrow$ MHA(4 Heads) $\rightarrow$ Dense | Time-Distributed BCE | **Lernbares Embedding** | ROC-AUC: 0,764 |
| **`transformer_exam_survival.py`** | Prüfungs-Sequenz | $(N, 40, 20)$ | Dense(64) $\rightarrow$ PosEnc $\rightarrow$ MHA(4 Heads) $\rightarrow$ Dense | Time-Distributed BCE | **Lernbares Embedding** | ROC-AUC: 0,875 |
| **`dynamic_deephit_model.py`** | Semester-Sequenz | $(N, 16, 20)$ | GRU(64) $\rightarrow$ Shared Sub-Net $\rightarrow$ Cause 1 & 2 | Cause-Specific BCE + Ranking | Implicit (Recurrent) | ROC-AUC: 0,774 |
| **`autoregressive_next_exam.py`** | Next-Exam ($t_{k+1}$) | Hist: $(N, 30, 12)$<br>Ctx: $(N, 6)$ | GRU(64) Trunk + Dense Context $\rightarrow$ Dual Output | MSE (Note) + BCE (Pass) | Implicit (Recurrent) | $R^2$: 0,443 / AUC: 0,937 |
| **`autoregressive_deep_transformer.py`**| Next-Exam ($t_{k+1}$) | Hist: $(N, 30, 12)$<br>Ctx: $(N, 6)$ | Dense(64) $\rightarrow$ SinCos $\rightarrow$ 3x MHA $\rightarrow$ Dual Output | MSE (Note) + BCE (Pass) | **Sin/Cos (Vaswani 2017)** | **$R^2$: 0,477 / AUC: 0,942** |
| **`deep_transformer_regression.py`** | Multi-Task Suite | $(N, 40, 20)$ | Dense(128) $\rightarrow$ 3x MHA(8 Heads) $\rightarrow$ AttnPooling | MSE / Masked BCE | **KEIN PosEnc (Lücke!)** | **UNDER REVISION** |

---

## 2. Klärung: Feature Builder vs. Positional Encoding auf Modellebene

> [!NOTE]
> **Audit-Befund zur Datenübergabe:**
> - Der [`feature_builder.py`](file:///c:/GitHub_public/Abschlussprojekt/src/feature_builder.py) liefert für **alle** Transformer-Modelle denselben vollständigen Tensor `(N, T, F)` inklusive der Zeitspalte `fachsemester` und `pruefungs_nr`.
> - Der Unterschied lag **ausschließlich auf Modellebene**:
>   - `transformer_survival_model.py`: Addiert im Modellcode eine lernbare Positionsmatrix `x = x + self.pos_embedding`.
>   - `autoregressive_deep_transformer.py`: Addiert im Modellcode eine analytische Sin/Cos-Funktion `x = x + SinCosPositionalEncoding()(x)`.
>   - `deep_transformer_regression.py`: Hatte **vergessen**, eine Positional-Encoding-Schicht im Keras-Graph einzubauen. Die reine Self-Attention ist permutationsinvariant – das Modell verlor die strikte zeitliche Reihenfolge.

---

## 3. Masterplan zur systematischen Modellüberarbeitung (5 Module)

```
REFINEMENT-ROADMAP (5 MODULE)
├── Modul 1: Noten-Regression: `gradeblind` als offizieller Primär-Standard
├── Modul 2: Sequenz-Backbone Head-to-Head (GRU vs. LSTM vs. Dilated TCN)
├── Modul 3: Transformer-Harmonisierung (Einheitliches SinCos + Causal Masking)
├── Modul 4: Regularisierung (ReduceLROnPlateau, EarlyStopping, L1/L2 Weight Decay)
└── Modul 5: Asymmetrische Verlustfunktionen (Focal Loss & Class Weights für Survival)
```

---

### Modul 1: `gradeblind` als verbindlicher Default für Notenregression
- **Rationale:** Wenn einem Notenregressor frühere Noten zur Vorhersage der Abschlussnote übergeben werden, lernt das Modell im Wesentlichen nur den CP-gewichteten Mittelwert (Tautologie / Leakage).
- **Entscheidung:**
  - **`gradeblind`** (nur ECTS, Versuchsanzahl, Semesterfolge, Support-Teilnahmen) wird der **offizielle Primär-Benchmark** für alle Notenvorhersagen.
  - Der `standard`-Modus wird nur als theoretische Obergrenze dokumentiert.

---

### Modul 2: Head-to-Head Sequenz-Benchmark (GRU vs. LSTM vs. Dilated TCN)
- **Hintergrund:** Das Recurrent Exam GRU ist aktuell der stärkste Survival-Klassifikator ($\text{ROC-AUC} = \mathbf{0,8930}$).
- **Experimenteller Vergleich (auf identischem Split):**
  1. **Standard GRU** (64 Units, 2 Layers)
  2. **Standard LSTM** (64 Units, 2 Layers mit explizitem Zellzustand $c_t$)
  3. **Temporal Convolutional Network (1D Dilated TCN):** Parallele zeitliche Faltungen mit exponentiellem Dilationsfaktor ($d \in \{1, 2, 4, 8\}$) – extrem schnell auf Multi-Core CPU!

---

### Modul 3: Transformer-Harmonisierung & Positional Alignment
- **Maßnahmen auf Modellebene:**
  1. **Einheitliches SinCosPositionalEncoding:** Alle Transformer-Modelle erhalten die analytische Sin/Cos-Codierung.
  2. **Kausales Attention-Masking:** Verhindert Look-Ahead ($M_{ij} = -\infty$ für $j > i$) bei sequenziellen Schätzungen.
  3. **Masked Loss:** Ignoriert gepaddete Zeitschritte im Gradientenfluss.
  4. **Schlankes $d_{\text{model}} = 64$:** Bietet bessere Generalisierung und $4\times$ schnellere Trainingszeiten als $d=128$.

---

### Modul 4: Erweiterte Regularisierung & Trainingsstabilität
- **Bestätigung:** Die Umstellung von BatchNorm auf `LayerNormalization` ist bereits in allen Sequenz- und Transformer-Netzen vollzogen.
- **Erweiterungen:**
  - **`ReduceLROnPlateau`:** Dynamische Absenkung der Lernrate um $50\,\%$ (`patience=5`, `min_lr=1e-6`), wenn der `val_loss` stagniert.
  - **`EarlyStopping` mit großzügiger Patience:** `patience=12` mit `restore_best_weights=True`.
  - **L1/L2-Weight Decay:** Einsatz von `kernel_regularizer=tf.keras.regularizers.l2(1e-4)` als Alternative/Ergänzung zu Dropout, um Overfitting auf Tabellendaten zu unterdrücken.

---

### Modul 5: Asymmetrische Verlustfunktionen (Focal Loss & Class Weights)
- **Hintergrund:** Im Personen-Semester-Panel liegt die Event-Rate bei nur $\approx 3,8\,\%$, im Prüfungs-Panel bei $\approx 12\,\%$. Standard Binary Cross-Entropy neigt dazu, die Mehrheitsklasse (Verbleib) zu bevorzugen.
- **Lösungsoptionen:**
  1. **Class Weights:** Automatische Gewichtung $w_{\text{dropout}} = \frac{1}{\pi_0}$ im Cross-Entropy-Loss.
  2. **Focal Loss (Lin et al.):**
     $$\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)$$
     Mit $\gamma = 2,0$ werden einfache "Non-Event"-Beispiele gedämpft und der Gradient fokussiert sich auf die schwierigen Abbruchmuster $\rightarrow$ **erheblicher Schub für den PR-AUC auf der Minderheitsklasse**.
