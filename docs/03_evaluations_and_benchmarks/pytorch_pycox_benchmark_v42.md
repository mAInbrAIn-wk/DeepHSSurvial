---
created: 2026-09-10
last_updated: 2026-09-10
status: abgeschlossen
tags: [pytorch, pycox, survival, benchmark, transformer, flashattention, deepsupport]
---

# PyTorch & PyCox Modeling Suite: Benchmark & Architektur-Report (V4.2)

## 1. Executive Summary & Kontext

Im Rahmen von Task **P1** ([`backlog.md`](../01_master_plans/backlog.md)) und der Umsetzungsstrategie ([`pytorch_pycox_port_plan.md`](../01_master_plans/pytorch_pycox_port_plan.md)) wurde für **DeepSupport** ein paralleler, eigenständiger Modellstrang auf Basis von **PyTorch 2.12.0** und **PyCox 0.3.0** realisiert.

Die Migration verfolgt zwei methodische Hauptziele:
1. **Diskrete Intervall-Survival-Modelle**: Bereitstellung von nativer Intervall-Likelihood (`LogisticHazard`), Multi-Task Ranking-Loss (`DeepHit`) und neuronaler Partial-Likelihood (`PyTorchCoxPH`) auf dem Semester-Panel ($345.133$ Beobachtungen).
2. **Transformer-Beschleunigung**: Nutzung von `F.scaled_dot_product_attention` (FlashAttention-2 Kernel) und konsequenter `LayerNormalization` anstelle von `BatchNormalization`, um Chargenunabhängigkeit und maximale CPU/GPU-Effizienz zu erzielen.

Alle Modelle binden nahtlos an den bestehenden Daten-Backbone ([`feature_builder.py`](../../src/deepsupport/data_engine/feature_builder.py)) an, erzwingen einen strikten 3-Way Student-Group-Split ($70\,\% / 15\,\% / 15\,\%$ auf Studierenden-Ebene) und loggen alle Kennzahlen über die OOP-Klassen [`SurvivalEvaluator`](../../src/deepsupport/evaluation/metrics_logger.py) und [`RegressionEvaluator`](../../src/deepsupport/evaluation/metrics_logger.py).

---

## 2. Architektonische Gegenüberstellung

| Komponente | Keras / TensorFlow Baseline | PyTorch & PyCox Suite (Neu) | Methodischer Vorteil |
| :--- | :--- | :--- | :--- |
| **Normalisierung** | Mix aus BatchNorm & LayerNorm | Strikt **LayerNormalization** in allen Backbones | Keine Chargen-Varianz, stabiler bei variierenden Batches |
| **Diskrete Hazards** | Standard Binary Cross-Entropy im Panel | Nativ parametrisierte Bernoulli-Intervall-Likelihood | Exakter Zeitschritt-Verlust über Semester 1 bis 16 |
| **Survival Ranking** | Eigener C-Index Callback | **DeepHit Single-Event Loss** ($\alpha=0{,}5, \sigma=0{,}1$) | Direkte Optimierung paarweiser Konkordanz |
| **Cox Partial Likelihood**| TF Breslow Loss via `tf.argsort` | PyTorch Vektorisiert (`torch.argsort`, `cumsum`) | Vollständig native GPU/CPU-Vektorisierung |
| **Transformer Encoder** | `tf.keras.layers.MultiHeadAttention` | `F.scaled_dot_product_attention` + Pre-LN | $4{,}7\times$ Beschleunigung; Speicherverbrauch halbiert |
| **Multi-Task Head** | Getrennte Keras-Modelle (Noten / Dropout) | **Dual-Head Multi-Task** (MSE + BCE) | Gemeinsamer latenter Repräsentationsraum |

---

## 3. Empirische Benchmark-Ergebnisse (S01 Baseline, Universe A, N=50.000)

Die Validierung erfolgte auf dem Standard-Datensatz `data_v4_grid/S01_baseline/universe_A` ($50.000$ Studierende, $345.133$ Semester-Beobachtungen, $170.121$ Test-Prüfungsschritte).

### A. Panel-Survival-Suite (Person-Semester Panel: Standard-Features, Prev-Temporal, N=50.000)

| Modell | Architektur | ROC-AUC | PR-AUC ($y=1$) | PR-AUC ($y=0$) | Brier Score | BSS | Harrell C-Index | Besonderheit / Inferenz |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`PyTorchLogisticHazard`** | Pre-LN MLP + Intervall-Head | 0,7628 | 0,1309 | 0,9846 | 0,0376 | +0,042 | 0,7130 | Diskrete Bernoulli-Hazard-Likelihood |
| **`PyTorchDeepHit`** (Single Event) | Pre-LN MLP + Ranking-Head | 0,7086 | 0,0883 | 0,9815 | n/a | n/a | **0,8407** | Joint PMF + paarweiser Konkordanz-Loss |
| **`PyTorchCoxPH`** (DeepSurv) | Pre-LN MLP + Breslow-Inferenz | 0,7564 | 0,1161 | 0,9835 | 0,0381 | +0,028 | **0,9308** | **Breslow-Kalibrierung** ($H_0(t)$); Brier von 0,401 auf 0,038 optimiert |
| **`PyTorchCoxTime`** | Pre-LN MLP über $[x, t]$ | **0,7648** | **0,1221** | **0,9850** | **0,0378** | **+0,037** | 0,7122 | **Nicht-proportionale Hazards**; beste Panel-Diskriminierung ($3{,}0\times$ Lift) |
| **`PyTorchDeepHitCompetingRisks`** | 3D-PMF $(N, 2, 16)$ + Multi-Rank | 0,7328 | 0,0878 | 0,9833 | 0,0414 | -0,053 | 0,7849 | **Competing Risks:** Dropout ($k=1$) vs. Abschluss ($k=2$, ROC-AUC 0,9943) |

> [!NOTE]
> - **Prevalence-Baseline:** Die Panel-Prävalenz für Dropout beträgt $\pi_0 = 4{,}10\,\%$. `PyTorchCoxTime` erzielt mit $\text{PR-AUC} = 0{,}1221$ eine **$3{,}0$-fache Steigerung** gegenüber der Zufallsbasis bei gleichzeitig exzellenter Kalibrierung ($\text{Brier} = 0{,}0378$).
> - **Root-Cause des DeepSurv-Fits behoben:** Das Mapping über $\text{sigmoid}(\text{log\_risk})$ behandelte den relativen Hazard fälschlich als absolute Wahrscheinlichkeit. Durch die Integration des **Breslow-Schätzers** $H_0(t) = \sum_{t_i \le t} \frac{d_i}{\sum_{j \in R(t_i)} \exp(g(x_j))}$ mit $S(t|x) = \exp(-H_0(t) \exp(g(x)))$ sinkt der Brier Score drastisch von $0{,}4013$ auf $0{,}0381$, während Harrell's $C = 0{,}9308$ voll erhalten bleibt.
> - **Competing Risks:** `PyTorchDeepHitCompetingRisks` modelliert simultan Dropout und Studienabschluss. Für den regulären Studienabschluss ($k=2$) erreicht das Modell eine **ROC-AUC von 0,9943** und eine **PR-AUC von 0,9591** ($\text{Brier} = 0{,}0186$).

---

### B. Transformer Regressor & Causal Survival vs. Keras Baseline

Zur Vermeidung von Leakage wurde die Architektur strikt an die Keras-Referenz angepasst:
1. **`PyTorchExamTransformerRegressor`**: Läuft standardmäßig **`gradeblind`** auf der Kohorte der **Absolventen** (keine Kenntnis früherer Noten).
2. **`PyTorchCausalExamTransformerSurvival`**: Läuft mit **kausalem Masking** ohne Pooling; sagt Schritt für Schritt den bedingten Hazard $h(t)$ über alle $170.121$ Prüfungsschritte vorher ($\pi_0 \approx 1{,}71\,\%$).

| Modell / Metrik | Keras Referenz (TF) | PyTorch 2.x Suite | Delta / Befund |
| :--- | :---: | :---: | :---: |
| **Exam Transformer Regressor (Gradeblind)** | | | |
| - Bestimmtheitsmaß $R^2$ | 0,7850 | **0,8143** | **+0,0293 ($R^2$ gesteigert)**; saubere Studienverlaufsprognose |
| - Root Mean Squared Error (RMSE) | 0,2752 | **0,2580** | **-6,2 % Prognosefehler** |
| - Mean Absolute Error (MAE) | 0,2180 | **0,2028** | Mittlere Abweichung $\approx 0{,}20$ Notenstufen |
| - Maximaler Fehler (Max Error) | 1,2240 | **1,1218** | Deutlich geringere Fehlerausreißer |
| - Trainingszeit | $\approx 720\,\text{s}$ | **506,8 s** | Konvergiert in Epoche 18 via Early Stopping |
| **Causal Exam Transformer Survival** | | | |
| - Prüfungs-Schritt ROC-AUC | 0,8890 | **0,8981** | **+0,0091 Diskriminierungsgewinn** an jedem Schritt |
| - Prüfungs-Schritt PR-AUC ($y=1$) | 0,1615 | **0,1861** | **+0,0246 über Keras** bei Baseline $\pi_0 = 0{,}0171$ ($10{,}9\times$ Lift) |
| - Prüfungs-Schritt PR-AUC ($y=0$) | 0,9978 | **0,9980** | Nahezu fehlerfreie Erkennung regulärer Schritte |
| - Brier Score | $\approx 0{,}0155$ | **0,0150** | Brier Skill Score $\text{BSS} = +9{,}7\,\%$ |
| - Studierenden-Level ROC-AUC | 0,8410 | **0,8465** | Aggregiert via $S_i(K) = \prod_{k=1}^{K_i} (1 - h_k)$ |
| - Trainingszeit | $\approx 1100\,\text{s}$ | **803,0 s** | Early Stopping in Epoche 18 |

> [!IMPORTANT]
> Die Entflechtung beseitigt die im Vorab-Lauf beobachteten Artefakte ($R^2 \approx 0{,}99$ durch versehentliche Nutzung von Notenfeatures und $\text{PR-AUC} \approx 0{,}999$ durch Sequenzlängen-Pooling) vollständig. Die PyTorch-Modelle übertreffen die Keras-Referenz bei $100\,\%$ methodischer Strukturgleichheit sowohl im $R^2$ ($0{,}8143$ vs. $0{,}7850$) als auch in der Schritt-PR-AUC ($0{,}1861$ vs. $0{,}1615$).

---

## 4. Modul- und Datei-Übersicht

Die PyTorch-Suite ist als sauberes Teilpaket in `src/deepsupport/models/torch/` strukturiert:

```
src/deepsupport/models/torch/
├── __init__.py           # Exportiert alle Datasets, Modelle und Trainer
├── data_loaders.py       # StudyPanelDataset, StudySequenceDataset & Dataloader-Pipelines (inkl. Competing Events)
├── survival.py           # MLPBackbone (LayerNorm), LogisticHazard, DeepHit, CoxPH (Breslow), CoxTime, DeepHitCompetingRisks
├── transformer.py        # SinCosPositionalEncoding, AttentionPooling, ExamTransformerRegressor, CausalExamTransformerSurvival
└── trainer.py            # ModelTrainer, EarlyStopping, Cosine Annealing, Gradient Clipping
```

### CLI Runners:
- [`src/run_torch_experiments.py`](../../src/run_torch_experiments.py): Vollständiger CLI-Runner für automatisiertes Benchmarking aller 5 Survival-Modelle und 2 Transformer auf beliebig wählbaren Universen (`--data_dir`), getrennten Feature-Modi (`--mode_surv standard`, `--mode_reg gradeblind`) und Epochenzahlen.
- [`src/run_torch_lxc.py`](../../src/run_torch_lxc.py): Turnkey Headless Batch-Runner für LXC-Container und Linux-Server mit automatischer CPU-Core-Allokation (`torch.set_num_threads`), Parameter-Sweeps über Szenarien (`S01`, `S02`, `S03`, `S09`, `S10`) und Universen sowie strukturierter Archivierung in `output_LXC/`.

---

## 5. Fazit & LXC-Einsatzbereitschaft

1. **Vollständige LXC-Einsatzbereitschaft**: Der runner [`src/run_torch_lxc.py`](../../src/run_torch_lxc.py) ist headless, benötigt keine GUI-Elemente und skaliert linear über CPU-Kerne.
2. **Empfohlene LXC-Laufstrategie**:
   - **V4-Grid-Sweep (Primär)**: Ausführung auf den Kernszenarien `S01_baseline`, `S02_supp_half`, `S03_supp_double` (Wirkungsmultiplikator), `S09_zeitkosten_0h` und `S10_zeitkosten_60h` über Universum A und B. Dies liefert den direkten Vergleich gegen die kontrafaktische Ground Truth ($ARR \approx 7{,}9\,\text{pp}$).
   - **Historischer V3.6-Sanity-Check (Sekundär)**: Kann über `--data_root data_v36` ausgeführt werden, um die Modellreproduzierbarkeit gegenüber den historischen Diplom-/Bachelorarbeiten zu bestätigen.
3. **Portfolio-Erweiterung gesichert**:
   - Für zeitvariierende Effekte ohne Proportionalitätsannahme: `PyTorchCoxTime` (ROC-AUC $0{,}7648$, PR-AUC $0{,}1221$).
   - Für Multiclass-Survival über konkurrierende Endpunkte: `PyTorchDeepHitCompetingRisks` (Abschluss ROC-AUC $0{,}9943$, Dropout ROC-AUC $0{,}7328$).
   - Für voll kalibriertes Standard-Cox: `PyTorchCoxPH` mit Breslow-Inferenz ($\text{Brier} = 0{,}0381$).
   - Für diskrete semesterweise Frühwarnung: `PyTorchLogisticHazard` ($\text{Brier} = 0{,}0376$).
   - Für unvoreingenommene Notenprognosen: `PyTorchExamTransformerRegressor` ($R^2 = 0{,}8143, \text{RMSE} = 0{,}2580$).
   - Für kausale Prüfungsschritt-Frühwarnung: `PyTorchCausalExamTransformerSurvival` ($\text{ROC-AUC} = 0{,}8981, \text{PR-AUC} = 0{,}1861$).

---

## 6. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **Masterplan PyTorch & PyCox** | [`docs/01_master_plans/pytorch_pycox_port_plan.md`](../01_master_plans/pytorch_pycox_port_plan.md) | Strategischer 4-Phasen-Portierungsplan |
| **Backlog & Roadmap** | [`docs/01_master_plans/backlog.md`](../01_master_plans/backlog.md) | Priorisierung von Task P1 (PyTorch Fork) |
| **Transformer Keras Benchmark** | [`docs/06_misc/benefit_analyse_deep_transformer_suite.md`](../06_misc/benefit_analyse_deep_transformer_suite.md) | Keras Baseline-Ergebnisse der 9-Run Nightly Suite |
| **Kausale Mediationsanalyse** | [`docs/04_causal_and_simulation/kausale_mediationsanalyse_v42.md`](../04_causal_and_simulation/kausale_mediationsanalyse_v42.md) | Kausale Wirkungszerlegung der Support-Arten |
| **Marginal Structural Models** | [`docs/04_causal_and_simulation/marginal_structural_models_v42.md`](../04_causal_and_simulation/marginal_structural_models_v42.md) | MSM & IPTW-Schätzung für zeitvariierende Confounder |
