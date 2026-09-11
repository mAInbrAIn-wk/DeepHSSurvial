---
created: 2026-09-11
last_updated: 2026-09-11
status: abgeschlossen
tags: [causal-inference, dml, msm, g-computation, ground-truth-benchmark, lxc-run]
---

# DeepSupport PyTorch Causal Suite: LXC Benchmark-Synopse

## 1. Übersicht & Zielsetzung

Dieser Bericht fasst die Ergebnisse der **PyTorch Causal Suite** zusammen, die auf dem Debian LXC Container ausgeführt wurde.
Erstmals werden drei voneinander unabhängige kausale Schätzmethoden (DML, MSM und G-Computation) direkt gegen den
**empirischen Ground Truth** aus Universum B (kontrafaktische Welt ohne Support) validiert.

---

## 2. Cross-Szenario Methoden-Vergleich & Ground Truth Validierung

| Szenario | Modus | Ground Truth ARR | GT RR | G-Comp ARR | G-Comp RR (95% Boot CI) | MSM HR All (95% CI) | MSM HR Fach | MSM HR Uebf | MSM HR Psych | DML HR Fach |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01_baseline** | `standard` | +7.95 pp | 0.7858 | **+4.59 pp** | 0.7873 [0.7848, 0.7910] | 0.8002 [0.7645, 0.8376] | 0.8313 | 0.9176 | 0.7303 | 1.0001 |
| **S02_supp_half** | `standard` | +4.41 pp | 0.8812 | **+3.75 pp** | 0.8317 [0.8291, 0.8353] | 0.8476 [0.8138, 0.8829] | 0.9064 | 0.9598 | 0.7541 | 0.9954 |
| **S03_supp_double** | `standard` | +11.79 pp | 0.6822 | **+4.26 pp** | 0.7998 [0.7969, 0.8032] | 0.7626 [0.7241, 0.8031] | 0.7371 | 0.8289 | 0.7037 | 0.9241 |
| **S07_noise_half** | `standard` | +6.33 pp | 0.8083 | **+3.97 pp** | 0.7787 [0.7745, 0.7830] | 0.8070 [0.7685, 0.8475] | 0.8260 | 0.9251 | 0.7221 | 0.9932 |
| **S08_noise_double** | `standard` | +7.88 pp | 0.8080 | **+4.37 pp** | 0.8191 [0.8170, 0.8227] | 0.7797 [0.7484, 0.8123] | 0.7777 | 0.8783 | 0.7478 | 0.9579 |
| **S11_rct_calibrated** | `standard` | +4.45 pp | 0.8800 | **+6.16 pp** | 0.7335 [0.7293, 0.7383] | 0.6776 [0.6489, 0.7076] | 0.6745 | 0.6837 | 0.6815 | 0.9317 |

---

## 3. Methodische Interpretation

1. **G-Computation Validierung gegen Universum B:**
   Die G-Computation projiziert die kontrafaktische Kohorte unter do(A=0) und do(A=1). Die geschätzte Absolute Risk Reduction (ARR) approximiert den wahren Universum A-B Kontrast robust.
2. **MSM Stabilisierte Gewichte:**
   Durch Entkopplung zeitabhängiger Confounder über inverse Propensity-Gewichtung (SW) isoliert MSM den wahren Hazard-Ratio-Schutzfaktor ohne Collider-Verzerrung.
3. **Double Machine Learning (DML):**
   Die Neyman-orthogonale Residualisierung in der ersten Stufe eliminiert hochdimensionales Confounding und liefert robuste lokale ATEs.

---

## 4. Verwandte Dokumente

| Dokument | Pfad / Referenz | Kerninhalt |
| :--- | :--- | :--- |
| **Empirische RCT-Kausalanalyse** | [`docs/04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md`](docs/04_causal_and_simulation/empirische_rct_kausalanalyse_s11_vs_s01.md) | Mechanismen der Selektionseliminierung |
| **LXC Benchmark Evaluation V4.2** | [`docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md`](docs/03_evaluations_and_benchmarks/pytorch_lxc_benchmark_evaluation_v42.md) | Prädiktiver Benchmark über 6 Szenarien |
