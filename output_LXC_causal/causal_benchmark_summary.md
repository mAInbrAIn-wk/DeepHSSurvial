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

| Szenario | Ground Truth ARR | G-Comp ARR | G-Comp RR (95% CI) | MSM HR (All Support) | DML ATE (Fachlich) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S01_baseline** | +7.95 pp | **+4.59 pp** | n/a | n/a | +0.0028 |
| **S02_supp_half** | +4.41 pp | **+3.75 pp** | n/a | n/a | +0.0055 |
| **S03_supp_double** | +11.79 pp | **+4.26 pp** | n/a | n/a | +0.0009 |
| **S07_noise_half** | +6.33 pp | **+3.97 pp** | n/a | n/a | +0.0034 |
| **S08_noise_double** | +7.88 pp | **+4.37 pp** | n/a | n/a | +0.0014 |
| **S11_rct_calibrated** | +4.45 pp | **+6.16 pp** | n/a | n/a | -0.0009 |

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
