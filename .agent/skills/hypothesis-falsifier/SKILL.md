---
name: hypothesis-falsifier
description: >-
  Systematically transforms observational explanations, narrative speculations, and model discrepancies into structured, falsifiable scientific hypotheses with concrete verification criteria and automated verification scripts. Use whenever analyzing complex simulation mechanics, causal effects, or discrepancies across model architectures and data versions.
---

# Hypothesis Falsifier: Empirical Verification & Falsification Protocol

## 1. Overview & Purpose

When analyzing complex machine learning models, counterfactual simulations, and causal inference results, narrative explanations (e.g., *"Model X probably predicts Y because the DGP mechanism changed Z"*) are useful as an initial brainstorming step, but dangerous if taken as truth.

The **Hypothesis Falsifier** enforces a disciplined scientific process:
1. **Deconstruct** narrative claims into discrete, falsifiable statements.
2. **Formulate** testable hypotheses ($H_0$ vs. $H_1$) at the correct level of granularity (Macro, Panel, Semester, Exam, Individual Student).
3. **Establish** explicit falsification criteria (what specific empirical evidence kills the hypothesis?).
4. **Implement & Execute** automated verification code on actual data or simulations.
5. **Synthesize** verified results into decision-grade documentation.

---

## 2. When to Activate This Skill

Activate this skill whenever:
- You or the user observe an unexpected metric, paradox, or discrepancy between model types (e.g., Linear Cox vs. Deep DML).
- Comparing results across dataset versions (e.g., V3.6 vs. V4.1) or across universes (e.g., Universe A vs. Universe C).
- Formulating explanations for why a treatment effect is detected, inverted, or attenuated.
- Evaluating whether a model finding reflects a genuine causal mechanism or an artifact of confounding/leakage.

---

## 3. The 5-Step Falsification Protocol

### Step 1: Claim Extraction & Taxonomy
Review the narrative analysis and extract every claim into one of three categories:
- **[DGP Claim]:** Assertions about the Data Generating Process (e.g., parameter values, distributions, thresholds).
- **[Confounder/Selection Claim]:** Assertions about who receives treatment and their latent traits (e.g., motivation gap, baseline hazards).
- **[Estimator Capability Claim]:** Assertions about model inductive bias (e.g., linear vs. non-linear, temporal attention, sample size requirements).

### Step 2: Formal Hypothesis Specification
For every extracted claim, formulate a hypothesis pair:
- **Null Hypothesis ($H_0$):** The observed difference or effect is purely stochastic noise, an artifact of sample selection, or unassociated with the claimed mechanism.
- **Alternative Hypothesis ($H_1$):** The claimed mechanism causes the shift, with a specified direction and minimum detectable effect size.
- **Granularity Level:**
  - `Macro`: Universe-wide dropout rate, ARR, NNT ($N = 50.000$).
  - `Student-Level`: Fixed attributes, final graduation status, career timeline.
  - `Person-Semester`: Longitudinal panel ($T = 1..16$, time-varying covariates).
  - `Exam-Level`: Micro-level grades, attempts, fail/pass binary outcomes ($N \approx 1.2M$ rows).

### Step 3: Falsification Criteria & Guardrails
Define what would **falsify** $H_1$ *before* inspecting the data:
- **Directional Violation:** e.g., If higher support dosage yields a higher hazard ratio ($HR > 1.0$) rather than protective ($HR < 1.0$), $H_1$ is falsified.
- **Effect Size Threshold:** e.g., If the difference in means between support users and non-users is $|\Delta| < 0.01$, the selection claim is falsified.
- **Anti-Hallucination Rule:** NEVER state a number without running code. If a historical number is recalled from memory, mark it as `[UNVERIFIED_MEMORY]` until confirmed against disk artifacts.

### Step 4: Verification Execution
Generate a standalone Python script in `scratch/`:
- **Mandatory Environment:**
  ```powershell
  $env:PYTHONPATH = "src"
  C:\GitHub_public\.venv\Scripts\python.exe <script_path>
  ```
- **Zero-Imputation Policy:** Missing evaluation metrics must be `None` / `null`, never imputed as `0.0`.
- **Reproducibility:** Log dataset paths, sample size $N$, RNG seed, and execution timestamp.

### Step 5: Decision Matrix & Documentation
Classify the hypothesis outcome:
- `CONFIRMED`: Empirical data strictly satisfies $H_1$ criteria.
- `FALSIFIED`: Data contradicts $H_1$ or satisfies $H_0$.
- `AMBIGUOUS`: Insufficient power, confounding variables present, or requires counterfactual re-simulation.

Update the relevant analysis document or artifact with a structured verdict table:

| ID | Hypothesis | Level | Test Metric | Falsification Bound | Empirical Result | Verdict |
|:---|:---|:---|:---|:---|:---|:---:|
| H1 | Dosis-Skalierung | Model | $\Delta HR_{\text{DML}}$ | $HR \ge 0.95$ | $HR = 0.85$ | **CONFIRMED** |
| H2 | Linearer Cox-Bias | Macro | $HR_{\text{Oracle}}$ | $HR < 0.95$ | $HR = 1.0037$ | **CONFIRMED** |

---

## 4. Standard Falsification Templates

### Template A: Treatment vs. Selection Disentanglement
```python
# Measure latent baseline traits BEFORE treatment uptake
support_users = df[df['has_support'] == 1]
non_users = df[df['has_support'] == 0]
delta_latent = support_users['hidden_motivation'].mean() - non_users['hidden_motivation'].mean()
print(f"Selection Gap (Latent): {delta_latent:.4f}")
# Falsification check: if delta_latent >= 0, negative selection is FALSIFIED
```

### Template B: Counterfactual Migration Tracking
```python
# Merge identical students across Universes (e.g. A vs C)
m = pd.merge(df_A[['studierenden_id', 'status']], df_C[['studierenden_id', 'status']], on='studierenden_id', suffixes=('_A', '_C'))
saved = ((m['status_A'] == 'dropout') & (m['status_C'] == 'graduated')).sum()
hurt = ((m['status_A'] == 'graduated') & (m['status_C'] == 'dropout')).sum()
print(f"Hurt: {hurt}, Saved: {saved}, Net: {hurt - saved}")
```

---

## 5. Checklists for Agents
- [ ] Did you check whether historical numbers belong to V3.1/V3.2, V3.6 clean, or V4.1?
- [ ] Did you inspect `generation_metadata.json` or git commits for the generator provenance?
- [ ] Did you test whether the empirical finding holds on both standard and oracle feature sets?
- [ ] Are all reported metrics verified with an executable script?
