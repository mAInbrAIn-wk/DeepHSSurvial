# Agent Guidelines & Execution Rules for DeepSupport

---
created: 2026-09-06
last_updated: 2026-09-06
status: active
tags: [agent-rules, environment, conventions]
---

## ⚠️ MANDATORY PYTHON ENVIRONMENT RULE (CRITICAL)

**All Python executions MUST use the dedicated virtual environment:**

```powershell
# Windows PowerShell Execution Pattern:
$env:PYTHONPATH = "src"
C:\GitHub_public\.venv\Scripts\python.exe <script_path> [args]
```

### Why?
1. **Windows Application Control Policy:** The system-wide Python interpreter (`C:\Users\wilfr\AppData\Local\Programs\Python\Python312\python.exe` or `python` from default PATH) is subject to strict Windows Defender Application Control policies. Core scientific native DLLs (such as `scipy.optimize._highspy._core` / HiGHS solver, C++ bindings, etc.) are **blocked** by security policy when invoked through the system Python.
2. **Whitelisted venv:** The virtual environment at `C:\GitHub_public\.venv` is fully configured, whitelisted, and contains all compatible wheels:
   - TensorFlow / Keras 3.x
   - PyTorch 2.x
   - Scikit-Learn, Scikit-Survival, Lifelines
   - DuckDB, Arrow, Pandas, NumPy
3. **RULE:** **NEVER** run bare `python <script>` or `py <script>`. Always invoke `C:\GitHub_public\.venv\Scripts\python.exe` or ensure `$env:PYTHONPATH = "src"` is set and `C:\GitHub_public\.venv\Scripts\Activate.ps1` is loaded.

---

## 📝 DOCUMENTATION & LINKING STANDARDS

1. **YAML Frontmatter:** Every markdown document in `docs/` must begin with standard YAML frontmatter:
   ```yaml
   ---
   created: YYYY-MM-DD
   last_updated: YYYY-MM-DD
   status: [abgeschlossen | in_bearbeitung | archiv]
   tags: [tag1, tag2, tag3]
   ---
   ```
2. **Relative Markdown Links:** Always use relative links (e.g. `[Text](../03_evaluations_and_benchmarks/master_synopse_v4_gesamt.md)`).
   - **DO NOT** use `file:///C:/...` URIs in committed markdown documents. Relative links work natively on GitHub and in local markdown readers.
3. **Cross-Referencing Table:** Major analyses and synopses must contain a `## Verwandte Dokumente` table at the end connecting related threads across the 8 documentation folders.

---

## 🧠 DATA & EVALUATION POLICIES

1. **Zero-Imputation Policy:** Missing values in evaluation metrics must be logged as `null` (`None`), **never** imputed as `0.0`.
2. **OOP Evaluator Classes:** All model evaluations must use the 5 standardized OOP classes in `src/deepsupport/evaluation/metrics_logger.py`:
   - `SurvivalEvaluator` (Dropout / Survival Models)
   - `RegressionEvaluator` (Continuous Grade / GPA Prediction)
   - `MulticlassEvaluator` (Multi-class Landmark Status Prediction)
   - `CausalEvaluator` (Double Machine Learning, Hazard Ratios, Bootstrap & Asymptotic CIs)
   - `DualHeadEvaluator` (Autoregressive Next-Exam Grade + Pass Multi-Task Models)
3. **Evaluation PR-AUC Policy:** Precision-Recall AUC must always be logged for **all classes** (both minority event $y=1$ and majority event $y=0$), together with baseline prevalence $\pi_0$.
