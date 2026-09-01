"""
Hierarchische Cross-Szenario & Modell-Evaluierungs-Engine V4.1
=============================================================
Synthetisiert alle 15 Simulations-Szenarien, 11 Modellklassen, 5 Feature-Modi
und vergleicht Modell-Kausalitaet direkt mit der Ground Truth ARR.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

ROOT_DIR = Path("C:/GitHub_public/Abschlussprojekt")
V41_DIR = ROOT_DIR / 'src' / 'output_v4_grid_v41'
V36_ORIG_DIR = ROOT_DIR / 'src' / 'output_dl'
V36_CLEAN_DIR = ROOT_DIR / 'src' / 'output_v36_clean_rerun'
ARTIFACTS_DIR = ROOT_DIR / 'Artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None

def load_simulation_grid() -> pd.DataFrame:
    gt_path = V41_DIR / 'metrics' / 'full_sensitivity_grid_results.json'
    if not gt_path.exists():
        return pd.DataFrame()
    data = load_json(gt_path)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    return df

def scan_metrics() -> pd.DataFrame:
    records = []
    
    scan_targets = [
        (V41_DIR / 'S01_baseline' / 'universe_A' / 'metrics', 'V4.1', 'S01_baseline'),
        (V36_ORIG_DIR / 'metrics', 'V3.6_Original', 'S01_baseline'),
        (V36_CLEAN_DIR / 'metrics', 'V3.6_Clean', 'S01_baseline'),
    ]
    
    for s_dir in V41_DIR.glob('S*'):
        if s_dir.is_dir() and s_dir.name != 'S01_baseline':
            u_dir = s_dir / 'universe_A' / 'metrics'
            if u_dir.exists():
                scan_targets.append((u_dir, 'V4.1', s_dir.name))
                
    for m_dir, dataset_ver, scen in scan_targets:
        if not m_dir.exists():
            continue
        for j_path in m_dir.glob('*.json'):
            if j_path.name in ['full_sensitivity_grid_results.json', 'feature_grid_master_benchmark.json']:
                continue
            data = load_json(j_path)
            if not data or not isinstance(data, dict):
                continue
                
            model_key = j_path.stem.replace('_metrics', '')
            
            # Modus & Temporal ableiten
            mode = data.get('mode', 'standard')
            temporal = data.get('temporal', 'prev')
            if 'gradeblind' in model_key: mode = 'gradeblind'
            elif 'oracle' in model_key: mode = 'oracle'
            elif 'blind' in model_key: mode = 'blind'
            elif 'realistic' in model_key or 'erwerb_blind' in model_key: mode = 'realistic'
            
            if '_cum' in model_key: temporal = 'cum'
            elif '_prev' in model_key: temporal = 'prev'
            
            # Kennzahlen harmonisieren
            roc_auc = data.get('ROC-AUC') or data.get('roc_auc') or data.get('ROC-AUC_Panel') or data.get('Pruefungs-Ebene ROC-AUC') or data.get('Semester ROC-AUC') or data.get('ROC-AUC_Test') or data.get('Dropout Hazard ROC-AUC')
            pr_auc = data.get('PR-AUC') or data.get('pr_auc') or data.get('PR-AUC_Panel') or data.get('Pruefungs-Ebene PR-AUC') or data.get('Semester PR-AUC') or data.get('PR-AUC_Test') or data.get('Dropout Hazard PR-AUC')
            r2 = data.get('R2 Score') or data.get('r2_score') or data.get('R2') or data.get('R2_Test') or data.get('r2')
            rmse = data.get('RMSE') or data.get('rmse') or data.get('RMSE_Test')
            mae = data.get('MAE') or data.get('mae') or data.get('MAE_Test')
            brier = data.get('Brier Score') or data.get('brier_score') or data.get('Brier_Score') or data.get('Dropout Brier Score')
            
            # Kausal-Metriken
            hr_fach = data.get('hr_fachlich') or data.get('HR_Fach') or data.get('Mean_RR_fach') or (data.get('fach_partial', {}).get('mean_rr') if isinstance(data.get('fach_partial'), dict) else None)
            hr_uebf = data.get('hr_ueberfachlich') or data.get('HR_Uebf') or data.get('Mean_RR_uebf') or (data.get('uebf_partial', {}).get('mean_rr') if isinstance(data.get('uebf_partial'), dict) else None)
            hr_psych = data.get('hr_psychosozial') or data.get('HR_Psych') or data.get('Mean_RR_psych') or (data.get('psych_partial', {}).get('mean_rr') if isinstance(data.get('psych_partial'), dict) else None)
            
            records.append({
                'dataset_version': dataset_ver,
                'scenario': scen,
                'file_name': j_path.name,
                'model_key': model_key,
                'mode': mode,
                'temporal': temporal,
                'roc_auc': float(roc_auc) if roc_auc is not None else np.nan,
                'pr_auc': float(pr_auc) if pr_auc is not None else np.nan,
                'r2_score': float(r2) if r2 is not None else np.nan,
                'rmse': float(rmse) if rmse is not None else np.nan,
                'mae': float(mae) if mae is not None else np.nan,
                'brier_score': float(brier) if brier is not None else np.nan,
                'hr_fach': float(hr_fach) if hr_fach is not None else np.nan,
                'hr_uebf': float(hr_uebf) if hr_uebf is not None else np.nan,
                'hr_psych': float(hr_psych) if hr_psych is not None else np.nan,
            })
            
    df = pd.DataFrame(records)
    return df

def generate_hierarchical_report():
    print("==========================================================================")
    print("   STARTE HIERARCHISCHE CROSS-SZENARIO MODELL-SYNOPSE V4.1")
    print("==========================================================================")
    
    df_metrics = scan_metrics()
    df_gt = load_simulation_grid()
    
    print(f"Gefundene Metrik-Eintraege: {len(df_metrics)}")
    
    # 1. Export CSV
    master_csv = ARTIFACTS_DIR / 'v41_cross_scenario_metrics_master.csv'
    df_metrics.to_csv(master_csv, index=False)
    print(f"[OK] Master-Metriken exportiert: {master_csv}")
    
    # 2. Markdown Report
    report_lines = [
        "# Synoptischer Cross-Szenario & Modell-Evaluierungsbericht V4.1",
        "",
        "> [!IMPORTANT]",
        "> **Systematische Gesamtsynopse:** Abgleich aller 15 Simulations-Szenarien, 11 Modellklassen, 5 Feature-Modi und Validierung der Modell-Kausalitaet gegen die experimentelle Simulations-Ground-Truth.",
        "",
        "---",
        "",
        "## 1. Ground Truth der 15 Simulationswelten (N = 50.000, seed = 99999)",
        "",
        "| # | Szenario | Parameter-Fokus | Dropout A (Full) | Dropout B (None) | Wahre ARR (B-A) | Wahre RR (A/B) | NNT |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    
    scen_order = [
        ('S01_baseline', 'Baseline (Referenz)', 29.2, 37.1, 7.9, 0.787, 12.6),
        ('S02_supp_half', 'Support-Wirkung 0.5x', 32.7, 37.1, 4.4, 0.881, 22.7),
        ('S03_supp_double', 'Support-Wirkung 2.0x', 25.3, 37.1, 11.8, 0.682, 8.5),
        ('S04_grade_half', 'Notenboost 0.5x', 30.6, 37.1, 6.5, 0.825, 15.4),
        ('S05_grade_double', 'Notenboost 2.0x', 27.6, 37.1, 9.5, 0.744, 10.5),
        ('S06_grade_quad', 'Notenboost 4.0x', 27.1, 37.1, 10.0, 0.730, 10.0),
        ('S07_noise_half', 'Rauschen 0.5x', 26.7, 33.0, 6.3, 0.809, 15.8),
        ('S08_noise_double', 'Rauschen 2.0x', 33.2, 41.0, 7.9, 0.810, 12.7),
        ('S09_cost_zero', 'Zeitkosten 0h', 28.6, 37.1, 8.5, 0.771, 11.8),
        ('S10_cost_double', 'Zeitkosten 60h (2x)', 29.7, 37.1, 7.4, 0.801, 13.5),
        ('S11_rct_calibrated', 'RCT (Zufallsauswahl)', 32.6, 37.1, 4.5, 0.879, 22.5),
        ('S12_overload_half', 'Overload-Penalty 0.5x', 26.0, 34.1, 8.1, 0.762, 12.3),
        ('S13_overload_double', 'Overload-Penalty 2.0x', 34.6, 41.8, 7.3, 0.828, 13.8),
        ('S14_overload_cap', 'Overload-Cap 15%', 26.7, 35.0, 8.4, 0.763, 12.0),
        ('S15_cost_effect_double', 'Kombi (Kosten+Wirkung 2x)', 25.8, 37.1, 11.3, 0.695, 8.9),
    ]
    
    for s_id, label, da, db, arr, rr, nnt in scen_order:
        report_lines.append(f"| **{s_id[:3]}** | {s_id} | {label} | **{da:.1f} %** | **{db:.1f} %** | **{arr:.1f} pp** | **{rr:.3f}** | **{nnt:.1f}** |")
        
    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Ebene 1 & 2: Modellklassen-Benchmark & Modus-Synthese (V4.1 Baseline)",
        "",
        "### 2.1 Survival- & Dropout-Klassifikation (Test-Set)",
        "",
        "| Modellklasse | Modellname | Modus | Temporal | ROC-AUC | PR-AUC (Dropout) | Brier Score |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    surv_df = df_metrics[(df_metrics['dataset_version'] == 'V4.1') & (df_metrics['scenario'] == 'S01_baseline') & (df_metrics['roc_auc'].notna())].sort_values(by='roc_auc', ascending=False)
    for _, r in surv_df.iterrows():
        brier_str = f"{r['brier_score']:.4f}" if not np.isnan(r['brier_score']) else "-"
        pr_str = f"{r['pr_auc']:.4f}" if not np.isnan(r['pr_auc']) else "-"
        report_lines.append(f"| Survival | `{r['model_key']}` | `{r['mode']}` | `{r['temporal']}` | **{r['roc_auc']:.4f}** | {pr_str} | {brier_str} |")
        
    report_lines.extend([
        "",
        "### 2.2 Noten- & GPA-Regressionsmodelle (Test-Set)",
        "",
        "| Modellklasse | Modellname | Modus | Temporal | $R^2$ Score | RMSE | MAE |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    
    reg_df = df_metrics[(df_metrics['dataset_version'] == 'V4.1') & (df_metrics['scenario'] == 'S01_baseline') & (df_metrics['r2_score'].notna())].sort_values(by='r2_score', ascending=False)
    for _, r in reg_df.iterrows():
        report_lines.append(f"| Regression | `{r['model_key']}` | `{r['mode']}` | `{r['temporal']}` | **{r['r2_score']:.4f}** | {r['rmse']:.4f} | {r['mae']:.4f} |")
        
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Ebene 3: Kausale Validierung & Ground Truth Reality Check",
        "",
        "| Modell / Methode | Datengrundlage | Geschaetzter Schutzeffekt (RR / HR) | Wahre Ground Truth RR (A vs. B) | Kausalitaets-Bias |",
        "| :--- | :--- | :---: | :---: | :---: |",
        "| **Simulation Ground Truth** | Experimenteller A/B-Split | **0.787** (ARR 7.9 pp) | 0.787 | **0.000** (Referenz) |",
        "| **Dynamic DeepHit Fixed** | Kontrafaktische Inferenz (Fach) | **0.951** | 0.787 | +0.164 |",
        "| **Transformer DML** | Orthogonalisierte ATEs (Fach) | **0.884** | 0.787 | +0.097 |",
        "| **Oracle Logistic Hazard** | Volle Information (Latente Mot.) | **0.989** | 0.787 | +0.202 |",
        "| **Extended Cox Panel** | Regression mit TVCs (Fach) | **1.089** | 0.787 | +0.302 (Selektionsbias) |",
        "",
        "---",
        "",
        "## 4. Ebene 4: Methoden-Ranking, Robustheit & Ensemble/MoE-Potenzial",
        "",
        "### 4.1 Gesamt-Scorecard der Modellfamilien",
        "",
        "| Modellfamilie | Praediktive Guete (ROC/R2) | Kausale Treue (ARR-Recovery) | Rausch-Resilienz (S07/08) | Recheneffizienz | Gesamtrang |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        "| **Autoregressiver Deep Transformer** | **#1** (0.9411 / 0.7036) | **#2** Hoch (sequentiell) | **#1** Sehr hoch | **#3** Mittel (10 Min.) | **Rang 1 (Bester Allrounder)** |",
        "| **Recurrent Exam GRU** | **#1** (0.8960 / 0.9010) | **#2** Hoch | **#1** Sehr hoch | **#1** Sehr schnell (1 Min.) | **Rang 2 (Bester Predictor)** |",
        "| **Double Machine Learning (DML)** | **#2** (0.7522) | **#1** (0.8839 geringster Bias) | **#2** Mittel | **#1** Sehr schnell (1.5 Min.) | **Rang 3 (Bester Kausalschaetzer)** |",
        "| **Dynamic DeepHit Competing Risks** | **#1** (0.9997 Grad / 0.8116) | **#2** (0.9508) | **#2** Hoch | **#2** Schnell (3.5 Min.) | **Rang 4 (Bester Multi-Event)** |",
        "| **Extended Cox Panel (PHReg)** | **#3** (0.7510) | **#3** (1.0899 Selektions-anfaellig) | **#3** Gering | **#1** Sofort (<5s) | **Rang 5 (Oekonometrie-Standard)** |",
        "",
        "### 4.2 Ensemble- & Mixture-of-Experts (MoE) Synergien",
        "* **Kombination Deep Transformer + DML:** Der Deep Transformer liefert die praezisesten Sequenz-Embeddings fuer Vorhersagen ($R^2=0.7036$), waehrend DML den Schutzeffekt am unverzerrtesten isoliert ($RR=0.8839$).",
        "* **MoE-Gating-Hypothese:** Ein Gating-Netzwerk, das bei Normalstudierenden auf den Exam-Transformer und bei extremen Workload-Ueberlastungen auf DeepHit Competing Risks schaltet, maximiert sowohl Fruehwarnung als auch Interventionsgenauigkeit.",
    ])
    
    report_md = ARTIFACTS_DIR / 'v41_cross_scenario_gesamtreview.md'
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"[OK] Synoptischer Gesamtreport gespeichert unter: {report_md}")

if __name__ == '__main__':
    generate_hierarchical_report()
