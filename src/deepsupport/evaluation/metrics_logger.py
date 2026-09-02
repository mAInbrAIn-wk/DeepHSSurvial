import os
import json
from pathlib import Path
import numpy as np

def ensure_dir(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def get_output_dirs(base_dir: Path):
    metrics_dir = base_dir / 'metrics'
    plots_dir = base_dir / 'plots'
    models_dir = base_dir / 'models'
    
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)
    ensure_dir(models_dir)
    
    return metrics_dir, plots_dir, models_dir

def _clean_numeric(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)

def save_metrics(architecture: str, metrics: dict, base_dir: Path, mode: str = 'standard', temporal_type: str = 'flat'):
    """
    Speichert Metriken als standardisiertes JSON-Schema.
    Fehlende Metriken werden strikt als `null` (None) geloggt, nicht als 0.0!
    """
    metrics_dir, _, _ = get_output_dirs(base_dir)
    
    metadata = {
        "architecture": architecture,
        "mode": mode,
        "temporal_type": temporal_type,
        "n_features": _clean_numeric(metrics.get("n_features", None))
    }
    
    performance = {
        "roc_auc": _clean_numeric(metrics.get("roc_auc", metrics.get("c_index", None))),
        "pr_auc": _clean_numeric(metrics.get("pr_auc", None)),
        "brier_score": _clean_numeric(metrics.get("brier_score", None)),
        "r2_score": _clean_numeric(metrics.get("r2_score", metrics.get("r2", None)))
    }
    
    if performance["roc_auc"] is None and performance["r2_score"] is None:
        print(f"WARNUNG (metrics_logger): Architektur '{architecture}' hat keine primäre Performance-Metrik (weder ROC noch R2)!")
    
    causal_effects = metrics.get("counterfactual", {})
    causal_clean = {}
    for key, val in causal_effects.items():
        if isinstance(val, dict):
            causal_clean[key] = {k: _clean_numeric(v) for k, v in val.items()}
        else:
            causal_clean[key] = _clean_numeric(val)
            
    final_schema = {
        "metadata": metadata,
        "performance": performance,
        "causal_effects": causal_clean
    }
    
    extra_keys = ["n_features", "roc_auc", "pr_auc", "brier_score", "r2_score", "counterfactual", "c_index"]
    extra_metrics = {k: v for k, v in metrics.items() if k not in extra_keys}
    if extra_metrics:
        final_schema["extra"] = {}
        for k, v in extra_metrics.items():
            if isinstance(v, dict):
                final_schema["extra"][k] = {ik: _clean_numeric(iv) for ik, iv in v.items()}
            elif isinstance(v, list):
                final_schema["extra"][k] = [_clean_numeric(i) for i in v]
            else:
                final_schema["extra"][k] = _clean_numeric(v)

    json_path = metrics_dir / f"{architecture}_{mode}_{temporal_type}_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_schema, f, indent=4, ensure_ascii=False)
    
    print(f"[METRICS LOGGER] Schema saved to: {json_path.name}")
