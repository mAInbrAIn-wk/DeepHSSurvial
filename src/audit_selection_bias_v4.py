"""
Stufe 2 Audit: Quantifizierung der Selektionsverzerrung (Confounding by Indication)
==================================================================================
Untersucht den Zustand von Studierenden im Semester vor der allerersten Inanspruchnahme (t0)
im Vergleich zu Nicht-Nutzern auf Basis des V4.2 Panel-Datensatzes.

Berechnet:
1. t0-Vergleich (Mittelwerte, Differenzen, Cohen's d, p-Werte) fuer alle 3 Support-Arten.
2. Logistische Selektionsmodelle: Pr(Support_t = 1 | X_{t-1}) zum Nachweis des DGP-Triggers.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import deepsupport.data_engine.feature_builder as fb


def compute_cohens_d(x1: np.ndarray, x0: np.ndarray) -> float:
    """Berechnet Cohen's d Effektstaerke zwischen zwei Stichproben."""
    n1, n0 = len(x1), len(x0)
    if n1 < 2 or n0 < 2:
        return 0.0
    var1, var0 = np.var(x1, ddof=1), np.var(x0, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
    if pooled_sd < 1e-9:
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled_sd)


def audit_selection_bias(data_dir: Path, output_dir: Path):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) if output_dir else data_dir
    diag_dir = output_dir / "diagnostics"
    metrics_dir = output_dir / "metrics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("   STUFE 2 AUDIT: SELEKTIONSVERZERRUNG (CONFOUNDING BY INDICATION)")
    print(f"   Datenverzeichnis : {data_dir}")
    print(f"   Output           : {output_dir}")
    print("=" * 80)

    # 1. Panel mit allen Oracle-Variablen laden
    panel_df, _, _, _ = fb.build_semester_panel_df(
        data_dir, mode='standard', temporal='prev', oracle=True
    )
    print(f"[INFO] Panel geladen: {len(panel_df):,} Person-Semester Zeilen von {panel_df['studierenden_id'].nunique():,} Studierenden.")

    support_types = {
        "fachlich": {
            "col": "fach_supp_count",
            "name": "Fachlicher Support",
            "primary_trigger": "fails_prev",
            "latent_trigger": "hidden_erwartete_note_prev"
        },
        "ueberfachlich": {
            "col": "uebf_supp_count",
            "name": "Überfachlicher Support",
            "primary_trigger": "cp_rueckstand",
            "latent_trigger": "hidden_motivation_prev"
        },
        "psychosozial": {
            "col": "psych_supp_count",
            "name": "Psychosozialer Support",
            "primary_trigger": "erwerbstaetigkeit_std",
            "latent_trigger": "hidden_soziale_integration_prev"
        }
    }

    variables_to_audit = [
        ("fails_prev", "Klausur-Fehlversuche (Vorsemester)", "beobachtbar"),
        ("gpa_prev", "Notendurchschnitt GPA (Vorsemester)", "beobachtbar"),
        ("delta_cp_prev", "Erworbene CP (Vorsemester)", "beobachtbar"),
        ("cp_rueckstand", "CP-Rückstand kumulativ", "beobachtbar"),
        ("hzb_note", "Abiturnote (HZB)", "beobachtbar"),
        ("erwerbstaetigkeit_std", "Erwerbstätigkeit (Std/Woche)", "beobachtbar"),
        ("hidden_motivation_prev", "Latente Motivation", "latent"),
        ("hidden_soziale_integration_prev", "Latente Soziale Integration", "latent"),
        ("hidden_erwartete_note_prev", "Latente Leistungserwartung", "latent"),
        ("hidden_overload_prev", "Latenter Workload / Overload", "latent"),
        ("hidden_zeit_puffer", "Individueller Zeitpuffer (Std)", "latent")
    ]

    audit_results = {}
    selection_models = {}

    for s_key, s_info in support_types.items():
        s_col = s_info["col"]
        s_name = s_info["name"]
        print(f"\n--- Analysiere Selektionsmechanismus für: {s_name.upper()} ---")

        # Finde Erstinanspruchnahme pro Studi
        users_df = panel_df[panel_df[s_col] > 0]
        first_use = users_df.groupby('studierenden_id')['fachsemester'].min().reset_index()
        first_use.rename(columns={'fachsemester': 'first_sem'}, inplace=True)

        # Merge mit Panel
        panel_merged = panel_df.merge(first_use, on='studierenden_id', how='left')

        # Gruppe 1: Zustand bei t0 (im Semester der Erstinanspruchnahme)
        t0_mask = (panel_merged['first_sem'].notna()) & (panel_merged['fachsemester'] == panel_merged['first_sem'])
        t0_users = panel_merged[t0_mask]

        # Gruppe 0: Studierende, die NIEMALS diesen Support genutzt haben
        never_users = panel_merged[panel_merged['first_sem'].isna()]

        n_t0 = len(t0_users)
        n_never = never_users['studierenden_id'].nunique()
        print(f"  Erstinanspruchnahmen (t0): {n_t0:,} | Nicht-Nutzer (Never): {n_never:,}")

        var_stats = {}
        for var_col, var_label, var_type in variables_to_audit:
            if var_col not in panel_merged.columns:
                continue
            v_t0 = t0_users[var_col].dropna().values
            v_never = never_users[var_col].dropna().values

            mean_t0, std_t0 = float(np.mean(v_t0)), float(np.std(v_t0))
            mean_never, std_never = float(np.mean(v_never)), float(np.std(v_never))
            diff = mean_t0 - mean_never
            d = compute_cohens_d(v_t0, v_never)
            ttest = stats.ttest_ind(v_t0, v_never, equal_var=False)

            var_stats[var_col] = {
                "label": var_label,
                "type": var_type,
                "mean_t0": mean_t0,
                "std_t0": std_t0,
                "mean_never": mean_never,
                "std_never": std_never,
                "diff": diff,
                "cohens_d": d,
                "p_value": float(ttest.pvalue)
            }

        # 2. Logistisches Selektionsmodell Pr(Treatment_t = 1 | X_{t-1})
        panel_df['treat_active'] = (panel_df[s_col] > 0).astype(int)
        formula = (
            f"treat_active ~ fails_prev + cp_rueckstand + hzb_note + erwerbstaetigkeit_std + "
            f"hidden_motivation_prev + hidden_soziale_integration_prev + hidden_overload_prev"
        )
        logit_mod = smf.logit(formula, data=panel_df).fit(disp=False)
        
        odds_ratios = {}
        for param, val in logit_mod.params.items():
            if param == 'Intercept':
                continue
            odds_ratios[param] = {
                "coef": float(val),
                "odds_ratio": float(np.exp(val)),
                "p_value": float(logit_mod.pvalues[param]),
                "ci_lower": float(np.exp(logit_mod.conf_int().loc[param, 0])),
                "ci_upper": float(np.exp(logit_mod.conf_int().loc[param, 1]))
            }

        audit_results[s_key] = {
            "name": s_name,
            "n_first_users": n_t0,
            "n_never_users": n_never,
            "comparison": var_stats,
            "selection_model_or": odds_ratios
        }

    # JSON exportieren
    json_path = metrics_dir / "selection_bias_audit_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=4, ensure_ascii=False)
    print(f"\n[INFO] Audit-Metriken gespeichert: {json_path}")

    # Markdown Report erstellen
    md_path = diag_dir / "selection_bias_audit_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Audit-Bericht: Quantifizierung der Selektionsverzerrung (Stufe 2)\n\n")
        f.write("Dieses Dokument belegt quantitativ die **Auswahlverzerrung (Confounding by Indication)**: ")
        f.write("Support-Empfänger befinden sich zum Zeitpunkt ihrer ersten Inanspruchnahme ($t_0$) ")
        f.write("in einer hochgradig vorbelasteten Krisensituation im Vergleich zu Studierenden, die nie Support beanspruchen.\n\n")

        for s_key, res in audit_results.items():
            f.write(f"## {res['name']}\n\n")
            f.write(f"- **Erstinanspruchnahmen ($t_0$)**: {res['n_first_users']:,}\n")
            f.write(f"- **Vergleichsgruppe (Nie Support genutzt)**: {res['n_never_users']:,}\n\n")

            f.write("### 1. Zustandsvergleich bei $t_0$ vs. Nicht-Nutzer\n\n")
            f.write("| Merkmal | Typ | Mittelwert $t_0$ (Nutzer) | Mittelwert Nie-Nutzer | Differenz | Cohen's $d$ | Signifikanz ($p$) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for v_col, v in res["comparison"].items():
                p_str = "< 0.0001" if v["p_value"] < 0.0001 else f"{v['p_value']:.4f}"
                f.write(f"| **{v['label']}** | `{v['type']}` | {v['mean_t0']:.3f} | {v['mean_never']:.3f} | {v['diff']:+.3f} | **{v['cohens_d']:+.3f}** | {p_str} |\n")

            f.write("\n### 2. Logistisches Selektionsmodell: Pr(Support in Semester $t$ = 1)\n\n")
            f.write("| Prädiktor im Vorsemester | Odds Ratio (OR) | 95% Konfidenzintervall | Interpretation |\n")
            f.write("| :--- | :---: | :---: | :--- |\n")

            for p_col, p_data in res["selection_model_or"].items():
                or_val = p_data["odds_ratio"]
                ci_l = p_data["ci_lower"]
                ci_u = p_data["ci_upper"]
                interp = "Erhöht Teilnahme" if or_val > 1.05 else ("Senkt Teilnahme" if or_val < 0.95 else "Neutral")
                f.write(f"| `{p_col}` | **{or_val:.3f}** | [{ci_l:.3f}, {ci_u:.3f}] | {interp} |\n")
            f.write("\n---\n\n")

    print(f"[INFO] Markdown-Bericht gespeichert: {md_path}")
    return audit_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audit Selection Bias V4.2")
    parser.add_argument('--data_dir', type=str, default='data_v4_grid/S01_baseline/universe_A')
    parser.add_argument('--output_dir', type=str, default='output_v4_models/S01_baseline/universe_A')
    args = parser.parse_args()

    audit_selection_bias(Path(args.data_dir), Path(args.output_dir))
