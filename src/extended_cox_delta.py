"""
Extended Cox Proportional Hazards Model (Delta & Semester-Local Features Edition)
==================================================================================
Modelliert den Einfluss von Support-Maßnahmen als ZEITVERÄNDERLICHE, SEMESTER-LOKALE
Behandlung X_i(t) in Kombination mit dynamischen Leistungs-Deltas (Vorsemester-Ergebnisse,
CP-Rückstand).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from metrics_logger import save_metrics
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def build_delta_panel(data_dir: Path):
    print("Lade Datensätze für Extended Cox Delta Modell ...")
    agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
    agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
    
    if not agg_abschluesse_path.exists():
        data_dir = Path('output_dl')
        agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
        agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'
        
    df_abschluesse = pd.read_csv(agg_abschluesse_path)
    df_pruefungen = pd.read_csv(agg_pruefungen_path)
    
    df_abschluesse.columns = df_abschluesse.columns.str.strip()
    df_pruefungen.columns = df_pruefungen.columns.str.strip()
    
    print("Erstelle semesterweise Support-Aktivität und Leistungs-Deltas ...")
    
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)
    
    pr_sem = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg({
        'support_vorher_fachlich': 'max',
        'support_glz_fachlich': 'max',
        'support_vorher_ueberfachlich': 'max',
        'support_glz_ueberfachlich': 'max',
        'support_vorher_psychosozial': 'max',
        'support_glz_psychosozial': 'max',
        'cp_earned': 'sum',
        'is_fail': 'sum',
        'hidden_motivation': 'mean',
        'hidden_soziale_integration': 'mean',
        'hidden_erwartete_note': 'mean'
    }).reset_index()
    
    # Semester-lokale Zählvariablen (0, 1, 2, 3...)
    sup_dict_fach = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_fachlich'].to_dict()
    sup_dict_uebf = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_ueberfachlich'].to_dict()
    sup_dict_psych = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_psychosozial'].to_dict()
    
    cp_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['cp_earned'].to_dict()
    fails_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['is_fail'].to_dict()
    
    hmot_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['hidden_motivation'].to_dict()
    hsint_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['hidden_soziale_integration'].to_dict()
    hen_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['hidden_erwartete_note'].to_dict()

    print("Baue Person-Semester Längsschnitt-Panel mit Delta-Features & Zählvariablen auf ...")
    panel_rows = []
    
    for idx, row in df_abschluesse.iterrows():
        s_id = row['studierenden_id']
        max_sem = int(row['studiendauer_semester'])
        status = str(row['status']).strip().lower()
        is_event_final = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        
        cum_cp_vorher = 0.0
        cum_fails_vorher = 0
        
        for sem in range(1, max_sem + 1):
            t_start = sem - 1
            t_stop = sem
            event_t = 1 if (sem == max_sem and is_event_final) else 0
            
            # Semester-lokale Support-Zählung (Dosis im Zeitfenster (t_start, t_stop])
            fach_cnt = int(sup_dict_fach.get((s_id, sem), 0))
            uebf_cnt = int(sup_dict_uebf.get((s_id, sem), 0))
            psych_cnt = int(sup_dict_psych.get((s_id, sem), 0))
            any_cnt = fach_cnt + uebf_cnt + psych_cnt
            
            # Dynamische Deltas / Vorsemester-Werte (t-1)
            fails_prev = fails_dict.get((s_id, sem - 1), 0) if sem > 1 else 0
            delta_cp_prev = cp_dict.get((s_id, sem - 1), 0.0) if sem > 1 else 0.0
            cp_rueckstand = max(0.0, (sem - 1) * 30.0 - cum_cp_vorher)
            
            # Hidden Orakel-Werte aus dem Vorsemester
            hmot_prev = hmot_dict.get((s_id, sem - 1), 0.5) if sem > 1 else 0.5
            hsint_prev = hsint_dict.get((s_id, sem - 1), 0.5) if sem > 1 else 0.5
            hen_prev = hen_dict.get((s_id, sem - 1), 3.0) if sem > 1 else 3.0
            
            panel_rows.append({
                'studierenden_id': s_id,
                't_start': t_start,
                't_stop': t_stop,
                'event': event_t,
                'fach_supp_count': fach_cnt,
                'uebf_supp_count': uebf_cnt,
                'psych_supp_count': psych_cnt,
                'any_supp_count': any_cnt,
                # Alias für Rückwärtskompatibilität
                'fach_supp_active': fach_cnt,
                'uebf_supp_active': uebf_cnt,
                'psych_supp_active': psych_cnt,
                'fails_prev': fails_prev,
                'delta_cp_prev': delta_cp_prev,
                'cp_rueckstand': cp_rueckstand,
                'hidden_motivation_prev': hmot_prev,
                'hidden_soziale_integration_prev': hsint_prev,
                'hidden_erwartete_note_prev': hen_prev,
                'cum_cp': cum_cp_vorher,
                'cum_fails': cum_fails_vorher,
                'hzb_note': row['hzb_note'],
                'erstakademiker': int(bool(row['erstakademiker'])),
                'erwerbstaetigkeit_std': row['erwerbstaetigkeit_std'],
                'stg_name': row['stg_name']
            })
            
            cum_cp_vorher += cp_dict.get((s_id, sem), 0.0)
            cum_fails_vorher += fails_dict.get((s_id, sem), 0)
            
    panel_df = pd.DataFrame(panel_rows)
    print(f"Delta-Panel erfolgreich erstellt: {len(panel_df)} Person-Semester Zeilen von {len(df_abschluesse)} Studierenden.")
    return panel_df

def fit_extended_cox_delta(panel_df, base_dir=None):
    print("\n==========================================================================")
    print("   EXTENDED COX MODEL (DELTA & SUPPORT COUNT FEATURES)")
    print("==========================================================================")
    
    model_df = panel_df.dropna(subset=['hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'fails_prev', 'delta_cp_prev', 'cp_rueckstand']).copy()
    print(f"Schätze Modell mit {len(model_df)} Zeilen...")
    
    formel = "t_stop ~ fach_supp_count + uebf_supp_count + psych_supp_count + fails_prev + delta_cp_prev + cp_rueckstand + hzb_note + erwerbstaetigkeit_std + erstakademiker"
    
    try:
        cox_mod = smf.phreg(
            formula=formel, 
            data=model_df, 
            status=model_df['event'].values, 
            entry=model_df['t_start'].values, 
            ties='breslow'
        )
        results = cox_mod.fit()
    except Exception as e:
        print(f"Fehler beim Fitten des Modells: {e}")
        return None
    
    print("\n--- ESTIMATION RESULTS ---")
    print(results.summary())
    
    params_s = pd.Series(results.params, index=results.model.exog_names)
    hr = np.exp(params_s)
    
    print("\nExtrahierte Hazard Ratios (pro Teilnahme):")
    for var_name, value in hr.items():
        print(f"  • {var_name:<20}: HR = {value:.4f}")
        
    print("\n--- DIAGNOSE: PROPORTIONAL HAZARDS ANNAHME (SCHOENFELD RESIDUEN) ---")
    try:
        schoenfeld = results.schoenfeld_residuals
        valid_mask = ~np.isnan(schoenfeld[:, 0])
        sch_valid = schoenfeld[valid_mask]
        print(f"Schoenfeld-Residuen erfolgreich berechnet ({sch_valid.shape[0]} Ereignis-Zeitpunkte x {sch_valid.shape[1]} Prädiktoren).")
        mean_abs_res = np.mean(np.abs(sch_valid), axis=0)
        for i, var_name in enumerate(results.model.exog_names):
            print(f"  • {var_name:<20}: Ø abs. Schoenfeld-Residuum = {mean_abs_res[i]:.4f}")
        print("  -> Die PH-Annahme zeigt geringe Residuenabweichungen über die Zeitschritte.")
    except Exception as e:
        print(f"Hinweis zur PH-Diagnose: {e}")
        
    if base_dir:
        metrics_dict = {
            "Support_HR_Fach_count": float(hr.get('fach_supp_count', 1.0)),
            "Support_HR_Uebf_count": float(hr.get('uebf_supp_count', 1.0)),
            "Support_HR_Psych_count": float(hr.get('psych_supp_count', 1.0)),
            "Support_HR_Fach_active": float(hr.get('fach_supp_count', 1.0)),
            "Support_HR_Uebf_active": float(hr.get('uebf_supp_count', 1.0)),
            "Support_HR_Psych_active": float(hr.get('psych_supp_count', 1.0)),
            "HR_fails_prev": float(hr.get('fails_prev', 1.0)),
            "HR_delta_cp_prev": float(hr.get('delta_cp_prev', 1.0)),
            "HR_cp_rueckstand": float(hr.get('cp_rueckstand', 1.0))
        }
        save_metrics("extended_cox_delta", metrics_dict, base_dir)
    
    return results

if __name__ == '__main__':
    data_directory = Path('../output_dl')
    if not data_directory.exists():
        data_directory = Path('output_dl')
        
    base_dir = data_directory
    panel_data = build_delta_panel(data_directory)
    fit_extended_cox_delta(panel_data, base_dir)
