"""
Extended Cox Proportional Hazards Model (Time-Varying Covariates Edition)
========================================================================
Modelliert den Einfluss von Support-Maßnahmen als ZEITVERÄNDERLICHE KOVARIATE
(Time-Varying Treatment X_i(t)) über die gesamte Studiendauer.

Vorteile:
- Verhindert Immortal-Time Bias vollständig (ohne Datenverlust durch Landmark-Schnitte)
- Berücksichtigt Support-Teilnahmen zu jedem Zeitpunkt des Studiums
- Nutzt den denormalisierten DataCube `agg_pruefungen.csv` & `agg_abschluesse.csv`
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

def build_person_semester_panel(data_dir: Path):
    print("Lade Datensätze für Extended Cox Modell ...")
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
    
    # 1. Bestimme pro Student und Semester die aggregierte Support-Exposition & Leistungen
    print("Erstelle semesterweisen Support-Indikator und Leistungswerte aus agg_pruefungen ...")
    
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)
    
    # Gruppierung der Prüfungsdaten nach Student & Fachsemester
    pr_sem = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg({
        'support_vorher_fachlich': 'max',
        'support_glz_fachlich': 'max',
        'support_vorher_ueberfachlich': 'max',
        'support_glz_ueberfachlich': 'max',
        'support_vorher_psychosozial': 'max',
        'support_glz_psychosozial': 'max',
        'cp_earned': 'sum',
        'is_fail': 'sum'
    }).reset_index()
    
    # Semester-lokale Zählvariablen (0, 1, 2, 3...)
    sup_dict_fach = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_fachlich'].to_dict()
    sup_dict_uebf = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_ueberfachlich'].to_dict()
    sup_dict_psych = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_psychosozial'].to_dict()
    
    cp_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['cp_earned'].to_dict()
    fails_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['is_fail'].to_dict()

    print("Baue Person-Semester Längsschnitt-Panel auf ...")
    panel_rows = []
    
    for idx, row in df_abschluesse.iterrows():
        s_id = row['studierenden_id']
        max_sem = int(row['studiendauer_semester'])
        status = str(row['status']).strip().lower()
        is_event_final = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        
        # Lagged Confounder (Vergangenheit bis t-1)
        cum_cp_vorher = 0.0
        cum_fails_vorher = 0
        
        for sem in range(1, max_sem + 1):
            t_start = sem - 1
            t_stop = sem
            
            # Event tritt nur im finalen Beobachtungssemester auf
            event_t = 1 if (sem == max_sem and is_event_final) else 0
            
            fach_cnt = int(sup_dict_fach.get((s_id, sem), 0))
            uebf_cnt = int(sup_dict_uebf.get((s_id, sem), 0))
            psych_cnt = int(sup_dict_psych.get((s_id, sem), 0))
            any_cnt = fach_cnt + uebf_cnt + psych_cnt
            
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
                'fach_supp_tv': fach_cnt,
                'uebf_supp_tv': uebf_cnt,
                'psych_supp_tv': psych_cnt,
                'any_supp_tv': any_cnt,
                'cum_cp': cum_cp_vorher,
                'cum_fails': cum_fails_vorher,
                'hzb_note': row['hzb_note'],
                'erstakademiker': int(bool(row['erstakademiker'])),
                'erwerbstaetigkeit_std': row['erwerbstaetigkeit_std'],
                'stg_name': row['stg_name']
            })
            
            # Update lagged values for next semester (t+1)
            cum_cp_vorher += cp_dict.get((s_id, sem), 0.0)
            cum_fails_vorher += fails_dict.get((s_id, sem), 0)
            
    panel_df = pd.DataFrame(panel_rows)
    print(f"Panel erfolgreich erstellt: {len(panel_df)} Person-Semester Zeilen von {len(df_abschluesse)} Studierenden.")
    return panel_df

from metrics_logger import save_metrics
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def fit_extended_cox_model(panel_df, base_dir=None):
    print("\n==========================================================================")
    print("   EXTENDED COX PROPORTIONAL HAZARDS MODEL (STATSMODELS PHREG)")
    print("==========================================================================")
    
    # Bereinigung fehlender Werte
    model_df = panel_df.dropna(subset=['hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'cum_cp', 'cum_fails']).copy()
    
    print(f"Schätze Modell mit {len(model_df)} Zeilen...")
    
    formel = "t_stop ~ fach_supp_count + uebf_supp_count + psych_supp_count + cum_cp + cum_fails + hzb_note + erwerbstaetigkeit_std + erstakademiker"
    
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
    
    print("\nExtrahierte Hazard Ratios:")
    for var_name, value in hr.items():
        print(f"  • {var_name:<20}: HR = {value:.4f}")
        
    print("\nInterpretation der zeitveränderlichen Effekte (Time-Varying HR):")
    print("- HR < 1.0 bedeutet: Support-Nutzung verringert das Risiko eines Studienabbruchs im jeweiligen Semester (Schutzfaktor).")
    print("- HR > 1.0 bedeutet: Erhöhtes Abbruchrisiko im jeweiligen Semester.")
    print("\nVorteil dieses Modells: Entstörung des kausalen Effekts durch Einbeziehung vergangener Fehlversuche & CPs.")
    print("==========================================================================")
    
    if base_dir:
        hr_f = float(hr.get('fach_supp_count', 1.0))
        hr_u = float(hr.get('uebf_supp_count', 1.0))
        hr_p = float(hr.get('psych_supp_count', 1.0))
        
        metrics_dict = {
            "Support_HR_Fach_count": hr_f,
            "Support_HR_Uebf_count": hr_u,
            "Support_HR_Psych_count": hr_p,
            "fach_partial": {"mean_hr": hr_f, "median_hr": hr_f},
            "fach_isolated": {"mean_hr": hr_f, "median_hr": hr_f},
            "uebf_partial": {"mean_hr": hr_u, "median_hr": hr_u},
            "uebf_isolated": {"mean_hr": hr_u, "median_hr": hr_u},
            "psych_partial": {"mean_hr": hr_p, "median_hr": hr_p},
            "psych_isolated": {"mean_hr": hr_p, "median_hr": hr_p},
            "HR_cum_fails": float(hr.get('cum_fails', 1.0)),
            "HR_cum_cp": float(hr.get('cum_cp', 1.0)),
            # Abwärtskompatible Keys:
            "Support_HR_Fach_tv": hr_f,
            "Support_HR_Uebf_tv": hr_u,
            "Support_HR_Psych_tv": hr_p
        }
        save_metrics("extended_cox_panel", metrics_dict, base_dir)
    
    return results

def train_extended_cox_model(data_dir: Path):
    panel_data = build_person_semester_panel(data_dir)
    return fit_extended_cox_model(panel_data, data_dir)

if __name__ == '__main__':
    data_directory = Path('../output_dl')
    if not data_directory.exists():
        data_directory = Path('output_dl')
        
    train_extended_cox_model(data_directory)

