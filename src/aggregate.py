"""
Datenaggregation für HSDS Datensatz (Python-Skript)
==================================================
Überführt die Logik aus Datenaggregation.ipynb in ein automatisierbares Skript.
Erzeugt:
  - agg_pruefungen.csv (Längsschnitt: Prüfungen mit Support-Expositions-Merkmalen)
  - agg_abschluesse.csv (Querschnitt: Aggregation auf Studierenden-Ebene mit allen Kontrollvariablen für EDA/Dashboards)
"""

from pathlib import Path
import pandas as pd
import numpy as np

def aggregiere_daten(output_dir: Path):
    print("Starte Datenaggregation ...")
    
    # 1. Daten laden
    studierende_df = pd.read_csv(output_dir / 'studierende.csv')
    studiengaenge_df = pd.read_csv(output_dir / 'studiengaenge.csv')
    module_df = pd.read_csv(output_dir / 'module.csv')
    pruefungen_df = pd.read_csv(output_dir / 'pruefungen.csv')
    semester_df = pd.read_csv(output_dir / 'semester.csv')
    einschreibungen_df = pd.read_csv(output_dir / 'einschreibungen.csv')
    support_angebote_df = pd.read_csv(output_dir / 'support_angebote.csv')
    support_modul_zuordnung_df = pd.read_csv(output_dir / 'support_modul_zuordnung.csv')
    support_teilnahmen_df = pd.read_csv(output_dir / 'support_teilnahmen.csv')
    abschluesse_df = pd.read_csv(output_dir / 'abschluesse.csv')
    
    # 2. Vorbereitung & Bezeichnungen anreichern
    studierende_df = studierende_df.merge(
        studiengaenge_df.rename(columns={'name': 'stg_name'})[['studiengang_id', 'stg_name']],
        on='studiengang_id',
        how='left'
    )
    
    pruefungen_df = pruefungen_df.merge(
        module_df.rename(columns={'name': 'modul_name'})[['modul_id', 'modul_name']],
        on='modul_id',
        how='left'
    ).merge(
        studierende_df[['studierenden_id', 'stg_name']],
        on='studierenden_id',
        how='left'
    )
    
    # Fachsemester zu Prüfungen matchen via Einschreibungen
    if 'fachsemester' not in pruefungen_df.columns:
        pruefungen_df = pruefungen_df.merge(
            einschreibungen_df[['studierenden_id', 'semester_id', 'fachsemester']],
            on=['studierenden_id', 'semester_id'],
            how='left'
        )
    
    support_mit_typ = support_teilnahmen_df.merge(
        support_angebote_df[['angebot_id', 'typ']],
        on='angebot_id',
        how='inner'
    )
    
    support_full = support_mit_typ.merge(
        support_modul_zuordnung_df,
        on='angebot_id',
        how='left'
    )
    
    # 3. Längsschnitt: agg_pruefungen_df
    agg_pruefungen_df = pruefungen_df.reset_index(drop=True)
    agg_pruefungen_df['pruefung_id'] = agg_pruefungen_df.index
    
    if not support_full.empty:
        merged = agg_pruefungen_df.merge(
            support_full,
            on='studierenden_id',
            suffixes=('_pruefung', '_support')
        )
        
        semester_nr_map = semester_df.set_index('semester_id')['semester_nr']
        merged['sem_nr_pruefung'] = merged['semester_id_pruefung'].map(semester_nr_map)
        merged['sem_nr_support'] = merged['semester_id_support'].map(semester_nr_map)
        
        zeit_vorher = merged['sem_nr_support'] < merged['sem_nr_pruefung']
        zeit_gleich = merged['semester_id_support'] == merged['semester_id_pruefung']
        
        ist_fachlich = (merged['typ'] == 'fachlich') & (merged['modul_id_support'] == merged['modul_id_pruefung'])
        ist_ueberfachlich = merged['typ'] == 'ueberfachlich'
        ist_psychosozial = merged['typ'] == 'psychosozial'
        
        bed_vorher = (zeit_vorher & ist_fachlich) | (zeit_vorher & ist_ueberfachlich) | (zeit_vorher & ist_psychosozial)
        bed_gleich = (zeit_gleich & ist_fachlich) | (zeit_gleich & ist_ueberfachlich) | (zeit_gleich & ist_psychosozial)
        
        filtered_vorher = merged[bed_vorher]
        filtered_gleich = merged[bed_gleich]
        
        fsv_kategorien = filtered_vorher.groupby(['pruefung_id', 'typ']).size().unstack(fill_value=0)
        fsg_kategorien = filtered_gleich.groupby(['pruefung_id', 'typ']).size().unstack(fill_value=0)
        
        fsv_kategorien = fsv_kategorien.add_prefix('support_vorher_')
        fsg_kategorien = fsg_kategorien.add_prefix('support_glz_')
        
        agg_pruefungen_df = agg_pruefungen_df.set_index('pruefung_id')
        agg_pruefungen_df = agg_pruefungen_df.join([fsv_kategorien, fsg_kategorien]).fillna(0)
        agg_pruefungen_df = agg_pruefungen_df.reset_index()
    else:
        for col in ['support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
                    'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial']:
            agg_pruefungen_df[col] = 0

    agg_pruefungen_df = agg_pruefungen_df.merge(
        module_df[['modul_id', 'cp', 'schwierigkeit']],
        on='modul_id',
        how='left'
    )
    
    for col in ['support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
                'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial']:
        if col not in agg_pruefungen_df.columns:
            agg_pruefungen_df[col] = 0
            
    # 4. Querschnitt: agg_abschluesse_df
    pr_grouped = agg_pruefungen_df.groupby('studierenden_id')
    
    stud_agg = pr_grouped.agg(
        Anz_Pruefungen=('pruefung_id', 'count'),
        AVG_Note=('note', 'mean'),
        Anz_Fehlversuche=('bestanden', lambda s: (~s).sum()),
        Anz_Bestanden=('bestanden', lambda s: s.sum()),
    ).reset_index()
    
    stud_agg['Fehlversuchsquote'] = stud_agg['Anz_Fehlversuche'] / stud_agg['Anz_Pruefungen']
    
    # Versuchs-spezifische Metriken und Spalten-Aliase für EDA/Dashboard-Kompatibilität
    alias_dict = {
        ('support_vorher_fachlich', 1): 'Fachlicher_Support_Vorher_ErstVersuche',
        ('support_vorher_fachlich', 2): 'Fachlicher_Support_Vorher_ZweitVersuche',
        ('support_vorher_fachlich', 3): 'Fachlicher_Support_Vorher_DrittVersuche',
        ('support_glz_fachlich', 1): 'Fachlicher_Support_GLZ_ErstVersuche',
        ('support_glz_fachlich', 2): 'Fachlicher_Support_GLZ_ZweitVersuche',
        ('support_glz_fachlich', 3): 'Fachlicher_Support_GLZ_DrittVersuche',
        
        ('support_vorher_ueberfachlich', 1): 'Ueberfachlicher_Support_Vorher_ErstVersuche',
        ('support_vorher_ueberfachlich', 2): 'Ueberfachlicher_Support_Vorher_ZweitVersuche',
        ('support_vorher_ueberfachlich', 3): 'Ueberfachlicher_Support_Vorher_DrittVersuche',
        ('support_glz_ueberfachlich', 1): 'Ueberfachlicher_Support_GLZ_ErstVersuche',
        ('support_glz_ueberfachlich', 2): 'Ueberfachlicher_Support_GLZ_ZweitVersuche',
        ('support_glz_ueberfachlich', 3): 'Ueberfachlicher_Support_GLZ_DrittVersuche',
        
        ('support_vorher_psychosozial', 1): 'Psychosozialer_Support_Vorher_ErstVersuche',
        ('support_vorher_psychosozial', 2): 'Psychosozialer_Support_Vorher_ZweitVersuche',
        ('support_vorher_psychosozial', 3): 'Psychosozialer_Support_Vorher_DrittVersuche',
        ('support_glz_psychosozial', 1): 'Psychosozialer_Support_GLZ_ErstVersuche',
        ('support_glz_psychosozial', 2): 'Psychosozialer_Support_GLZ_ZweitVersuche',
        ('support_glz_psychosozial', 3): 'Psychosozialer_Support_GLZ_DrittVersuche',
    }

    for v in [1, 2, 3]:
        v_df = agg_pruefungen_df[agg_pruefungen_df['versuch'] == v]
        v_grouped = v_df.groupby('studierenden_id')
        
        cnt_col = 'Anz_ZweitVersuche' if v == 2 else ('Anz_DrittVersuche' if v == 3 else 'Anz_ErstVersuche')
        avg_col = 'AVG_ErstVersucheNote' if v == 1 else ('AVG_ZweitVersucheNote' if v == 2 else 'AVG_DrittVersucheNote')
        
        cnt = v_grouped['note'].count().rename(cnt_col)
        avg_n = v_grouped['note'].mean().rename(avg_col)
        
        stud_agg = stud_agg.merge(cnt, on='studierenden_id', how='left').merge(avg_n, on='studierenden_id', how='left')
        stud_agg[cnt_col] = stud_agg[cnt_col].fillna(0).astype(int)
        
        for sup_col in ['support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
                        'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial']:
            col_name = alias_dict.get((sup_col, v), f'{sup_col}_V{v}')
            sup_sum = v_grouped[sup_col].sum().rename(col_name)
            stud_agg = stud_agg.merge(sup_sum, on='studierenden_id', how='left')
            stud_agg[col_name] = stud_agg[col_name].fillna(0).astype(int)

    # 5. Phasen/Semester-Aggregations-Metriken (Sem 1-2 & Sem 1-4)
    sem12_pr = agg_pruefungen_df[agg_pruefungen_df['fachsemester'] <= 2].groupby('studierenden_id')
    sem14_pr = agg_pruefungen_df[agg_pruefungen_df['fachsemester'] <= 4].groupby('studierenden_id')
    
    avg_note_sem12 = sem12_pr['note'].mean().rename('AVG_note_sem1-2')
    avg_note_sem14 = sem14_pr['note'].mean().rename('AVG_note_sem1-4')
    
    avg_cp_sem12 = (sem12_pr.apply(lambda df: df[df['bestanden']]['cp'].sum()) / 2.0).rename('AVG_cp_sem1-2')
    avg_cp_sem14 = (sem14_pr.apply(lambda df: df[df['bestanden']]['cp'].sum()) / 4.0).rename('AVG_cp_sem1-4')
    fehlversuche_sem12 = sem12_pr.apply(lambda df: (~df['bestanden']).sum()).rename('fehlversuche_sem12')
    
    stud_agg = stud_agg.merge(avg_note_sem12, on='studierenden_id', how='left') \
                       .merge(avg_note_sem14, on='studierenden_id', how='left') \
                       .merge(avg_cp_sem12, on='studierenden_id', how='left') \
                       .merge(avg_cp_sem14, on='studierenden_id', how='left') \
                       .merge(fehlversuche_sem12, on='studierenden_id', how='left')
    stud_agg['fehlversuche_sem12'] = stud_agg['fehlversuche_sem12'].fillna(0).astype(int)

    # Support-Teilnahme Flags (Fach_supp, Uebf_supp, Psych_supp)
    if not support_teilnahmen_df.empty:
        if 'fachsemester' not in support_teilnahmen_df.columns:
            support_teilnahmen_df = support_teilnahmen_df.merge(
                einschreibungen_df[['studierenden_id', 'semester_id', 'fachsemester']],
                on=['studierenden_id', 'semester_id'],
                how='left'
            )
            
        sup_studi = support_teilnahmen_df.merge(support_angebote_df[['angebot_id', 'typ']], on='angebot_id', how='left')
        sup_grouped = sup_studi.groupby('studierenden_id')
        
        fach_supp = sup_grouped.apply(lambda df: (df['typ'] == 'fachlich').any()).rename('Fach_supp')
        uebf_supp = sup_grouped.apply(lambda df: (df['typ'] == 'ueberfachlich').any()).rename('Uebf_supp')
        psych_supp = sup_grouped.apply(lambda df: (df['typ'] == 'psychosozial').any()).rename('Psych_supp')
        sup_cnt = sup_grouped.size().rename('support_exposure_count')
        
        # Pre-Landmark (Sem 1-2) Support Flags
        sup_sem12 = sup_studi[sup_studi['fachsemester'] <= 2]
        sup_sem12_grouped = sup_sem12.groupby('studierenden_id')
        
        fach_supp_sem12 = sup_sem12_grouped.apply(lambda df: (df['typ'] == 'fachlich').any()).rename('Fach_supp_sem12')
        uebf_supp_sem12 = sup_sem12_grouped.apply(lambda df: (df['typ'] == 'ueberfachlich').any()).rename('Uebf_supp_sem12')
        psych_supp_sem12 = sup_sem12_grouped.apply(lambda df: (df['typ'] == 'psychosozial').any()).rename('Psych_supp_sem12')
        sup_cnt_sem12 = sup_sem12_grouped.size().rename('support_exposure_count_sem12')
        
        stud_agg = stud_agg.merge(fach_supp, on='studierenden_id', how='left') \
                           .merge(uebf_supp, on='studierenden_id', how='left') \
                           .merge(psych_supp, on='studierenden_id', how='left') \
                           .merge(sup_cnt, on='studierenden_id', how='left') \
                           .merge(fach_supp_sem12, on='studierenden_id', how='left') \
                           .merge(uebf_supp_sem12, on='studierenden_id', how='left') \
                           .merge(psych_supp_sem12, on='studierenden_id', how='left') \
                           .merge(sup_cnt_sem12, on='studierenden_id', how='left')
    else:
        stud_agg['Fach_supp'] = False
        stud_agg['Uebf_supp'] = False
        stud_agg['Psych_supp'] = False
        stud_agg['support_exposure_count'] = 0
        stud_agg['Fach_supp_sem12'] = False
        stud_agg['Uebf_supp_sem12'] = False
        stud_agg['Psych_supp_sem12'] = False
        stud_agg['support_exposure_count_sem12'] = 0

    stud_agg['Fach_supp'] = stud_agg['Fach_supp'].fillna(False).astype(bool)
    stud_agg['Uebf_supp'] = stud_agg['Uebf_supp'].fillna(False).astype(bool)
    stud_agg['Psych_supp'] = stud_agg['Psych_supp'].fillna(False).astype(bool)
    stud_agg['support_exposure_count'] = stud_agg['support_exposure_count'].fillna(0).astype(int)
    stud_agg['any_support'] = stud_agg['support_exposure_count'] > 0

    stud_agg['Fach_supp_sem12'] = stud_agg['Fach_supp_sem12'].fillna(False).astype(bool)
    stud_agg['Uebf_supp_sem12'] = stud_agg['Uebf_supp_sem12'].fillna(False).astype(bool)
    stud_agg['Psych_supp_sem12'] = stud_agg['Psych_supp_sem12'].fillna(False).astype(bool)
    stud_agg['support_exposure_count_sem12'] = stud_agg['support_exposure_count_sem12'].fillna(0).astype(int)
    stud_agg['any_support_sem12'] = stud_agg['support_exposure_count_sem12'] > 0
    
    # Exposure Groups
    def exp_group(c):
        if c == 0: return 'keine Supportexposition'
        elif c <= 2: return 'geringe Supportexposition'
        elif c <= 5: return 'mittlere Supportexposition'
        else: return 'hohe Supportexposition'
        
    stud_agg['support_exposure_group'] = stud_agg['support_exposure_count'].apply(exp_group)
            
    hidden_cols = [c for c in studierende_df.columns if c.startswith('hidden_')]
    merge_stud_cols = ['studierenden_id', 'hzb_note', 'hzb_typ', 'erstakademiker', 'erwerbstaetigkeit_std', 'stg_name'] + hidden_cols
    merge_stud_cols = list(dict.fromkeys(merge_stud_cols)) # unique
    
    agg_abschluesse_df = abschluesse_df.merge(
        studierende_df[merge_stud_cols],
        on='studierenden_id',
        how='left'
    ).merge(
        stud_agg,
        on='studierenden_id',
        how='left'
    )
    
    # Speichern der aggregierten CSVs
    agg_pruefungen_df.to_csv(output_dir / 'agg_pruefungen.csv', index=False)
    agg_abschluesse_df.to_csv(output_dir / 'agg_abschluesse.csv', index=False)
    
    print(f"  [OK] agg_pruefungen.csv ({len(agg_pruefungen_df)} Zeilen)")
    print(f"  [OK] agg_abschluesse.csv ({len(agg_abschluesse_df)} Zeilen)")

if __name__ == '__main__':
    aggregiere_daten(Path('../output_dl'))
