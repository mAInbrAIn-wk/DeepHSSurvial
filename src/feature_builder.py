"""
Feature Builder & Harmonization Module (Feature Factory)
=========================================================
Stellt standardisierte und konsistente Datenstrukturen (3D-Tensoren für Sequenzmodelle,
2D-DataFrames für Panel- & Landmark-Modelle) für alle 8 Modell-Klassen bereit.

Unterstützt beliebig kombinierbare Flags bzw. Modi:
- standard:    Vollständiges, bereinigtes Standard-Feature-Set.
- gradeblind:  Entfernt alle Noten-Leistungsdaten (gpa, note, delta_gpa), behält aber CP, Fehlversuche und HZB.
- blind:       Entfernt jeglichen akademischen Fortschritt (Noten, CP, Fails, Rückstand). Reine Eingangsprognose.
- oracle:      Fügt latente DGP-Variablen (Motivation, Soziale Integration, Erwartete Note) hinzu.
- realistic:   Entfernt sensible/nicht erfassbare Merkmale (Migration, Erstakademiker, Erwerb, Psych. Support, Schwierigkeit).
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd

PADDING_VALUE = -99.0

# Kanonische Mappings
HZB_ORDINAL_MAP = {
    'Allg. Hochschulreife': 3.0,
    'Fachgebundene HR': 2.0,
    'Fachhochschulreife': 1.0,
    'Berufl. Qualifikation': 0.0
}

STUDIENGAENGE_LIST = ['Informatik', 'BWL', 'Maschinenbau', 'Psychologie', 'Soziale Arbeit']


def _resolve_modes(mode: str = 'standard',
                   gradeblind: bool = False,
                   blind: bool = False,
                   oracle: bool = False,
                   realistic: bool = False) -> Tuple[bool, bool, bool, bool]:
    """Löst mode-String und boolesche Flags konsistent auf."""
    m = mode.lower().strip()
    if 'gradeblind' in m:
        gradeblind = True
    if 'blind' in m and 'gradeblind' not in m:
        blind = True
    if 'oracle' in m:
        oracle = True
    if 'realistic' in m or 'dsgvo' in m:
        realistic = True

    if blind:
        gradeblind = True

    return gradeblind, blind, oracle, realistic


def _load_raw_data(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lädt agg_abschluesse.csv und agg_pruefungen.csv mit Fallback-Prüfung."""
    data_dir = Path(data_dir)
    agg_abschluesse_path = data_dir / 'agg_abschluesse.csv'
    agg_pruefungen_path = data_dir / 'agg_pruefungen.csv'

    if not agg_abschluesse_path.exists():
        candidates = [Path('output_dl'), Path('../output_dl'), Path('src/output_dl'), Path('output_dl_v3')]
        for c in candidates:
            if (c / 'agg_abschluesse.csv').exists():
                agg_abschluesse_path = c / 'agg_abschluesse.csv'
                agg_pruefungen_path = c / 'agg_pruefungen.csv'
                break

    df_abschluesse = pd.read_csv(agg_abschluesse_path)
    df_pruefungen = pd.read_csv(agg_pruefungen_path)

    df_abschluesse.columns = df_abschluesse.columns.str.strip()
    df_pruefungen.columns = df_pruefungen.columns.str.strip()

    # Fallback merge of demographic columns from studierende.csv if missing
    studi_path = agg_abschluesse_path.parent / 'studierende.csv'
    if studi_path.exists() and 'migrationshintergrund' not in df_abschluesse.columns:
        df_studi = pd.read_csv(studi_path)
        df_studi.columns = df_studi.columns.str.strip()
        demog_cols = [c for c in ['studierenden_id', 'migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std', 'hzb_typ', 'hzb_note'] if c in df_studi.columns]
        merge_cols = [c for c in demog_cols if c == 'studierenden_id' or c not in df_abschluesse.columns]
        if len(merge_cols) > 1:
            df_abschluesse = pd.merge(df_abschluesse, df_studi[merge_cols], on='studierenden_id', how='left')

    return df_abschluesse, df_pruefungen


# =========================================================================
# 1. SEMESTER SEQUENCE TENSOR (Klasse 6: GRU, Transformer, DeepHit)
# =========================================================================
def build_semester_sequence_tensor(
    data_dir: Union[str, Path],
    max_semesters: int = 16,
    mode: str = 'standard',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Optional[int]]]:
    """
    Erstellt den 3D-Sequenztensor für Semester-Modelle: (N, max_semesters, n_features).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)

    agg_dict = {
        'sem_cp': ('cp_earned', 'sum'),
        'sem_fails': ('is_fail', 'sum'),
        'sem_gpa': ('note', 'mean'),
        'fach_supp_count': ('support_glz_fachlich', 'sum'),
        'uebf_supp_count': ('support_glz_ueberfachlich', 'sum'),
        'psych_supp_count': ('support_glz_psychosozial', 'sum')
    }
    if 'hidden_motivation' in df_pruefungen.columns:
        agg_dict['hidden_motivation'] = ('hidden_motivation', 'mean')
        agg_dict['hidden_soziale_integration'] = ('hidden_soziale_integration', 'mean')
        agg_dict['hidden_erwartete_note'] = ('hidden_erwartete_note', 'mean')

    sem_agg = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg(**agg_dict).reset_index()
    sem_lookup = sem_agg.set_index(['studierenden_id', 'fachsemester'])

    feature_names: List[str] = [
        'hzb_note',
        'hzb_typ_ord',
        'stg_Informatik', 'stg_BWL', 'stg_Maschinenbau', 'stg_Psychologie', 'stg_Soziale_Arbeit'
    ]

    if not realistic:
        feature_names.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    if not blind:
        feature_names.extend(['cum_fails_vorher', 'cp_rueckstand_vorher', 'sem_cp', 'sem_fails'])
        if not gradeblind:
            feature_names.append('sem_gpa')

    feature_names.extend(['fach_supp_count', 'uebf_supp_count'])
    if not realistic:
        feature_names.append('psych_supp_count')

    if oracle:
        feature_names.extend([
            'hidden_motivation_prev',
            'hidden_soziale_integration_prev',
            'hidden_erwartete_note_prev'
        ])

    feature_indices: Dict[str, Optional[int]] = {
        'fach_supp': feature_names.index('fach_supp_count') if 'fach_supp_count' in feature_names else None,
        'uebf_supp': feature_names.index('uebf_supp_count') if 'uebf_supp_count' in feature_names else None,
        'psych_supp': feature_names.index('psych_supp_count') if 'psych_supp_count' in feature_names else None,
    }

    n_features = len(feature_names)
    studis = df_abschluesse['studierenden_id'].unique()
    num_studis = len(studis)

    X_seq = np.full((num_studis, max_semesters, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((num_studis, max_semesters, 1), PADDING_VALUE, dtype=np.float32)
    studi_events = np.zeros(num_studis, dtype=int)

    for i, row in enumerate(df_abschluesse.itertuples(index=False)):
        s_id = row.studierenden_id
        max_sem = min(int(row.studiendauer_semester), max_semesters)
        status = str(row.status).strip().lower()
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        studi_events[i] = 1 if is_dropout else 0

        hzb = float(row.hzb_note)
        hzb_ord = HZB_ORDINAL_MAP.get(getattr(row, 'hzb_typ', 'Allg. Hochschulreife'), 3.0)
        stg = str(getattr(row, 'stg_name', 'Informatik'))
        stg_ohe = [1.0 if s == stg else 0.0 for s in STUDIENGAENGE_LIST]

        mig = 1.0 if bool(getattr(row, 'migrationshintergrund', False)) else 0.0
        erst = 1.0 if bool(getattr(row, 'erstakademiker', False)) else 0.0
        erw = float(getattr(row, 'erwerbstaetigkeit_std', 0.0))

        cum_cp_vorher = 0.0
        cum_fails_vorher = 0.0
        hmot_prev, hsint_prev, hen_prev = 0.5, 0.5, 3.0

        for sem in range(1, max_sem + 1):
            t_idx = sem - 1

            if (s_id, sem) in sem_lookup.index:
                s_data = sem_lookup.loc[(s_id, sem)]
                gpa = float(s_data['sem_gpa']) if not np.isnan(s_data['sem_gpa']) else 3.0
                cp = float(s_data['sem_cp'])
                fails = float(s_data['sem_fails'])
                fach_cnt = float(s_data['fach_supp_count'])
                uebf_cnt = float(s_data['uebf_supp_count'])
                psych_cnt = float(s_data['psych_supp_count'])
                if 'hidden_motivation' in s_data:
                    hmot_curr = float(s_data['hidden_motivation'])
                    hsint_curr = float(s_data['hidden_soziale_integration'])
                    hen_curr = float(s_data['hidden_erwartete_note'])
                else:
                    hmot_curr, hsint_curr, hen_curr = 0.5, 0.5, 3.0
            else:
                gpa, cp, fails = 3.0, 0.0, 0.0
                fach_cnt, uebf_cnt, psych_cnt = 0.0, 0.0, 0.0
                hmot_curr, hsint_curr, hen_curr = 0.5, 0.5, 3.0

            cp_rueckstand = max(0.0, (sem - 1) * 30.0 - cum_cp_vorher)

            row_feats = [hzb, hzb_ord] + stg_ohe

            if not realistic:
                row_feats.extend([mig, erst, erw])

            if not blind:
                row_feats.extend([cum_fails_vorher, cp_rueckstand, cp, fails])
                if not gradeblind:
                    row_feats.append(gpa)

            row_feats.extend([fach_cnt, uebf_cnt])
            if not realistic:
                row_feats.append(psych_cnt)

            if oracle:
                row_feats.extend([hmot_prev, hsint_prev, hen_prev])

            X_seq[i, t_idx, :] = row_feats
            y_seq[i, t_idx, 0] = 1.0 if (sem == max_sem and is_dropout) else 0.0

            cum_cp_vorher += cp
            cum_fails_vorher += fails
            hmot_prev, hsint_prev, hen_prev = hmot_curr, hsint_curr, hen_curr

    return studis, X_seq, y_seq, studi_events, feature_names, feature_indices


# =========================================================================
# 2. EXAM SEQUENCE TENSOR (Klasse 7: Exam-GRU, Exam-Transformer)
# =========================================================================
def build_exam_sequence_tensor(
    data_dir: Union[str, Path],
    max_exams: int = 50,
    mode: str = 'standard',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Optional[int]]]:
    """
    Erstellt den 3D-Sequenztensor für Prüfungs-Modelle: (N, max_exams, n_features).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_pruefungen = df_pruefungen.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    df_pruefungen['is_fail'] = (~df_pruefungen['bestanden']).astype(int)
    df_pruefungen['fails_cum'] = df_pruefungen.groupby('studierenden_id')['is_fail'].cumsum()
    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['cp_cum'] = df_pruefungen.groupby('studierenden_id')['cp_earned'].cumsum()
    df_pruefungen['note_clean'] = df_pruefungen['note'].fillna(3.0)
    df_pruefungen['gpa_cum'] = df_pruefungen.groupby('studierenden_id')['note_clean'].expanding().mean().reset_index(level=0, drop=True)

    status_dict = df_abschluesse.set_index('studierenden_id')['status'].to_dict()
    demog_dict = df_abschluesse.set_index('studierenden_id').to_dict(orient='index')

    feature_names: List[str] = [
        'hzb_note',
        'hzb_typ_ord',
        'stg_Informatik', 'stg_BWL', 'stg_Maschinenbau', 'stg_Psychologie', 'stg_Soziale_Arbeit'
    ]

    if not realistic:
        feature_names.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    feature_names.extend(['fachsemester', 'versuch', 'cp_value'])
    if not realistic:
        feature_names.append('schwierigkeit')

    if not blind:
        feature_names.extend(['fails_cum', 'cp_cum', 'cp_rueckstand'])
        if not gradeblind:
            feature_names.append('gpa_cum')

    feature_names.extend([
        'support_vorher_fachlich', 'support_glz_fachlich',
        'support_vorher_ueberfachlich', 'support_glz_ueberfachlich'
    ])
    if not realistic:
        feature_names.extend(['support_vorher_psychosozial', 'support_glz_psychosozial'])

    if oracle:
        feature_names.extend(['hidden_motivation', 'hidden_soziale_integration', 'hidden_erwartete_note'])

    feature_indices: Dict[str, Optional[int]] = {
        'fach_glz': feature_names.index('support_glz_fachlich') if 'support_glz_fachlich' in feature_names else None,
        'fach_vorher': feature_names.index('support_vorher_fachlich') if 'support_vorher_fachlich' in feature_names else None,
        'uebf_glz': feature_names.index('support_glz_ueberfachlich') if 'support_glz_ueberfachlich' in feature_names else None,
        'uebf_vorher': feature_names.index('support_vorher_ueberfachlich') if 'support_vorher_ueberfachlich' in feature_names else None,
        'psych_glz': feature_names.index('support_glz_psychosozial') if 'support_glz_psychosozial' in feature_names else None,
        'psych_vorher': feature_names.index('support_vorher_psychosozial') if 'support_vorher_psychosozial' in feature_names else None,
    }

    n_features = len(feature_names)
    studis = df_abschluesse['studierenden_id'].unique()
    num_studis = len(studis)

    X_seq = np.full((num_studis, max_exams, n_features), PADDING_VALUE, dtype=np.float32)
    y_seq = np.full((num_studis, max_exams, 1), PADDING_VALUE, dtype=np.float32)
    studi_events = np.zeros(num_studis, dtype=int)

    pr_grouped = df_pruefungen.groupby('studierenden_id')

    for i, s_id in enumerate(studis):
        status = str(status_dict.get(s_id, '')).strip().lower()
        is_dropout = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
        studi_events[i] = 1 if is_dropout else 0

        if s_id not in pr_grouped.groups:
            continue

        studi_pr = pr_grouped.get_group(s_id)
        d_info = demog_dict.get(s_id, {})

        hzb = float(d_info.get('hzb_note', 2.5))
        hzb_ord = HZB_ORDINAL_MAP.get(d_info.get('hzb_typ', 'Allg. Hochschulreife'), 3.0)
        stg = str(d_info.get('stg_name', 'Informatik'))
        stg_ohe = [1.0 if s == stg else 0.0 for s in STUDIENGAENGE_LIST]
        mig = 1.0 if bool(d_info.get('migrationshintergrund', False)) else 0.0
        erst = 1.0 if bool(d_info.get('erstakademiker', False)) else 0.0
        erw = float(d_info.get('erwerbstaetigkeit_std', 0.0))

        for k, row in enumerate(studi_pr.itertuples(index=False)):
            if k >= max_exams:
                break

            fsem = float(row.fachsemester)
            cp_rueckstand = max(0.0, (fsem - 1) * 30.0 - float(row.cp_cum))

            row_feats = [hzb, hzb_ord] + stg_ohe

            if not realistic:
                row_feats.extend([mig, erst, erw])

            row_feats.extend([fsem, float(row.versuch), float(row.cp)])
            if not realistic:
                row_feats.append(float(row.schwierigkeit))

            if not blind:
                row_feats.extend([float(row.fails_cum), float(row.cp_cum), cp_rueckstand])
                if not gradeblind:
                    row_feats.append(float(row.gpa_cum))

            row_feats.extend([
                float(row.support_vorher_fachlich), float(row.support_glz_fachlich),
                float(row.support_vorher_ueberfachlich), float(row.support_glz_ueberfachlich)
            ])
            if not realistic:
                row_feats.extend([
                    float(row.support_vorher_psychosozial), float(row.support_glz_psychosozial)
                ])

            if oracle:
                row_feats.extend([
                    float(getattr(row, 'hidden_motivation', 0.5)),
                    float(getattr(row, 'hidden_soziale_integration', 0.5)),
                    float(getattr(row, 'hidden_erwartete_note', 3.0))
                ])

            X_seq[i, k, :] = row_feats
            y_seq[i, k, 0] = 1.0 if (k == len(studi_pr) - 1 and is_dropout) else 0.0

    return studis, X_seq, y_seq, studi_events, feature_names, feature_indices


# =========================================================================
# 3. SEMESTER PANEL DATAFRAME (Klasse 5: Extended Cox, DeepSurv, DML)
# =========================================================================
def build_semester_panel_df(
    data_dir: Union[str, Path],
    mode: str = 'standard',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, Optional[str]]]:
    """
    Erstellt ein Person-Semester Längsschnitt-Panel im Counting Process Format
    (t_start, t_stop, event, X_features...).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)

    pr_sem = df_pruefungen.groupby(['studierenden_id', 'fachsemester']).agg({
        'support_glz_fachlich': 'sum',
        'support_glz_ueberfachlich': 'sum',
        'support_glz_psychosozial': 'sum',
        'cp_earned': 'sum',
        'is_fail': 'sum',
        'note': 'mean',
        'hidden_motivation': 'mean',
        'hidden_soziale_integration': 'mean',
        'hidden_erwartete_note': 'mean'
    }).reset_index()

    sup_dict_fach = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_fachlich'].to_dict()
    sup_dict_uebf = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_ueberfachlich'].to_dict()
    sup_dict_psych = pr_sem.set_index(['studierenden_id', 'fachsemester'])['support_glz_psychosozial'].to_dict()
    cp_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['cp_earned'].to_dict()
    fails_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['is_fail'].to_dict()
    gpa_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['note'].to_dict()

    hmot_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['hidden_motivation'].to_dict()
    hsint_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['hidden_soziale_integration'].to_dict()
    hen_dict = pr_sem.set_index(['studierenden_id', 'fachsemester'])['hidden_erwartete_note'].to_dict()

    panel_rows = []

    for idx, row in df_abschluesse.iterrows():
        s_id = row['studierenden_id']
        max_sem = int(row['studiendauer_semester'])
        status = str(row['status']).strip().lower()
        is_event_final = status in ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']

        hzb = float(row['hzb_note'])
        hzb_ord = HZB_ORDINAL_MAP.get(row.get('hzb_typ', 'Allg. Hochschulreife'), 3.0)
        stg = str(row.get('stg_name', 'Informatik'))
        mig = 1.0 if bool(row.get('migrationshintergrund', False)) else 0.0
        erst = 1.0 if bool(row.get('erstakademiker', False)) else 0.0
        erw = float(row.get('erwerbstaetigkeit_std', 0.0))

        cum_cp_vorher = 0.0
        cum_fails_vorher = 0

        for sem in range(1, max_sem + 1):
            t_start = sem - 1
            t_stop = sem
            event_t = 1 if (sem == max_sem and is_event_final) else 0

            fach_cnt = int(sup_dict_fach.get((s_id, sem), 0))
            uebf_cnt = int(sup_dict_uebf.get((s_id, sem), 0))
            psych_cnt = int(sup_dict_psych.get((s_id, sem), 0))

            fails_prev = fails_dict.get((s_id, sem - 1), 0) if sem > 1 else 0
            delta_cp_prev = cp_dict.get((s_id, sem - 1), 0.0) if sem > 1 else 0.0
            gpa_prev = gpa_dict.get((s_id, sem - 1), 3.0) if sem > 1 else 3.0
            cp_rueckstand = max(0.0, (sem - 1) * 30.0 - cum_cp_vorher)

            hmot_prev = hmot_dict.get((s_id, sem - 1), 0.5) if sem > 1 else 0.5
            hsint_prev = hsint_dict.get((s_id, sem - 1), 0.5) if sem > 1 else 0.5
            hen_prev = hen_dict.get((s_id, sem - 1), 3.0) if sem > 1 else 3.0

            p_row = {
                'studierenden_id': s_id,
                't_start': t_start,
                't_stop': t_stop,
                'event': event_t,
                'hzb_note': hzb,
                'hzb_typ_ord': hzb_ord,
                'stg_name': stg,
                'fach_supp_count': fach_cnt,
                'uebf_supp_count': uebf_cnt,
                'psych_supp_count': psych_cnt,
                'fach_supp_active': fach_cnt,
                'uebf_supp_active': uebf_cnt,
                'psych_supp_active': psych_cnt,
                'fails_prev': fails_prev,
                'delta_cp_prev': delta_cp_prev,
                'cp_rueckstand': cp_rueckstand,
                'cum_fails': cum_fails_vorher,
                'cum_cp': cum_cp_vorher,
                'gpa_prev': gpa_prev if not np.isnan(gpa_prev) else 3.0,
                'migrationshintergrund': mig,
                'erstakademiker': erst,
                'erwerbstaetigkeit_std': erw,
                'hidden_motivation_prev': hmot_prev,
                'hidden_soziale_integration_prev': hsint_prev,
                'hidden_erwartete_note_prev': hen_prev
            }

            for s_name in STUDIENGAENGE_LIST[1:]:
                col_key = f"stg_{s_name.replace(' ', '_')}"
                p_row[col_key] = 1.0 if stg == s_name else 0.0

            panel_rows.append(p_row)

            cum_cp_vorher += cp_dict.get((s_id, sem), 0.0)
            cum_fails_vorher += fails_dict.get((s_id, sem), 0)

    panel_df = pd.DataFrame(panel_rows)

    feature_cols: List[str] = ['hzb_note', 'hzb_typ_ord'] + [f"stg_{s.replace(' ', '_')}" for s in STUDIENGAENGE_LIST[1:]]

    if not realistic:
        feature_cols.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    if not blind:
        feature_cols.extend(['fails_prev', 'delta_cp_prev', 'cp_rueckstand'])
        if not gradeblind:
            feature_cols.append('gpa_prev')

    feature_cols.extend(['fach_supp_count', 'uebf_supp_count'])
    if not realistic:
        feature_cols.append('psych_supp_count')

    if oracle:
        feature_cols.extend([
            'hidden_motivation_prev',
            'hidden_soziale_integration_prev',
            'hidden_erwartete_note_prev'
        ])

    feature_indices = {
        'fach_supp': 'fach_supp_count',
        'uebf_supp': 'uebf_supp_count',
        'psych_supp': 'psych_supp_count' if not realistic else None
    }

    return panel_df, feature_cols, 'event', feature_indices


# =========================================================================
# 4. LANDMARK DATASET (Klassen 1, 2a, 4: S1-S2 Landmark bis T0=2)
# =========================================================================
def build_landmark_dataset(
    data_dir: Union[str, Path],
    t0: int = 2,
    mode: str = 'standard',
    gradeblind: bool = False,
    blind: bool = False,
    oracle: bool = False,
    realistic: bool = False
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, Optional[str]]]:
    """
    Erstellt ein statisches Landmark-Dataset mit Aggregaten bis Semester T0 (Default: 2).
    """
    gradeblind, blind, oracle, realistic = _resolve_modes(mode, gradeblind, blind, oracle, realistic)
    df_abschluesse, df_pruefungen = _load_raw_data(data_dir)

    df_valid = df_abschluesse[df_abschluesse['studiendauer_semester'] >= t0].copy()

    df_pruefungen['cp_earned'] = np.where(df_pruefungen['bestanden'], df_pruefungen['cp'], 0)
    df_pruefungen['is_fail'] = np.where(~df_pruefungen['bestanden'], 1, 0)
    pr_t0 = df_pruefungen[df_pruefungen['fachsemester'] <= t0]

    lm_agg = pr_t0.groupby('studierenden_id').agg({
        'cp_earned': 'sum',
        'is_fail': 'sum',
        'note': 'mean',
        'support_glz_fachlich': 'sum',
        'support_glz_ueberfachlich': 'sum',
        'support_glz_psychosozial': 'sum',
        'hidden_motivation': 'mean',
        'hidden_soziale_integration': 'mean',
        'hidden_erwartete_note': 'mean'
    }).reset_index().rename(columns={
        'cp_earned': 'cp_s1s2',
        'is_fail': 'fails_s1s2',
        'note': 'gpa_s1s2',
        'support_glz_fachlich': 'fach_supp_s1s2',
        'support_glz_ueberfachlich': 'uebf_supp_s1s2',
        'support_glz_psychosozial': 'psych_supp_s1s2',
        'hidden_motivation': 'hidden_motivation_s1s2',
        'hidden_soziale_integration': 'hidden_soziale_integration_s1s2',
        'hidden_erwartete_note': 'hidden_erwartete_note_s1s2'
    })

    df_lm = pd.merge(df_valid, lm_agg, on='studierenden_id', how='left')
    df_lm['cp_s1s2'] = df_lm['cp_s1s2'].fillna(0.0)
    df_lm['fails_s1s2'] = df_lm['fails_s1s2'].fillna(0.0)
    df_lm['gpa_s1s2'] = df_lm['gpa_s1s2'].fillna(3.0)
    df_lm['fach_supp_s1s2'] = df_lm['fach_supp_s1s2'].fillna(0.0)
    df_lm['uebf_supp_s1s2'] = df_lm['uebf_supp_s1s2'].fillna(0.0)
    df_lm['psych_supp_s1s2'] = df_lm['psych_supp_s1s2'].fillna(0.0)

    df_lm['hzb_typ_ord'] = df_lm['hzb_typ'].map(HZB_ORDINAL_MAP).fillna(3.0)
    for s_name in STUDIENGAENGE_LIST:
        df_lm[f"stg_{s_name.replace(' ', '_')}"] = (df_lm['stg_name'] == s_name).astype(float)

    df_lm['is_dropout'] = df_lm['status'].str.strip().str.lower().isin(
        ['abgebrochen', 'exmatrikuliert', 'zeitueberschreitung']
    ).astype(int)

    feature_cols: List[str] = ['hzb_note', 'hzb_typ_ord'] + [f"stg_{s.replace(' ', '_')}" for s in STUDIENGAENGE_LIST]

    if not realistic:
        feature_cols.extend(['migrationshintergrund', 'erstakademiker', 'erwerbstaetigkeit_std'])

    if not blind:
        feature_cols.extend(['cp_s1s2', 'fails_s1s2'])
        if not gradeblind:
            feature_cols.append('gpa_s1s2')

    feature_cols.extend(['fach_supp_s1s2', 'uebf_supp_s1s2'])
    if not realistic:
        feature_cols.append('psych_supp_s1s2')

    if oracle:
        feature_cols.extend([
            'hidden_motivation_s1s2',
            'hidden_soziale_integration_s1s2',
            'hidden_erwartete_note_s1s2'
        ])

    feature_indices = {
        'fach_supp': 'fach_supp_s1s2',
        'uebf_supp': 'uebf_supp_s1s2',
        'psych_supp': 'psych_supp_s1s2' if not realistic else None
    }

    return df_lm, feature_cols, 'is_dropout', feature_indices