"""
Zeitreihen-Analyse: Variante 1 (Semester-basierte Zeitreihe)
===========================================================
Aggregiert Prüfungs- und Supportdaten pro Studierendem auf Semester-Ebene.

Schrittweite: 1 Semester (t = 1..T_max)

Features pro Semester (F):
- sem_avg_note: Notendurchschnitt im Semester
- sem_cp_earned: Summe bestandener CPs im Semester
- sem_cp_attempted: Summe angemeldeter CPs im Semester
- sem_fail_count: Anzahl Nichtbestehen im Semester
- SEPARATE Support-Counts:
  - sem_support_fachlich_relevant: Modulbezogener fachlicher Support
  - sem_support_fachlich_sonst: Sonstiger fachlicher Support
  - sem_support_ueberfachlich: Überfachlicher Support (Lerncoaching, Zeitmanagement)
  - sem_support_psychosozial: Psychosozialer Support (Beratung, Peer-Group)
- Statistische & Demografische Merkmale (hzb_note, erwerbstaetigkeit_std, stg_name OHE)

Modell: Keras LSTM / GRU mit Masking-Layer zur Vorhersage des semesterweisen Notenverlaufs.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, LSTM, GRU, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_parity_plot

PADDING_VALUE = -99.0

def create_semester_timeseries_dataset(output_dir: Path):
    print("Lade Datensätze für semesterweise Zeitreihen-Transformation ...")
    studierende_df = pd.read_csv(output_dir / 'studierende.csv')
    studiengaenge_df = pd.read_csv(output_dir / 'studiengaenge.csv')
    einschreibungen_df = pd.read_csv(output_dir / 'einschreibungen.csv')
    pruefungen_df = pd.read_csv(output_dir / 'pruefungen.csv')
    module_df = pd.read_csv(output_dir / 'module.csv')
    support_angebote_df = pd.read_csv(output_dir / 'support_angebote.csv')
    support_zuordnung_df = pd.read_csv(output_dir / 'support_modul_zuordnung.csv')
    support_teilnahmen_df = pd.read_csv(output_dir / 'support_teilnahmen.csv')
    
    # Anreichern
    studierende_df = studierende_df.merge(
        studiengaenge_df.rename(columns={'name': 'stg_name'})[['studiengang_id', 'stg_name']],
        on='studiengang_id', how='left'
    )
    
    pruefungen_df = pruefungen_df.merge(
        module_df[['modul_id', 'cp']], on='modul_id', how='left'
    ).merge(
        einschreibungen_df[['studierenden_id', 'semester_id', 'fachsemester']],
        on=['studierenden_id', 'semester_id'], how='left'
    )
    
    # Support-Klassifikation (fachlich_relevant, fachlich_sonst, ueberfachlich, psychosozial)
    sup_full = support_teilnahmen_df.merge(
        support_angebote_df[['angebot_id', 'typ']], on='angebot_id', how='left'
    ).merge(
        einschreibungen_df[['studierenden_id', 'semester_id', 'fachsemester']],
        on=['studierenden_id', 'semester_id'], how='left'
    )
    
    # Fachlich relevant: Hat das Angebot eine Zuordnung zu einem im Semester belegten Modul?
    sup_zuord = sup_full.merge(support_zuordnung_df, on='angebot_id', how='left')
    sem_pr_modules = pruefungen_df[['studierenden_id', 'semester_id', 'modul_id']].drop_duplicates()
    
    sup_zuord = sup_zuord.merge(
        sem_pr_modules,
        left_on=['studierenden_id', 'semester_id', 'modul_id'],
        right_on=['studierenden_id', 'semester_id', 'modul_id'],
        how='left',
        indicator=True
    )
    sup_zuord['is_relevant'] = sup_zuord['_merge'] == 'both'
    
    # Aggregation pro Student & Fachsemester
    print("Aggregiere Features pro Student & Fachsemester ...")
    sem_pr_agg = pruefungen_df.groupby(['studierenden_id', 'fachsemester']).agg(
        sem_avg_note=('note', 'mean'),
        sem_cp_earned=('cp', lambda cps: cps[pruefungen_df.loc[cps.index, 'bestanden']].sum()),
        sem_cp_attempted=('cp', 'sum'),
        sem_fail_count=('bestanden', lambda b: (~b).sum()),
    ).reset_index()
    
    # Support-Counts getrennt aggrigieren
    sup_fach_rel = sup_zuord[(sup_zuord['typ'] == 'fachlich') & sup_zuord['is_relevant']].groupby(['studierenden_id', 'fachsemester']).size().rename('sem_support_fachlich_relevant')
    sup_fach_sonst = sup_zuord[(sup_zuord['typ'] == 'fachlich') & (~sup_zuord['is_relevant'])].groupby(['studierenden_id', 'fachsemester']).size().rename('sem_support_fachlich_sonst')
    sup_ueberfachlich = sup_full[sup_full['typ'] == 'ueberfachlich'].groupby(['studierenden_id', 'fachsemester']).size().rename('sem_support_ueberfachlich')
    sup_psychosozial = sup_full[sup_full['typ'] == 'psychosozial'].groupby(['studierenden_id', 'fachsemester']).size().rename('sem_support_psychosozial')
    
    sem_pr_agg = sem_pr_agg.merge(sup_fach_rel, on=['studierenden_id', 'fachsemester'], how='left') \
                           .merge(sup_fach_sonst, on=['studierenden_id', 'fachsemester'], how='left') \
                           .merge(sup_ueberfachlich, on=['studierenden_id', 'fachsemester'], how='left') \
                           .merge(sup_psychosozial, on=['studierenden_id', 'fachsemester'], how='left')
                           
    sem_cols_sup = ['sem_support_fachlich_relevant', 'sem_support_fachlich_sonst', 'sem_support_ueberfachlich', 'sem_support_psychosozial']
    for c in sem_cols_sup:
        sem_pr_agg[c] = sem_pr_agg[c].fillna(0).astype(int)
        
    # Statische Merkmale vorbereiten
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_feats = ohe.fit_transform(studierende_df[['stg_name', 'hzb_typ']])
    cat_cols = ohe.get_feature_names_out(['stg_name', 'hzb_typ']).tolist()
    cat_df = pd.DataFrame(cat_feats, columns=cat_cols)
    
    stat_df = pd.concat([
        studierende_df[['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker']],
        cat_df
    ], axis=1)
    
    # Skalierung der kontinuierlichen Variablen wurde nach hinten verschoben (vermeidet Data Leakage)
    num_seq_cols = ['sem_cp_earned', 'sem_cp_attempted', 'sem_fail_count'] + sem_cols_sup
    
    # Statische Merkmale:
    # hzb_note und erwerbstaetigkeit_std werden später skaliert
    
    # 3D Tensor (N x T_max x F) aufbauen
    max_semesters = int(sem_pr_agg['fachsemester'].max()) # z.B. 16
    studi_list = studierende_df['studierenden_id'].tolist()
    n_studis = len(studi_list)
    
    seq_feature_cols = num_seq_cols
    stat_feature_cols = ['hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker'] + cat_cols
    f_total = len(seq_feature_cols) + len(stat_feature_cols)
    
    print(f"Konstruiere 3D Tensor: {n_studis} Studierende x {max_semesters} Semester x {f_total} Features ...")
    
    X_tensor = np.full((n_studis, max_semesters, f_total), PADDING_VALUE, dtype=np.float32)
    y_target = np.zeros(n_studis, dtype=np.float32)
    
    stat_dict = stat_df.set_index('studierenden_id').to_dict('index')
    sem_grouped = {s_id: group for s_id, group in sem_pr_agg.groupby('studierenden_id')}
    
    for i, s_id in enumerate(studi_list):
        if s_id in sem_grouped:
            s_data = sem_grouped[s_id].sort_values('fachsemester')
            y_target[i] = s_data['sem_avg_note'].mean()
            s_stat = list(stat_dict[s_id].values())
            for row in s_data.itertuples(index=False):
                t = int(row.fachsemester) - 1
                if t < max_semesters:
                    seq_vals = [getattr(row, c) for c in seq_feature_cols]
                    X_tensor[i, t, :] = np.array(seq_vals + s_stat, dtype=np.float32)

    return X_tensor, y_target, max_semesters, f_total

def build_semester_lstm(max_semesters: int, num_features: int):
    model = Sequential([
        Masking(mask_value=PADDING_VALUE, input_shape=(max_semesters, num_features)),
        LSTM(64, return_sequences=True),
        LayerNormalization(),
        Dropout(0.3),
        LSTM(32),
        LayerNormalization(),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("=" * 70)
    print("ZEITREIHEN-ANALYSE (VARIANTE 1: SEMESTER-SCHRITTWEITE & LSTM)")
    print("=" * 70)
    
    output_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    X, y, max_sem, n_features = create_semester_timeseries_dataset(output_dir)
    
    # 3-Wege Split (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"\nDatensatz-Aufteilung:")
    print(f"  - Training Set:   {X_train.shape[0]} Sequenzen")
    print(f"  - Validation Set: {X_val.shape[0]} Sequenzen")
    print(f"  - Test Set:       {X_test.shape[0]} Sequenzen")
    
    print("\nSkaliere Features nach dem Split (vermeidet Leakage) ...")
    
    # Indizes der Features
    # seq_feature_cols hat 7 Features: ['sem_cp_earned', 'sem_cp_attempted', 'sem_fail_count'] + 4 Support
    # stat_feature_cols startet danach. 'hzb_note' und 'erwerbstaetigkeit_std' sind die ersten beiden statischen.
    num_seq_feats = 7
    stat_start = num_seq_feats
    
    # 1. Sequentielle Features skalieren
    valid_mask_train = X_train[:, :, 0] != PADDING_VALUE
    train_seq_valid = X_train[valid_mask_train][:, :num_seq_feats]
    
    scaler_seq = StandardScaler()
    scaler_seq.fit(train_seq_valid)
    
    # 2. Statische Features skalieren (nur einmal pro validem Student)
    student_mask_train = valid_mask_train.any(axis=1)
    # Nimm den ersten Zeitschritt für jeden validen Studenten (statische Features sind konstant)
    # Suche den ersten validen Zeitschritt für jeden Studenten
    first_valid_idx = np.argmax(valid_mask_train[student_mask_train], axis=1)
    train_stat_valid = X_train[student_mask_train, first_valid_idx, stat_start:stat_start+2]
    
    scaler_stat = StandardScaler()
    scaler_stat.fit(train_stat_valid)
    
    # Transformation anwenden auf alle Sets
    for X_split in [X_train, X_val, X_test]:
        valid_mask = X_split[:, :, 0] != PADDING_VALUE
        X_split[valid_mask, :num_seq_feats] = scaler_seq.transform(X_split[valid_mask, :num_seq_feats])
        X_split[valid_mask, stat_start:stat_start+2] = scaler_stat.transform(X_split[valid_mask, stat_start:stat_start+2])
    
    print("\nTrainiere Keras Semester-LSTM ...")
    model = build_semester_lstm(max_sem, n_features)
    model.summary()
    
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluierung
    test_preds = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    
    print("\n" + "=" * 70)
    print("ERGEBNISSE SEMESTER-LSTM (TEST-SET)")
    print("=" * 70)
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print("=" * 70)
    
    # Metriken & Modell speichern
    metrics_dict = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
    save_metrics("timeseries_semester_lstm", metrics_dict, output_dir)
    save_keras_model(model, "timeseries_semester_lstm", output_dir)
    plot_learning_curve(history.history, "timeseries_semester_lstm", output_dir, metric_name='mae')
    plot_parity_plot(y_test, test_preds, "timeseries_semester_lstm", output_dir)
    
    # Lernkurven plotten
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss (MSE)', color='#2980b9', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)', color='#e74c3c', linewidth=2)
    plt.title('Semester-LSTM: Learning Curve (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    out_fig = Path('output_dl/learning_curves_semester_lstm.png') if Path('output_dl').exists() else Path('../output_dl/learning_curves_semester_lstm.png')
    plt.savefig(out_fig, dpi=300)
    print(f"[INFO] Lernkurve gespeichert unter: {out_fig.resolve()}")

if __name__ == '__main__':
    main()
