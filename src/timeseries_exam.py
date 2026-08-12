"""
Zeitreihen-Analyse: Variante 2 (Prüfungs-basierte Zeitreihe)
===========================================================
Erstellt eine Zeitreihe, bei der jeder Zeitschritt eine EINZELNE PRÜFUNG darstellt.
Semester-Informationen bleiben erhalten und werden nicht aggregiert.

Schrittweite: 1 Einzelprüfung (t = 1..T_exam_max)

Features pro Prüfungs-Schritt (F):
- fachsemester: Das Fachsemester, in dem die Prüfung abgelegt wurde
- versuch: Prüfungsversuch (1, 2 oder 3)
- cp: Credit Points des Moduls
- schwierigkeit: Modulschwierigkeit
- Support-Expositionen (vorher/gleichzeitig getrennt):
  - support_vorher_fachlich, support_glz_fachlich
  - support_vorher_ueberfachlich, support_glz_ueberfachlich
  - support_vorher_psychosozial, support_glz_psychosozial
- support_genutzt: Binäres Flag, ob Support in Anspruch genommen wurde
- Statische Merkmale (hzb_note, erwerbstaetigkeit_std, stg_name OHE)

Modell: Keras GRU / LSTM mit Masking-Layer (für unterschiedlich lange Prüfungssequenzen).
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
from tensorflow.keras.layers import Dense, Dropout, Masking, GRU, LSTM, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

from metrics_logger import save_metrics, save_keras_model, plot_learning_curve, plot_parity_plot

PADDING_VALUE = -99.0

def create_exam_timeseries_dataset(output_dir: Path):
    print("Lade Datensätze für prüfungsweise Zeitreihen-Transformation ...")
    if not (output_dir / 'studierende.csv').exists():
        output_dir = Path('../output_dl') if Path('../output_dl/studierende.csv').exists() else Path('output_dl')
    if not (output_dir / 'studierende.csv').exists():
        output_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    studierende_df = pd.read_csv(output_dir / 'studierende.csv')
    studiengaenge_df = pd.read_csv(output_dir / 'studiengaenge.csv')
    agg_pruefungen_df = pd.read_csv(output_dir / 'agg_pruefungen.csv')
    
    # Anreichern
    studierende_df = studierende_df.merge(
        studiengaenge_df.rename(columns={'name': 'stg_name'})[['studiengang_id', 'stg_name']],
        on='studiengang_id', how='left'
    )
    
    # Neue rollierende Features (V2) berechnen
    agg_pruefungen_df = agg_pruefungen_df.sort_values(['studierenden_id', 'pruefung_id']).reset_index(drop=True)
    agg_pruefungen_df['is_fail'] = (~agg_pruefungen_df['bestanden']).astype(int)
    agg_pruefungen_df['fails_cum'] = agg_pruefungen_df.groupby('studierenden_id')['is_fail'].cumsum()
    agg_pruefungen_df['cp_earned'] = np.where(agg_pruefungen_df['bestanden'], agg_pruefungen_df['cp'], 0)
    agg_pruefungen_df['cp_cum'] = agg_pruefungen_df.groupby('studierenden_id')['cp_earned'].cumsum()
    agg_pruefungen_df['note_clean'] = agg_pruefungen_df['note'].fillna(3.0)
    agg_pruefungen_df['gpa_cum'] = agg_pruefungen_df.groupby('studierenden_id')['note_clean'].expanding().mean().reset_index(level=0, drop=True)

    # LAGGING (Shift by 1) zur Vermeidung von Data Leakage
    agg_pruefungen_df['fails_cum_lag'] = agg_pruefungen_df.groupby('studierenden_id')['fails_cum'].shift(1).fillna(0)
    agg_pruefungen_df['cp_cum_lag'] = agg_pruefungen_df.groupby('studierenden_id')['cp_cum'].shift(1).fillna(0)
    agg_pruefungen_df['gpa_cum_lag'] = agg_pruefungen_df.groupby('studierenden_id')['gpa_cum'].shift(1).fillna(3.0)

    # Skalierung der Prüfungsmerkmale
    exam_num_cols = [
        'fachsemester', 'versuch', 'cp', 'schwierigkeit',
        'support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
        'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial',
        'fails_cum_lag', 'cp_cum_lag'
    ]
    
    # Skalierung der Prüfungsmerkmale wird nach hinten verschoben (vermeidet Data Leakage)
    # Ergänze Spalten falls fehlend
    for c in exam_num_cols:
        if c not in agg_pruefungen_df.columns:
            agg_pruefungen_df[c] = 0.0
    
    # Statistische Merkmale vorbereiten
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_feats = ohe.fit_transform(studierende_df[['stg_name', 'hzb_typ']])
    cat_cols = ohe.get_feature_names_out(['stg_name', 'hzb_typ']).tolist()
    cat_df = pd.DataFrame(cat_feats, columns=cat_cols)
    
    stat_df = pd.concat([
        studierende_df[['studierenden_id', 'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker']],
        cat_df
    ], axis=1)
    
    # Skalierung der statischen Merkmale wird nach hinten verschoben    
    # Bestimme maximale Prüfungsanzahl pro Student (T_exam_max)
    exam_counts = agg_pruefungen_df.groupby('studierenden_id').size()
    max_exams = int(exam_counts.max())  # z.B. 25-30
    
    studi_list = studierende_df['studierenden_id'].tolist()
    n_studis = len(studi_list)
    
    stat_feature_cols = ['hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker'] + cat_cols
    f_total = len(exam_num_cols) + len(stat_feature_cols) + 1  # +1 für support_genutzt flag
    
    print(f"Konstruiere 3D Tensor: {n_studis} Studierende x {max_exams} Prüfungen x {f_total} Features ...")
    
    X_tensor = np.full((n_studis, max_exams, f_total), PADDING_VALUE, dtype=np.float32)
    y_target = np.zeros(n_studis, dtype=np.float32)
    
    stat_dict = stat_df.set_index('studierenden_id').to_dict('index')
    
    exam_grouped = {s_id: group for s_id, group in agg_pruefungen_df.groupby('studierenden_id')}
    
    for i, s_id in enumerate(studi_list):
        if s_id in exam_grouped:
            s_exams = exam_grouped[s_id].sort_values(['fachsemester', 'pruefung_id'])
            y_target[i] = s_exams['note'].mean()
            s_stat = list(stat_dict[s_id].values())
            for t, row in enumerate(s_exams.itertuples(index=False)):
                if t < max_exams:
                    exam_vals = [getattr(row, c) for c in exam_num_cols] + [float(row.support_genutzt)]
                    X_tensor[i, t, :] = np.array(exam_vals + s_stat, dtype=np.float32)

    return X_tensor, y_target, max_exams, f_total

def build_exam_gru(max_exams: int, num_features: int):
    model = Sequential([
        Masking(mask_value=PADDING_VALUE, input_shape=(max_exams, num_features)),
        GRU(64, return_sequences=True),
        LayerNormalization(),
        Dropout(0.3),
        GRU(32),
        LayerNormalization(),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("=" * 70)
    print("ZEITREIHEN-ANALYSE (VARIANTE 2: PRÜFUNGS-SCHRITTWEITE & GRU)")
    print("=" * 70)
    
    output_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    X, y, max_exams, n_features = create_exam_timeseries_dataset(output_dir)
    
    # 3-Wege Split (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    print(f"\nDatensatz-Aufteilung:")
    print(f"  - Training Set:   {X_train.shape[0]} Sequenzen")
    print(f"  - Validation Set: {X_val.shape[0]} Sequenzen")
    print(f"  - Test Set:       {X_test.shape[0]} Sequenzen")
    
    print("\nSkaliere Features nach dem Split (vermeidet Leakage) ...")
    
    # Indizes der Features
    # exam_num_cols hat 12 Features (Indizes 0 bis 11)
    # support_genutzt ist Index 12
    # stat_feature_cols startet bei Index 13. 'hzb_note' und 'erwerbstaetigkeit_std' sind 13 und 14.
    num_seq_feats = 12
    stat_start = 13
    
    # 1. Sequentielle Features skalieren
    # Da PADDING_VALUE -99.0 ist und wir bei Index 0 (fachsemester) sicher positive Werte haben,
    # können wir auf != PADDING_VALUE prüfen.
    valid_mask_train = X_train[:, :, 0] != PADDING_VALUE
    train_seq_valid = X_train[valid_mask_train][:, :num_seq_feats]
    
    scaler_exam = StandardScaler()
    scaler_exam.fit(train_seq_valid)
    
    # 2. Statische Features skalieren (nur einmal pro validem Student)
    student_mask_train = valid_mask_train.any(axis=1)
    # Finde den ersten validen Zeitschritt für jeden Studenten
    first_valid_idx = np.argmax(valid_mask_train[student_mask_train], axis=1)
    train_stat_valid = X_train[student_mask_train, first_valid_idx, stat_start:stat_start+2]
    
    scaler_stat = StandardScaler()
    scaler_stat.fit(train_stat_valid)
    
    # Transformation anwenden auf alle Sets
    for X_split in [X_train, X_val, X_test]:
        valid_mask = X_split[:, :, 0] != PADDING_VALUE
        X_split[valid_mask, :num_seq_feats] = scaler_exam.transform(X_split[valid_mask, :num_seq_feats])
        X_split[valid_mask, stat_start:stat_start+2] = scaler_stat.transform(X_split[valid_mask, stat_start:stat_start+2])
    
    print("\nTrainiere Keras Prüfungs-GRU (Sequential Exam Recurrent Network) ...")
    model = build_exam_gru(max_exams, n_features)
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
    print("ERGEBNISSE PRÜFUNGS-GRU (TEST-SET)")
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
    output_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    save_metrics("timeseries_exam_gru", metrics_dict, output_dir)
    save_keras_model(model, "timeseries_exam_gru", output_dir)
    plot_learning_curve(history.history, "timeseries_exam_gru", output_dir, metric_name='mae')
    plot_parity_plot(y_test, test_preds, "timeseries_exam_gru", output_dir)

if __name__ == '__main__':
    main()
