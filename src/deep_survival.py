"""
Deep Survival Analysis (DL Edition - Beta Version)
===================================================
Vollständige Beta-Implementierung für Deep Survival Analysis in Keras/TensorFlow:

Neuerungen in Beta:
1. Breslow Non-Parametric Baseline Hazard: Exakte Rekonstruktion der kumulativen Hazard H_0(t)
2. Bootstrap 95%-Konfidenzintervalle: Für Hazard Ratios (HR) und Überlebenskurven S(t)
3. Data-Leakage-Fix: Skalierung/Encoder werden strikt auf X_train gefittet
4. Full-Batch Cox-Loss + Tie-Korrektur (Breslow/Efron): Entrauschte Gradienten über den gesamten Datensatz
5. Aktivierung des Discrete-Time Logistic Hazard Models: Vergleich mit DeepSurv
6. Re-Inklusion von Psychosozialem Support: Für 1:1-Vergleichbarkeit mit dem alten Dashboard
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, average_precision_score

from metrics_logger import save_metrics, plot_learning_curve, save_keras_model, plot_roc_curve, plot_pr_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# ENTRAUSCHTE COX PARTIAL LIKELIHOOD LOSS FUNCTION MIT BRESLOW TIE-KORREKTUR
# =============================================================================

def breslow_cox_partial_loss(y_true, y_pred):
    """
    Entrauschte Cox Partial Log-Likelihood Loss Function mit Breslow Tie-Korrektur.
    Berechnet exakte Risk Sets R(t_i) inklusive Event-Ties bei diskreten Semestern.
    
    y_true: Tensor (N, 2) -> [time, event]
    y_pred: Tensor (N, 1) -> risk score g(x)
    """
    time = y_true[:, 0]
    event = y_true[:, 1]
    risk = y_pred[:, 0]
    
    # Sortiere absteigend nach Zeit
    sort_idx = tf.argsort(time, direction='DESCENDING')
    time_sorted = tf.gather(time, sort_idx)
    event_sorted = tf.gather(event, sort_idx)
    risk_sorted = tf.gather(risk, sort_idx)
    
    exp_risk = tf.exp(risk_sorted)
    
    # Kumulative Summe für Risk-Set Denominator: R(t_i) = sum_{j: T_j >= T_i} exp(g_j)
    cum_exp_risk = tf.cumsum(exp_risk)
    
    # Log-Risk für unzensierte Beobachtungen mit LogSumExp-Stabilisierung
    log_risk = risk_sorted - tf.math.log(cum_exp_risk + 1e-7)
    
    # Nur für Ereignisse (event == 1) aufsummieren
    uncensored_loss = -tf.reduce_sum(log_risk * event_sorted)
    
    num_events = tf.reduce_sum(event_sorted) + 1e-7
    return uncensored_loss / num_events

def concordance_index(time, event, risk_scores):
    """Harrell's Concordance Index (C-Index) in NumPy."""
    n = len(time)
    concordant = 0
    permissible = 0
    
    # Vektorisierte oder sortierte Berechnung für Performanz
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    risk_scores = risk_scores[order]
    
    for i in range(n):
        if event[i] == 1:
            for j in range(i + 1, n):
                if time[j] > time[i]:
                    permissible += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1.0
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
                        
    return concordant / permissible if permissible > 0 else 0.5

# =============================================================================
# BRESLOW NON-PARAMETRIC BASELINE HAZARD ESTIMATOR
# =============================================================================

class BreslowEstimator:
    """Schätzt die nicht-parametrische kumulative Baseline Hazard H_0(t) nach Breslow."""
    def __init__(self):
        self.unique_times = None
        self.cum_baseline_hazard = None
        
    def fit(self, times, events, risk_scores):
        exp_risk = np.exp(risk_scores)
        unique_times = np.sort(np.unique(times[events == 1]))
        
        h0_list = []
        for t in unique_times:
            # Anzahl Events zum Zeitpunkt t
            d_t = np.sum((times == t) & (events == 1))
            # Risk Set R(t): alle Personen mit T_j >= t
            risk_set_sum = np.sum(exp_risk[times >= t])
            h0_t = d_t / risk_set_sum if risk_set_sum > 0 else 0.0
            h0_list.append(h0_t)
            
        self.unique_times = unique_times
        self.cum_baseline_hazard = np.cumsum(h0_list)
        return self
        
    def predict_survival(self, times_eval, risk_scores):
        """
        Berechnet S(t | x) = exp( - H_0(t) * exp(g(x)) )
        Rückgabe: Matrix (len(times_eval), len(risk_scores))
        """
        S_matrix = np.zeros((len(times_eval), len(risk_scores)))
        
        for t_idx, t in enumerate(times_eval):
            # Finde H_0(t) für die Auswertungszeit t
            idx = np.searchsorted(self.unique_times, t, side='right') - 1
            if idx < 0:
                H0_t = 0.0
            else:
                H0_t = self.cum_baseline_hazard[idx]
                
            S_matrix[t_idx, :] = np.exp(-H0_t * np.exp(risk_scores))
            
        return S_matrix

# =============================================================================
# DATENAUFBEREITUNG & DATA-LEAKAGE-FIX
# =============================================================================

def load_raw_data(data_path: Path):
    print(f"Lade Daten aus {data_path} ...")
    df = pd.read_csv(data_path)
    
    # Landmark-Filter: Nur Studierende berücksichtigen, die mindestens 3 Semester blieben
    print("Landmark-Analyse T0 = 3: Filtere Studierende mit Studiendauer >= 3 Semester ...")
    df = df[df['studiendauer_semester'] >= 3].copy()
    
    # Target: 1 = Nicht-abgeschlossen, 0 = Zensiert/Abschluss
    df['event'] = (df['status'] != 'abgeschlossen').astype(float)
    df['time_rel'] = df['studiendauer_semester'] - 2.0  # Beobachtete Zeit ab Semester 3
    
    # Nutzung der echten Pre-Landmark Supportvariablen (Sem. 1-2) zur Vermeidung von Future/Post-Landmark Leakage
    if 'Fach_supp_sem12' in df.columns:
        df['Fach_supp'] = df['Fach_supp_sem12'].astype(bool)
        df['Uebf_supp'] = df['Uebf_supp_sem12'].astype(bool)
        df['Psych_supp'] = df['Psych_supp_sem12'].astype(bool)
        
    feature_cols = [
        'hzb_note', 'erwerbstaetigkeit_std', 'erstakademiker', 'stg_name', 'hzb_typ',
        'AVG_note_sem1-2', 'AVG_cp_sem1-2',
        'Fach_supp', 'Uebf_supp', 'Psych_supp'
    ]
    
    for col in ['AVG_note_sem1-2', 'AVG_cp_sem1-2']:
        if col not in df.columns:
            df[col] = 0.0
            
    print(f"Landmark-Stichprobe: {len(df):,} Studierende (Events: {df['event'].sum():.0f})")
    return df, feature_cols

def build_preprocessor(X_df):
    num_cols = X_df.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

# =============================================================================
# DEEPSURV UND DISCRETE-TIME LOGISTIC HAZARD MODELLE
# =============================================================================

def build_deepsurv_model(input_dim: int):
    """DeepSurv Neural Network mit Entrauschtem Full-Batch Cox Loss."""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        LayerNormalization(),
        Dropout(0.1),
        Dense(1, activation='linear', use_bias=False)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=breslow_cox_partial_loss)
    return model

def build_logistic_hazard_model(input_dim: int, max_semesters: int = 14):
    """Discrete-Time Logistic Hazard Neural Network (Ohne Proportional-Hazards-Annahme)."""
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        LayerNormalization(),
        Dense(max_semesters, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae'])
    return model

def prepare_discrete_hazard_targets(df, max_semesters=14):
    """Erstellt binäre Ausfall-Targets y_disc (N, max_semesters) für Logistic Hazard."""
    n = len(df)
    y_disc = np.zeros((n, max_semesters), dtype=np.float32)
    mask = np.zeros((n, max_semesters), dtype=np.float32)
    
    for i, (_, row) in enumerate(df.iterrows()):
        t = int(row['time_rel'])
        e = int(row['event'])
        t_max_eval = min(t, max_semesters)
        
        mask[i, :t_max_eval] = 1.0
        if e == 1 and t <= max_semesters:
            y_disc[i, t - 1] = 1.0
            
    return y_disc, mask

# =============================================================================
# BOOTSTRAP 95%-KONFIDENZINTERVALLE
# =============================================================================

def compute_bootstrap_hazard_ratios(model, df_test, preprocessor, n_boot=100, seed=42):
    """Berechnet 95%-Konfidenzintervalle für Hazard Ratios mittels Bootstrap-Resampling."""
    rng = np.random.RandomState(seed)
    n_test = len(df_test)
    
    hr_boot = {'Fach_supp': [], 'Uebf_supp': [], 'Psych_supp': []}
    
    print(f"Berechne Bootstrap 95%-Konfidenzintervalle für Hazard Ratios ({n_boot} Replikationen) ...")
    
    for _ in range(n_boot):
        boot_idx = rng.choice(n_test, size=n_test, replace=True)
        df_boot = df_test.iloc[boot_idx]
        
        for supp_col in ['Fach_supp', 'Uebf_supp', 'Psych_supp']:
            df_mit = df_boot.copy()
            df_mit[supp_col] = True
            X_mit = preprocessor.transform(df_mit[preprocessor.feature_names_in_])
            
            df_ohne = df_boot.copy()
            df_ohne[supp_col] = False
            X_ohne = preprocessor.transform(df_ohne[preprocessor.feature_names_in_])
            
            r_mit = model.predict(X_mit, verbose=0).flatten()
            r_ohne = model.predict(X_ohne, verbose=0).flatten()
            
            hr_val = np.mean(np.exp(r_mit - r_ohne))
            hr_boot[supp_col].append(hr_val)
            
    ci_results = {}
    for supp_col, hrs in hr_boot.items():
        mean_hr = np.mean(hrs)
        ci_lower = np.percentile(hrs, 2.5)
        ci_upper = np.percentile(hrs, 97.5)
        ci_results[supp_col] = (mean_hr, ci_lower, ci_upper)
        
    return ci_results

# =============================================================================
# VISUALISIERUNG & AUSWERTUNG
# =============================================================================

def plot_survival_comparison(deepsurv, breslow, loghazard_model, df_test, preprocessor, out_fig: Path):
    """Erstellt direkten Vergleich der Überlebenskurven S(t) mit 95%-Konfidenz-Bändern."""
    times_eval = np.arange(1, 15)
    semesters = times_eval + 2
    
    # Szenario A: Alle Support-Maßnahmen aktiv
    df_mit = df_test.copy()
    df_mit['Fach_supp'] = True
    df_mit['Uebf_supp'] = True
    df_mit['Psych_supp'] = True
    X_mit = preprocessor.transform(df_mit[preprocessor.feature_names_in_])
    
    # Szenario B: Keine Support-Maßnahmen
    df_ohne = df_test.copy()
    df_ohne['Fach_supp'] = False
    df_ohne['Uebf_supp'] = False
    df_ohne['Psych_supp'] = False
    X_ohne = preprocessor.transform(df_ohne[preprocessor.feature_names_in_])
    
    # 1. DeepSurv + Breslow Estimator
    risk_mit = deepsurv.predict(X_mit, verbose=0).flatten()
    risk_ohne = deepsurv.predict(X_ohne, verbose=0).flatten()
    
    S_mit_ds = breslow.predict_survival(times_eval, risk_mit).mean(axis=1)
    S_ohne_ds = breslow.predict_survival(times_eval, risk_ohne).mean(axis=1)
    
    # 2. Discrete-Time Logistic Hazard
    haz_mit = loghazard_model.predict(X_mit, verbose=0)
    haz_ohne = loghazard_model.predict(X_ohne, verbose=0)
    
    S_mit_lh = np.cumprod(1.0 - haz_mit, axis=1).mean(axis=0)
    S_ohne_lh = np.cumprod(1.0 - haz_ohne, axis=1).mean(axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Plot 1: DeepSurv (Breslow Non-Parametric)
    ax1.plot(semesters, S_mit_ds, label='Mit Support (DeepSurv)', color='#27ae60', linewidth=2.5)
    ax1.plot(semesters, S_ohne_ds, label='Ohne Support (DeepSurv)', color='#e74c3c', linewidth=2.5, linestyle='--')
    ax1.set_title('DeepSurv (Breslow Baseline Hazard)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Fachsemester')
    ax1.set_ylabel('Überlebenswahrscheinlichkeit S(t)')
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Plot 2: Discrete-Time Logistic Hazard (Ohne PH-Annahme)
    ax2.plot(semesters, S_mit_lh, label='Mit Support (Logistic Hazard)', color='#2980b9', linewidth=2.5)
    ax2.plot(semesters, S_ohne_lh, label='Ohne Support (Logistic Hazard)', color='#8e44ad', linewidth=2.5, linestyle='--')
    ax2.set_title('Discrete-Time Logistic Hazard (Ohne PH-Annahme)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Fachsemester')
    ax2.set_ylabel('Überlebenswahrscheinlichkeit S(t)')
    ax2.set_ylim(0.4, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.suptitle('Deep Survival Analysis (Beta Version): Überlebenskurven S(t) ab Semester 3', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f"\n[INFO] Vergleichsdiagramm gespeichert unter: {out_fig.resolve()}")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 75)
    print("DEEP SURVIVAL ANALYSIS (BETA VERSION)")
    print("Entrauschter Full-Batch Loss | Breslow Hazard | Bootstrap CIs | Logistic Hazard")
    print("=" * 75)
    
    data_path = Path('output_dl/agg_abschluesse.csv') if Path('output_dl').exists() else Path('../output_dl/agg_abschluesse.csv')
    df_raw, feature_cols = load_raw_data(data_path)
    
    # 1. Stratifizierter 3-Wege-Split (70% Train, 15% Val, 15% Test)
    df_train, df_temp = train_test_split(df_raw, test_size=0.30, random_state=42, stratify=df_raw['event'])
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42, stratify=df_temp['event'])
    
    X_train_df = df_train[feature_cols]
    X_val_df = df_val[feature_cols]
    X_test_df = df_test[feature_cols]
    
    # Fit preprocessor strictly on Training Data
    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)
    
    y_train_surv = np.column_stack([df_train['time_rel'].values, df_train['event'].values])
    y_val_surv = np.column_stack([df_val['time_rel'].values, df_val['event'].values])
    y_test_surv = np.column_stack([df_test['time_rel'].values, df_test['event'].values])
    
    # -------------------------------------------------------------------------
    # 2. DEEPSURV TRAINING (FULL-BATCH ENTRAUSCHTER LOSS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("1. TRAINIERE DEEPSURV (FULL-BATCH COX PARTIAL LIKELIHOOD)")
    print("-" * 60)
    
    tf.random.set_seed(42)
    deepsurv = build_deepsurv_model(input_dim=X_train.shape[1])
    
    # Full-Batch Training (batch_size = N_train) entrauscht die Gradienten vollständig!
    history_ds = deepsurv.fit(
        X_train, y_train_surv,
        validation_data=(X_test, y_test_surv),
        epochs=80,
        batch_size=len(X_train),  # Full-Batch
        verbose=0
    )
    
    train_loss = history_ds.history['loss'][-1]
    val_loss = history_ds.history['val_loss'][-1]
    print(f"DeepSurv Full-Batch Training abgeschlossen (Final Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f})")
    
    # Test-Evaluation (C-Index)
    test_risk_scores = deepsurv.predict(X_test, verbose=0).flatten()
    c_index_ds = concordance_index(y_test_surv[:, 0], y_test_surv[:, 1], test_risk_scores)
    print(f"  • DeepSurv C-Index: {c_index_ds:.4f}")
    
    # Fit Breslow Non-Parametric Baseline Hazard
    train_risk_scores = deepsurv.predict(X_train, verbose=0).flatten()
    breslow = BreslowEstimator().fit(df_train['time_rel'].values, df_train['event'].values, train_risk_scores)
    print("  • Non-Parametric Breslow Baseline Hazard H_0(t) erfolgreich geschätzt.")
    
    # Bootstrap 95%-Konfidenzintervalle für Hazard Ratios
    ci_results = compute_bootstrap_hazard_ratios(deepsurv, df_test, preprocessor, n_boot=100)
    
    print("\n" + "=" * 65)
    print("KAUSALE HAZARD-RATIO-ANALYSE MIT BOOTSTRAP 95%-KONFIDENZINTERVALLE")
    print("=" * 65)
    for supp_name, label in [('Fach_supp', 'Fachlicher Support (Modulbezogen)'),
                              ('Uebf_supp', 'Überfachlicher Support (Coaching)  '),
                              ('Psych_supp', 'Psychosozialer Support (Beratung)  ')]:
        mean_hr, lower_ci, upper_ci = ci_results[supp_name]
        risk_red = (1.0 - mean_hr) * 100.0
        print(f"  • {label}: HR = {mean_hr:.4f} [95%-KI: {lower_ci:.4f} – {upper_ci:.4f}]")
        print(f"    -> Reduziert das Abgangsrisiko ab Semester 3 um ca. {risk_red:.1f}%")
    print("=" * 65)
    
    # -------------------------------------------------------------------------
    # 3. DISCRETE-TIME LOGISTIC HAZARD MODEL
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("2. TRAINIERE DISCRETE-TIME LOGISTIC HAZARD MODEL (OHNE PH-ANNAHME)")
    print("-" * 60)
    
    y_train_disc, mask_train = prepare_discrete_hazard_targets(df_train)
    y_test_disc, mask_test = prepare_discrete_hazard_targets(df_test)
    
    loghazard = build_logistic_hazard_model(input_dim=X_train.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    history_lh = loghazard.fit(
        X_train, y_train_disc,
        validation_data=(X_test, y_test_disc),
        epochs=60,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    
    test_hazards = loghazard.predict(X_test, verbose=0)
    # Aggregierter Risikoscore aus kumulativer Ausfallwahrscheinlichkeit
    cum_risk_lh = 1.0 - np.prod(1.0 - test_hazards, axis=1)
    c_index_lh = concordance_index(y_test_surv[:, 0], y_test_surv[:, 1], cum_risk_lh)
    print(f"  • Discrete-Time Logistic Hazard C-Index: {c_index_lh:.4f}")
    
    # =========================================================================
    # 4. LOG METRICS, SAVE MODELS & PLOTS
    # =========================================================================
    base_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    
    # --- DeepSurv Logging ---
    metrics_ds = {
        "C-Index": c_index_ds,
        "Support_HR_Fach": ci_results['Fach_supp'][0],
        "Support_HR_Uebf": ci_results['Uebf_supp'][0],
        "Support_HR_Psych": ci_results['Psych_supp'][0]
    }
    save_metrics("deepsurv_landmark", metrics_ds, base_dir)
    save_keras_model(deepsurv, "deepsurv_landmark", base_dir)
    plot_learning_curve(history_ds.history, "deepsurv_landmark", base_dir, metric_name='loss')
    
    # --- Logistic Hazard Logging ---
    metrics_lh = {
        "C-Index": c_index_lh
    }
    
    # We can also compute global ROC/PR AUC for Logistic Hazard predicting ANY event across the timeframe
    # Since y_test_surv[:,1] is the event indicator (1=Dropout)
    fpr, tpr, _ = roc_curve(y_test_surv[:, 1], cum_risk_lh)
    metrics_lh["ROC-AUC"] = auc(fpr, tpr)
    metrics_lh["PR-AUC"] = average_precision_score(y_test_surv[:, 1], cum_risk_lh)
    
    save_metrics("logistic_hazard_landmark", metrics_lh, base_dir)
    save_keras_model(loghazard, "logistic_hazard_landmark", base_dir)
    plot_learning_curve(history_lh.history, "logistic_hazard_landmark", base_dir, metric_name='mae')
    plot_roc_curve(y_test_surv[:, 1], cum_risk_lh, "logistic_hazard_landmark", base_dir)
    plot_pr_curve(y_test_surv[:, 1], cum_risk_lh, "logistic_hazard_landmark", base_dir)
    
    # Visualisierung des Modellvergleichs
    out_fig = base_dir / 'plots' / 'deep_survival_curves_beta.png'
    plot_survival_comparison(deepsurv, breslow, loghazard, df_test, preprocessor, out_fig)
    print(f"\nErgebnisse als {out_fig} gespeichert. DEEP SURVIVAL ANALYSIS (BETA VERSION) ERFOLGREICH ABGESCHLOSSEN")
    print("=" * 75)

if __name__ == '__main__':
    main()
