"""
Training Script: Abschlussnoten-Vorhersage (Regression)
======================================================
Vorhersage der Abschlussnote ('abschlussnote') auf Basis von 'agg_abschluesse.csv'.

Modelle:
1. Linear Regression (Ridge Baseline)
2. Support Vector Regression (SVR)
3. Random Forest Regressor
4. Keras Multi-Layer Perceptron (MLP Regressor)

Behandlung von Abbrechern (NaNs bei Abschlussnote):
- Modus 'graduates_only' (Standard): Filtert auf erfolgreiche Abschlüsse (Noten 1.0–4.0).
  Dies entspricht der echten Notenprognose für Absolventen.
- Modus 'impute_5.0': Ersetzt NaNs bei Abbrechern durch 5.0 ('nicht bestanden').
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from metrics_logger import save_metrics, plot_learning_curve, save_keras_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# FEATURE-MASKEN & KONFIGURATION
# =============================================================================

# Target Spalte
TARGET_COL = "abschlussnote"

# 1. Zwingende Leakage-Spalten (müssen IMMER maskiert werden)
LEAKAGE_COLUMNS = [
    "studierenden_id",
    "abschlussnote",           # Target
    "bachelorarbeitsnote",     # Direkt korreliertes Teilmerkmal
    "status",                  # Verrät Abbruch direkt
    "studiendauer_semester",   # Verrät exakte Semesterdauer
    "abschluss_semester_id",   # Verrät Endsemester
    "anomalie_typ",            # Enthält explizites Abbruch-Labeling
]

# Spalten, die Future Leakage, Lifetime-Aggregat oder Ground-Truth darstellen
STRICT_MASKED_COLUMNS = [
    # Full-Study / Lifetime Aggregates
    "AVG_Note", "Anz_Pruefungen", "Anz_Bestanden", "Anz_Fehlversuche", "Fehlversuchsquote", "ECTS_bestanden",
    # Attempt-Level Lifetime Breakdown (Future Leakage)
    "Anz_ErstVersuche", "AVG_ErstVersucheNote", 
    "Anz_ZweitVersuche", "AVG_ZweitVersucheNote", 
    "Anz_DrittVersuche", "AVG_DrittVersucheNote",
    # Post-Landmark (Sem 3-4) Aggregates (Future Leakage)
    "AVG_note_sem1-4", "AVG_cp_sem1-4",
    # Lifetime Support Aggregates (Future Leakage)
    "Fach_supp", "Uebf_supp", "Psych_supp", "support_exposure_count", "any_support", "support_exposure_group",
]

def load_and_preprocess_regression_data(
    data_path: Path, 
    impute_strategy: str = "graduates_only"  # 'graduates_only' oder 'impute_5.0'
):
    print(f"Lade Daten aus {data_path} ...")
    df = pd.read_csv(data_path)
    
    # Behandlung von NaNs in der Abschlussnote
    if impute_strategy == "graduates_only":
        print("Modus: 'graduates_only' -> Filtere auf Studierende mit geglücktem Abschluss (Noten 1.0–4.0).")
        df = df[df[TARGET_COL].notna()].copy()
    elif impute_strategy == "impute_5.0":
        print("Modus: 'impute_5.0' -> Ersetze NaNs bei Abbrechern durch 5.0 (Nicht bestanden).")
        df[TARGET_COL] = df[TARGET_COL].fillna(5.0)
    else:
        raise ValueError(f"Unbekannte impute_strategy: {impute_strategy}")
        
    y = df[TARGET_COL].values
    
    # Automatische Erkennung aller hidden_* Ground-Truth-Spalten & Per-Attempt Support Spalten
    hidden_and_attempt_cols = [col for col in df.columns if col.startswith('hidden_') or 'Support_' in col or 'Versuche' in col]
    exclude_set = set(LEAKAGE_COLUMNS + STRICT_MASKED_COLUMNS + hidden_and_attempt_cols)
    feature_cols = [col for col in df.columns if col not in exclude_set and col != TARGET_COL]
    
    print(f"Nutze {len(feature_cols)} ehrliche Features (Demographie + Sem 1-2 Landmark) für die Noten-Regression.")
    print("Verwendete Features:", feature_cols)
    
    print(f"Datensatzgröße: {len(df)} Zeilen.")
    print(f"Nutze {len(feature_cols)} Features für die Noten-Regression.")
    print("Maskierte Spalten:", sorted(list(exclude_set.intersection(df.columns))))
    
    X_df = df[feature_cols]
    
    num_cols = X_df.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    return X_df, y, feature_cols, preprocessor

def build_mlp_regressor(input_dim: int):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dropout(0.1),
        Dense(1, activation='linear')  # Kontinuierliche Noten-Ausgabe
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_regressor(model_name: str, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name:<25} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def plot_regression_results(history, y_test, test_predictions, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Keras Loss Curve (MSE)
    axes[0].plot(history.history['loss'], label='Train MSE', color='#2980b9', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val MSE', color='#e74c3c', linewidth=2)
    axes[0].set_title('Keras MLP Regressor: MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Parity Plot: Actual vs. Predicted (Keras MLP)
    y_pred_mlp = test_predictions["Keras MLP"]
    axes[1].scatter(y_test, y_pred_mlp, alpha=0.3, color='#8e44ad', edgecolors='none', s=20)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfekte Diagonale')
    axes[1].set_title('Parity Plot (Keras MLP): Tatsächlich vs. Vorhergesagt')
    axes[1].set_xlabel('Tatsächliche Abschlussnote')
    axes[1].set_ylabel('Vorhergesagte Abschlussnote')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. R² Vergleich der Modelle
    models = list(test_predictions.keys())
    r2_scores = [r2_score(y_test, test_predictions[m]) for m in models]
    
    colors = ['#34495e', '#27ae60', '#2980b9', '#8e44ad']
    axes[2].bar(models, r2_scores, color=colors[:len(models)], alpha=0.85)
    axes[2].set_title('Modellvergleich: R²-Bestimmtheitsmaß (Test-Set)')
    axes[2].set_ylabel('R² Score (höher = besser)')
    axes[2].set_ylim(min(0, min(r2_scores) - 0.05), 1.0)
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.6)
    for i, r2 in enumerate(r2_scores):
        axes[2].text(i, r2 + 0.02, f"{r2:.3f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[INFO] Regressions-Diagramm gespeichert unter: {output_path.resolve()}")

def main():
    print("=" * 70)
    print("NOTEN-REGRESSION: BASELINES (RIDGE, SVR, RF) VS. KERAS MLP REGRESSOR")
    print("=" * 70)
    
    data_path = Path('output_dl/agg_abschluesse.csv')
    if not data_path.exists():
        data_path = Path('../output_dl/agg_abschluesse.csv')
    if not data_path.exists():
        print(f"Fehler: Datei {data_path} nicht gefunden!")
        return
        
    # Preprocessing
    # Wählen: 'graduates_only' (Standard für echten Notenvergleich) oder 'impute_5.0'
    impute_strategy = "graduates_only"
    X_df, y, feature_cols, preprocessor = load_and_preprocess_regression_data(data_path, impute_strategy=impute_strategy)
    
    # 3-Wege Split (70% Train, 15% Val, 15% Test)
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(X_df, y, test_size=0.30, random_state=42)
    X_val_df, X_test_df, y_val, y_test = train_test_split(X_temp_df, y_temp, test_size=0.50, random_state=42)
    
    # Preprocessing anwenden (Fit nur auf Train)
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)
    
    print(f"\nDatensatz-Aufteilung:")
    print(f"  - Training Set:   {X_train.shape[0]} Muster (70%)")
    print(f"  - Validation Set: {X_val.shape[0]} Muster (15%)")
    print(f"  - Test Set:       {X_test.shape[0]} Muster (15%)")
    
    val_metrics = {}
    test_metrics = {}
    test_predictions = {}
    
    # =========================================================================
    # 1. BASELINE 1: LINEARE REGRESSION (RIDGE)
    # =========================================================================
    print("\n" + "-" * 50)
    print("1. TRAINIERE LINEARE REGRESSION (RIDGE)")
    print("-" * 50)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    val_metrics["Linear (Ridge)"] = evaluate_regressor("Linear (Ridge) [Val]", y_val, ridge_model.predict(X_val))
    test_predictions["Linear (Ridge)"] = ridge_model.predict(X_test)
    
    # =========================================================================
    # 2. BASELINE 2: SUPPORT VECTOR REGRESSION (SVR)
    # =========================================================================
    print("\n" + "-" * 50)
    print("2. TRAINIERE SUPPORT VECTOR REGRESSION (SVR)")
    print("-" * 50)
    svr_model = SVR(kernel='rbf', C=1.0)
    svr_model.fit(X_train, y_train)
    val_metrics["SVR"] = evaluate_regressor("SVR [Val]", y_val, svr_model.predict(X_val))
    test_predictions["SVR"] = svr_model.predict(X_test)
    
    # =========================================================================
    # 3. BASELINE 3: RANDOM FOREST REGRESSOR
    # =========================================================================
    print("\n" + "-" * 50)
    print("3. TRAINIERE RANDOM FOREST REGRESSOR")
    print("-" * 50)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    val_metrics["Random Forest"] = evaluate_regressor("Random Forest [Val]", y_val, rf_model.predict(X_val))
    test_predictions["Random Forest"] = rf_model.predict(X_test)
    
    # =========================================================================
    # 4. KERAS MLP REGRESSOR (DEEP LEARNING)
    # =========================================================================
    print("\n" + "-" * 50)
    print("4. TRAINIERE KERAS MLP REGRESSOR")
    print("-" * 50)
    tf.random.set_seed(42)
    mlp_regressor = build_mlp_regressor(input_dim=X_train.shape[1])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    history = mlp_regressor.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    val_preds_mlp = mlp_regressor.predict(X_val, verbose=0).flatten()
    val_metrics["Keras MLP"] = evaluate_regressor("Keras MLP [Val]", y_val, val_preds_mlp)
    
    test_preds_mlp = mlp_regressor.predict(X_test, verbose=0).flatten()
    test_predictions["Keras MLP"] = test_preds_mlp
    
    # =========================================================================
    # 5. ABSCHLIESSENDE EVALUIERUNG & METRIK-LOGGING
    # =========================================================================
    print("\n" + "#" * 70)
    print("  ABSCHLIESSENDE EVALUIERUNG AUF DEM HELD-OUT TEST-SET (15%)")
    print("#" * 70)
    
    base_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    from metrics_logger import plot_parity_plot
    
    print(f"\n{'Modell':<20} | {'RMSE':<8} | {'MAE':<8} | {'R² Score':<8}")
    print("-" * 50)
    for m_name in ["Linear (Ridge)", "SVR", "Random Forest", "Keras MLP"]:
        y_pred = test_predictions[m_name]
        metrics = evaluate_regressor(m_name, y_test, y_pred)
        test_metrics[m_name] = metrics
        
        metrics_dict = {
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "R2": metrics["r2"]
        }
        
        clean_name = m_name.replace(' ', '_').replace('(', '').replace(')', '').lower() + "_regression"
        plot_parity_plot(y_test, y_pred, clean_name, base_dir)
        
        if m_name == "Keras MLP":
            save_keras_model(mlp_regressor, "mlp_baseline_regression", base_dir)
            plot_learning_curve(history.history, "mlp_baseline_regression", base_dir, metric_name='mae')
            
        save_metrics(clean_name, metrics_dict, base_dir)

    output_fig = Path('output_dl/regression_results.png') if Path('output_dl').exists() else Path('../output_dl/regression_results.png')
    plot_regression_results(history, y_test, test_predictions, output_fig)

if __name__ == "__main__":
    main()
