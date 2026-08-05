"""
Training Script: Status-Vorhersage mit 3-Wege-Split (Train/Val/Test) & Lernkurven
==================================================================================
Vorhersage des Studienabschluss-Status ('status') auf Basis von 'agg_abschluesse.csv'.

Features:
- Detaillierter 3-Wege Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
- Evaluierung aller Modelle auf Validation-Set (Modellauswahl) und abschließend auf Test-Set
- Generierung von Classification Reports & Confusion Matrizen für ALLE 4 Modelle:
  1. Naive Bayes Classifier
  2. Random Forest Classifier
  3. Support Vector Machine (SVC)
  4. Keras Multi-Layer Perceptron (MLP)
- Speichern der Lernkurven (Loss & Accuracy) als 'output_dl/learning_curves.png'
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc, 
    average_precision_score, roc_auc_score
)
from metrics_logger import save_metrics, plot_roc_curve, plot_pr_curve, plot_learning_curve, save_keras_model, plot_confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# FEATURE-MASKEN & KONFIGURATION
# =============================================================================

LEAKAGE_COLUMNS = [
    "studierenden_id",
    "status",                  # Target
    "abschlussnote",           # Nur bei Abschluss vorhanden
    "bachelorarbeitsnote",     # Nur bei Abschluss vorhanden
    "studiendauer_semester",   # Verrät exakte Semesterdauer
    "abschluss_semester_id",   # Verrät Endsemester
    "anomalie_typ",            # Enthält explizite Abbruch-Labeling
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

def load_and_preprocess_data(data_path: Path, target_col: str = "status", binary_target: bool = False, blind: bool = False):
    print(f"Lade Daten aus {data_path} (blind={blind}) ...")
    df = pd.read_csv(data_path)
    
    if binary_target:
        df['target'] = (df[target_col] == 'abgeschlossen').astype(int)
        class_names = ['nicht_abgeschlossen', 'abgeschlossen']
    else:
        le = LabelEncoder()
        df['target'] = le.fit_transform(df[target_col].astype(str))
        class_names = [str(c) for c in le.classes_]
        
    y = df['target'].values
    
    # Automatische Erkennung aller hidden_* Ground-Truth-Spalten & Per-Attempt Support Spalten
    hidden_and_attempt_cols = [col for col in df.columns if col.startswith('hidden_') or 'Support_' in col or 'Versuche' in col]
    exclude_set = set(LEAKAGE_COLUMNS + STRICT_MASKED_COLUMNS + hidden_and_attempt_cols)
    if blind:
        note_cols = [c for c in df.columns if 'note' in c.lower() or 'gpa' in c.lower()]
        exclude_set.update(note_cols)
        
    feature_cols = [col for col in df.columns if col not in exclude_set and col != 'target']
    
    print(f"Nutze {len(feature_cols)} ehrliche Features (Demographie + Sem 1-2 Landmark, blind={blind}) für die Vorhersage.")
    print("Verwendete Features:", feature_cols)
    
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
    
    return X_df, y, class_names, feature_cols, preprocessor

def build_mlp_model(input_dim: int, num_classes: int):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        LayerNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        LayerNormalization(),
        Dropout(0.2),
        Dense(1 if num_classes == 2 else num_classes, activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    
    loss_fn = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return model

def print_eval_results(model_name: str, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n" + "=" * 60)
    print(f" MODELL: {model_name} (Test Accuracy: {acc * 100:.2f}%)")
    print("=" * 60)
    
    rep = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n[1] Classification Report:")
    print(rep)
    
    print("[2] Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in class_names], columns=[f"Pred: {c}" for c in class_names])
    print(cm_df.to_string())
    print("-" * 60)
    return acc, rep

def plot_learning_curves(history, val_scores_dict, test_scores_dict, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Keras Loss Curve
    axes[0].plot(history.history['loss'], label='Train Loss', color='#2980b9', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    axes[0].set_title('Keras MLP: Loss (Train vs. Validation)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Keras Accuracy Curve
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='#27ae60', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='#f39c12', linewidth=2)
    axes[1].set_title('Keras MLP: Accuracy (Train vs. Validation)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Model Comparison Bar Chart (Val vs. Test Accuracy)
    models = list(val_scores_dict.keys())
    x = np.arange(len(models))
    width = 0.35
    
    val_accs = [val_scores_dict[m] * 100 for m in models]
    test_accs = [test_scores_dict[m] * 100 for m in models]
    
    axes[2].bar(x - width/2, val_accs, width, label='Validation Acc (%)', color='#3498db')
    axes[2].bar(x + width/2, test_accs, width, label='Test Acc (%)', color='#2ecc71')
    axes[2].set_title('Modellvergleich: Validation vs. Test Accuracy')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15, ha='right')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_ylim(70, 100)
    axes[2].legend()
    axes[2].grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n[INFO] Lernkurven-Diagramm erfolgreich gespeichert unter: {output_path.resolve()}")

def main(blind: bool = False):
    suffix = "_blind" if blind else ""
    print("=" * 70)
    print(f"STATUS-VORHERSAGE: 3-WEGE SPLIT (TRAIN / VAL / TEST, blind={blind})")
    print("=" * 70)
    
    data_path = Path('output_dl/agg_abschluesse.csv')
    if not data_path.exists():
        data_path = Path('../output_dl/agg_abschluesse.csv')
    if not data_path.exists():
        print(f"Fehler: Datei {data_path} nicht gefunden!")
        return
        
    X_df, y, class_names, feature_cols, preprocessor = load_and_preprocess_data(data_path, binary_target=False, blind=blind)
    num_classes = len(class_names)
    
    # 3-Wege Stratifizierter Split (70% Train, 15% Val, 15% Test)
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # Preprocessor wird NUR auf den Trainingsdaten gefittet (kein Data-Leakage!)
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)
    
    print(f"\nDatensatz-Aufteilung:")
    print(f"  - Training Set:   {X_train.shape[0]} Muster (70%)")
    print(f"  - Validation Set: {X_val.shape[0]} Muster (15%)")
    print(f"  - Test Set:       {X_test.shape[0]} Muster (15%)")
    
    val_scores = {}
    test_scores = {}
    test_predictions = {}
    
    # 1. NAIVE BAYES BASELINE
    print("\n" + "-" * 50)
    print("1. TRAINIERE NAIVE BAYES BASELINE")
    print("-" * 50)
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    val_preds_nb = nb_model.predict(X_val)
    val_scores["Naive Bayes"] = accuracy_score(y_val, val_preds_nb)
    test_predictions["Naive Bayes"] = nb_model.predict(X_test)
    print(f"Validation Accuracy: {val_scores['Naive Bayes'] * 100:.2f}%")
    
    # 2. RANDOM FOREST CLASSIFIER
    print("\n" + "-" * 50)
    print("2. TRAINIERE RANDOM FOREST CLASSIFIER")
    print("-" * 50)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    val_preds_rf = rf_model.predict(X_val)
    val_scores["Random Forest"] = accuracy_score(y_val, val_preds_rf)
    test_predictions["Random Forest"] = rf_model.predict(X_test)
    print(f"Validation Accuracy: {val_scores['Random Forest'] * 100:.2f}%")
    
    # 3. SUPPORT VECTOR MACHINE (SVC)
    print("\n" + "-" * 50)
    print("3. TRAINIERE SUPPORT VECTOR MACHINE (SVC)")
    print("-" * 50)
    svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42, max_iter=3000)
    svm_model.fit(X_train, y_train)
    val_preds_svm = svm_model.predict(X_val)
    val_scores["SVM"] = accuracy_score(y_val, val_preds_svm)
    test_predictions["SVM"] = svm_model.predict(X_test)
    print(f"Validation Accuracy: {val_scores['SVM'] * 100:.2f}%")
    
    # 4. KERAS MLP (DEEP LEARNING)
    print("\n" + "-" * 50)
    print("4. TRAINIERE KERAS MULTI-LAYER PERCEPTRON (MLP)")
    print("-" * 50)
    tf.random.set_seed(42)
    mlp_model = build_mlp_model(input_dim=X_train.shape[1], num_classes=num_classes)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = mlp_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )
    
    val_preds_proba = mlp_model.predict(X_val, verbose=0)
    val_preds_mlp = np.argmax(val_preds_proba, axis=1) if num_classes > 2 else (val_preds_proba > 0.5).astype(int).flatten()
    val_scores["Keras MLP"] = accuracy_score(y_val, val_preds_mlp)
    
    test_preds_proba = mlp_model.predict(X_test, verbose=0)
    test_predictions["Keras MLP"] = np.argmax(test_preds_proba, axis=1) if num_classes > 2 else (test_preds_proba > 0.5).astype(int).flatten()
    print(f"Validation Accuracy: {val_scores['Keras MLP'] * 100:.2f}%")
    
    # 5. ABSCHLIESSENDE EVALUIERUNG & METRIK-LOGGING
    print("\n" + "#" * 70)
    print("  ABSCHLIESSENDE EVALUIERUNG AUF DEM HELD-OUT TEST-SET (15% = 1.500 Muster)")
    print("#" * 70)
    
    base_dir = Path('output_dl') if Path('output_dl').exists() else Path('../output_dl')
    
    for m_name in ["Naive Bayes", "Random Forest", "SVM", "Keras MLP"]:
        y_pred = test_predictions[m_name]
        test_acc, report_str = print_eval_results(m_name, y_test, y_pred, class_names)
        test_scores[m_name] = test_acc
        
        # Calculate standard metrics
        metrics_dict = {
            "Accuracy": test_acc,
            "Precision_Macro": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "Recall_Macro": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "F1_Macro": f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
        
        # Calculate ROC and PR for Keras MLP
        if m_name == "Keras MLP":
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, test_preds_proba)
                metrics_dict["ROC-AUC"] = auc(fpr, tpr)
                metrics_dict["PR-AUC"] = average_precision_score(y_test, test_preds_proba)
                
                plot_roc_curve(y_test, test_preds_proba, f"MLP_Baseline_Classification{suffix}", base_dir)
                plot_pr_curve(y_test, test_preds_proba, f"MLP_Baseline_Classification{suffix}", base_dir)
            else:
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=range(num_classes))
                try:
                    metrics_dict["ROC-AUC_Macro"] = roc_auc_score(y_test_bin, test_preds_proba, average='macro', multi_class='ovr')
                    metrics_dict["PR-AUC_Macro"] = average_precision_score(y_test_bin, test_preds_proba, average='macro')
                except ValueError:
                    pass
            
            save_keras_model(mlp_model, f"mlp_baseline_classification{suffix}", base_dir)
            
        save_name = f"{m_name.replace(' ', '_').lower()}_baseline{suffix}"
        plot_confusion_matrix(y_test, y_pred, save_name, base_dir)
        save_metrics(save_name, metrics_dict, base_dir, report_str=report_str)

    # 6. GESAMTZUSAMMENFASSUNG & SPEICHERN DER LERNKURVEN
    print("\n" + "=" * 70)
    print(" MODELLVERGLEICH (VALIDATION VS. TEST ACCURACY)")
    print("=" * 70)
    print(f"{'Modell':<25} | {'Validation Acc':<15} | {'Test Acc':<15}")
    print("-" * 60)
    for m in ["Naive Bayes", "Random Forest", "SVM", "Keras MLP"]:
        print(f"{m:<25} | {val_scores[m]*100:>13.2f}% | {test_scores[m]*100:>13.2f}%")
    print("=" * 70)

    output_fig = Path('output_dl/learning_curves.png') if Path('output_dl').exists() else Path('../output_dl/learning_curves.png')
    plot_learning_curves(history, val_scores, test_scores, output_fig)
    plot_learning_curve(history.history, "mlp_baseline_classification", base_dir, metric_name='accuracy')

if __name__ == "__main__":
    main()
