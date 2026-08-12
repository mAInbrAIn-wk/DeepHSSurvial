import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
    TimeDistributed, Masking, Add
)
import tensorflow.keras.backend as K

from recurrent_survival_model import build_recurrent_survival_dataset, PADDING_VALUE
from transformer_survival_model import PositionalEncoding
from extended_cox_delta import build_delta_panel

def build_deep_causal_transformer_model(sequence_length, feature_dim, d_model=64, num_heads=4, num_blocks=2):
    """
    Größerer Deep Causal Transformer mit 2 gestapelten Attention-Blöcken
    und tieferem Feed-Forward-Netzwerk.
    """
    inputs = Input(shape=(sequence_length, feature_dim))
    
    # 1. Masking Layer
    masked_inputs = Masking(mask_value=PADDING_VALUE)(inputs)
    
    # 2. Linear Projection auf d_model (Trainierbare Matrix W: 8 -> 64)
    x = TimeDistributed(Dense(d_model, activation='relu'))(masked_inputs)
    
    # 3. Positional Encoding
    x = PositionalEncoding(sequence_length, d_model)(x)
    
    # 4. Gestapelte Causal Transformer Blöcke (num_blocks = 2)
    for _ in range(num_blocks):
        # Attention Block
        attn_out = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1
        )(query=x, value=x, key=x, use_causal_mask=True)
        
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        
        # Tieferes Feed-Forward Block (128 -> 64)
        ff_out = TimeDistributed(Dense(128, activation='relu'))(x)
        ff_out = TimeDistributed(Dropout(0.1))(ff_out)
        ff_out = TimeDistributed(Dense(d_model, activation='relu'))(ff_out)
        
        x = Add()([x, ff_out])
        x = LayerNormalization()(x)

    # Output Head
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    data_dir = Path(r"c:\GitHub_public\Abschlussprojekt\output_dl")
    output_dir = data_dir / "analysis"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("================================================================================")
    print("DEEP TRANSFORMER-DML BENCHMARK: ENLARGED ARCHITECTURE (2 BLOCKS, d_model=64)")
    print("================================================================================")

    # 1. Lade 3D-Zeitreihen-Panel
    studis, X_3d, y_3d, studi_events = build_recurrent_survival_dataset(data_dir, max_semesters=16, blind=False)
    n_samples, sequence_length, feature_dim = X_3d.shape

    # 2. Bauen & Pretrainen des größeren Deep Causal Transformers
    d_model = 64
    deep_transformer = build_deep_causal_transformer_model(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        d_model=d_model,
        num_heads=4,
        num_blocks=2
    )

    def masked_bce(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, PADDING_VALUE), tf.float32)
        y_true_clean = tf.maximum(y_true, 0.0)
        bce = K.binary_crossentropy(y_true_clean, y_pred)
        return tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-7)

    deep_transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=masked_bce)

    print("\n[Stage 1] Pretraining Deep Causal Transformer Encoder (2 Blocks, d_model=64)...")
    deep_transformer.fit(X_3d, y_3d, epochs=20, batch_size=64, verbose=0)

    # Extrahiere Hidden Features der vorletzten Layer Norm Schicht
    feature_extractor = Model(inputs=deep_transformer.input, outputs=deep_transformer.layers[-3].output)
    hidden_features = feature_extractor.predict(X_3d, verbose=0) # (N, T, 64)

    # 3. Flachklopfen & Match mit 2D DML Panel
    df_panel = build_delta_panel(data_dir)
    
    flat_rows = []
    for i in range(n_samples):
        s_id = studis[i]
        for t in range(sequence_length):
            if y_3d[i, t, 0] != PADDING_VALUE:
                h_vec = hidden_features[i, t, :]
                event = y_3d[i, t, 0]
                flat_rows.append({
                    "studierenden_id": s_id,
                    "t_stop": t + 1,
                    "event": event,
                    **{f"h_{k}": h_vec[k] for k in range(d_model)}
                })
                
    df_flat = pd.DataFrame(flat_rows)
    df_merged = df_flat.merge(df_panel[["studierenden_id", "t_stop", "fach_supp_active", "uebf_supp_active", "psych_supp_active"]], on=["studierenden_id", "t_stop"], how="left").fillna(0)

    h_cols = [f"h_{k}" for k in range(d_model)]
    X_h = df_merged[h_cols].values
    Y = df_merged["event"].values
    base_event_rate = float(np.mean(Y))

    # 4. Double Machine Learning Orthogonalisierung für alle 3 Support-Typen
    print("\n[Stage 2] Double Machine Learning (Orthogonalisierung via Deep Transformer Embeddings)...")
    
    treatment_cols = {
        "fachlich": "fach_supp_active",
        "ueberfachlich": "uebf_supp_active",
        "psychosozial": "psych_supp_active"
    }

    results = {
        "empirische_event_rate": base_event_rate,
        "treatments": {}
    }

    print("\n================================================================================")
    print("DEEP TRANSFORMER-DML ERGEBNISSE (ALLE SUPPORT-TYPEN):")
    print("================================================================================")

    for typ_name, col_name in treatment_cols.items():
        A = df_merged[col_name].values
        
        prop_model = LogisticRegression(max_iter=1000)
        prop_model.fit(X_h, A)
        p_hat = prop_model.predict_proba(X_h)[:, 1]
        p_hat = np.clip(p_hat, 0.01, 0.99)
        A_res = A - p_hat

        out_model = Ridge(alpha=1.0)
        out_model.fit(X_h, Y)
        y_hat = out_model.predict(X_h)
        Y_res = Y - y_hat

        effect_model = Ridge(alpha=0.001)
        effect_model.fit(A_res.reshape(-1, 1), Y_res)
        beta = float(effect_model.coef_[0])

        relative_risk = float((base_event_rate + beta) / base_event_rate if base_event_rate > 0 else 1.0)

        results["treatments"][typ_name] = {
            "beta": beta,
            "relative_risk": relative_risk,
            "treatment_rate": float(np.mean(A))
        }

        print(f"--- {typ_name.upper()} SUPPORT ({col_name}) ---")
        print(f"  Geschätzter Kausaler Effekt (Beta): {beta:.6f}")
        print(f"  Geschätztes Relatives Risiko (RR): {relative_risk:.4f}")

    # Rückwärtskompatible Felder für fachlichen Support
    results["deep_transformer_dml_rr"] = results["treatments"]["fachlich"]["relative_risk"]
    results["deep_transformer_beta"] = results["treatments"]["fachlich"]["beta"]
    print("================================================================================")

    with open(output_dir / "deep_transformer_dml_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
