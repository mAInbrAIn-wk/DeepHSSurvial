"""
DeepSupport PyTorch Transformer Suite
=====================================
Enthält den hochperformanten autoregressiven Prüfungs-Transformer
(Next-Exam Multi-Task Regressor) mit:
- Analytischem SinCos-Positional-Encoding
- FlashAttention / F.scaled_dot_product_attention (PyTorch 2.x)
- Konsequenter LayerNormalization
- Attention-Pooling für ungleich lange Prüfungssequenzen
- Dual-Task-Heads (Notenregression MSE + Bestehens-Klassifikation BCE)
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinCosPositionalEncoding(nn.Module):
    """
    Klassisches Sinus-Cosinus Positional Encoding nach Vaswani et al. (2017).
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Addiert Positions-Encoding zu (N, seq_len, d_model)."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class AttentionPooling(nn.Module):
    """
    Gelerntes Aufmerksamkeits-Pooling über variable Sequenzlängen.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (N, S, d_model)
        mask: (N, S) True wo valide, False wo gepaddet
        """
        scores = self.score_proj(x)  # (N, S, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)  # (N, S, 1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.sum(x * weights, dim=1)  # (N, d_model)
        return pooled


class TransformerEncoderBlock(nn.Module):
    """
    Schlanker, hochoptimierter Transformer-Block mit F.scaled_dot_product_attention,
    Pre-LayerNorm-Architektur und Unterstützung für kausales Masking.
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
        is_causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal
        assert d_model % n_heads == 0, "d_model muss durch n_heads teilbar sein."

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (N, S, d_model)
        key_padding_mask: (N, S) mit True für valide Positionen, False für gepaddete Positionen
        """
        normed = self.norm1(x)
        N, S, D = normed.shape

        qkv = self.qkv_proj(normed).reshape(N, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # jeweils (N, n_heads, S, head_dim)

        attn_mask = None
        if self.is_causal:
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device)).unsqueeze(0).unsqueeze(0)
            if key_padding_mask is not None:
                attn_mask = causal & key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_mask[:, :, :, 0] = True
            else:
                attn_mask = causal
        elif key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask[:, :, :, 0] = True

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(N, S, D)
        x = x + self.dropout(self.out_proj(attn_out))
        x = x + self.ffn(self.norm2(x))
        return x


class PyTorchExamTransformerRegressor(nn.Module):
    """
    Prüfungs-Transformer zur Vorhersage der Abschlussnote (Absolventenkohorte).
    Strukturgleich zu Deep_Exam_Transformer_Regressor aus deep_transformer_regression.py:
    - Pre-LN Encoder mit SinCosPositionalEncoding
    - AttentionPooling über Zeitschritte
    - Regressionskopf mit LayerNorm
    - Default-Modus: gradeblind
    """
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(feat_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = SinCosPositionalEncoding(d_model=d_model, max_len=max_seq_len + 10)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, is_causal=False)
            for _ in range(n_layers)
        ])

        self.pooler = AttentionPooling(d_model=d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        sequence: (N, S, feat_dim)
        mask: (N, S) True für valide Positionen
        Rückgabe: (N,) vorhergesagte Abschlussnote
        """
        h = self.input_proj(sequence)
        h = self.input_norm(h)
        h = self.pos_encoder(h)

        for block in self.blocks:
            h = block(h, key_padding_mask=mask)

        pooled = self.pooler(h, mask=mask)
        out = self.head(pooled).squeeze(-1)
        return out

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["sequence"]
        mask = batch.get("mask")
        y_true = batch["target"]

        pred = self.forward(seq, mask=mask)
        valid = ~torch.isnan(y_true)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq.device, requires_grad=True)
        return F.mse_loss(pred[valid], y_true[valid])


class PyTorchCausalExamTransformerSurvival(nn.Module):
    """
    Kausales Prüfungs-Transformer Survival-Modell.
    Strukturgleich zu Deep_Exam_Causal_Transformer_Survival aus deep_transformer_regression.py:
    - Kausales Attention-Masking (kein Blick in spätere Prüfungen)
    - KEIN POOLING! TimeDistributed Hazard-Head an jedem Schritt
    - Evaluierung über alle aktiven Prüfungsschritte (pi_0 ~ 1.71%)
    - Kumulatives Studierenden-Überleben S(K) = prod_{k=1}^K (1 - h_k)
    """
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(feat_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = SinCosPositionalEncoding(d_model=d_model, max_len=max_seq_len + 10)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, is_causal=True)
            for _ in range(n_layers)
        ])

        # TimeDistributed Hazard Head (an jedem Zeitschritt t)
        self.time_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_logits(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        sequence: (N, S, feat_dim)
        mask: (N, S) True für valide Positionen
        Rückgabe: (N, S) unnormalisierte Logits für bedingten Hazard
        """
        h = self.input_proj(sequence)
        h = self.input_norm(h)
        h = self.pos_encoder(h)

        for block in self.blocks:
            h = block(h, key_padding_mask=mask)

        logits = self.time_head(h).squeeze(-1)  # (N, S)
        return logits

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Rückgabe: (N, S) bedingte Hazard-Wahrscheinlichkeiten h(t) in (0, 1)
        """
        logits = self.forward_logits(sequence, mask=mask)
        return torch.sigmoid(logits)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Masked BCE über alle aktiven, unpadded Prüfungsschritte."""
        seq = batch["sequence"]
        mask = batch.get("mask")
        target = batch["target"]  # (N, S) oder (N, S, 1)
        if target.dim() == 3:
            target = target.squeeze(-1)

        logits = self.forward_logits(seq, mask=mask)

        valid = (target != -99.0)
        if mask is not None:
            valid = valid & mask

        if valid.sum() == 0:
            return torch.tensor(0.0, device=seq.device, requires_grad=True)

        return F.binary_cross_entropy_with_logits(logits[valid], target[valid])

    @torch.no_grad()
    def predict_step_hazard(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Liefert Schritt-Hazards h(k) als NumPy-Array (N, S)."""
        self.eval()
        hazards = self.forward(sequence, mask=mask)
        return hazards.cpu().numpy()

    @torch.no_grad()
    def predict_student_survival(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet für jeden Studierenden die kumulative Überlebenswahrscheinlichkeit
        S_i(K) = prod_{k=1}^K (1 - h_k) und das Abbruchrisiko 1 - S_i(K).
        Rückgabe: (surv_probs, risk_probs) jeweils (N,)
        """
        self.eval()
        hazards = self.forward(sequence, mask=mask).cpu().numpy()
        mask_np = mask.cpu().numpy()

        N = hazards.shape[0]
        surv_probs = np.ones(N, dtype=np.float32)

        for i in range(N):
            s_len = int(np.sum(mask_np[i]))
            if s_len > 0:
                h_k = np.clip(hazards[i, :s_len], 1e-6, 1.0 - 1e-6)
                surv_probs[i] = np.prod(1.0 - h_k)

        risk_probs = 1.0 - surv_probs
        return surv_probs, risk_probs


class PyTorchNextExamTransformer(nn.Module):
    """
    Autoregressiver Next-Exam Transformer mit Multi-Task Heads:
    1. Notenvorhersage (kontinuierlich, 1.0 - 5.0) via MSE Loss
    2. Bestehenswahrscheinlichkeit (binär, 0 oder 1) via BCEWithLogits Loss
    """
    def __init__(
        self,
        feat_dim: int,
        context_dim: int = 0,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        loss_weight_pass: float = 1.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.context_dim = context_dim
        self.d_model = d_model
        self.loss_weight_pass = loss_weight_pass

        self.input_proj = nn.Linear(feat_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = SinCosPositionalEncoding(d_model=d_model, max_len=max_seq_len + 10)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, is_causal=False)
            for _ in range(n_layers)
        ])

        if context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.Linear(context_dim, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
            )
            pooled_dim = d_model + (d_model // 2)
        else:
            self.context_proj = None
            pooled_dim = d_model

        self.pooler = AttentionPooling(d_model=d_model)

        self.shared_dense = nn.Sequential(
            nn.Linear(pooled_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_note = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.head_pass = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(sequence)
        h = self.input_norm(h)
        h = self.pos_encoder(h)

        for block in self.blocks:
            h = block(h, key_padding_mask=mask)

        pooled_seq = self.pooler(h, mask=mask)

        if self.context_proj is not None and context is not None:
            ctx_h = self.context_proj(context)
            features = torch.cat([pooled_seq, ctx_h], dim=-1)
        else:
            features = pooled_seq

        shared = self.shared_dense(features)
        pred_note = self.head_note(shared).squeeze(-1)
        logits_pass = self.head_pass(shared).squeeze(-1)

        return pred_note, logits_pass

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = batch["sequence"]
        mask = batch.get("mask")
        ctx = batch.get("context")
        y_note = batch.get("target_note")
        y_pass = batch.get("target_pass")

        pred_note, logits_pass = self.forward(seq, mask=mask, context=ctx)

        loss_note = torch.tensor(0.0, device=seq.device)
        if y_note is not None:
            valid_note = ~torch.isnan(y_note)
            if valid_note.sum() > 0:
                loss_note = F.mse_loss(pred_note[valid_note], y_note[valid_note])

        loss_pass = torch.tensor(0.0, device=seq.device)
        if y_pass is not None:
            loss_pass = F.binary_cross_entropy_with_logits(logits_pass, y_pass)

        return loss_note + self.loss_weight_pass * loss_pass
