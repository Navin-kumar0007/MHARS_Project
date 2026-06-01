"""
MHARS — Phase 3: Temporal Fusion Transformer (TFT)
=================================================
A lightweight, custom Temporal Fusion Transformer for interpretable,
multivariate time-series forecasting.

Features:
- Gated Residual Networks (GRN) for variable selection
- Multi-head temporal self-attention
- Quantile output heads (10th, 50th, 90th percentiles) for built-in confidence intervals

Usage:
    from stage2_ml.tft_predictor import TFTPredictor
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model * 2)

    def forward(self, x):
        x = self.linear(x)
        val, gate = x.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        skip = self.skip(x)
        x = F.elu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.norm(skip + x)


class VariableSelectionNetwork(nn.Module):
    """Calculates variable selection weights and processes inputs via GRNs."""
    def __init__(self, num_vars, d_model, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.flatten_grn = GRN(num_vars * d_model, d_model, num_vars)
        self.single_grns = nn.ModuleList([GRN(d_model, d_model, d_model, dropout) for _ in range(num_vars)])

    def forward(self, x):
        # x: (batch, seq_len, num_vars, d_model)
        batch, seq, _, _ = x.shape
        flat = x.view(batch, seq, -1)
        
        # Variable selection weights
        weights = self.flatten_grn(flat)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1)  # (batch, seq, num_vars, 1)

        # Apply single GRNs
        processed = torch.stack([grn(x[:, :, i, :]) for i, grn in enumerate(self.single_grns)], dim=2)
        
        # Combine
        weighted = (processed * weights).sum(dim=2)  # (batch, seq, d_model)
        return weighted, weights.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TFTPredictor(nn.Module):
    """
    Lightweight Temporal Fusion Transformer.
    Predicts a specific horizon with multiple quantiles (e.g., 0.1, 0.5, 0.9).
    """
    def __init__(self, num_vars=5, d_model=32, n_heads=4, num_quantiles=3, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        
        # Input embeddings for each variable
        self.var_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_vars)])
        
        # Variable Selection
        self.vsn = VariableSelectionNetwork(num_vars, d_model, dropout)
        
        # Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Post-attention processing
        self.post_attn_grn = GRN(d_model, d_model, d_model, dropout)
        
        # Output quantiles (e.g., p10, p50, p90)
        self.quantile_head = nn.Linear(d_model, num_quantiles)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, num_vars)
        Returns:
            quantiles: (batch_size, num_quantiles)
            var_weights: (batch_size, seq_len, num_vars) - feature importance
            attn_weights: (batch_size, seq_len, seq_len) - temporal importance
        """
        batch, seq, _ = x.shape
        
        # Embed variables
        embedded = torch.stack([
            emb(x[:, :, i].unsqueeze(-1)) for i, emb in enumerate(self.var_embeddings)
        ], dim=2)  # (batch, seq, num_vars, d_model)
        
        # Variable Selection
        vsn_out, var_weights = self.vsn(embedded)  # (batch, seq, d_model)
        
        # Add positional encoding
        vsn_out = self.pos_encoder(vsn_out)
        
        # Self-Attention
        attn_out, attn_weights = self.attention(vsn_out, vsn_out, vsn_out)
        
        # Post-processing (we only care about the last sequence step for prediction)
        final_state = self.post_attn_grn(attn_out[:, -1, :])  # (batch, d_model)
        
        # Quantile predictions
        quantiles = self.quantile_head(final_state)  # (batch, num_quantiles)
        
        return quantiles, var_weights, attn_weights

    def predict_point(self, x):
        """Helper to get just the median (50th percentile) prediction."""
        quantiles, _, _ = self.forward(x)
        return quantiles[:, 1]  # Assuming num_quantiles=3, index 1 is median (p50)
