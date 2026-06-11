"""
MHARS Models
=============
Shared PyTorch model definitions used across the framework.
Imported by both the training scripts and the inference pipeline.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ThermalLSTM(nn.Module):
        """12-step LSTM thermal predictor. Validated by Kalla & Smith (2024)."""
        def __init__(self, input_size=1, hidden_size=128, dropout=0.2):
            super().__init__()
            self.lstm   = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.drop   = nn.Dropout(dropout)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out     = self.drop(out[:, -1, :])
            return self.linear(out).squeeze(-1)

    class ThermalAutoencoder(nn.Module):
        """Autoencoder for within-range anomaly detection. Seq_len=12."""
        def __init__(self, seq_len=12):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(seq_len, 6), nn.ReLU(),
                nn.Linear(6, 3),       nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 6),       nn.ReLU(),
                nn.Linear(6, seq_len),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def reconstruction_error(self, x):
            return ((x - self.forward(x)) ** 2).mean(dim=1)

    class VibrationDetector(nn.Module):
        """Autoencoder for vibration-based anomaly detection (5 features)."""
        def __init__(self, n_features=5):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 16), nn.ReLU(),
                nn.Linear(16, 8),          nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 16),          nn.ReLU(),
                nn.Linear(16, n_features),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

        def reconstruction_error(self, x):
            return ((x - self.forward(x)) ** 2).mean(dim=1)

    # ── V2 Enhanced Models (Phase 1 Deep Analysis) ─────────────────────────

    class ThermalLSTMv2(nn.Module):
        """
        Bidirectional LSTM with temporal attention for multivariate thermal prediction.

        Improvements over ThermalLSTM:
          1. Multivariate input (5 thermal sensors instead of 1)
          2. 2-layer BiLSTM for richer temporal representation
          3. Temporal attention mechanism for interpretability (XAI)
          4. Dropout between layers to reduce overfitting

        Architecture:
          Input (batch, seq_len, 5) → BiLSTM(128, 2-layer)
            → Attention over timesteps → Dropout → Linear → prediction
        """
        def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.2,
                     output_horizon=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.output_horizon = output_horizon   # P1.5: direct multi-horizon forecast
            self.lstm = nn.LSTM(
                input_size, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            # Temporal attention: learn which timesteps matter most
            self.attention_weight = nn.Linear(hidden_size * 2, 1)
            self.drop = nn.Dropout(dropout)
            # Head emits `output_horizon` steps ahead (t+1 … t+H).
            self.linear = nn.Linear(hidden_size * 2, output_horizon)

        def _context(self, x):
            out, _ = self.lstm(x)                              # (batch, seq_len, hidden*2)
            attn_scores = self.attention_weight(out)           # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)   # (batch, seq_len, 1)
            context = (out * attn_weights).sum(dim=1)          # (batch, hidden*2)
            return self.drop(context), attn_weights

        def forward(self, x):
            context, _ = self._context(x)
            out = self.linear(context)                         # (batch, H)
            # Backward-compatible: H==1 returns (batch,), else (batch, H).
            return out.squeeze(-1) if self.output_horizon == 1 else out

        def forward_with_attention(self, x):
            """Return prediction AND attention weights for XAI visualization."""
            context, attn_weights = self._context(x)
            out = self.linear(context)
            pred = out.squeeze(-1) if self.output_horizon == 1 else out
            return pred, attn_weights.squeeze(-1)  # (batch,)/(batch,H), (batch, seq_len)

    class ThermalAutoencoderLSTM(nn.Module):
        """
        LSTM-based autoencoder for temporal anomaly detection.

        Improvements over ThermalAutoencoder:
          1. Temporal-aware: catches sequence-level anomalies
          2. Works across ALL machine types (no hardcoded bypass)
          3. Per-sensor reconstruction error for XAI explainability
          4. Larger bottleneck preserves failure signatures

        Architecture:
          Encoder: LSTM(input, 32) → take last hidden
          Decoder: RepeatVector → LSTM(32, 32) → Linear(32, input)
        """
        def __init__(self, input_size=5, hidden_size=32, seq_len=12):
            super().__init__()
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            self.input_size = input_size
            self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.decoder_output = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            # x: (batch, seq_len, input_size)
            _, (h_n, c_n) = self.encoder_lstm(x)
            bottleneck = h_n[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
            dec_out, _ = self.decoder_lstm(bottleneck, (h_n, c_n))
            return self.decoder_output(dec_out)

        def reconstruction_error(self, x):
            """Per-sample mean reconstruction error across timesteps and sensors."""
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=(1, 2))

        def per_sensor_error(self, x):
            """Per-sensor reconstruction error for XAI explainability."""
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)  # (batch, input_size)

    class RULPredictor(nn.Module):
        """
        Dedicated Remaining Useful Life prediction head.
        Replaces linear temperature extrapolation with a learned BiLSTM
        that predicts cycles-to-failure from sensor sequences.

        Architecture:
          Input (batch, seq_len, 5) → BiLSTM(64, 2-layer)
            → last hidden → MLP → ReLU (non-negative RUL)
        """
        def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.ReLU(),  # RUL must be non-negative
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(-1)

    class FaultClassifier(nn.Module):
        """
        P2.4 — Supervised fault-type classifier.

        Replaces the rule-based ``_fingerprint_anomaly`` heuristic with a learned
        model that maps a per-tick feature vector (temperature dynamics + the
        per-model anomaly scores + urgency) to a concrete fault class. Trained on
        the serving distribution with each anomaly type injected on a schedule, so
        it predicts the actual failure mode (heat spike, bearing wear, fan
        blockage, sensor drift, power surge) with a calibrated confidence.
        """
        def __init__(self, n_features=10, n_classes=6, hidden=32, dropout=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, n_classes),
            )

        def forward(self, x):
            return self.net(x)               # logits (batch, n_classes)

        def predict(self, x):
            """Return (class_idx, confidence) for a single normalized feature row."""
            with torch.no_grad():
                probs = torch.softmax(self.forward(x), dim=-1)
                conf, idx = torch.max(probs, dim=-1)
            return int(idx.item()), float(conf.item())

    try:
        from stage2_ml.tft_predictor import TFTPredictor
    except ImportError:
        class TFTPredictor:
            def __init__(self, *args, **kwargs):
                raise ImportError("TFTPredictor not found in stage2_ml.")



else:
    # Stub classes when torch is not installed
    class ThermalLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")

    class ThermalAutoencoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")

    class VibrationDetector:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")

    class ThermalLSTMv2:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")

    class ThermalAutoencoderLSTM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")

    class RULPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")

    class FaultClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")

    class TFTPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required: pip install torch")