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