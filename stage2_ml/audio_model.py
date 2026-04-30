"""
MHARS — Stage 2: Audio Anomaly Detector
=============================================
A third modality for the attention fusion layer. Provides an
audio-based anomaly score that captures acoustic signatures
of machine degradation (e.g., grinding, whining).

Architecture:
  Audio MFCC features → MLP (13 → 8 → 13) → reconstruction error → anomaly score
  Trained on normal acoustic data only.
"""

import numpy as np
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AudioDetector(nn.Module):
    def __init__(self, n_features=13):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)


def generate_audio_data(n_units=100, seed=42):
    rng = np.random.default_rng(seed)
    features_normal = []
    features_all = []

    for unit_id in range(n_units):
        max_cycle = rng.integers(150, 350)
        baseline_mfcc = rng.uniform(-10, 10, size=13)

        for cycle in range(max_cycle):
            rul = max_cycle - cycle
            frac = 1 - (rul / max_cycle)

            # Degradation logic: as failure approaches, high-frequency MFCCs shift
            mfcc = baseline_mfcc.copy()
            mfcc[0] += frac * 5.0 + rng.normal(0, 0.5)  # Overall energy
            mfcc[1:5] -= frac * 2.0 + rng.normal(0, 0.2) # Low-mid freq shifts
            mfcc[8:] += frac * 3.0 + rng.normal(0, 0.3)  # High freq whine/grinding
            
            features_all.append(mfcc)
            if frac < 0.3:
                features_normal.append(mfcc)

    return {
        "features_normal": np.array(features_normal, dtype=np.float32),
        "features_all": np.array(features_all, dtype=np.float32)
    }

def run_training():
    print("\n── Training Audio Detector ─────────────────────────────────")
    data = generate_audio_data()
    X_train = torch.tensor(data["features_normal"])
    
    # Scale to [0, 1] range based on training data
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-6
    X_train = (X_train - mean) / std

    dataset = TensorDataset(X_train, X_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AudioDetector(n_features=13)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

    # Calculate threshold (95th percentile)
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(X_train).numpy()
    threshold = float(np.percentile(errors, 95))
    
    print(f"  Training complete. Threshold (95th perc): {threshold:.4f}")
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, "audio_model.pt"))
    with open(os.path.join(model_dir, "audio_model_meta.json"), "w") as f:
        json.dump({"threshold": threshold, "mean": mean.numpy().tolist(), "std": std.numpy().tolist()}, f)

if __name__ == "__main__":
    run_training()
