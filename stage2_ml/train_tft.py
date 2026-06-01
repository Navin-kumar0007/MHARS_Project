"""
TFT Training Script
===================
Trains the Phase 3 Temporal Fusion Transformer on simulated data.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

try:
    from mhars.config import Config
    from stage2_ml.tft_predictor import TFTPredictor
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from mhars.config import Config
    from stage2_ml.tft_predictor import TFTPredictor

def generate_dummy_data(samples=1000, seq_len=12, num_vars=5):
    X = torch.rand(samples, seq_len, num_vars)
    # Target is some non-linear combination of the last timestep
    y_p50 = X[:, -1, :].mean(dim=-1) + 0.1 * torch.randn(samples)
    y_p10 = y_p50 - 0.1
    y_p90 = y_p50 + 0.1
    y = torch.stack([y_p10, y_p50, y_p90], dim=1)
    return X, y

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds: (batch, num_quantiles)
        target: (batch, num_quantiles) or (batch,) if only point target
        """
        if target.dim() == 1:
            target = target.unsqueeze(-1).expand_as(preds)
            
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target[:, i] - preds[:, i]
            loss = torch.max((q - 1) * err, q * err)
            losses.append(loss)
        return torch.stack(losses, dim=1).mean()

def train():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    X_train, y_train = generate_dummy_data(2000)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = TFTPredictor(
        num_vars=Config.LSTM_INPUT_SIZE_V2,
        d_model=Config.TFT_D_MODEL,
        n_heads=Config.TFT_N_HEADS,
        num_quantiles=Config.TFT_NUM_QUANTILES
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = QuantileLoss()
    
    print("Training TFT Predictor...")
    for epoch in range(5):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds, _, _ = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 - Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), Config.TFT_MODEL)
    print(f"Model saved to {Config.TFT_MODEL}")

if __name__ == "__main__":
    train()
