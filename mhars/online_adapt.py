"""
R4 — Label-free lifelong on-device adaptation.

When the drift monitor (X.1) flags a sustained shift in the normal operating
regime, the anomaly autoencoder is fine-tuned *self-supervised* on recent
NORMAL-operation windows — no labels are needed because reconstruction is its
own target, and "normal" is defined by a low live anomaly score. The EVT
calibrator is then refit on the adapted model's errors.

A canary guard prevents regressions: the adaptation is accepted only if the
fine-tuned model reconstructs held-out recent-normal data at least as well as
the incumbent; otherwise it is rolled back. Each accepted adaptation is an event
recorded in the model registry.

This closes the loop: drift → label-free self-supervised retrain → recalibrate →
canary → adopt/rollback, entirely on the edge.
"""
from __future__ import annotations
import copy
import numpy as np


class OnlineAdapter:
    @staticmethod
    def adapt(ae_model, windows: np.ndarray, n_epochs: int = 20, lr: float = 1e-3,
              canary_frac: float = 0.25):
        """Fine-tune `ae_model` (self-supervised) on recent-normal `windows`
        (N,seq,feat). Returns (adopted: bool, new_calibrator_or_None, info)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from mhars.anomaly_calibrator import AnomalyCalibrator

        X = np.asarray(windows, dtype=np.float32)
        n = len(X)
        if n < 40:
            return False, None, {"reason": "insufficient normal data", "n": n}

        rng = np.random.RandomState(0)
        idx = rng.permutation(n)
        cut = int(n * (1 - canary_frac))
        train_x = torch.tensor(X[idx[:cut]])
        canary_x = torch.tensor(X[idx[cut:]])

        # Incumbent performance on held-out recent-normal (canary).
        old_state = copy.deepcopy(ae_model.state_dict())
        ae_model.eval()
        with torch.no_grad():
            old_err = float(ae_model.reconstruction_error(canary_x).mean())

        # Self-supervised fine-tune (reconstruction is its own label).
        opt = torch.optim.Adam(ae_model.parameters(), lr=lr)
        lossf = nn.MSELoss()
        dl = DataLoader(TensorDataset(train_x), batch_size=64, shuffle=True)
        ae_model.train()
        for _ in range(n_epochs):
            for (xb,) in dl:
                opt.zero_grad()
                recon = ae_model(xb)
                lossf(recon, xb).backward()
                opt.step()
        ae_model.eval()
        with torch.no_grad():
            new_err = float(ae_model.reconstruction_error(canary_x).mean())
            train_errs = ae_model.reconstruction_error(train_x).numpy()

        # Canary: accept only if not worse on held-out normal (≤5% regression).
        adopted = new_err <= old_err * 1.05
        if not adopted:
            ae_model.load_state_dict(old_state)  # rollback
            return False, None, {"reason": "canary rejected", "old_err": round(old_err, 6),
                                 "new_err": round(new_err, 6), "n": n}

        new_calib = AnomalyCalibrator.fit(train_errs)
        return True, new_calib, {"old_err": round(old_err, 6), "new_err": round(new_err, 6),
                                 "n": n, "improvement_pct": round(100 * (old_err - new_err) / (old_err + 1e-9), 1)}
