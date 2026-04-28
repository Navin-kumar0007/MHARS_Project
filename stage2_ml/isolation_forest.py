from mhars.config import Config
"""
MHARS — Stage 2: Isolation Forest (Noise Filter)
==================================================
Trains on normal CMAPSS thermal readings. Flags corrupted or
anomalous sensor values before they reach the LSTM or CNN.

Why Isolation Forest over K-Means:
  - No need to specify K in advance
  - Works on streaming data (O(n log n))
  - Handles asymmetric distributions that thermal data produces
  - Validated by Liu et al. (2008) IEEE ICDM
"""

import numpy as np
import os, sys, pickle

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from stage1_simulation.load_cmapss import load_cmapss, preprocess, THERMAL_SENSORS


def build_feature_matrix(df):
    norm_cols = [f"{s}_norm" for s in THERMAL_SENSORS]
    return df[norm_cols].values


def train_isolation_forest(X_normal, contamination=0.03, random_state=Config.SEED):
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_normal)
    return clf


def evaluate(clf, X_normal_test, X_anomaly):
    """
    IsolationForest predict(): +1 = normal, -1 = anomaly.
    We want: clean data → mostly +1, near-failure data → mostly -1.
    """
    preds_normal  = clf.predict(X_normal_test)
    preds_anomaly = clf.predict(X_anomaly)
    false_pos_rate = (preds_normal  == -1).mean()
    detection_rate = (preds_anomaly == -1).mean()
    return false_pos_rate, detection_rate


def get_anomaly_score(clf, X):
    """
    Returns normalized anomaly scores [0, 1].
    0 = definitely normal, 1 = highly anomalous.
    Used by attention fusion downstream.
    """
    raw   = clf.decision_function(X)
    score = -raw   # flip: more negative = more anomalous → higher score
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    return score.astype(np.float32)


def save_model(clf, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Saved → {path}")


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "isolation_forest.pkl")):
    print("\n── Isolation Forest ──────────────────────────────────────────")
    df  = load_cmapss()
    df  = preprocess(df)
    X   = build_feature_matrix(df)

    normal_mask  = df["near_failure"] == 0
    anomaly_mask = df["near_failure"] == 1
    X_normal  = X[normal_mask]
    X_anomaly = X[anomaly_mask]
    X_train, X_test = train_test_split(X_normal, test_size=0.2, random_state=Config.SEED)

    print(f"  Train (normal):  {len(X_train):,}")
    print(f"  Test  (normal):  {len(X_test):,}")
    print(f"  Anomaly samples: {len(X_anomaly):,}")

    clf = train_isolation_forest(X_train)
    fpr, dr = evaluate(clf, X_test, X_anomaly)

    print(f"\n  False positive rate : {fpr*100:.1f}%  (target < 3%)")
    print(f"  Detection rate      : {dr*100:.1f}%")

    status = "✓" if fpr <= 0.03 else "⚠ above target — lower contamination to 0.03"
    print(f"  {status}")

    save_model(clf, model_path)
    return clf, fpr, dr


if __name__ == "__main__":
    run_training()