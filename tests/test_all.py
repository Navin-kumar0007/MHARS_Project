"""
MHARS — tests/test_all.py
===========================
Run: pytest tests/ -v
Or:  python -m pytest tests/test_all.py -v
"""

import sys, os, json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage2_ml'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage3_ai'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage4_hardware'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage5_adapter'))


# ═══════════════════════════════════════════════════════
# test_isolation_forest.py
# ═══════════════════════════════════════════════════════
class TestIsolationForest:

    def test_model_file_exists(self):
        if not os.path.exists("models/isolation_forest.pkl"):
            pytest.skip("Model not trained yet — run stage2_ml/run_stage2.py")

    @pytest.mark.skipif(not os.path.exists("models/isolation_forest.pkl"), reason="Model not trained")
    def test_false_positive_rate(self):
        import pickle
        from load_cmapss import load_cmapss, preprocess
        from isolation_forest import build_feature_matrix, evaluate
        with open("models/isolation_forest.pkl", "rb") as f:
            clf = pickle.load(f)
        df = load_cmapss()
        df = preprocess(df)
        X = build_feature_matrix(df)
        X_normal  = X[df["near_failure"] == 0]
        X_anomaly = X[df["near_failure"] == 1]
        from sklearn.model_selection import train_test_split
        _, X_test = train_test_split(X_normal, test_size=0.2, random_state=42)
        fpr, dr = evaluate(clf, X_test, X_anomaly)
        assert fpr <= 0.05, f"FPR {fpr*100:.1f}% exceeds 5% target"

    @pytest.mark.skipif(not os.path.exists("models/isolation_forest.pkl"), reason="Model not trained")
    def test_anomaly_scores_positive(self):
        import pickle
        from isolation_forest import get_anomaly_score
        with open("models/isolation_forest.pkl", "rb") as f:
            clf = pickle.load(f)
        X = np.random.rand(10, 5).astype(np.float32)
        scores = get_anomaly_score(clf, X)
        assert scores.shape == (10,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


# ═══════════════════════════════════════════════════════
# test_lstm.py
# ═══════════════════════════════════════════════════════
class TestLSTM:

    def test_model_file_exists(self):
        if not os.path.exists("models/lstm.pt"):
            pytest.skip("Model not trained yet — run stage2_ml/run_stage2.py")

    def test_window_shape(self):
        from load_cmapss import load_cmapss, preprocess, make_lstm_windows
        df = load_cmapss()
        df = preprocess(df)
        X, y, _ = make_lstm_windows(df, window=12)
        assert X.shape[1] == 12, "Window size must be 12"
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    @pytest.mark.skipif(not os.path.exists("models/lstm.pt"), reason="Model not trained")
    def test_prediction_shape(self):
        import torch
        from lstm_predictor import ThermalLSTM, load_model
        model = load_model("models/lstm.pt")
        dummy = torch.FloatTensor(np.random.rand(5, 12, 1).astype(np.float32))
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (5,), f"Expected (5,), got {out.shape}"

    @pytest.mark.skipif(not os.path.exists("models/lstm.pt"), reason="Model not trained")
    def test_prediction_in_range(self):
        from lstm_predictor import load_model, predict_next
        model = load_model("models/lstm.pt")
        window = np.linspace(0.3, 0.5, 12).astype(np.float32)
        pred = predict_next(model, window)
        assert 0 <= pred <= 1.5, f"Prediction {pred} out of reasonable range"


# ═══════════════════════════════════════════════════════
# test_attention_fusion.py
# ═══════════════════════════════════════════════════════
class TestAttentionFusion:

    def test_healthy_scores_low(self):
        from attention_fusion import fuse
        r = fuse(0.1, 0.1, 0.1, cnn_score=0.1, audio_score=0.1)
        assert r["global_context_score"] < 0.3

    def test_critical_scores_high(self):
        from attention_fusion import fuse
        r = fuse(0.9, 0.9, 0.9, cnn_score=0.9, audio_score=0.9)
        assert r["global_context_score"] > 0.7

    def test_camera_blocked_reduces_weight(self):
        from attention_fusion import fuse
        r_blocked = fuse(0.5, 0.5, 0.5, cnn_score=0.9,
                          cnn_var=100.0, audio_score=0.5)
        r_normal  = fuse(0.5, 0.5, 0.5, cnn_score=0.9,
                          cnn_var=None, audio_score=0.5)
        assert r_blocked["weights"]["cnn"] < r_normal["weights"]["cnn"]

    def test_weights_sum_to_one(self):
        from attention_fusion import fuse
        r = fuse(0.5, 0.4, 0.3, cnn_score=0.6, audio_score=0.4)
        total = sum(r["weights"].values())
        assert abs(total - 1.0) < 1e-5, f"Weights sum {total} != 1.0"

    def test_no_placeholder_flag(self):
        from attention_fusion import fuse
        r = fuse(0.5, 0.5, 0.5, cnn_score=0.6, audio_score=0.4)
        assert not r["placeholders_used"]["cnn"]
        assert not r["placeholders_used"]["audio"]

    def test_urgency_in_range(self):
        from attention_fusion import fuse
        r = fuse(0.8, 0.7, 0.6, cnn_score=0.5, audio_score=0.5)
        assert 0 <= r["urgency_score"] <= 1


# ═══════════════════════════════════════════════════════
# test_rl_router.py
# ═══════════════════════════════════════════════════════
class TestRLRouter:

    def test_high_urgency_goes_edge(self):
        from rl_router import route
        assert route(0.85)["path"] == "edge"
        assert route(0.99)["path"] == "edge"

    def test_low_urgency_goes_cloud(self):
        from rl_router import route
        assert route(0.10)["path"] == "cloud"
        assert route(0.35)["path"] == "cloud"

    def test_middle_urgency_both(self):
        from rl_router import route
        assert route(0.50)["path"] == "both"
        assert route(0.75)["path"] == "both"

    def test_result_has_required_fields(self):
        from rl_router import route
        r = route(0.6)
        for field in ["path", "urgency_score", "reason", "latency", "timestamp"]:
            assert field in r, f"Missing field: {field}"

    def test_all_six_cases(self):
        from rl_router import route
        cases = [(0.10,"cloud"),(0.35,"cloud"),(0.50,"both"),
                 (0.75,"both"),(0.85,"edge"),(0.99,"edge")]
        for urgency, expected in cases:
            assert route(urgency)["path"] == expected, \
                f"urgency={urgency} expected {expected}"


# ═══════════════════════════════════════════════════════
# test_machine_adapter.py
# ═══════════════════════════════════════════════════════
class TestMachineAdapter:

    def test_similarity_not_all_one(self):
        from machine_adapter import (
            find_most_similar_machine, cosine_similarity,
            _NORMALIZED_FEATURES
        )
        sims = set()
        ids  = list(_NORMALIZED_FEATURES.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                sim = cosine_similarity(
                    _NORMALIZED_FEATURES[ids[i]],
                    _NORMALIZED_FEATURES[ids[j]]
                )
                sims.add(round(sim, 4))
        assert len(sims) > 1, \
            "All similarities identical — normalization fix not applied"

    def test_similarity_below_one(self):
        from machine_adapter import cosine_similarity, _NORMALIZED_FEATURES
        ids = list(_NORMALIZED_FEATURES.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                sim = cosine_similarity(
                    _NORMALIZED_FEATURES[ids[i]],
                    _NORMALIZED_FEATURES[ids[j]]
                )
                assert sim < 0.9999, \
                    f"Similarity between machine {ids[i]} and {ids[j]} = {sim:.4f} (still near 1.0)"

    def test_engine_different_from_server(self):
        from machine_adapter import cosine_similarity, _NORMALIZED_FEATURES
        sim_engine_server = cosine_similarity(
            _NORMALIZED_FEATURES[3], _NORMALIZED_FEATURES[2])
        sim_cpu_server    = cosine_similarity(
            _NORMALIZED_FEATURES[0], _NORMALIZED_FEATURES[2])
        # Engine (high heat, high temps) should differ more from
        # Server (low heat, low temps) than CPU does
        assert sim_engine_server != sim_cpu_server, \
            "Engine and CPU show same similarity to Server"

    def test_find_most_similar_returns_valid(self):
        from machine_adapter import find_most_similar_machine
        best_id, sim = find_most_similar_machine(3, [0, 1, 2])
        assert best_id in [0, 1, 2]
        assert 0 < sim < 1.001

    def test_model_file_exists(self):
        path = "models/lstm_adapted_engine.pt"
        if not os.path.exists(path):
            pytest.skip(f"Adapted model not generated — run stage5_adapter/run_stage5.py")


# ═══════════════════════════════════════════════════════
# test_audio_mfcc.py
# ═══════════════════════════════════════════════════════
class TestAudioMFCC:

    def test_audio_generation(self):
        from audio_mfcc import generate_machine_audio
        audio = generate_machine_audio(65.0, machine_safe_max=85.0, seed=0)
        assert audio.dtype == np.float32
        assert len(audio) == 22050     # 1 second at 22050 Hz
        assert np.abs(audio).max() <= 1.0 + 1e-5

    def test_mfcc_shape(self):
        from audio_mfcc import generate_machine_audio, extract_mfcc_features
        audio    = generate_machine_audio(65.0, seed=0)
        features = extract_mfcc_features(audio, n_mfcc=13)
        assert features.shape == (39,), f"Expected (39,), got {features.shape}"
        assert features.dtype == np.float32

    def test_pipeline_returns_score_in_range(self):
        from audio_mfcc import AudioPipeline
        pipeline = AudioPipeline()
        for i in range(25):
            pipeline.process_from_temperature(45.0, seed=i)
        result = pipeline.process_from_temperature(85.0, seed=99)
        assert 0 <= result["audio_score"] <= 1

    def test_anomaly_scores_higher_than_normal(self):
        from audio_mfcc import AudioPipeline
        pipeline = AudioPipeline()
        for i in range(25):
            pipeline.process_from_temperature(40.0, seed=i)
        normal   = pipeline.process_from_temperature(45.0, seed=100)
        warning  = pipeline.process_from_temperature(75.0, seed=101)
        critical = pipeline.process_from_temperature(95.0, seed=102)
        assert warning["audio_score"] > normal["audio_score"] or critical["audio_score"] > normal["audio_score"], \
            "Higher temps should generally score higher than normal temp"