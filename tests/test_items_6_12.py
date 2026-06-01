"""
MHARS — Tests for Items 6 & 12 fixes
======================================
Item 6:  Evaluation metrics (F1, AUC-ROC, MAE, MAPE, NASA scoring, etc.)
Item 12: Multi-fault injection & heteroscedastic noise in ThermalEnvV2
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage1_simulation'))


# ══════════════════════════════════════════════════════════════════════════════
#  Item 6: Evaluation Metrics Framework
# ══════════════════════════════════════════════════════════════════════════════

class TestRegressionMetrics:
    """Tests for RMSE, MAE, MAPE, R²."""

    def test_rmse_perfect(self):
        from benchmarks.evaluation_metrics import rmse
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_known(self):
        from benchmarks.evaluation_metrics import rmse
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error of 1 on last
        assert abs(rmse(y_true, y_pred) - np.sqrt(1/3)) < 1e-6

    def test_mae_perfect(self):
        from benchmarks.evaluation_metrics import mae
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_mae_known(self):
        from benchmarks.evaluation_metrics import mae
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # each off by 1
        assert abs(mae(y_true, y_pred) - 1.0) < 1e-6

    def test_mape_known(self):
        from benchmarks.evaluation_metrics import mape
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        assert abs(mape(y_true, y_pred) - 10.0) < 0.1  # 10%

    def test_r_squared_perfect(self):
        from benchmarks.evaluation_metrics import r_squared
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(r_squared(y, y) - 1.0) < 1e-6

    def test_regression_report(self):
        from benchmarks.evaluation_metrics import regression_report
        y_true = np.random.rand(100)
        y_pred = y_true + np.random.normal(0, 0.01, 100)
        report = regression_report(y_true, y_pred)
        assert "rmse" in report
        assert "mae" in report
        assert "mape" in report
        assert "r_squared" in report
        assert report["r_squared"] > 0.9  # very close predictions


class TestAnomalyMetrics:
    """Tests for Precision, Recall, F1, AUC-ROC, AUC-PR."""

    def test_precision_perfect(self):
        from benchmarks.evaluation_metrics import precision
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert precision(y_true, y_pred) == 1.0

    def test_recall_perfect(self):
        from benchmarks.evaluation_metrics import recall
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert recall(y_true, y_pred) == 1.0

    def test_f1_perfect(self):
        from benchmarks.evaluation_metrics import f1_score
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0])
        assert f1_score(y_true, y_pred) == 1.0

    def test_f1_zero_when_all_wrong(self):
        from benchmarks.evaluation_metrics import f1_score
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        assert f1_score(y_true, y_pred) == 0.0

    def test_auc_roc_perfect(self):
        from benchmarks.evaluation_metrics import auc_roc
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert auc_roc(y_true, y_scores) > 0.95

    def test_auc_roc_random(self):
        from benchmarks.evaluation_metrics import auc_roc
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_scores = rng.random(1000)
        auc = auc_roc(y_true, y_scores)
        assert 0.3 < auc < 0.7  # random classifier ≈ 0.5

    def test_auc_pr_perfect(self):
        from benchmarks.evaluation_metrics import auc_pr
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert auc_pr(y_true, y_scores) > 0.80

    def test_anomaly_report(self):
        from benchmarks.evaluation_metrics import anomaly_report
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        scores = np.array([0.9, 0.4, 0.1, 0.6, 0.8, 0.2])
        report = anomaly_report(y_true, y_pred, scores)
        assert "precision" in report
        assert "recall" in report
        assert "f1_score" in report
        assert "auc_roc" in report
        assert "auc_pr" in report


class TestRULMetrics:
    """Tests for NASA scoring function and timeliness."""

    def test_nasa_score_asymmetric(self):
        from benchmarks.evaluation_metrics import nasa_scoring_function
        y_true = np.array([50.0, 50.0])
        y_early = np.array([40.0, 40.0])  # d = -10
        y_late  = np.array([60.0, 60.0])  # d = +10
        score_early = nasa_scoring_function(y_true, y_early)
        score_late = nasa_scoring_function(y_true, y_late)
        assert score_late > score_early, "Late predictions should be penalized more"

    def test_nasa_score_perfect(self):
        from benchmarks.evaluation_metrics import nasa_scoring_function
        y = np.array([50.0, 50.0])
        score = nasa_scoring_function(y, y)
        assert score == 0.0

    def test_timeliness(self):
        from benchmarks.evaluation_metrics import rul_timeliness
        y_true = np.array([50.0, 50.0, 50.0, 50.0])
        y_pred = np.array([45.0, 42.0, 60.0, 30.0])  # 2 timely, 2 not
        t = rul_timeliness(y_true, y_pred, early_window=10)
        assert abs(t - 0.5) < 1e-6  # 45 and 42 are within window

    def test_rul_report(self):
        from benchmarks.evaluation_metrics import rul_report
        y_true = np.random.rand(50) * 100
        y_pred = y_true + np.random.normal(0, 5, 50)
        report = rul_report(y_true, y_pred)
        assert "nasa_score" in report
        assert "timeliness_10" in report
        assert "timeliness_20" in report
        assert "rmse" in report


class TestConformalMetrics:
    """Tests for coverage probability and interval width."""

    def test_coverage_perfect(self):
        from benchmarks.evaluation_metrics import coverage_probability
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        assert coverage_probability(y_true, lower, upper) == 1.0

    def test_coverage_none(self):
        from benchmarks.evaluation_metrics import coverage_probability
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([5.0, 5.0, 5.0])
        upper = np.array([6.0, 6.0, 6.0])
        assert coverage_probability(y_true, lower, upper) == 0.0

    def test_interval_width(self):
        from benchmarks.evaluation_metrics import average_interval_width
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([1.0, 2.0, 3.0])
        assert abs(average_interval_width(lower, upper) - 1.0) < 1e-6

    def test_conformal_report(self):
        from benchmarks.evaluation_metrics import conformal_report
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        report = conformal_report(y_true, lower, upper)
        assert "coverage_probability" in report
        assert "average_interval_width" in report


class TestRLMetrics:
    """Tests for safety violation rate and energy efficiency."""

    def test_safety_violation_rate(self):
        from benchmarks.evaluation_metrics import safety_violation_rate
        temps = np.array([80, 85, 90, 95, 100])
        rate = safety_violation_rate(temps, safe_max=85.0)
        assert abs(rate - 0.6) < 1e-6  # 3 of 5 exceed 85

    def test_energy_efficiency(self):
        from benchmarks.evaluation_metrics import energy_efficiency
        fans = np.array([0.5, 0.5, 0.5])
        eff = energy_efficiency(fans)
        assert abs(eff - 0.25) < 1e-6  # 0.5^2 = 0.25

    def test_full_benchmark_runner(self):
        from benchmarks.evaluation_metrics import run_full_benchmark
        results = run_full_benchmark(
            lstm_y_true=np.random.rand(50),
            lstm_y_pred=np.random.rand(50),
            anomaly_y_true=np.random.randint(0, 2, 50),
            anomaly_y_pred=np.random.randint(0, 2, 50),
            anomaly_scores=np.random.rand(50),
            rl_temps=np.random.uniform(60, 95, 100),
            rl_fan_speeds=np.random.uniform(0, 1, 100),
        )
        assert "lstm_regression" in results
        assert "anomaly_detection" in results
        assert "rl_control" in results


# ══════════════════════════════════════════════════════════════════════════════
#  Item 12: Multi-Fault Injection & Heteroscedastic Noise
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiFaultInjection:
    """Tests for simultaneous fault injection in ThermalEnvV2."""

    def test_faults_can_occur(self):
        """Over enough steps, at least one fault should trigger."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        env.reset(seed=42)
        
        faults_seen = set()
        for _ in range(500):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            faults_seen.update(info.get("active_faults", []))
            if term or trunc:
                env.reset(seed=None)
        
        assert len(faults_seen) > 0, "Expected at least one fault type over 500 steps"

    def test_simultaneous_faults_possible(self):
        """Both bearing_failure and cooling_loss can co-occur."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        
        simultaneous = False
        for seed in range(50):
            env.reset(seed=seed)
            for _ in range(500):
                _, _, term, trunc, info = env.step(env.action_space.sample())
                faults = info.get("active_faults", [])
                if len(faults) >= 2:
                    simultaneous = True
                    break
                if term or trunc:
                    break
            if simultaneous:
                break
        
        assert simultaneous, "Expected simultaneous faults at least once in 50 episodes"

    def test_bearing_failure_increases_vibration(self):
        """When bearing failure is active, vibration score should spike."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        
        max_vib_during_fault = 0.0
        found_fault = False
        
        for seed in range(20):
            env.reset(seed=seed)
            for _ in range(500):
                _, _, term, trunc, info = env.step(env.action_space.sample())
                if "bearing_failure" in info.get("active_faults", []):
                    found_fault = True
                    max_vib_during_fault = max(max_vib_during_fault, info["vib_score"])
                if term or trunc:
                    break
            if found_fault:
                break
        
        if found_fault:
            assert max_vib_during_fault > 0.2, \
                f"Vibration should spike during bearing failure, got {max_vib_during_fault}"

    def test_active_faults_in_info(self):
        """Step info dict should contain 'active_faults' key."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        env.reset(seed=42)
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "active_faults" in info


class TestHeteroscedasticNoise:
    """Tests for temperature-dependent sensor noise."""

    def test_noise_increases_with_temperature(self):
        """Noise standard deviation should be higher at higher temperatures."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        env.reset(seed=42)
        
        # Low temperature noise
        noise_low = [env._heteroscedastic_noise(30.0) for _ in range(500)]
        # High temperature noise
        noise_high = [env._heteroscedastic_noise(90.0) for _ in range(500)]
        
        std_low = np.std(noise_low)
        std_high = np.std(noise_high)
        
        assert std_high > std_low, \
            f"High temp noise std ({std_high:.4f}) should exceed low temp ({std_low:.4f})"

    def test_noise_mean_near_zero(self):
        """Noise should be zero-mean (unbiased sensor)."""
        from stage1_simulation.gym_env import ThermalEnvV2
        env = ThermalEnvV2(machine_type_id=0, variable_episodes=False)
        env.reset(seed=42)
        
        noise_samples = [env._heteroscedastic_noise(60.0) for _ in range(2000)]
        mean_noise = np.mean(noise_samples)
        assert abs(mean_noise) < 0.1, f"Noise should be zero-mean, got {mean_noise:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
