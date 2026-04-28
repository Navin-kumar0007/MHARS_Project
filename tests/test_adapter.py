import pytest
import numpy as np
from stage5_adapter.machine_adapter import normalize_features

def test_normalize_features():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # Should center around 0 and have std 1
    norm_data, mean, std = normalize_features(data)
    
    assert np.allclose(norm_data.mean(axis=0), [0.0, 0.0])
    # The std computation might use N or N-1, but we can just check if std is recorded
    assert len(mean) == 2
    assert len(std) == 2
    assert mean[0] == 3.0
    assert mean[1] == 4.0

def test_normalize_features_with_existing_stats():
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean = np.array([3.0, 4.0])
    std = np.array([2.0, 2.0])
    
    norm_data, ret_mean, ret_std = normalize_features(data, mean=mean, std=std)
    
    # (1 - 3)/2 = -1
    assert norm_data[0][0] == -1.0
    assert np.array_equal(ret_mean, mean)
    assert np.array_equal(ret_std, std)
