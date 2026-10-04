import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from vctk16_normalization import global_statistics, apply_cmvn
from run_vctk16_isolated_suite import plan, output_root

def test_disabled_zscore_preserves_features_and_cross_microphone_scale():
    train = np.arange(60, dtype=np.float32).reshape(2, 3, 10)
    target = train + 100
    mean, std = global_statistics(train, False)
    np.testing.assert_array_equal((train - mean) / std, train)
    np.testing.assert_array_equal((target - mean) / std, target)

def test_enabled_zscore_uses_training_statistics_for_target():
    train = np.arange(60, dtype=np.float32).reshape(2, 3, 10)
    mean, std = global_statistics(train, True)
    normalized = (train - mean) / std
    np.testing.assert_allclose(normalized.mean(axis=(0, 2)), 0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=(0, 2)), 1, atol=1e-6)
    assert np.all(((train + 100 - mean) / std).mean(axis=(0, 2)) > 1)

def test_cmvn_is_preserved_without_second_standardization():
    features = np.arange(90 * 111, dtype=np.float32).reshape(90, 111)
    features = apply_cmvn(features)
    mean, std = global_statistics(features[None], False)
    np.testing.assert_array_equal((features[None] - mean) / std, features[None])

def test_plan_excludes_xvector_combinations_and_old_combined_outputs():
    stages = plan()
    assert len(stages) == 8
    assert {s['architecture'] for s in stages} == {'cnn', 'temporal_cnn'}
    assert {s['condition'] for s in stages} == {'zscore', 'cmn', 'cmvn', 'rasta'}
    for stage in stages:
        path = output_root(stage['architecture'], stage['condition'])
        assert ('baseline_' if stage['condition'] == 'zscore' else 'isolated_') in path.name
