"""Verify per-recording CMVN without loading a model or touching the GPU."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import librosa

spec = importlib.util.spec_from_file_location(
    'vctk16_normalization', Path(__file__).parents[1] / 'experiments/vctk16_normalization.py')
normalization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalization)


@pytest.mark.parametrize('n_mfcc', [30, 40])
def test_cmvn_matches_derivatives_after_normalization(n_mfcc):
    rng = np.random.default_rng(19)
    static = rng.normal(size=(n_mfcc, 111)).astype(np.float32)
    delta = librosa.feature.delta(static, width=9)
    delta2 = librosa.feature.delta(static, width=9, order=2)
    matrix = np.concatenate((static, delta, delta2))
    original = matrix.copy()
    result = normalization.apply_cmvn(matrix, n_mfcc)
    np.testing.assert_allclose(result[:n_mfcc].mean(axis=1), 0, atol=2e-7)
    np.testing.assert_allclose(result[:n_mfcc].std(axis=1), 1, atol=2e-7)
    np.testing.assert_allclose(result[n_mfcc:2*n_mfcc],
                               librosa.feature.delta(result[:n_mfcc], width=9), atol=2e-7)
    np.testing.assert_allclose(result[2*n_mfcc:],
                               librosa.feature.delta(result[:n_mfcc], width=9, order=2), atol=2e-7)
    np.testing.assert_array_equal(matrix, original)


def test_cmvn_removes_positive_scale_and_offset_of_each_coefficient():
    rng = np.random.default_rng(42)
    static = rng.normal(size=(30, 111))
    transformed = static * rng.uniform(.1, 5, (30, 1)) + rng.normal(size=(30, 1))*10
    np.testing.assert_allclose(normalization.apply_cmvn(static),
                               normalization.apply_cmvn(transformed), atol=1e-12)


def test_cmvn_handles_constant_coefficients_and_rejects_invalid_values():
    matrix = np.zeros((90, 111), dtype=np.float32)
    matrix[:30] = 7
    np.testing.assert_array_equal(normalization.apply_cmvn(matrix), 0)
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        normalization.apply_cmvn(matrix)
