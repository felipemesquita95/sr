import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from logmel_speech_normalization import normalize_log_mel, speech_mask

def test_speech_stats_ignore_silence_and_normalize_each_band():
    x = np.array([[100., 1., 3., 100.], [-500., 10., 30., -500.]])
    mask = np.array([False, True, True, False])
    y = normalize_log_mel(x, mask)
    np.testing.assert_allclose(y[:, mask].mean(axis=1), 0, atol=1e-7)
    np.testing.assert_allclose(y[:, mask].std(axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(y[:, mask], [[-1, 1], [-1, 1]])

def test_constant_bands_and_no_speech_fallback_are_finite():
    assert np.isfinite(normalize_log_mel(np.ones((128, 20)), np.zeros(20, bool))).all()

def test_mask_uses_frame_centers():
    mask = speech_mask([dict(start=256, end=768)], 4, 16000)
    np.testing.assert_array_equal(mask, [True, True, False, False])
