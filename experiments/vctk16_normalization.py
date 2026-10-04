"""Per-recording CMVN for static MFCCs and their temporal derivatives."""
import numpy as np

CMVN_PROTOCOL = 'static_mean_std_per_window_deltas_scaled_same_std_v1'


def global_statistics(train_x: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    """Training-only z-score statistics, or identity for isolated conditions."""
    if enabled:
        return (train_x.mean(axis=(0, 2), keepdims=True),
                train_x.std(axis=(0, 2), keepdims=True) + 1e-8)
    mean = np.zeros((1, train_x.shape[1], 1), dtype=train_x.dtype)
    return mean, np.ones_like(mean)


def apply_cmvn(matrix: np.ndarray, n_mfcc: int = 30) -> np.ndarray:
    """Normalize static coefficients; scale Δ/ΔΔ consistently.

    Statistics use only the current recording's selected window. Dividing
    each derivative by the corresponding static standard deviation matches
    recomputing derivatives after the static affine transformation.
    """
    if matrix.ndim != 2 or matrix.shape[0] not in (n_mfcc, 3*n_mfcc):
        raise ValueError('CMVN feature rows do not match n_mfcc')
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError('CMVN requires finite, non-empty features')
    result = matrix.copy()
    static = matrix[:n_mfcc]
    mean = static.mean(axis=1, keepdims=True)
    std = np.maximum(static.std(axis=1, keepdims=True), 1e-8)
    result[:n_mfcc] = (static - mean) / std
    if matrix.shape[0] == 3*n_mfcc:
        result[n_mfcc:2*n_mfcc] /= std
        result[2*n_mfcc:3*n_mfcc] /= std
    return result
