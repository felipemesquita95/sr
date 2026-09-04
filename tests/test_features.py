"""Testes do alinhamento de comprimento e das garantias contra vazamento."""

from __future__ import annotations

import numpy as np

from sr.features import FeatureAdjustmentSubsystem


def test_truncate_keeps_leading_frames():
    """Matrizes longas devem ser cortadas, preservando o início."""
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)

    result = FeatureAdjustmentSubsystem.pad_or_truncate(matrix, 2)

    assert result.shape == (3, 2)
    assert np.array_equal(result, matrix[:, :2])


def test_extend_repeats_content_without_zeros():
    """Matrizes curtas devem ser estendidas por repetição, nunca com zeros.

    Preencher com zeros criaria silêncio artificial de duração correlacionada ao
    enunciado original — exatamente o artefato que o trabalho investiga.
    """
    matrix = np.arange(1, 7, dtype=np.float32).reshape(2, 3)

    result = FeatureAdjustmentSubsystem.pad_or_truncate(matrix, 8)

    assert result.shape == (2, 8)
    assert not np.any(result == 0), 'o preenchimento introduziu zeros'
    assert np.array_equal(result[:, :3], matrix)
    assert np.array_equal(result[:, 3:6], matrix)


def test_exact_length_is_unchanged():
    """Matrizes já no comprimento pedido devem passar inalteradas."""
    matrix = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert np.array_equal(FeatureAdjustmentSubsystem.pad_or_truncate(matrix, 3), matrix)
