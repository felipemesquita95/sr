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


def _split_of(items_per_set, permute):
    """Monta os três conjuntos por meio de ``_finalize``, com ou sem permutação."""
    from sr.config import Settings

    subsystem = FeatureAdjustmentSubsystem(
        Settings(num_speakers=4, permute_labels=permute, permutation_seed=7))
    groups = []
    label = 0
    for count in items_per_set:
        group = []
        for index in range(count):
            group.append((np.full((3, 5), float(label), dtype=np.float32), label % 4))
            label += 1
        groups.append(group)
    return subsystem._finalize(*groups, label='teste')


def test_permutation_preserves_class_counts():
    """A permutação redistribui os rótulos, sem criar nem destruir exemplos de classe."""
    honest = _split_of((12, 4, 4), permute=False)
    permuted = _split_of((12, 4, 4), permute=True)

    pooled = lambda s: np.sort(np.concatenate([s.train_y, s.validation_y, s.test_y]))

    assert np.array_equal(pooled(honest), pooled(permuted))
    assert honest.train_x.shape == permuted.train_x.shape


def test_permutation_breaks_the_recording_label_association():
    """Sob permutação, a gravação deixa de predizer o próprio rótulo."""
    honest = _split_of((12, 4, 4), permute=False)
    permuted = _split_of((12, 4, 4), permute=True)

    # As matrizes chegam na mesma ordem; só os rótulos mudam de dono. Se a
    # permutação não movesse nada, o controle não controlaria coisa alguma.
    assert not np.array_equal(honest.train_y, permuted.train_y)
