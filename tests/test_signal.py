"""Testes das operações de processamento de sinal.

Verificam propriedades que devem valer independentemente do sinal de entrada, e não
valores numéricos específicos: são as propriedades que, se quebradas, invalidariam
silenciosamente os experimentos.
"""

from __future__ import annotations

import numpy as np
import pytest

from sr.preprocessing import signal


@pytest.fixture
def tone() -> np.ndarray:
    """Senoide de 1 kHz amostrada a 48 kHz, com duração de um segundo."""
    t = np.arange(48_000) / 48_000
    return np.sin(2 * np.pi * 1000 * t).astype(np.float32)


def test_resample_produces_expected_length(tone):
    """A reamostragem deve produzir a razão exata entre as taxas."""
    assert len(signal.resample(tone, 48_000, 8_000)) == 8_000
    assert len(signal.resample(tone, 48_000, 16_000)) == 16_000


def test_resample_is_identity_at_same_rate(tone):
    """Reamostrar para a taxa corrente não deve alterar o sinal."""
    assert np.array_equal(signal.resample(tone, 48_000, 48_000), tone)


def test_antialias_filter_attenuates_above_cutoff():
    """Componentes acima do novo Nyquist devem ser atenuados antes da decimação.

    Sem esta propriedade, a decimação rebateria essas componentes para dentro da banda
    útil, contaminando os coeficientes cepstrais com energia que não existia no sinal.
    """
    t = np.arange(48_000) / 48_000
    # 6 kHz está acima do Nyquist de 4 kHz correspondente à taxa alvo de 8 kHz.
    above = np.sin(2 * np.pi * 6000 * t).astype(np.float32)

    filtered = signal.antialias_filter(above, 48_000, 8_000)

    # Ignora as bordas, como no teste da banda passante: a filtragem de fase zero
    # deixa um transitório nas últimas amostras, que não representa a resposta em
    # regime do filtro. Aqui ele afeta cerca de 15 amostras em 48 000.
    interior = slice(2_000, -2_000)
    assert np.max(np.abs(filtered[interior])) < 0.05 * np.max(np.abs(above))


def test_antialias_filter_preserves_passband():
    """Componentes bem abaixo do corte devem atravessar o filtro praticamente intactas."""
    t = np.arange(48_000) / 48_000
    below = np.sin(2 * np.pi * 500 * t).astype(np.float32)

    filtered = signal.antialias_filter(below, 48_000, 8_000)

    # Ignora as bordas, onde o filtro de fase zero tem transitório.
    interior = slice(2_000, -2_000)
    assert np.max(np.abs(filtered[interior])) > 0.9 * np.max(np.abs(below[interior]))


def test_pre_emphasis_disabled_is_identity(tone):
    """Coeficiente nulo deve desativar a pré-ênfase sem alterar o sinal."""
    assert np.array_equal(signal.pre_emphasis(tone, 0.0), tone)


def test_pre_emphasis_preserves_length(tone):
    """A pré-ênfase não deve alterar o número de amostras."""
    assert len(signal.pre_emphasis(tone, 0.97)) == len(tone)


def test_pre_emphasis_boosts_high_frequencies():
    """A pré-ênfase deve elevar as componentes agudas em relação às graves."""
    t = np.arange(16_000) / 16_000
    low = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 3000 * t).astype(np.float32)

    low_gain = np.std(signal.pre_emphasis(low, 0.97)) / np.std(low)
    high_gain = np.std(signal.pre_emphasis(high, 0.97)) / np.std(high)

    assert high_gain > low_gain


def test_silence_and_speech_are_complementary():
    """Fala e silêncio devem particionar o sinal, sem sobreposição nem perda.

    O diagnóstico de canal depende desta propriedade: se os dois recortes se
    sobrepusessem, a condição 'só silêncio' conteria fala e o resultado perderia
    o sentido.
    """
    rng = np.random.default_rng(0)
    quiet = np.zeros(8_000, dtype=np.float32)
    loud = rng.standard_normal(8_000).astype(np.float32)
    audio = np.concatenate([quiet, loud, quiet])

    speech = signal.remove_silence(audio, top_db=30)
    silence = signal.extract_silence(audio, top_db=30)

    assert len(speech) + len(silence) == len(audio)


def test_extract_mfccs_shape():
    """A matriz de MFCCs deve ter uma linha por coeficiente pedido."""
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(8_000).astype(np.float32)

    mfccs = signal.extract_mfccs(audio, 8_000, num_mfccs=40, frame_size=256)

    assert mfccs.shape[0] == 40
    # Com salto de 128 amostras, um segundo a 8 kHz rende cerca de 63 quadros.
    assert 60 <= mfccs.shape[1] <= 66
