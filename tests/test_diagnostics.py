"""Testes da assinatura de canal.

O que se protege aqui é a distinção entre "não havia material" e "o material não
identificava o locutor". Se um recorte curto demais produzisse um vetor qualquer em
vez de nada, o diagnóstico mediria ruído numérico e reportaria como resultado.
"""

from __future__ import annotations

import numpy as np
import pytest

from sr.diagnostics import MIN_SAMPLES, channel_signature, extract_condition, signatures_of

SAMPLING_RATE = 16_000
NUM_MFCCS = 13


@pytest.fixture
def sinal() -> np.ndarray:
    """Sinal com um trecho alto entre dois trechos silenciosos."""
    rng = np.random.default_rng(0)
    quieto = np.zeros(8_000, dtype=np.float32)
    alto = rng.standard_normal(8_000).astype(np.float32)
    return np.concatenate([quieto, alto, quieto])


# ----------------------------------------------------------------------
# Recorte
# ----------------------------------------------------------------------

def test_full_returns_the_signal_untouched(sinal):
    assert np.array_equal(extract_condition(sinal, 'full', top_db=30), sinal)


def test_speech_and_silence_partition_the_signal(sinal):
    """Os dois recortes somam o sinal inteiro, sem sobreposição.

    O diagnóstico depende disso: se os recortes se sobrepusessem, a condição
    'só silêncio' conteria fala e o resultado perderia o sentido.
    """
    fala = extract_condition(sinal, 'speech', top_db=30)
    silencio = extract_condition(sinal, 'silence', top_db=30)

    assert len(fala) + len(silencio) == len(sinal)


def test_unknown_condition_is_rejected(sinal):
    with pytest.raises(ValueError, match='Condição desconhecida'):
        extract_condition(sinal, 'ruido', top_db=30)


# ----------------------------------------------------------------------
# Assinatura
# ----------------------------------------------------------------------

def test_signature_has_a_mean_and_a_deviation_per_coefficient():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16_000).astype(np.float32)

    assinatura = channel_signature(audio, SAMPLING_RATE, NUM_MFCCS)

    assert assinatura.shape == (2 * NUM_MFCCS,)


def test_short_segment_yields_nothing_rather_than_noise():
    """Recorte curto demais devolve None, e não uma estimativa instável.

    Devolver um vetor aqui faria o classificador aprender sobre gravações que, na
    prática, não tinham silêncio algum a oferecer.
    """
    curto = np.zeros(MIN_SAMPLES - 1, dtype=np.float32)

    assert channel_signature(curto, SAMPLING_RATE, NUM_MFCCS) is None


def test_signature_is_scale_sensitive():
    """A assinatura muda com o ganho, que é parte da identidade do canal.

    Não é um defeito: ganho de gravação é exatamente uma das pistas de sessão que o
    diagnóstico se propõe a detectar.
    """
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16_000).astype(np.float32)

    original = channel_signature(audio, SAMPLING_RATE, NUM_MFCCS)
    atenuado = channel_signature(audio * 0.1, SAMPLING_RATE, NUM_MFCCS)

    assert not np.allclose(original, atenuado)


# ----------------------------------------------------------------------
# As três condições em uma passada
# ----------------------------------------------------------------------

def test_all_conditions_are_computed_when_material_exists(sinal):
    assinaturas = signatures_of(sinal, SAMPLING_RATE, NUM_MFCCS, top_db=30)

    assert set(assinaturas) == {'silence', 'speech', 'full'}
    assert all(v.shape == (2 * NUM_MFCCS,) for v in assinaturas.values())
    assert all(v.dtype == np.float32 for v in assinaturas.values())


def test_condition_without_material_is_omitted_not_zeroed():
    """Sinal sem silêncio algum não produz entrada ``silence``.

    A ausência da chave é o que permite ao diagnóstico distinguir a gravação que não
    tinha silêncio da gravação cujo silêncio não identificava ninguém.
    """
    rng = np.random.default_rng(0)
    so_fala = rng.standard_normal(16_000).astype(np.float32)

    assinaturas = signatures_of(so_fala, SAMPLING_RATE, NUM_MFCCS, top_db=30)

    assert 'silence' not in assinaturas
    assert 'full' in assinaturas
