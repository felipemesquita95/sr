"""Testes das arquiteturas e da agregação temporal.

Não se verifica desempenho aqui — isso é o que os experimentos medem. Verifica-se
que cada arquitetura constrói, produz uma distribuição sobre os locutores, e que as
camadas de agregação calculam o que afirmam calcular. Uma agregação errada não
falharia: treinaria normalmente e devolveria um número um pouco pior, sem indício
de que a causa era a camada e não a arquitetura.
"""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops

from sr.models.pooling import AttentiveStatisticsPooling, StatisticsPooling
from sr.models.registry import ARCHITECTURES, build_model


def to_numpy(tensor) -> np.ndarray:
    """Traz um tensor de saída para a memória do processo.

    Necessário porque o backend é o PyTorch com ROCm: os tensores ficam na GPU, e
    ``np.asarray`` sobre eles falha. ``ops.convert_to_numpy`` é agnóstico de backend
    e continuaria valendo se o backend mudasse.
    """
    return np.asarray(ops.convert_to_numpy(tensor))


NUM_MFCCS = 13
NUM_FRAMES = 64
NUM_CLASSES = 5
INPUT_SHAPE = (NUM_MFCCS, NUM_FRAMES)


# ----------------------------------------------------------------------
# Agregação
# ----------------------------------------------------------------------

def test_statistics_pooling_computes_mean_and_deviation():
    """A saída é a média seguida do desvio, por canal."""
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((4, 20, 3)).astype(np.float32)

    saida = to_numpy(StatisticsPooling()(batch))

    assert saida.shape == (4, 6)
    assert np.allclose(saida[:, :3], batch.mean(axis=1), atol=1e-5)
    assert np.allclose(saida[:, 3:], batch.std(axis=1), atol=1e-5)


def test_statistics_pooling_survives_a_constant_channel():
    """Canal constante tem variância zero, e a raiz quadrada ali tem derivada infinita.

    Sem o piso de variância o treino morreria com NaN ao encontrar um trecho de
    silêncio — que é justamente o que este trabalho passa o tempo todo analisando.
    """
    constante = np.ones((2, 10, 3), dtype=np.float32)

    saida = to_numpy(StatisticsPooling()(constante))

    assert np.all(np.isfinite(saida))
    assert np.allclose(saida[:, :3], 1.0)
    assert np.allclose(saida[:, 3:], 0.0, atol=1e-3)


def test_statistics_pooling_distinguishes_what_the_mean_cannot():
    """Duas sequências de mesma média e dispersões diferentes têm de sair diferentes.

    É a razão de existir da camada: a média global as tornaria idênticas.
    """
    calma = np.zeros((1, 10, 1), dtype=np.float32)
    agitada = np.tile(np.array([[-1.0], [1.0]], dtype=np.float32), (5, 1))[None, ...]

    a = to_numpy(StatisticsPooling()(calma))
    b = to_numpy(StatisticsPooling()(agitada))

    assert np.allclose(a[:, 0], b[:, 0], atol=1e-6)  # mesma média
    assert not np.allclose(a[:, 1], b[:, 1])  # desvios distintos


def test_attentive_pooling_has_the_same_output_width():
    """A variante atenta muda como pondera, não o formato do que entrega."""
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((2, 20, 8)).astype(np.float32)

    simples = to_numpy(StatisticsPooling()(batch))
    atenta = to_numpy(AttentiveStatisticsPooling()(batch))

    assert atenta.shape == simples.shape == (2, 16)
    assert np.all(np.isfinite(atenta))


# ----------------------------------------------------------------------
# Arquiteturas
# ----------------------------------------------------------------------

@pytest.mark.parametrize('architecture', sorted(ARCHITECTURES))
def test_every_architecture_builds_and_predicts(architecture):
    """Toda arquitetura registrada constrói e devolve uma distribuição por locutor."""
    rng = np.random.default_rng(0)
    lote = rng.standard_normal((2, *INPUT_SHAPE)).astype(np.float32)

    model = build_model(architecture, INPUT_SHAPE, NUM_CLASSES, learning_rate=1e-3)
    saida = to_numpy(model.predict(lote, verbose=0))

    assert saida.shape == (2, NUM_CLASSES)
    assert np.allclose(saida.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(saida >= 0)


@pytest.mark.parametrize('architecture', sorted(ARCHITECTURES))
def test_every_architecture_trains_one_step(architecture):
    """Um passo de treino tem de rodar sem NaN, em toda arquitetura.

    Pega erros de forma e instabilidades numéricas que a mera construção não revela.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, *INPUT_SHAPE)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=4)

    model = build_model(architecture, INPUT_SHAPE, NUM_CLASSES, learning_rate=1e-3)
    history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)

    assert np.isfinite(history.history['loss'][0])


def test_unknown_architecture_names_the_available_ones():
    """A mensagem tem de listar as opções, senão o erro não ajuda a corrigir o perfil."""
    with pytest.raises(KeyError, match='temporal_cnn'):
        build_model('lstm', INPUT_SHAPE, NUM_CLASSES, learning_rate=1e-3)


def test_pooling_choice_is_validated():
    """Agregação desconhecida falha na construção, e não silenciosamente na média."""
    from sr.models.convolutional import build_temporal_cnn

    with pytest.raises(ValueError, match='Agregação desconhecida'):
        build_temporal_cnn(INPUT_SHAPE, NUM_CLASSES, pooling='mediana')


def test_xvector_rejects_mismatched_block_specifications():
    """Listas de comprimentos diferentes produziriam uma rede distinta da pedida."""
    from sr.models.tdnn import build_xvector

    with pytest.raises(ValueError, match='mesmo comprimento'):
        build_xvector(INPUT_SHAPE, NUM_CLASSES, filters=(64, 64), kernels=(3,), dilations=(1,))


def test_statistics_pooling_doubles_the_representation_width():
    """A CNN temporal com estatísticas tem de ter mais parâmetros densos que a com média.

    Confirma que a troca de agregação de fato chegou ao modelo, em vez de ser
    silenciosamente ignorada.
    """
    media = build_model('temporal_cnn', INPUT_SHAPE, NUM_CLASSES, learning_rate=1e-3)
    estatisticas = build_model('temporal_cnn_stats', INPUT_SHAPE, NUM_CLASSES, learning_rate=1e-3)

    assert estatisticas.count_params() > media.count_params()
