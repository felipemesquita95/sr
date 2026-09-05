"""Registro das arquiteturas disponíveis e sua construção a partir do nome.

Centralizar a construção em um único ponto permite que os experimentos iterem
sobre nomes de arquitetura sem conhecer as suas implementações, e garante que
todas recebam otimizador, função de perda e métricas idênticos — condição para
que a comparação entre elas isole a arquitetura como única variável.
"""

from __future__ import annotations

import logging
from typing import Callable

from keras import Model
from keras.optimizers import Adam

from sr.models.attention import build_attention_model, build_temporal_attention
from sr.models.convolutional import (build_cepstral_cnn, build_temporal_cnn,
                                     build_temporal_cnn_statistics,
                                     build_temporal_cnn_wide)
from sr.models.tdnn import (build_attentive_xvector, build_temporal_resnet,
                            build_xvector)

logger = logging.getLogger(__name__)

#: Arquiteturas disponíveis, mapeadas pelo nome usado nos perfis de configuração.
#:
#: As entradas formam três pares de ablação, e não uma coleção solta de modelos:
#: ``cnn`` contra ``temporal_cnn`` isola o eixo da convolução; ``temporal_cnn``
#: contra ``temporal_cnn_stats`` isola a agregação; ``xvector`` contra
#: ``xvector_attentive`` isola a ponderação dos quadros. Comparações assim atribuem a
#: diferença observada a uma causa; uma tabela de modelos arbitrários, não.
ARCHITECTURES: dict[str, Callable[..., Model]] = {
    'cnn': build_cepstral_cnn,
    'temporal_cnn': build_temporal_cnn,
    'temporal_cnn_stats': build_temporal_cnn_statistics,
    'attention': build_attention_model,
    'xvector': build_xvector,
    'xvector_attentive': build_attentive_xvector,
    'temporal_attention': build_temporal_attention,
    'temporal_cnn_wide': build_temporal_cnn_wide,
    'temporal_resnet': build_temporal_resnet,
}


def build_model(
    architecture: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    learning_rate: float,
    **kwargs,
) -> Model:
    """Constrói e compila a arquitetura indicada.

    Args:
        architecture: Nome da arquitetura; deve constar de :data:`ARCHITECTURES`.
        input_shape: Forma de uma amostra, ``(num_mfccs, num_quadros)``.
        num_classes: Número de locutores.
        learning_rate: Taxa de aprendizado inicial do otimizador Adam.
        **kwargs: Hiperparâmetros específicos da arquitetura, repassados ao construtor.

    Returns:
        Modelo compilado com perda de entropia cruzada categórica esparsa.

    Raises:
        KeyError: Se o nome da arquitetura for desconhecido.
    """
    if architecture not in ARCHITECTURES:
        raise KeyError(
            f'Arquitetura desconhecida: {architecture!r}. '
            f'Disponíveis: {", ".join(sorted(ARCHITECTURES))}.'
        )

    model = ARCHITECTURES[architecture](input_shape, num_classes, **kwargs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    logger.info('Arquitetura %r: %d parâmetros treináveis, entrada %s.',
                architecture, model.count_params(), input_shape)
    return model
