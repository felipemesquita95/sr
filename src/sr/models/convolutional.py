"""Arquiteturas convolucionais, diferindo no eixo sobre o qual convoluem.

A entrada tem forma ``(num_mfccs, num_quadros)``. A ``Conv1D`` do Keras interpreta
o primeiro eixo como passos e o segundo como canais, então **a orientação do tensor
decide sobre o que a rede convolui** — e as duas escolhas produzem redes
radicalmente diferentes, tanto em pressuposto quanto em número de parâmetros.

Convolução cepstral (:func:`build_cepstral_cnn`)
    Passos são os coeficientes; os quadros são canais. Cada quadro recebe pesos
    próprios, de modo que não há invariância temporal: a rede pode aprender que
    determinado instante da gravação tem determinada energia.

Convolução temporal (:func:`build_temporal_cnn`)
    Passos são os quadros; os coeficientes são canais. É a orientação convencional
    em processamento de fala, e a agregação por média global torna a decisão
    invariante a deslocamentos no tempo.

Uma ressalva sobre a primeira: **o eixo cepstral não possui estrutura de vizinhança**.
Coeficientes adjacentes são projeções sobre bases distintas da DCT, não posições
vizinhas em um espaço métrico. Convolução pressupõe localidade — que a informação
relevante esteja em janelas contíguas — e essa premissa não se sustenta aqui. A
arquitetura é mantida por ser o baseline histórico do trabalho e por permitir a
comparação direta com a alternativa por atenção, que não faz tal suposição.
"""

from __future__ import annotations

from keras import Sequential
from keras.layers import (Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D,
                          Input, MaxPooling1D, Permute)
from keras.regularizers import l2


def build_cepstral_cnn(
    input_shape: tuple[int, ...],
    num_classes: int,
    conv_filters: int = 32,
    conv_kernel_size: int = 4,
    conv_stride: int = 1,
    conv_activation: str = 'tanh',
    conv_l2: float = 0.0,
    dense_units: int = 256,
    dense_activation: str = 'tanh',
    dense_l2: float = 0.04,
    dense_dropout: float = 0.0,
) -> Sequential:
    """Constrói a CNN que convolui ao longo do eixo dos coeficientes cepstrais.

    Com os quadros como canais, esta rede não compartilha pesos ao longo do tempo:
    o número de parâmetros da primeira camada cresce proporcionalmente ao número de
    quadros da entrada. Após a convolução, o achatamento leva todas as ativações
    para a camada densa, o que concentra a maior parte da capacidade do modelo em
    uma única matriz não estruturada.

    Args:
        input_shape: Forma de uma amostra, ``(num_mfccs, num_quadros)``.
        num_classes: Número de locutores.
        conv_filters: Número de filtros convolucionais.
        conv_kernel_size: Extensão do kernel, em coeficientes.
        conv_stride: Passo da convolução.
        conv_activation: Função de ativação da camada convolucional.
        conv_l2: Coeficiente de regularização L2 da convolução.
        dense_units: Unidades da camada densa oculta.
        dense_activation: Função de ativação da camada densa.
        dense_l2: Coeficiente de regularização L2 da camada densa.
        dense_dropout: Fração de dropout após a camada densa.

    Returns:
        Modelo compilado, pronto para treino.
    """
    return Sequential([
        Input(shape=input_shape),
        Conv1D(conv_filters, conv_kernel_size, strides=conv_stride, padding='valid',
               activation=conv_activation, kernel_regularizer=l2(conv_l2)),
        Flatten(),
        Dense(dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2)),
        Dropout(dense_dropout),
        Dense(num_classes, activation='softmax'),
    ], name='cepstral_cnn')


def build_temporal_cnn(
    input_shape: tuple[int, ...],
    num_classes: int,
    conv_filters: tuple[int, ...] = (64, 128),
    conv_kernel_size: int = 5,
    pool_size: int = 2,
    conv_activation: str = 'relu',
    conv_l2: float = 0.0,
    dense_units: int = 256,
    dense_activation: str = 'tanh',
    dense_l2: float = 0.04,
    dense_dropout: float = 0.3,
) -> Sequential:
    """Constrói a CNN que convolui ao longo do tempo, com agregação por média global.

    A transposição inicial coloca os quadros no eixo da convolução e os coeficientes
    como canais, de modo que os filtros são compartilhados ao longo do tempo. A
    agregação por ``GlobalAveragePooling1D`` substitui o achatamento e reduz cada
    mapa de ativação a um único valor.

    Essa substituição tem consequência direta sobre a comparabilidade das
    arquiteturas. Achatar uma sequência longa produz um vetor cujo comprimento cresce
    com a duração da entrada, e a camada densa seguinte passa a concentrar milhões de
    parâmetros — número que, em regimes de poucas amostras por classe, inviabiliza o
    treino. A média global torna o custo independente do comprimento, permitindo que
    a diferença observada entre as arquiteturas seja atribuída ao eixo escolhido, e
    não ao tamanho do modelo.

    Args:
        input_shape: Forma de uma amostra, ``(num_mfccs, num_quadros)``.
        num_classes: Número de locutores.
        conv_filters: Número de filtros de cada bloco convolucional, em ordem.
        conv_kernel_size: Extensão do kernel, em quadros.
        pool_size: Fator de subamostragem entre blocos.
        conv_activation: Função de ativação das camadas convolucionais.
        conv_l2: Coeficiente de regularização L2 das convoluções.
        dense_units: Unidades da camada densa oculta.
        dense_activation: Função de ativação da camada densa.
        dense_l2: Coeficiente de regularização L2 da camada densa.
        dense_dropout: Fração de dropout após a camada densa.

    Returns:
        Modelo compilado, pronto para treino.
    """
    layers = [
        Input(shape=input_shape),
        Permute((2, 1)),  # (num_mfccs, quadros) -> (quadros, num_mfccs)
    ]
    for filters in conv_filters:
        layers.append(Conv1D(filters, conv_kernel_size, padding='same',
                             activation=conv_activation, kernel_regularizer=l2(conv_l2)))
        layers.append(MaxPooling1D(pool_size))

    layers += [
        GlobalAveragePooling1D(),
        Dense(dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2)),
        Dropout(dense_dropout),
        Dense(num_classes, activation='softmax'),
    ]
    return Sequential(layers, name='temporal_cnn')
