"""Rede temporal com dilatação — a arquitetura de referência do campo.

O ``x-vector`` (Snyder et al., 2018) é o baseline convencional em reconhecimento de
locutor, e a sua ausência é uma fragilidade concreta em um trabalho que se propõe a
discutir o que a área mede: sem ele, qualquer resultado pode ser descartado como
artefato de uma arquitetura idiossincrática.

A estrutura tem três partes, e cada uma responde a uma exigência do problema:

1. **Convoluções dilatadas ao longo do tempo.** As dilatações crescentes ampliam o
   campo receptivo sem multiplicar parâmetros, de modo que camadas rasas enxergam
   contexto de centenas de milissegundos — a escala em que traços de locutor se
   manifestam, maior que a de um fonema e menor que a de uma frase.

2. **Agregação por estatísticas.** A rede opera quadro a quadro e precisa produzir
   uma decisão por gravação. Média e desvio ao longo do tempo fazem essa ponte, e o
   desvio é essencial: é ele que carrega a dispersão que a média descarta.

3. **Camadas densas de embedding.** Após a agregação, duas camadas estreitam a
   representação antes da classificação. É delas que, no uso original, se extrai o
   vetor de locutor.

Uma observação que este trabalho precisa registrar: o x-vector foi projetado quando
o confundidor de canal já era conhecido, e a agregação por estatísticas **não** o
resolve. Média e desvio de um trecho de silêncio descrevem o canal tão bem quanto
descrevem a voz. A arquitetura é mais capaz, não mais honesta.
"""

from __future__ import annotations

from keras import Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Input, Permute
from keras.regularizers import l2

from sr.models.pooling import AttentiveStatisticsPooling, StatisticsPooling

#: Dilatações de cada bloco convolucional, na formulação original.
DEFAULT_DILATIONS = (1, 2, 3, 1, 1)

#: Extensão do kernel de cada bloco, pareada com :data:`DEFAULT_DILATIONS`.
DEFAULT_KERNELS = (5, 3, 3, 1, 1)


def build_xvector(
    input_shape: tuple[int, ...],
    num_classes: int,
    filters: tuple[int, ...] = (256, 256, 256, 256, 768),
    kernels: tuple[int, ...] = DEFAULT_KERNELS,
    dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    embedding_units: int = 256,
    dense_l2: float = 0.0,
    dropout: float = 0.2,
    attentive: bool = False,
) -> Sequential:
    """Constrói a rede temporal dilatada com agregação por estatísticas.

    Os filtros são reduzidos em relação à formulação original (512 e 1500), que foi
    dimensionada para milhares de horas de fala. Aqui o regime é de dezenas de
    minutos por locutor, e a capacidade original levaria a decorar o conjunto de
    treino em poucas épocas.

    Args:
        input_shape: Forma de uma amostra, ``(num_mfccs, num_quadros)``.
        num_classes: Número de locutores.
        filters: Número de filtros de cada bloco.
        kernels: Extensão do kernel de cada bloco.
        dilations: Fator de dilatação de cada bloco.
        embedding_units: Largura das camadas densas após a agregação.
        dense_l2: Regularização L2 das camadas densas.
        dropout: Fração de dropout entre as camadas densas.
        attentive: Se verdadeiro, pondera a agregação por atenção sobre os quadros.

    Returns:
        Modelo pronto para compilação.

    Raises:
        ValueError: Se ``filters``, ``kernels`` e ``dilations`` tiverem comprimentos
            diferentes — o que produziria uma rede silenciosamente distinta da pedida.
    """
    if not (len(filters) == len(kernels) == len(dilations)):
        raise ValueError(
            f'filters, kernels e dilations devem ter o mesmo comprimento; '
            f'recebi {len(filters)}, {len(kernels)} e {len(dilations)}.')

    layers = [
        Input(shape=input_shape),
        Permute((2, 1)),  # (coeficientes, quadros) -> (quadros, coeficientes)
    ]
    for size, kernel, dilation in zip(filters, kernels, dilations):
        layers.append(Conv1D(size, kernel, dilation_rate=dilation,
                             padding='same', activation='relu'))
        layers.append(BatchNormalization())

    layers.append(AttentiveStatisticsPooling() if attentive else StatisticsPooling())
    layers += [
        Dense(embedding_units, activation='relu', kernel_regularizer=l2(dense_l2)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(embedding_units, activation='relu', kernel_regularizer=l2(dense_l2)),
        Dropout(dropout),
        Dense(num_classes, activation='softmax'),
    ]
    return Sequential(layers, name='xvector_attentive' if attentive else 'xvector')


def build_attentive_xvector(input_shape: tuple[int, ...], num_classes: int, **kwargs):
    """Variante do x-vector com agregação ponderada por atenção sobre os quadros.

    A diferença em relação a :func:`build_xvector` é uma única camada, o que torna a
    comparação entre as duas uma ablação da agregação — e não uma troca de modelo.

    Args:
        input_shape: Forma de uma amostra.
        num_classes: Número de locutores.
        **kwargs: Repassados a :func:`build_xvector`.

    Returns:
        Modelo pronto para compilação.
    """
    return build_xvector(input_shape, num_classes, attentive=True, **kwargs)
