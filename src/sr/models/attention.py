"""Arquitetura baseada em atenção sobre o eixo dos coeficientes cepstrais.

A motivação é geométrica. Convolução pressupõe **localidade**: que a informação
relevante se concentre em janelas de posições contíguas, e que a noção de "vizinho"
seja significativa. Essa premissa vale para o eixo do tempo, onde quadros adjacentes
são de fato instantes próximos, mas **não vale para o eixo cepstral**: os coeficientes
são projeções sobre bases distintas da transformada discreta do cosseno, e o índice
que os ordena é um número de base, não uma coordenada em um espaço métrico. Não há
sentido físico em somar os coeficientes 4 a 7 com pesos compartilhados, e trocar a
ordem dos coeficientes mudaria o resultado de uma convolução sem que nada no sinal
tivesse mudado.

Atenção não faz essa suposição. O mecanismo é **equivariante a permutação**: relações
entre pares quaisquer de posições são aprendidas diretamente, e a proximidade de
índice não confere nenhum privilégio. Para um eixo sem estrutura de vizinhança, é o
operador adequado — a dependência entre um coeficiente de ordem baixa e um de ordem
alta é modelada tão prontamente quanto entre dois adjacentes.

Um efeito colateral útil para a análise: os pesos de atenção formam uma matriz
``num_mfccs × num_mfccs`` diretamente inspecionável, que revela quais coeficientes o
modelo relaciona entre si.
"""

from __future__ import annotations

from keras import Model
from keras.layers import (Dense, Dropout, Flatten, Input, Layer, LayerNormalization, Permute,
                          MultiHeadAttention)
from keras.regularizers import l2
from keras.saving import register_keras_serializable

from sr.models.pooling import StatisticsPooling


@register_keras_serializable(package='sr')
class PositionalEmbedding(Layer):
    """Soma à entrada um viés aprendido, específico de cada posição.

    A atenção é equivariante a permutação, o que é desejável no eixo cepstral mas
    apaga por completo a identidade de cada coeficiente: sem esta camada, o modelo
    não distinguiria o coeficiente de ordem 2 do de ordem 30. O viés posicional
    devolve essa identidade sem reintroduzir a suposição de que índices próximos
    sejam semanticamente próximos — cada posição recebe um vetor independente, e não
    uma função suave do índice.
    """

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Cria a matriz de vieses, uma linha por posição do eixo de entrada.

        Args:
            input_shape: Forma da entrada, ``(lote, posições, dimensões)``.
        """
        self.positional_weights = self.add_weight(
            name='positional_weights',
            shape=(input_shape[1], input_shape[2]),
            initializer='random_normal',
            trainable=True,
        )

    def call(self, inputs):
        """Soma o viés posicional à entrada.

        Args:
            inputs: Tensor de forma ``(lote, posições, dimensões)``.

        Returns:
            Tensor de mesma forma, com o viés somado.
        """
        return inputs + self.positional_weights


def build_attention_model(
    input_shape: tuple[int, ...],
    num_classes: int,
    model_dim: int = 128,
    num_heads: int = 4,
    num_blocks: int = 2,
    ffn_units: int = 256,
    attention_dropout: float = 0.1,
    dense_units: int = 256,
    dense_activation: str = 'tanh',
    dense_l2: float = 0.04,
    dense_dropout: float = 0.1,
) -> Model:
    """Constrói o codificador por atenção sobre o eixo dos coeficientes cepstrais.

    A entrada ``(num_mfccs, num_quadros)`` é tratada como uma sequência de
    ``num_mfccs`` posições, cada uma descrita pela sua trajetória ao longo dos quadros.
    A projeção inicial leva essa trajetória a uma representação de dimensão
    ``model_dim``, e os blocos de atenção subsequentes aprendem, para cada coeficiente,
    quais outros coeficientes são informativos a seu respeito.

    Cada bloco segue a formulação usual: atenção multi-cabeça com conexão residual e
    normalização de camada, seguida de uma rede densa posição a posição com a mesma
    estrutura residual.

    Args:
        input_shape: Forma de uma amostra, ``(num_mfccs, num_quadros)``.
        num_classes: Número de locutores.
        model_dim: Dimensão da representação interna.
        num_heads: Número de cabeças de atenção por bloco.
        num_blocks: Número de blocos empilhados.
        ffn_units: Unidades da camada oculta da rede densa de cada bloco.
        attention_dropout: Fração de dropout na atenção e na rede densa dos blocos.
        dense_units: Unidades da camada densa de classificação.
        dense_activation: Função de ativação da camada densa de classificação.
        dense_l2: Coeficiente de regularização L2 da camada densa.
        dense_dropout: Fração de dropout antes da camada de saída.

    Returns:
        Modelo compilado, pronto para treino.
    """
    inputs = Input(shape=input_shape)

    x = Dense(model_dim)(inputs)
    x = PositionalEmbedding()(x)

    for _ in range(num_blocks):
        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=attention_dropout,
        )(x, x)
        x = LayerNormalization()(x + attention)

        projected = Dense(ffn_units, activation='relu')(x)
        projected = Dropout(attention_dropout)(projected)
        projected = Dense(model_dim)(projected)
        x = LayerNormalization()(x + projected)

    x = Flatten()(x)
    x = Dense(dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2))(x)
    x = Dropout(dense_dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='cepstral_attention')


def build_temporal_attention(
    input_shape: tuple[int, ...],
    num_classes: int,
    model_dim: int = 128,
    num_heads: int = 4,
    num_blocks: int = 2,
    ffn_units: int = 256,
    attention_dropout: float = 0.1,
    dense_units: int = 256,
    dense_activation: str = 'tanh',
    dense_l2: float = 0.04,
    dense_dropout: float = 0.1,
) -> Model:
    """Atenção sobre o eixo do **tempo**, e não sobre o dos coeficientes cepstrais.

    Preenche a célula que faltava na grade que o trabalho declara como objeto de
    estudo. Com esta arquitetura, eixo e mecanismo passam a variar independentemente:

    ==============  ======================  =====================
    mecanismo       eixo cepstral           eixo temporal
    ==============  ======================  =====================
    convolução      ``cnn``                 ``temporal_cnn``
    atenção         ``attention``           ``temporal_attention``
    ==============  ======================  =====================

    A distinção importa porque os dois eixos têm naturezas opostas. O cepstral não
    possui estrutura de vizinhança — coeficientes adjacentes são projeções de bases
    distintas da DCT —, e é por isso que a atenção, equivariante a permutação, foi
    proposta ali. O eixo temporal tem vizinhança e ordem, e a atenção sobre ele
    responde a outra pergunta: quais **instantes** da gravação carregam identidade.

    Essa pergunta não é neutra neste trabalho. Se o modelo puder escolher onde olhar,
    e a identidade se predisser melhor a partir dos trechos sem fala — como o
    diagnóstico de canal indica —, a atenção temporal é o mecanismo mais capaz de
    explorar o confundidor, e não uma defesa contra ele. Um resultado alto dentro de
    um mesmo microfone deve ser lido com essa possibilidade em mente.

    A agregação é por média e desvio, e não por achatamento. Achatar o eixo temporal
    produziria um vetor proporcional à duração e concentraria milhões de parâmetros na
    camada seguinte, o que confundiria o efeito do eixo com o do tamanho do modelo —
    exatamente o que a comparação existe para separar. Também é a agregação usada pela
    ``temporal_cnn_stats`` e pelo ``xvector``, o que mantém o contraste sobre o
    mecanismo.

    O viés posicional é mantido. A atenção continua equivariante a permutação, e sem
    ele a rede não distinguiria o início do fim da gravação.

    Args:
        input_shape: Forma de uma amostra, ``(num_mfccs, num_quadros)``.
        num_classes: Número de locutores.
        model_dim: Dimensão da representação interna.
        num_heads: Número de cabeças de atenção por bloco.
        num_blocks: Número de blocos empilhados.
        ffn_units: Unidades da camada oculta da rede densa de cada bloco.
        attention_dropout: Fração de dropout na atenção e na rede densa dos blocos.
        dense_units: Unidades da camada densa de classificação.
        dense_activation: Função de ativação da camada densa de classificação.
        dense_l2: Coeficiente de regularização L2 da camada densa.
        dense_dropout: Fração de dropout antes da camada de saída.

    Returns:
        Modelo compilado, pronto para treino.
    """
    inputs = Input(shape=input_shape)

    # (coeficientes, quadros) -> (quadros, coeficientes): os quadros viram as posições
    # atendidas, e cada um é descrito pelo seu vetor cepstral.
    x = Permute((2, 1))(inputs)
    x = Dense(model_dim)(x)
    x = PositionalEmbedding()(x)

    for _ in range(num_blocks):
        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=model_dim // num_heads,
            dropout=attention_dropout,
        )(x, x)
        x = LayerNormalization()(x + attention)

        projected = Dense(ffn_units, activation='relu')(x)
        projected = Dropout(attention_dropout)(projected)
        projected = Dense(model_dim)(projected)
        x = LayerNormalization()(x + projected)

    x = StatisticsPooling()(x)
    x = Dense(dense_units, activation=dense_activation, kernel_regularizer=l2(dense_l2))(x)
    x = Dropout(dense_dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='temporal_attention')
