"""Agregação de uma sequência temporal em um vetor de comprimento fixo.

A camada de agregação é onde uma rede de locutor decide **o que de uma gravação
inteira importa**, e a escolha tem consequência direta sobre o que o modelo pode
aprender.

``GlobalAveragePooling1D``, usada pela CNN temporal, reduz cada canal à sua média ao
longo do tempo — e com isso descarta a variância. Isso é uma perda concreta neste
problema: parte substancial do que distingue locutores está em **quanto** uma
característica oscila ao longo da fala, não apenas em seu valor médio. Duas vozes com
a mesma coloração espectral média e dispersões diferentes ficam indistinguíveis
depois da média.

A agregação por estatísticas (:class:`StatisticsPooling`) concatena média e desvio,
recuperando essa dimensão descartada ao custo de dobrar a largura da saída. É a
agregação padrão das arquiteturas de referência em reconhecimento de locutor, e a
diferença entre ela e a média simples isola uma única variável — o que a torna uma
ablação limpa, e não apenas uma troca de modelo.
"""

from __future__ import annotations

from keras import ops
from keras.layers import Dense, Layer
from keras.saving import register_keras_serializable

#: Piso da variância antes da raiz quadrada.
#:
#: Não é cosmético: a derivada de ``sqrt`` diverge em zero, e um canal constante ao
#: longo do tempo — que ocorre de fato em trechos de silêncio — produziria gradiente
#: infinito e derrubaria o treino com NaN.
VARIANCE_FLOOR = 1e-8


@register_keras_serializable(package='sr')
class StatisticsPooling(Layer):
    """Agrega a sequência em média e desvio padrão por canal, concatenados.

    A entrada tem forma ``(lote, passos, canais)`` e a saída ``(lote, 2 * canais)``:
    as médias de cada canal seguidas dos seus desvios.
    """

    def call(self, inputs):
        """Calcula média e desvio ao longo do eixo temporal.

        Args:
            inputs: Tensor de forma ``(lote, passos, canais)``.

        Returns:
            Tensor de forma ``(lote, 2 * canais)``.
        """
        mean = ops.mean(inputs, axis=1)
        centered = inputs - ops.expand_dims(mean, axis=1)
        variance = ops.mean(ops.square(centered), axis=1)
        deviation = ops.sqrt(ops.maximum(variance, VARIANCE_FLOOR))
        return ops.concatenate([mean, deviation], axis=-1)

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Forma da saída, com a largura dobrada pela concatenação."""
        return (input_shape[0], input_shape[2] * 2)


@register_keras_serializable(package='sr')
class AttentiveStatisticsPooling(Layer):
    """Agrega em média e desvio ponderados por atenção, calculada por canal.

    A agregação por estatísticas trata todos os quadros como igualmente informativos,
    o que não é verdade: quadros de silêncio, de plosivas e de vogais sustentadas
    contribuem de maneira muito diferente para a identificação. Aqui o modelo aprende
    um peso por quadro **e por canal**, e a média e o desvio passam a ser ponderados.

    Há uma tensão que o trabalho precisa registrar: a atenção pode perfeitamente
    aprender a **privilegiar** os quadros de silêncio, se for deles que a identidade
    do locutor for mais facilmente predita. A camada não é uma defesa contra o
    confundidor de canal — é um mecanismo mais expressivo, e por isso mesmo mais
    capaz de explorá-lo.

    Args:
        attention_units: Largura da camada oculta que calcula os pesos.
    """

    def __init__(self, attention_units: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attention_units = attention_units

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Cria as duas projeções que produzem os pesos de atenção.

        Args:
            input_shape: Forma da entrada, ``(lote, passos, canais)``.
        """
        self.hidden = Dense(self.attention_units, activation='tanh')
        self.scores = Dense(input_shape[-1])
        self.hidden.build(input_shape)
        self.scores.build((*input_shape[:-1], self.attention_units))
        super().build(input_shape)

    def call(self, inputs):
        """Calcula média e desvio ponderados pela atenção.

        Args:
            inputs: Tensor de forma ``(lote, passos, canais)``.

        Returns:
            Tensor de forma ``(lote, 2 * canais)``.
        """
        logits = self.scores(self.hidden(inputs))
        weights = ops.softmax(logits, axis=1)

        mean = ops.sum(weights * inputs, axis=1)
        squared = ops.sum(weights * ops.square(inputs), axis=1)
        variance = squared - ops.square(mean)
        deviation = ops.sqrt(ops.maximum(variance, VARIANCE_FLOOR))
        return ops.concatenate([mean, deviation], axis=-1)

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Forma da saída, com a largura dobrada pela concatenação."""
        return (input_shape[0], input_shape[2] * 2)

    def get_config(self) -> dict:
        """Configuração serializável da camada."""
        return {**super().get_config(), 'attention_units': self.attention_units}
