"""Treinamento das arquiteturas, com parada antecipada guiada pela validação.

O ponto crítico deste subsistema é qual conjunto governa as decisões de treino.
A parada antecipada e a redução da taxa de aprendizado observam **exclusivamente a
validação**. Usar o teste nessa função — como é fácil fazer passando-o como
``validation_data`` — faz com que a melhor época seja escolhida medindo no próprio
conjunto de avaliação, e a acurácia reportada deixa de ser uma estimativa de
desempenho para se tornar o máximo obtido sobre o teste, sistematicamente otimista.
"""

from __future__ import annotations

import logging
from pathlib import Path

from keras import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import set_random_seed

from sr.config import Settings
from sr.features import DataSplit
from sr.models import build_model
from sr.training.callbacks import ProgressRecorder

logger = logging.getLogger(__name__)

#: Fator de redução da taxa de aprendizado quando a validação estagna.
LR_REDUCTION_FACTOR = 0.2

#: Piso da taxa de aprendizado.
MIN_LEARNING_RATE = 1e-6


class TrainingSubsystem:
    """Constrói e treina uma arquitetura sobre uma partição dos dados.

    Args:
        settings: Configuração do experimento.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def train(self, architecture: str, split: DataSplit, output: Path,
              fold: int = 1, *, seed: int | None = None) -> tuple[Model, object]:
        """Treina uma arquitetura sobre uma partição.

        Args:
            architecture: Nome da arquitetura, conforme o registro de modelos.
            split: Conjuntos de treino, validação e teste já normalizados.
            output: Diretório onde gravar o modelo e os artefatos de acompanhamento.
            fold: Índice da partição, usado apenas para rotular os artefatos.
            seed: Semente de Python, NumPy e backend, aplicada antes de construir
                a rede e embaralhar os lotes. Omitida, preserva os protocolos antigos.

        Returns:
            O modelo treinado, com os pesos da melhor época restaurados, e o
            histórico de treino devolvido pelo Keras.
        """
        output.mkdir(parents=True, exist_ok=True)

        if seed is not None:
            set_random_seed(seed)

        model = build_model(
            architecture,
            input_shape=split.input_shape,
            num_classes=self.settings.num_speakers,
            learning_rate=self.settings.learning_rate,
        )

        history = model.fit(
            split.train_x, split.train_y,
            validation_data=(split.validation_x, split.validation_y),
            epochs=self.settings.epochs,
            batch_size=self.settings.batch_size,
            callbacks=self._callbacks(architecture, output, fold),
            verbose=2,
        )

        model.save(output / 'modelo.keras')
        logger.info('Modelo gravado em %s', output / 'modelo.keras')
        return model, history

    def _callbacks(self, architecture: str, output: Path, fold: int) -> list:
        """Monta a lista de callbacks do treino.

        A parada antecipada monitora a acurácia de validação e restaura os pesos da
        melhor época, de modo que o modelo avaliado não é o da última época — que,
        após a paciência decorrida, é por construção pior que o melhor observado.

        Args:
            architecture: Nome da arquitetura, usado nos artefatos de acompanhamento.
            output: Diretório dos artefatos.
            fold: Índice da partição.

        Returns:
            Lista de callbacks do Keras.
        """
        patience = self.settings.early_stopping_patience

        return [
            ReduceLROnPlateau(
                monitor='val_accuracy', mode='max',
                factor=LR_REDUCTION_FACTOR, patience=patience,
                min_lr=MIN_LEARNING_RATE, verbose=1,
            ),
            EarlyStopping(
                monitor='val_accuracy', mode='max',
                patience=patience, restore_best_weights=True, verbose=1,
            ),
            ProgressRecorder(output, architecture, fold),
        ]
