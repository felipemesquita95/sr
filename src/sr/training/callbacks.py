"""Callbacks de acompanhamento do treino.

Treinos longos precisam ser observáveis sem interromper a execução: os experimentos
deste trabalho levam horas, e descobrir apenas ao final que a perda divergiu na
terceira época desperdiça o restante do tempo.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import Callback

logger = logging.getLogger(__name__)


class ProgressRecorder(Callback):
    """Grava o andamento do treino em disco a cada época.

    Mantém dois artefatos atualizados: um PNG com as curvas correntes, que pode ser
    aberto em um visualizador enquanto o treino corre, e um JSON com o estado, para
    consumo por outros programas.

    A escrita do JSON é atômica — grava em arquivo temporário e renomeia — de modo
    que um leitor concorrente nunca observe um documento pela metade.

    Args:
        output: Diretório onde gravar os artefatos.
        architecture: Nome da arquitetura em treino, usado nos títulos.
        fold: Índice da partição em treino.
    """

    def __init__(self, output: Path, architecture: str, fold: int) -> None:
        super().__init__()
        self.output = output
        self.architecture = architecture
        self.fold = fold
        self.history: dict[str, list[float]] = {
            'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [],
        }

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Registra as métricas da época e reescreve os artefatos de acompanhamento.

        Args:
            epoch: Índice da época concluída, base zero.
            logs: Métricas da época, fornecidas pelo Keras.
        """
        logs = logs or {}
        for key in self.history:
            self.history[key].append(float(logs.get(key, float('nan'))))

        self._write_status(epoch + 1)
        self._write_curves()

    def _write_status(self, epoch: int) -> None:
        """Grava o estado corrente em JSON, de forma atômica."""
        payload = {
            'arquitetura': self.architecture,
            'particao': self.fold,
            'epoca': epoch,
            'historico': self.history,
        }
        temporary = self.output / 'progresso.json.tmp'
        temporary.write_text(json.dumps(payload, ensure_ascii=False))
        temporary.replace(self.output / 'progresso.json')

    def _write_curves(self) -> None:
        """Reescreve o PNG com as curvas correntes de perda e acurácia."""
        epochs = range(1, len(self.history['loss']) + 1)

        figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(14, 5))
        figure.suptitle(f'{self.architecture} — partição {self.fold}')

        loss_axis.plot(epochs, self.history['loss'], label='treino')
        loss_axis.plot(epochs, self.history['val_loss'], label='validação')
        loss_axis.set_xlabel('Época')
        loss_axis.set_ylabel('Perda')
        loss_axis.legend()
        loss_axis.grid(True, alpha=0.3)

        accuracy_axis.plot(epochs, self.history['accuracy'], label='treino')
        accuracy_axis.plot(epochs, self.history['val_accuracy'], label='validação')
        accuracy_axis.set_xlabel('Época')
        accuracy_axis.set_ylabel('Acurácia')
        accuracy_axis.legend()
        accuracy_axis.grid(True, alpha=0.3)

        figure.tight_layout()
        figure.savefig(self.output / 'progresso.png', dpi=120)
        plt.close(figure)
