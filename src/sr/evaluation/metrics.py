"""Cálculo das métricas de desempenho sobre o conjunto de teste.

A métrica principal é a acurácia, mas ela sozinha é insuficiente em um problema de
muitas classes: uma acurácia alta pode conviver com classes inteiras nunca preditas.
Por isso o F1 macro é reportado em paralelo — por dar peso igual a cada locutor,
independentemente de quantos enunciados ele contribua, ele expõe esse desequilíbrio
que a acurácia global esconde.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Desempenho de um modelo em um conjunto de teste.

    Attributes:
        accuracy: Fração de acertos, entre 0 e 1.
        precision: Precisão média macro.
        recall: Revocação média macro.
        f1: F1 médio macro.
        predictions: Rótulos preditos, base zero.
        targets: Rótulos verdadeiros, base zero.
        num_classes: Número total de classes do problema.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    predictions: np.ndarray
    targets: np.ndarray
    num_classes: int

    @property
    def chance_level(self) -> float:
        """Acurácia esperada de um classificador aleatório, entre 0 e 1."""
        return 1.0 / self.num_classes

    @property
    def times_chance(self) -> float:
        """Quantas vezes a acurácia obtida supera o nível do acaso.

        Em problemas com muitas classes, é a leitura que dá escala ao resultado: 20%
        de acurácia soa baixo, mas com 110 locutores equivale a vinte e duas vezes o
        acaso, e portanto a um sinal real e substancial.
        """
        return self.accuracy / self.chance_level

    def confusion(self) -> np.ndarray:
        """Matriz de confusão, com todas as classes representadas.

        Returns:
            Matriz de forma ``(num_classes, num_classes)``, em que a linha é o rótulo
            verdadeiro e a coluna o predito. Classes ausentes do teste aparecem como
            linhas nulas, e não são omitidas.
        """
        return confusion_matrix(
            self.targets, self.predictions, labels=np.arange(self.num_classes))

    def per_class_accuracy(self) -> np.ndarray:
        """Acurácia de cada classe individualmente.

        Returns:
            Vetor de tamanho ``num_classes``, com ``NaN`` nas classes sem amostras de
            teste — distinguindo "nunca testada" de "sempre errada", que a acurácia
            global confunde.
        """
        matrix = self.confusion()
        support = matrix.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(support > 0, np.diag(matrix) / support, np.nan)

    def describe(self) -> str:
        """Resumo textual das métricas, para registro no log."""
        return (f'acurácia={self.accuracy * 100:.2f}% '
                f'({self.times_chance:.1f}× o acaso de {self.chance_level * 100:.2f}%), '
                f'F1={self.f1:.4f}')


def evaluate_model(model, test_x: np.ndarray, test_y: np.ndarray, num_classes: int) -> EvaluationResult:
    """Avalia o modelo no conjunto de teste.

    Esta é a única leitura do conjunto de teste em todo o experimento. Ele não
    participou da escolha da época de parada nem das estatísticas de normalização,
    de modo que a acurácia aqui obtida estima desempenho em dados novos.

    Args:
        model: Modelo treinado.
        test_x: Tensor de teste.
        test_y: Rótulos verdadeiros, base zero.
        num_classes: Número de classes do problema.

    Returns:
        As métricas e as predições, para posterior geração de figuras.
    """
    probabilities = model.predict(test_x, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        test_y, predictions, output_dict=True, zero_division=0)
    accuracy = float(np.mean(predictions == test_y))

    result = EvaluationResult(
        accuracy=accuracy,
        precision=float(report['macro avg']['precision']),
        recall=float(report['macro avg']['recall']),
        f1=float(report['macro avg']['f1-score']),
        predictions=predictions,
        targets=test_y,
        num_classes=num_classes,
    )
    logger.info('Avaliação: %s', result.describe())
    return result
