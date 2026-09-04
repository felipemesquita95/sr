"""Testes das métricas de avaliação.

O que se verifica aqui não é o cálculo da acurácia — esse vem do scikit-learn —
mas o tratamento das classes **ausentes do conjunto de teste**. Com 80 ou 110
locutores e poucos enunciados por partição, classes sem amostra de teste são
situação normal, e é onde uma métrica mal construída passa a mentir.
"""

from __future__ import annotations

import numpy as np
import pytest

from sr.evaluation.metrics import EvaluationResult


def make_result(predictions, targets, num_classes: int) -> EvaluationResult:
    """Monta um resultado com as métricas agregadas zeradas.

    Os testes deste módulo exercitam apenas o que é derivado de ``predictions`` e
    ``targets``; os campos agregados vêm do scikit-learn e não são recalculados aqui.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return EvaluationResult(
        accuracy=float(np.mean(predictions == targets)),
        precision=0.0,
        recall=0.0,
        f1=0.0,
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
    )


def test_chance_level_is_the_inverse_of_the_class_count():
    result = make_result([0], [0], num_classes=80)

    assert result.chance_level == pytest.approx(0.0125)


def test_times_chance_scales_the_accuracy():
    """A leitura em múltiplos do acaso é o que dá escala ao resultado.

    Com 80 locutores, 57,5% de acurácia são 46 vezes o acaso — a razão é a forma
    honesta de comparar desempenhos entre corpora com números de classes distintos.
    """
    result = make_result([0, 0, 0, 0], [0, 0, 0, 1], num_classes=80)

    assert result.accuracy == pytest.approx(0.75)
    assert result.times_chance == pytest.approx(60.0)


def test_confusion_covers_every_class():
    """A matriz deve ter uma linha por classe do problema, não por classe testada.

    Sem fixar os rótulos, o scikit-learn dimensiona a matriz pelas classes que
    aparecem nos dados. A matriz de uma partição com 40 locutores testados ficaria
    40×40 e seria lida como se o problema tivesse 40 classes, desalinhando a figura
    do relatório em relação aos rótulos verdadeiros.
    """
    result = make_result([0, 2], [0, 2], num_classes=5)

    matrix = result.confusion()

    assert matrix.shape == (5, 5)
    assert matrix[0, 0] == 1
    assert matrix[2, 2] == 1
    # As classes 1, 3 e 4 não foram testadas: linhas nulas, e não ausentes.
    assert matrix[[1, 3, 4]].sum() == 0


def test_confusion_orients_rows_as_truth():
    """Linha é o rótulo verdadeiro e coluna o predito, como diz a documentação."""
    result = make_result(predictions=[1], targets=[0], num_classes=2)

    assert result.confusion()[0, 1] == 1
    assert result.confusion()[1, 0] == 0


def test_per_class_accuracy_marks_untested_classes_as_nan():
    """Classe sem amostra de teste deve ser NaN, nunca zero.

    Zero significaria "sempre errada", que é uma afirmação sobre o modelo. NaN
    significa "nunca testada", que é uma afirmação sobre a partição. Confundir as
    duas rebaixaria artificialmente a média por classe.
    """
    result = make_result(predictions=[0, 1], targets=[0, 0], num_classes=3)

    per_class = result.per_class_accuracy()

    assert per_class.shape == (3,)
    assert per_class[0] == pytest.approx(0.5)  # duas amostras, um acerto
    assert np.isnan(per_class[1])  # predita, mas nunca foi o rótulo verdadeiro
    assert np.isnan(per_class[2])  # ausente do teste


def test_per_class_accuracy_distinguishes_wrong_from_untested():
    """Uma classe testada e sempre errada vale zero, e não NaN."""
    result = make_result(predictions=[1], targets=[0], num_classes=2)

    per_class = result.per_class_accuracy()

    assert per_class[0] == 0.0
    assert np.isnan(per_class[1])
