"""Geração dos relatórios e figuras de resultado.

Cada partição produz um relatório próprio — métricas, matriz de confusão, acurácia
por locutor e curvas de treino — e o conjunto das partições produz um resumo com
média e desvio. O desvio entre partições não é um detalhe de apresentação: em
conjuntos pequenos ele costuma ser de vários pontos percentuais, e reportar apenas
a média de uma única partição transmite uma precisão que o experimento não tem.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sr.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)


def write_fold_report(result: EvaluationResult, history, output: Path) -> None:
    """Grava métricas e figuras de uma partição.

    Args:
        result: Métricas e predições da partição.
        history: Histórico de treino devolvido pelo Keras, ou ``None``.
        output: Diretório onde gravar; criado se não existir.
    """
    output.mkdir(parents=True, exist_ok=True)

    metrics = {
        'acuracia': result.accuracy,
        'precisao_macro': result.precision,
        'revocacao_macro': result.recall,
        'f1_macro': result.f1,
        'acaso': result.chance_level,
        'vezes_o_acaso': result.times_chance,
        'num_classes': result.num_classes,
        'num_amostras_teste': int(len(result.targets)),
    }
    (output / 'metricas.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    _plot_confusion_matrix(result, output / 'matriz_confusao.png')
    _plot_per_class_accuracy(result, output / 'acuracia_por_locutor.png')
    if history is not None:
        _plot_training_curves(history, output / 'curvas_treino.png')

    logger.info('Relatório da partição gravado em %s', output)


def write_summary(
    architecture: str,
    accuracies: list[float],
    f1_scores: list[float],
    chance_level: float,
    output: Path,
) -> None:
    """Grava o resumo de uma arquitetura sobre todas as partições executadas.

    Args:
        architecture: Nome da arquitetura avaliada.
        accuracies: Acurácia de cada partição, entre 0 e 1.
        f1_scores: F1 macro de cada partição.
        chance_level: Acurácia do classificador aleatório, entre 0 e 1.
        output: Diretório onde gravar.
    """
    output.mkdir(parents=True, exist_ok=True)
    accuracy = np.asarray(accuracies)
    f1 = np.asarray(f1_scores)

    lines = [f'Arquitetura: {architecture}', '']
    for index, (a, f) in enumerate(zip(accuracy, f1), start=1):
        lines.append(f'Partição {index}: acurácia = {a * 100:.2f}%, F1 = {f:.4f}')
    lines += [
        '',
        f'Acurácia média: {accuracy.mean() * 100:.2f}% ± {accuracy.std() * 100:.2f}',
        f'F1 macro médio: {f1.mean():.4f} ± {f1.std():.4f}',
        f'Acaso: {chance_level * 100:.2f}%',
        f'Razão sobre o acaso: {accuracy.mean() / chance_level:.1f}×',
    ]
    if len(accuracy) == 1:
        lines.append('')
        lines.append('Atenção: uma única partição executada; o desvio não é estimável.')

    (output / 'resumo.txt').write_text('\n'.join(lines) + '\n')
    (output / 'resumo.json').write_text(json.dumps({
        'arquitetura': architecture,
        'acuracias': [float(a) for a in accuracy],
        'f1_scores': [float(f) for f in f1],
        'acuracia_media': float(accuracy.mean()),
        'acuracia_desvio': float(accuracy.std()),
        'f1_medio': float(f1.mean()),
        'f1_desvio': float(f1.std()),
        'acaso': chance_level,
    }, indent=2, ensure_ascii=False))

    _plot_fold_summary(architecture, accuracy, f1, chance_level, output / 'resumo_particoes.png')

    logger.info('%s: acurácia média %.2f%% ± %.2f (%d partições).',
                architecture, accuracy.mean() * 100, accuracy.std() * 100, len(accuracy))


def _plot_confusion_matrix(result: EvaluationResult, output: Path) -> None:
    """Desenha a matriz de confusão normalizada por linha.

    A normalização por linha é o que torna a figura legível com muitas classes: sem
    ela, locutores com mais enunciados de teste dominam a escala de cor e a diagonal
    aparenta uniformidade que não existe.
    """
    matrix = result.confusion().astype(float)
    support = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.where(support > 0, matrix / support, 0.0)

    size = max(8, result.num_classes * 0.08)
    plt.figure(figsize=(size, size))
    plt.imshow(normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Fração das amostras do locutor')
    plt.title(f'Matriz de confusão — acurácia {result.accuracy * 100:.2f}%')
    plt.xlabel('Locutor predito')
    plt.ylabel('Locutor verdadeiro')
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_per_class_accuracy(result: EvaluationResult, output: Path) -> None:
    """Desenha a acurácia de cada locutor, ordenada pelo índice do locutor."""
    per_class = result.per_class_accuracy()

    plt.figure(figsize=(max(10, result.num_classes * 0.12), 5))
    plt.bar(np.arange(1, result.num_classes + 1), per_class * 100, color='steelblue')
    plt.axhline(np.nanmean(per_class) * 100, color='crimson', linestyle='--',
                label=f'média {np.nanmean(per_class) * 100:.1f}%')
    plt.axhline(result.chance_level * 100, color='gray', linestyle=':',
                label=f'acaso {result.chance_level * 100:.2f}%')
    plt.title('Acurácia por locutor no conjunto de teste')
    plt.xlabel('Locutor')
    plt.ylabel('Acurácia (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_training_curves(history, output: Path) -> None:
    """Desenha as curvas de perda e acurácia de treino e validação.

    A distância entre as duas curvas é o diagnóstico visual de sobreajuste, e é o que
    justifica a parada antecipada adotada.
    """
    epochs = range(1, len(history.history['loss']) + 1)

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(14, 5))

    loss_axis.plot(epochs, history.history['loss'], label='treino')
    loss_axis.plot(epochs, history.history.get('val_loss', []), label='validação')
    loss_axis.set_title('Perda')
    loss_axis.set_xlabel('Época')
    loss_axis.set_ylabel('Entropia cruzada')
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)

    accuracy_axis.plot(epochs, history.history['accuracy'], label='treino')
    accuracy_axis.plot(epochs, history.history.get('val_accuracy', []), label='validação')
    accuracy_axis.set_title('Acurácia')
    accuracy_axis.set_xlabel('Época')
    accuracy_axis.set_ylabel('Acurácia')
    accuracy_axis.legend()
    accuracy_axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_fold_summary(
    architecture: str,
    accuracy: np.ndarray,
    f1: np.ndarray,
    chance_level: float,
    output: Path,
) -> None:
    """Desenha acurácia e F1 de cada partição, com a média e o acaso como referência."""
    positions = np.arange(1, len(accuracy) + 1)
    width = 0.38

    plt.figure(figsize=(max(8, len(positions) * 1.2), 5))
    plt.bar(positions - width / 2, accuracy * 100, width, label='Acurácia', color='steelblue')
    plt.bar(positions + width / 2, f1 * 100, width, label='F1 macro', color='darkorange')
    plt.axhline(accuracy.mean() * 100, color='crimson', linestyle='--',
                label=f'acurácia média {accuracy.mean() * 100:.1f}%')
    plt.axhline(chance_level * 100, color='gray', linestyle=':',
                label=f'acaso {chance_level * 100:.2f}%')
    plt.xticks(positions, [f'Partição {p}' for p in positions])
    plt.title(f'Arquitetura {architecture}: resultado por partição')
    plt.ylabel('%')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
