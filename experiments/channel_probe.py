#!/usr/bin/env python3
"""Diagnóstico de canal: quanto da identidade do locutor está fora da fala.

O experimento responde a uma pergunta que a acurácia de um classificador de locutor
não responde sozinha: **a partir de que parte do sinal a decisão está sendo tomada?**

Um classificador é treinado três vezes, sobre três recortes do mesmo áudio:

``silence``
    Somente os trechos **sem** voz. Se a identidade do locutor puder ser predita
    daqui, ela não está sendo lida da fala — está na assinatura do equipamento, do
    ambiente e do ruído de fundo. É o controle negativo do experimento principal.

``speech``
    Somente os trechos **com** voz, com o silêncio interno removido. É a condição em
    que a decisão só pode se apoiar em conteúdo vocal.

``full``
    O sinal completo, como usado no experimento principal. Note que esta condição
    **contém** a anterior e o silêncio: se o silêncio identifica o locutor, a
    condição completa tem acesso a essa pista mesmo quando há fala em abundância.

A comparação informativa é ``full`` contra ``speech``. Comparar ``full`` contra
``silence`` isola menos do que parece, justamente porque a primeira contém a segunda.

O classificador é deliberadamente simples — regressão logística sobre a média e o
desvio de cada coeficiente cepstral. Uma rede profunda poderia extrair a assinatura
de canal de formas mais sutis, então um resultado positivo aqui é um limite
*inferior* para o tamanho do efeito, e não uma estimativa dele.

Uso::

    SR_CONFIG=configs/brsd.env python experiments/channel_probe.py
    SR_CONFIG=configs/brsd.env python experiments/channel_probe.py --condition speech
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from sr.config import Settings, load_settings  # noqa: E402
from sr.datasets import build_index  # noqa: E402
from sr.diagnostics import CONDITIONS, channel_signature, extract_condition  # noqa: E402

logger = logging.getLogger('channel_probe')


def build_features(settings: Settings, condition: str) -> dict[tuple[int, int], np.ndarray]:
    """Extrai a assinatura de cada gravação do corpus, sob a condição pedida.

    Args:
        settings: Configuração do experimento.
        condition: Recorte de sinal a utilizar.

    Returns:
        Mapeamento de ``(locutor, enunciado)`` para vetor de assinatura.
    """
    import librosa

    recordings = build_index(settings)
    features: dict[tuple[int, int], np.ndarray] = {}
    skipped = 0

    for position, recording in enumerate(recordings, start=1):
        audio, _ = librosa.load(recording.path, sr=settings.source_sampling_rate)
        segment = extract_condition(audio, condition, settings.vad_top_db)
        signature = channel_signature(segment, settings.source_sampling_rate, settings.num_mfccs)

        if signature is None:
            skipped += 1
            continue

        features[(recording.speaker, recording.utterance)] = signature
        if position % 50 == 0:
            logger.info('  %d/%d gravações processadas.', position, len(recordings))

    if skipped:
        logger.warning('%d gravações sem %s suficiente foram descartadas.', skipped, condition)
    return features


def run_cross_validation(
    features: dict[tuple[int, int], np.ndarray],
    num_folds: int,
) -> np.ndarray:
    """Avalia a separabilidade dos locutores por validação cruzada.

    A partição espelha a do experimento principal — grupos definidos pela posição do
    enunciado dentro de cada locutor — para que os números sejam comparáveis.

    Args:
        features: Assinaturas indexadas por ``(locutor, enunciado)``.
        num_folds: Número de partições.

    Returns:
        Vetor com a acurácia de cada partição, entre 0 e 1.
    """
    utterances_by_speaker: dict[int, list[int]] = {}
    for speaker, utterance in features:
        utterances_by_speaker.setdefault(speaker, []).append(utterance)

    fold_of: dict[tuple[int, int], int] = {}
    for speaker, utterances in utterances_by_speaker.items():
        for position, utterance in enumerate(sorted(utterances)):
            fold_of[(speaker, utterance)] = position % num_folds

    accuracies = []
    for fold in range(num_folds):
        train_x, train_y, test_x, test_y = [], [], [], []
        for key, vector in features.items():
            if fold_of[key] == fold:
                test_x.append(vector)
                test_y.append(key[0] - 1)
            else:
                train_x.append(vector)
                train_y.append(key[0] - 1)

        train_x, train_y = np.asarray(train_x), np.asarray(train_y)
        test_x, test_y = np.asarray(test_x), np.asarray(test_y)

        # Padroniza com estatísticas do treino, como no experimento principal.
        scaler = StandardScaler().fit(train_x)
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(scaler.transform(train_x), train_y)

        accuracy = classifier.score(scaler.transform(test_x), test_y)
        accuracies.append(accuracy)
        logger.info('Partição %d/%d: acurácia = %.2f%%', fold + 1, num_folds, accuracy * 100)

    return np.asarray(accuracies)


def write_report(
    accuracies: np.ndarray,
    condition: str,
    settings: Settings,
    output: Path,
) -> None:
    """Grava o resultado do diagnóstico em texto, JSON e figura.

    Args:
        accuracies: Acurácia de cada partição.
        condition: Condição avaliada.
        settings: Configuração do experimento.
        output: Diretório onde gravar.
    """
    output.mkdir(parents=True, exist_ok=True)
    chance = 1.0 / settings.num_speakers
    mean = float(accuracies.mean())

    lines = [
        f'Diagnóstico de canal — condição: {condition}',
        f'Corpus: {settings.dataset_format}, {settings.num_speakers} locutores',
        '',
    ]
    for index, accuracy in enumerate(accuracies, start=1):
        lines.append(f'Partição {index}: acurácia = {accuracy * 100:.2f}%')
    lines += [
        '',
        f'Acurácia média: {mean * 100:.2f}% ± {accuracies.std() * 100:.2f}',
        f'Acaso ({settings.num_speakers} locutores): {chance * 100:.2f}%',
        f'Razão sobre o acaso: {mean / chance:.1f}×',
    ]

    (output / f'diagnostico_{condition}.txt').write_text('\n'.join(lines) + '\n')
    (output / f'diagnostico_{condition}.json').write_text(json.dumps({
        'condicao': condition,
        'corpus': settings.dataset_format,
        'num_locutores': settings.num_speakers,
        'acuracias': [float(a) for a in accuracies],
        'acuracia_media': mean,
        'acuracia_desvio': float(accuracies.std()),
        'acaso': chance,
        'vezes_o_acaso': mean / chance,
    }, indent=2, ensure_ascii=False))

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(accuracies) + 1), accuracies * 100, color='crimson', label=condition)
    plt.axhline(mean * 100, color='darkred', linestyle='--', label=f'média {mean * 100:.1f}%')
    plt.axhline(chance * 100, color='gray', linestyle=':', label=f'acaso {chance * 100:.2f}%')
    plt.title(f'Diagnóstico de canal ({settings.dataset_format}) — condição "{condition}"')
    plt.xlabel('Partição')
    plt.ylabel('Acurácia (%)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / f'diagnostico_{condition}.png', dpi=150)
    plt.close()

    logger.info('Condição %r: %.2f%% ± %.2f (acaso %.2f%%, %.1f× o acaso).',
                condition, mean * 100, accuracies.std() * 100, chance * 100, mean / chance)
    logger.info('Relatório gravado em %s', output)


def main() -> int:
    """Executa o diagnóstico para as condições pedidas.

    Returns:
        Código de saída: ``0`` em caso de sucesso, ``1`` em caso de erro.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', '-c', default=None, help='Perfil do experimento.')
    parser.add_argument('--condition', '-k', action='append', choices=CONDITIONS,
                        help='Condição a avaliar; pode repetir. Padrão: todas.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as error:
        logger.error('%s', error)
        return 1

    conditions = args.condition or list(CONDITIONS)
    output = settings.models_path / 'diagnostico_canal'
    summary: dict[str, float] = {}

    for condition in conditions:
        logger.info('=== Condição %r ===', condition)
        features = build_features(settings, condition)
        if not features:
            logger.error('Nenhuma gravação utilizável na condição %r.', condition)
            continue
        accuracies = run_cross_validation(features, settings.num_folds)
        write_report(accuracies, condition, settings, output)
        summary[condition] = float(accuracies.mean())

    if len(summary) > 1:
        _write_comparison(summary, settings, output)
    return 0


def _write_comparison(summary: dict[str, float], settings: Settings, output: Path) -> None:
    """Grava a figura comparativa entre as condições avaliadas.

    É a figura que sustenta a conclusão: a altura da barra ``silence`` acima do acaso
    mede o confundidor, e a distância entre ``full`` e ``speech`` mede o quanto dele
    a condição completa efetivamente aproveita.
    """
    chance = 1.0 / settings.num_speakers
    order = [c for c in CONDITIONS if c in summary]
    values = [summary[c] * 100 for c in order]
    labels = {'silence': 'só silêncio', 'speech': 'só fala', 'full': 'sinal completo'}

    plt.figure(figsize=(8, 5))
    bars = plt.bar([labels[c] for c in order], values,
                   color=['crimson', 'seagreen', 'steelblue'][:len(order)])
    plt.axhline(chance * 100, color='gray', linestyle=':', label=f'acaso {chance * 100:.2f}%')
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.1f}%',
                 ha='center', fontsize=10)
    plt.title(f'Identidade do locutor por recorte do sinal ({settings.dataset_format})')
    plt.ylabel('Acurácia (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'comparacao_condicoes.png', dpi=150)
    plt.close()

    (output / 'comparacao_condicoes.json').write_text(
        json.dumps({'acaso': chance, **summary}, indent=2, ensure_ascii=False))
    logger.info('Comparação entre condições gravada em %s', output)


if __name__ == '__main__':
    raise SystemExit(main())
