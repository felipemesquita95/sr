#!/usr/bin/env python3
"""Diagnóstico de canal atravessando o microfone: o confundidor sobrevive à troca?

O diagnóstico de canal (``channel_probe.py``) mostra que o locutor é identificável a
partir de trechos **sem fala**. Ele não diz, porém, *de que* essa pista é feita — e a
distinção decide como ler o protocolo cross-microfone.

Duas hipóteses produzem o mesmo resultado dentro de uma trilha:

``transdutor``
    A pista é a resposta em frequência do microfone. Nesse caso ela **não** atravessa
    a troca de trilha: treinar o classificador de silêncio em ``mic1`` e avaliá-lo em
    ``mic2`` deve derrubar a acurácia ao acaso.

``sessão``
    A pista é o ambiente, o ganho, a postura, a respiração — tudo o que é particular
    da sessão única em que aquele locutor foi gravado. As duas trilhas do VCTK são
    **simultâneas**: gravam o mesmo instante por dois transdutores. Uma pista de
    sessão atravessa a troca intacta.

A consequência é direta e é a razão de este experimento vir antes dos treinos. Se a
identidade sobreviver à travessia a partir do **silêncio**, então o que resta de
acurácia no protocolo cross-microfone não é voz isolada: é voz mais o resíduo de
sessão que o protocolo não remove. O número deixa de ser um piso honesto do que o
sistema sabe sobre vozes e passa a ser um teto — mede quanto ainda há de confundidor,
não quanto há de sinal.

O classificador é o mesmo do diagnóstico dentro da trilha — regressão logística sobre
média e desvio dos coeficientes cepstrais — para que os números sejam comparáveis. Um
resultado positivo é limite *inferior* do efeito: um modelo mais capaz extrairia mais.

As assinaturas são lidas dos arquivos ``assinaturas.npz`` gravados durante a ingestão.
O áudio bruto do VCTK é apagado à medida que é processado, então esta é a única via
possível sem decodificar o corpus inteiro de novo.

Uso::

    SR_CONFIG=configs/vctk_cross_mic.env python experiments/channel_transfer.py
    SR_CONFIG=configs/vctk_cross_mic.env python experiments/channel_transfer.py -k silence
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
from sr.diagnostics import CONDITIONS  # noqa: E402

logger = logging.getLogger('channel_transfer')

#: Nome do arquivo de assinaturas gravado ao lado das features de cada gravação.
SIGNATURES_FILE = 'assinaturas.npz'


def load_signatures(
    features_path: Path,
    condition: str,
    num_speakers: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Lê as assinaturas de uma trilha, sob uma condição.

    Gravações cujo recorte foi curto demais não têm a condição no arquivo — o
    silêncio de um enunciado sem pausas, por exemplo. Elas são omitidas, e não
    preenchidas com zeros: a ausência de silêncio mensurável é um fato sobre a
    gravação, e inventar um vetor nulo criaria uma classe artificial de pontos
    idênticos que o classificador aprenderia a separar.

    Args:
        features_path: Diretório da trilha, com um subdiretório por locutor.
        condition: Recorte de sinal, uma de :data:`CONDITIONS`.
        num_speakers: Índice máximo de locutor a considerar.

    Returns:
        Mapeamento de ``(locutor, enunciado)`` para vetor de assinatura.

    Raises:
        FileNotFoundError: Se o diretório da trilha não existir.
    """
    if not features_path.is_dir():
        raise FileNotFoundError(f'Trilha inexistente: {features_path}')

    signatures: dict[tuple[int, int], np.ndarray] = {}
    missing = 0

    for speaker in range(1, num_speakers + 1):
        speaker_path = features_path / str(speaker)
        if not speaker_path.is_dir():
            continue

        for utterance_path in speaker_path.iterdir():
            if not utterance_path.is_dir() or not utterance_path.name.isdigit():
                continue

            archive = utterance_path / SIGNATURES_FILE
            if not archive.exists():
                missing += 1
                continue

            with np.load(archive) as data:
                if condition not in data:
                    missing += 1
                    continue
                signatures[(speaker, int(utterance_path.name))] = data[condition]

    if missing:
        logger.info('  %d gravações sem assinatura de %r em %s.',
                    missing, condition, features_path.name)
    return signatures


def paired_matrices(
    train: dict[tuple[int, int], np.ndarray],
    test: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alinha as duas trilhas sobre as gravações que ambas possuem.

    A interseção é obrigatória. Se uma trilha tiver silêncio mensurável em uma
    gravação e a outra não, comparar as duas populações mediria também a diferença
    entre elas, e não apenas o efeito da travessia.

    Args:
        train: Assinaturas da trilha de treino.
        test: Assinaturas da trilha de teste.

    Returns:
        Tupla ``(x_treino, y_treino, x_teste, y_teste)``, com rótulos base zero e as
        linhas na mesma ordem de chaves nas duas trilhas.
    """
    shared = sorted(set(train) & set(test))
    x_train = np.asarray([train[key] for key in shared])
    x_test = np.asarray([test[key] for key in shared])
    labels = np.asarray([key[0] - 1 for key in shared])
    return x_train, labels, x_test, labels


def evaluate_transfer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Treina em uma trilha e avalia na outra.

    Não há validação cruzada: a partição é dada pelo transdutor, e cada gravação
    aparece uma vez de cada lado. O treino vê ``mic1`` inteiro e o teste é ``mic2``
    inteiro — o mesmo locutor dizendo a mesma frase no mesmo instante.

    Args:
        x_train: Assinaturas de treino.
        y_train: Rótulos de treino.
        x_test: Assinaturas de teste.
        y_test: Rótulos de teste.

    Returns:
        Acurácia na trilha de teste, entre 0 e 1.
    """
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(scaler.transform(x_train), y_train)
    return float(classifier.score(scaler.transform(x_test), y_test))


def evaluate_within(
    x: np.ndarray,
    y: np.ndarray,
    keys: list[tuple[int, int]],
    num_folds: int,
) -> float:
    """Mede a separabilidade dentro da própria trilha, como referência.

    Sem esse número a travessia não se interpreta: uma acurácia cruzada baixa pode
    significar que a pista não atravessou, ou que nunca houve pista nenhuma naquela
    condição. A partição espelha a do experimento principal — grupos pela posição do
    enunciado dentro do locutor.

    Args:
        x: Assinaturas da trilha.
        y: Rótulos base zero.
        keys: Chaves ``(locutor, enunciado)`` na ordem das linhas de ``x``.
        num_folds: Número de partições.

    Returns:
        Acurácia média sobre as partições, entre 0 e 1.
    """
    utterances_by_speaker: dict[int, list[int]] = {}
    for speaker, utterance in keys:
        utterances_by_speaker.setdefault(speaker, []).append(utterance)

    fold_of: dict[tuple[int, int], int] = {}
    for speaker, utterances in utterances_by_speaker.items():
        for position, utterance in enumerate(sorted(utterances)):
            fold_of[(speaker, utterance)] = position % num_folds

    folds = np.asarray([fold_of[key] for key in keys])
    accuracies = []
    for fold in range(num_folds):
        held_out = folds == fold
        scaler = StandardScaler().fit(x[~held_out])
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(scaler.transform(x[~held_out]), y[~held_out])
        accuracies.append(classifier.score(scaler.transform(x[held_out]), y[held_out]))

    return float(np.mean(accuracies))


def write_report(
    results: dict[str, dict[str, float]],
    settings: Settings,
    output: Path,
    num_speakers: int,
) -> None:
    """Grava o resultado da travessia em texto, JSON e figura.

    Args:
        results: Por condição, as chaves ``dentro``, ``travessia`` e ``pares``.
        settings: Configuração do experimento.
        output: Diretório onde gravar.
        num_speakers: Locutores efetivamente presentes nas duas trilhas.
    """
    output.mkdir(parents=True, exist_ok=True)
    chance = 1.0 / num_speakers

    lines = [
        'Diagnóstico de canal atravessando o microfone',
        f'Treino: {settings.features_path_train.name}   '
        f'Teste: {settings.features_path_test.name}',
        f'{num_speakers} locutores, acaso {chance * 100:.2f}%',
        '',
        f'{"condição":<16}{"dentro":>10}{"travessia":>12}{"retido":>10}{"pares":>10}',
    ]
    for condition, result in results.items():
        retained = result['travessia'] / result['dentro'] if result['dentro'] else 0.0
        lines.append(
            f'{condition:<16}{result["dentro"] * 100:>9.2f}%'
            f'{result["travessia"] * 100:>11.2f}%'
            f'{retained * 100:>9.1f}%{result["pares"]:>10.0f}'
        )

    lines += [
        '',
        'A coluna "dentro" é a validação cruzada na trilha de treino; "travessia" é o',
        'mesmo classificador avaliado na outra trilha; "retido" é a razão entre as',
        'duas. Retenção alta na condição "silence" indica confundidor de sessão, que',
        'o protocolo cross-microfone não remove. Retenção próxima do acaso indica que',
        'a pista era do transdutor, e que a troca de trilha de fato a elimina.',
    ]

    (output / 'travessia_canal.txt').write_text('\n'.join(lines) + '\n')
    (output / 'travessia_canal.json').write_text(json.dumps({
        'trilha_treino': str(settings.features_path_train),
        'trilha_teste': str(settings.features_path_test),
        'num_locutores': num_speakers,
        'acaso': chance,
        'condicoes': results,
    }, indent=2, ensure_ascii=False))

    order = [c for c in CONDITIONS if c in results]
    labels = {'silence': 'só silêncio', 'speech': 'só fala', 'full': 'sinal completo'}
    positions = np.arange(len(order))
    width = 0.38

    plt.figure(figsize=(8, 5))
    within = [results[c]['dentro'] * 100 for c in order]
    across = [results[c]['travessia'] * 100 for c in order]
    bars_a = plt.bar(positions - width / 2, within, width,
                     color='steelblue', label='dentro da trilha')
    bars_b = plt.bar(positions + width / 2, across, width,
                     color='crimson', label='atravessando o microfone')
    for bar, value in zip(list(bars_a) + list(bars_b), within + across):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.1f}',
                 ha='center', fontsize=9)
    plt.axhline(chance * 100, color='gray', linestyle=':',
                label=f'acaso {chance * 100:.2f}%')
    plt.xticks(positions, [labels[c] for c in order])
    plt.ylabel('Acurácia (%)')
    plt.ylim(0, 100)
    plt.title('Identidade do locutor pela assinatura de canal, antes e depois da troca')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'travessia_canal.png', dpi=150)
    plt.close()

    logger.info('Relatório gravado em %s', output)


def main() -> int:
    """Executa a travessia para as condições pedidas.

    Returns:
        Código de saída: ``0`` em caso de sucesso, ``1`` em caso de erro.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', '-c', default=None, help='Perfil do experimento.')
    parser.add_argument('--condition', '-k', action='append', choices=CONDITIONS,
                        help='Condição a avaliar; pode repetir. Padrão: todas.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                        datefmt='%H:%M:%S')

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as error:
        logger.error('%s', error)
        return 1

    if not (settings.features_path_train and settings.features_path_test):
        logger.error('O perfil precisa definir FEATURES_PATH_TRAIN e FEATURES_PATH_TEST.')
        return 1

    results: dict[str, dict[str, float]] = {}
    speakers_seen = 0

    for condition in args.condition or list(CONDITIONS):
        logger.info('=== Condição %r ===', condition)
        train = load_signatures(settings.features_path_train, condition, settings.num_speakers)
        test = load_signatures(settings.features_path_test, condition, settings.num_speakers)

        x_train, y_train, x_test, y_test = paired_matrices(train, test)
        if len(x_train) < settings.num_folds:
            logger.warning('Pares insuficientes na condição %r; ignorada.', condition)
            continue

        keys = sorted(set(train) & set(test))
        speakers_seen = max(speakers_seen, len({key[0] for key in keys}))
        logger.info('  %d pares em %d locutores.', len(keys), len({k[0] for k in keys}))

        within = evaluate_within(x_train, y_train, keys, settings.num_folds)
        across = evaluate_transfer(x_train, y_train, x_test, y_test)
        results[condition] = {'dentro': within, 'travessia': across, 'pares': float(len(keys))}

        logger.info('  dentro da trilha: %.2f%%   atravessando: %.2f%%   retido: %.1f%%',
                    within * 100, across * 100, across / within * 100 if within else 0.0)

    if not results:
        logger.error('Nenhuma condição pôde ser avaliada.')
        return 1

    write_report(results, settings, settings.models_path / 'diagnostico_travessia',
                 speakers_seen)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
