#!/usr/bin/env python3
"""A pista fora da fala é do transdutor, ou é da sessão em outras coordenadas?

O diagnóstico de travessia mostrou que a identidade predita a partir de trechos sem
fala cai de 85,4% para 4,3% quando o microfone muda, e disso foi concluído que a
pista era do transdutor e não da sessão. **A inferência não se sustenta**, e o motivo
é a própria estrutura do corpus.

Todos os locutores foram gravados com o mesmo modelo de microfone, na mesma câmara.
Um modelo de microfone é idêntico para todos, e portanto não pode distinguir
locutores: se a pista fosse o transdutor enquanto equipamento, ela seria constante
entre as classes e a acurácia seria a do acaso. O que varia por pessoa é a
**realização daquela sessão** — o ganho ajustado para aquele locutor, a posição e a
distância dele em relação ao microfone, o corpo, a respiração, as condições daquele
dia.

E as duas trilhas gravam a mesma sessão **ao mesmo tempo**. Tudo o que é de sessão
está nas duas. Se a pista não transfere, a explicação não pode ser que ela não exista
do outro lado.

A explicação compatível com os dois fatos é que os dois microfones têm cadeias de
ganho independentes, ajustadas por locutor. A informação de sessão existe nas duas
trilhas, mas em coordenadas diferentes, e a correspondência entre elas muda de pessoa
para pessoa. Um classificador linear treinado no espaço de uma trilha simplesmente
não sabe ler o espaço da outra.

Este experimento decide entre as duas leituras. Se uma transformação afim entre os
espaços das trilhas — aprendida em locutores que o classificador nunca verá, e sem
usar rótulo de identidade — devolver a acurácia, então a informação estava lá o tempo
todo. Nesse caso o protocolo cross-microfone **não removeu** o confundidor de sessão:
apenas o tornou ilegível, e o piso de identidade vocal que ele reporta precisa ser
reinterpretado.

Se a transformação não devolver nada, a leitura original se sustenta.

O desenho tem três exigências, e as três existem para que o resultado signifique o
que afirma:

1. A transformação é estimada em locutores **disjuntos** dos avaliados. Ajustá-la nos
   mesmos locutores seria aprender a identidade pelo caminho de trás.
2. A transformação não recebe rótulo de locutor. Ela mapeia assinatura em assinatura,
   e nada mais.
3. A regularização é fixada antes de olhar o resultado, e a condição ``speech`` roda
   como controle de que a transformação de fato funciona — se ela não recuperar nada
   nem na fala, o instrumento está quebrado e o resultado do silêncio não se lê.

Uso::

    python experiments/session_or_transducer.py
    python experiments/session_or_transducer.py --condicao silence
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

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

logger = logging.getLogger('session_or_transducer')

#: Nome do arquivo de assinaturas gravado ao lado das features de cada gravação.
SIGNATURES_FILE = 'assinaturas.npz'

#: Locutores reservados para estimar a transformação entre trilhas.
CALIBRATION_SPEAKERS = 36

#: Regularização da transformação afim, fixada antes de qualquer avaliação.
RIDGE_ALPHA = 1.0

#: Enunciados de cada locutor reservados para teste do classificador de identidade.
TEST_UTTERANCES = 40

#: Sementes das rotações de calibração. Não são amostras independentes: os grupos de
#: avaliação se sobrepõem, e a dispersão entre elas descreve sensibilidade à escolha
#: dos locutores de calibração, não incerteza estatística.
ROTATIONS = (11, 23, 37)


def load_signatures(
    features_path: Path,
    condition: str,
    num_speakers: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Lê as assinaturas de uma trilha, sob uma condição.

    Args:
        features_path: Diretório da trilha.
        condition: Recorte de sinal.
        num_speakers: Índice máximo de locutor.

    Returns:
        Mapeamento de ``(locutor, enunciado)`` para vetor de assinatura.
    """
    signatures: dict[tuple[int, int], np.ndarray] = {}
    for speaker in range(1, num_speakers + 1):
        speaker_path = features_path / str(speaker)
        if not speaker_path.is_dir():
            continue
        for utterance_path in speaker_path.iterdir():
            archive = utterance_path / SIGNATURES_FILE
            if not utterance_path.name.isdigit() or not archive.exists():
                continue
            with np.load(archive) as data:
                if condition in data:
                    signatures[(speaker, int(utterance_path.name))] = data[condition]
    return signatures


def fit_transform(
    source: dict[tuple[int, int], np.ndarray],
    target: dict[tuple[int, int], np.ndarray],
    speakers: set[int],
) -> Ridge:
    """Estima a transformação afim que leva o espaço de uma trilha ao da outra.

    Recebe apenas pares de assinaturas da mesma gravação, sem qualquer rótulo de
    identidade. O que ela aprende é a correspondência entre as duas cadeias de
    captação — resposta em frequência, ganho, piso de ruído —, e não quem falou.

    Args:
        source: Assinaturas da trilha a transformar.
        target: Assinaturas da trilha de destino.
        speakers: Locutores de calibração; nenhum deles pode ser avaliado depois.

    Returns:
        Regressão ajustada, aplicável a assinaturas de outros locutores.
    """
    keys = sorted(k for k in set(source) & set(target) if k[0] in speakers)
    x = np.asarray([source[k] for k in keys])
    y = np.asarray([target[k] for k in keys])
    return Ridge(alpha=RIDGE_ALPHA).fit(x, y)


def identity_accuracy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> float:
    """Treina o classificador de identidade e mede a acurácia na condição dada.

    Args:
        train_x: Assinaturas de treino, sempre da trilha de origem.
        train_y: Rótulos de treino.
        test_x: Assinaturas de teste, na condição avaliada.
        test_y: Rótulos de teste.

    Returns:
        Acurácia entre 0 e 1.
    """
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(scaler.transform(train_x), train_y)
    return float(classifier.score(scaler.transform(test_x), test_y))


def run_rotation(
    origin: dict[tuple[int, int], np.ndarray],
    crossed: dict[tuple[int, int], np.ndarray],
    num_speakers: int,
    seed: int,
) -> dict[str, float]:
    """Executa uma rotação completa: calibra, treina a identidade e avalia três condições.

    A divisão de enunciados é a mesma nas três condições, e o classificador é o mesmo
    — só muda de onde vem a assinatura de teste. É essa igualdade que torna as três
    acurácias comparáveis entre si.

    Args:
        origin: Assinaturas da trilha onde o classificador de identidade é treinado.
        crossed: Assinaturas da outra trilha.
        num_speakers: Total de locutores disponíveis.
        seed: Semente da escolha dos locutores de calibração.

    Returns:
        Acurácias das três condições e o número de locutores avaliados.
    """
    rng = np.random.default_rng(seed)
    speakers = np.arange(1, num_speakers + 1)
    calibration = set(int(s) for s in rng.choice(speakers, CALIBRATION_SPEAKERS, replace=False))
    evaluation = sorted(set(int(s) for s in speakers) - calibration)

    mapping = fit_transform(crossed, origin, calibration)

    shared = set(origin) & set(crossed)
    train_keys, test_keys = [], []
    for speaker in evaluation:
        utterances = sorted(k[1] for k in shared if k[0] == speaker)
        if len(utterances) <= TEST_UTTERANCES:
            continue
        test_keys += [(speaker, u) for u in utterances[-TEST_UTTERANCES:]]
        train_keys += [(speaker, u) for u in utterances[:-TEST_UTTERANCES]]

    train_x = np.asarray([origin[k] for k in train_keys])
    train_y = np.asarray([k[0] for k in train_keys])
    test_y = np.asarray([k[0] for k in test_keys])

    same = np.asarray([origin[k] for k in test_keys])
    other = np.asarray([crossed[k] for k in test_keys])

    return {
        'mesma_trilha': identity_accuracy(train_x, train_y, same, test_y),
        'outra_trilha': identity_accuracy(train_x, train_y, other, test_y),
        'outra_transformada': identity_accuracy(
            train_x, train_y, mapping.predict(other), test_y),
        'locutores': float(len(set(test_y))),
        'teste': float(len(test_keys)),
    }


def write_report(results: dict, output: Path) -> None:
    """Grava o resultado em texto, JSON e figura.

    Args:
        results: Por condição, as rotações e as médias.
        output: Diretório onde gravar.
    """
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        'A pista fora da fala é do transdutor, ou é da sessão em outras coordenadas?',
        '',
        'Transformação afim aprendida em 36 locutores de calibração, sem rótulo de',
        'identidade, e aplicada aos locutores avaliados, que ela nunca viu.',
        '',
        f'{"condição":<12}{"locutores":>10}{"acaso":>8}{"mesma":>9}{"outra":>9}'
        f'{"transf.":>10}{"recupera":>10}',
    ]
    for condition, block in results.items():
        speakers = block['locutores']
        chance = 100.0 / speakers
        same, other = block['mesma_trilha'] * 100, block['outra_trilha'] * 100
        mapped = block['outra_transformada'] * 100
        recovered = (mapped - other) / (same - other) if same > other else 0.0
        lines.append(
            f'{condition:<12}{speakers:>10.0f}{chance:>7.2f}%{same:>8.1f}%'
            f'{other:>8.1f}%{mapped:>9.1f}%{recovered * 100:>9.1f}%'
        )

    lines += [
        '',
        '"recupera" é a fração da queda que a transformação devolve. Alta na condição',
        '"silence" significa que a informação de sessão estava presente na outra trilha,',
        'apenas em outro sistema de coordenadas — e que o protocolo cross-microfone não',
        'removeu o confundidor, apenas o tornou ilegível para aquele classificador.',
        'Próxima de zero significa que a pista de fato não sobrevive à troca de captação.',
        '',
        'A condição "speech" é controle do instrumento, e não um resultado. Se a',
        'transformação não recuperar nada nela, ela não está funcionando, e o valor',
        'medido em "silence" não se interpreta.',
        '',
        'As rotações compartilham locutores de avaliação e não são amostras',
        'independentes: a dispersão entre elas descreve sensibilidade à escolha dos',
        'locutores de calibração, e não incerteza estatística.',
    ]
    (output / 'sessao_ou_transdutor.txt').write_text('\n'.join(lines) + '\n')
    (output / 'sessao_ou_transdutor.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False))

    order = [c for c in ('silence', 'speech', 'full') if c in results]
    labels = {'silence': 'só silêncio', 'speech': 'só fala', 'full': 'sinal completo'}
    positions = np.arange(len(order))
    width = 0.26

    plt.figure(figsize=(9, 5))
    for offset, key, color, name in (
        (-width, 'mesma_trilha', 'steelblue', 'mesma trilha'),
        (0.0, 'outra_trilha', 'crimson', 'outra trilha'),
        (width, 'outra_transformada', 'seagreen', 'outra trilha, transformada'),
    ):
        values = [results[c][key] * 100 for c in order]
        bars = plt.bar(positions + offset, values, width, color=color, label=name)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.1f}',
                     ha='center', fontsize=8)

    chance = 100.0 / results[order[0]]['locutores']
    plt.axhline(chance, color='gray', linestyle=':', label=f'acaso {chance:.2f}%')
    plt.xticks(positions, [labels[c] for c in order])
    plt.ylabel('Acurácia (%)')
    plt.ylim(0, 105)
    plt.title('A informação sobrevive à troca de captação, em outras coordenadas?')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'sessao_ou_transdutor.png', dpi=150)
    plt.close()
    logger.info('Relatório gravado em %s', output)


def main() -> int:
    """Executa as rotações para as condições pedidas.

    Returns:
        Código de saída: ``0`` em caso de sucesso, ``1`` em caso de erro.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--origem', type=Path,
                        default=Path('runs/features/vctk_mic1'),
                        help='Trilha onde o classificador de identidade é treinado.')
    parser.add_argument('--cruzada', type=Path,
                        default=Path('runs/features/vctk_mic2'),
                        help='Trilha de onde vêm as assinaturas transformadas.')
    parser.add_argument('--condicao', action='append', choices=('silence', 'speech', 'full'),
                        help='Condição a avaliar; pode repetir. Padrão: silence e speech.')
    parser.add_argument('--locutores', type=int, default=108, help='Total de locutores.')
    parser.add_argument('--saida', type=Path,
                        default=Path('runs/models/sessao_ou_transdutor'),
                        help='Diretório do relatório.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                        datefmt='%H:%M:%S')

    results: dict[str, dict] = {}
    for condition in args.condicao or ['silence', 'speech']:
        logger.info('=== Condição %r ===', condition)
        origin = load_signatures(args.origem, condition, args.locutores)
        crossed = load_signatures(args.cruzada, condition, args.locutores)
        if not origin or not crossed:
            logger.error('Sem assinaturas de %r nas trilhas indicadas.', condition)
            return 1

        rotations = [run_rotation(origin, crossed, args.locutores, seed) for seed in ROTATIONS]
        block = {
            key: float(np.mean([r[key] for r in rotations]))
            for key in ('mesma_trilha', 'outra_trilha', 'outra_transformada')
        }
        block['desvio_transformada'] = float(
            np.std([r['outra_transformada'] for r in rotations]))
        block['locutores'] = rotations[0]['locutores']
        block['por_rotacao'] = rotations
        results[condition] = block

        logger.info('  mesma trilha %.2f%%  outra %.2f%%  transformada %.2f%% ± %.2f',
                    block['mesma_trilha'] * 100, block['outra_trilha'] * 100,
                    block['outra_transformada'] * 100, block['desvio_transformada'] * 100)

    write_report(results, args.saida)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
