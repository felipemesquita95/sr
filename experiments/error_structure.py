#!/usr/bin/env python3
"""Estrutura dos erros: os acertos que atravessam o microfone são identidade ou grupo?

A matriz de transferência estabelece que a acurácia cai de cerca de 95% para algo
entre 35% e 42% quando o transdutor muda. O número agregado não diz **de que** esses
acertos são feitos, e há duas leituras incompatíveis:

``identidade``
    O que sobrevive à troca de microfone distingue pessoas. O resíduo é pequeno, mas
    é o que o título de um sistema de reconhecimento de locutor promete medir.

``grupo``
    O que sobrevive distingue apenas traço vocal grosso — gênero, sotaque — e os
    acertos são o que se obtém ao acertar o grupo e sortear dentro dele. Nesse caso o
    resíduo não é identidade, e a conclusão do trabalho muda.

Nenhuma acurácia agregada separa as duas, e é por isso que este instrumento existe.

O teste é sobre **para onde vão os erros**. Se a rede opera por grupo, um erro tende
a cair em alguém do mesmo grupo do alvo; se opera por identidade, o erro não tem
motivo para respeitar essa fronteira. A comparação exige cuidado: contar erros dentro
do grupo favorece grupos grandes, porque há mais candidatos onde errar. O referencial
usado aqui preserva **tanto** a distribuição dos grupos dos alvos **quanto** a
frequência das classes preditas, e mede o excesso sobre ela.

A ressalva que o instrumento não resolve: errar entre pessoas do mesmo grupo também é
compatível com um identificador que discrimina indivíduos e tem dificuldade com vozes
parecidas. Gênero e sotaque não são alternativas mutuamente exclusivas à identidade.
Por isso o resultado se lê como excesso sobre o acaso estrutural, e não como prova.

Não há treino nem inferência: as predições por gravação foram persistidas pela matriz
de transferência, o que torna esta análise gratuita e exatamente reproduzível.

Uso::

    python experiments/error_structure.py
    python experiments/error_structure.py --matriz runs/models/vctk_transfer_matrix
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

logger = logging.getLogger('error_structure')

#: Trilhas da matriz, na ordem em que os relatórios as apresentam.
TRACKS = ('mic1', 'mic2')

#: Campos de metadado avaliados, e a coluna correspondente no speaker-info.
GROUPINGS = ('genero', 'sotaque')


def read_speaker_info(path: Path) -> dict[str, dict[str, str]]:
    """Lê o ``speaker-info.txt`` do VCTK.

    O arquivo é de largura livre e a região pode conter espaços, então só as quatro
    primeiras colunas são posicionais e o restante da linha é a região.

    Args:
        path: Caminho do arquivo de metadados.

    Returns:
        Mapeamento do nome do locutor no corpus para os seus campos.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f'Metadados ausentes: {path}. O speaker-info.txt não acompanha as features; '
            'ver a procedência registrada no relatório.')

    speakers: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0].upper() == 'ID':
            continue
        # O arquivo do corpus identifica os locutores como ``p225`` e ``s5``; espelhos
        # de terceiros costumam gravar apenas o número. Aceitar as duas formas evita
        # que a origem do arquivo mude silenciosamente quem entra na análise — foi
        # assim que o locutor s5 ficou de fora de uma execução anterior.
        identifier = fields[0] if not fields[0].isdigit() else f'p{fields[0]}'
        if not fields[1].isdigit():
            continue
        speakers[identifier] = {
            'idade': fields[1],
            'genero': fields[2],
            'sotaque': fields[3],
            'regiao': ' '.join(fields[4:]),
        }
    return speakers


def index_to_group(
    manifest: Path,
    info: dict[str, dict[str, str]],
    field: str,
) -> dict[int, str]:
    """Traduz o índice interno de locutor para o valor de um campo de metadado.

    Locutores sem metadado são **omitidos**, e não recebem um rótulo de ausência.
    Um grupo "desconhecido" seria tratado pela análise como qualquer outro e poderia
    aparecer como estrutura onde só há lacuna de cobertura.

    Args:
        manifest: Caminho do manifesto do VCTK.
        info: Metadados por nome de locutor.
        field: Campo a projetar.

    Returns:
        Mapeamento de índice de locutor para o valor do campo.
    """
    names = json.loads(manifest.read_text(encoding='utf-8'))['locutores']
    groups = {}
    for index, name in names.items():
        if name in info:
            groups[int(index)] = info[name][field]
    return groups


def load_predictions(matrix_path: Path) -> list[dict]:
    """Reúne as predições de todos os ajustes da matriz de transferência.

    Args:
        matrix_path: Diretório de saída da matriz.

    Returns:
        Lista de ajustes, cada um com a origem, a semente e as gravações.

    Raises:
        FileNotFoundError: Se nenhum arquivo de predições for encontrado.
    """
    files = sorted(matrix_path.glob('*/*/predicoes.json'))
    if not files:
        raise FileNotFoundError(f'Sem predições em {matrix_path}. Rode transfer_matrix.py antes.')
    return [json.loads(path.read_text(encoding='utf-8')) for path in files]


def within_group_rate(
    targets: list[int],
    predictions: list[int],
    groups: dict[int, str],
) -> tuple[float, float, int]:
    """Mede a fração de erros que cai dentro do grupo do alvo, e o seu acaso estrutural.

    O acaso não é ``1/num_grupos``. Ele é calculado sobre as marginais efetivamente
    observadas: a probabilidade de o alvo estar no grupo *g* vezes a probabilidade de
    a predição estar em *g*, somada sobre os grupos. Isso é a taxa esperada quando a
    predição é independente do alvo, mas as duas mantêm as frequências que de fato
    têm. Sem essa correção, um sistema que apenas prefere prever classes de grupos
    grandes já pareceria organizado por grupo.

    Args:
        targets: Índices de locutor verdadeiros das gravações erradas.
        predictions: Índices preditos, na mesma ordem.
        groups: Mapeamento de índice para grupo.

    Returns:
        Tupla ``(taxa observada, taxa esperada, número de erros considerados)``.
    """
    pairs = [(t, p) for t, p in zip(targets, predictions) if t in groups and p in groups]
    if not pairs:
        return 0.0, 0.0, 0

    observed = sum(groups[t] == groups[p] for t, p in pairs) / len(pairs)

    target_share = Counter(groups[t] for t, _ in pairs)
    predicted_share = Counter(groups[p] for _, p in pairs)
    total = len(pairs)
    expected = sum(
        (target_share[g] / total) * (predicted_share[g] / total)
        for g in set(target_share) | set(predicted_share)
    )
    return observed, expected, len(pairs)


def analyse(
    fits: list[dict],
    groupings: dict[str, dict[int, str]],
) -> dict:
    """Percorre os ajustes e mede a estrutura dos erros de cada célula cruzada.

    Só as células **cruzadas** interessam — a captura diferente da origem do treino.
    Na célula de referência quase não há erro, e a estrutura do que sobra descreveria
    o resíduo do protocolo convencional, não a transferência.

    Args:
        fits: Ajustes carregados, com predições por gravação.
        groupings: Por campo, o mapeamento de índice para grupo.

    Returns:
        Estrutura com o resumo por origem e campo, e a cobertura dos metadados.
    """
    results: dict[str, dict] = {}

    for origin in TRACKS:
        crossed = next(t for t in TRACKS if t != origin)
        by_field: dict[str, list[dict]] = {field: [] for field in groupings}
        accuracies = []

        for fit in (f for f in fits if f['origem'] == origin):
            targets, predictions = [], []
            for recording in fit['gravacoes']:
                targets.append(recording['alvo_base_zero'] + 1)
                predictions.append(recording['predicoes_base_zero'][crossed] + 1)

            hits = sum(t == p for t, p in zip(targets, predictions))
            accuracies.append(hits / len(targets))

            wrong = [(t, p) for t, p in zip(targets, predictions) if t != p]
            for field, groups in groupings.items():
                observed, expected, considered = within_group_rate(
                    [t for t, _ in wrong], [p for _, p in wrong], groups)
                by_field[field].append({
                    'semente': fit['semente'],
                    'observado': observed,
                    'esperado': expected,
                    'excesso': observed - expected,
                    'erros': considered,
                })

        results[origin] = {
            'captura_avaliada': crossed,
            'acuracia_media': float(np.mean(accuracies)),
            'campos': {
                field: {
                    'observado': float(np.mean([r['observado'] for r in runs])),
                    'esperado': float(np.mean([r['esperado'] for r in runs])),
                    'excesso': float(np.mean([r['excesso'] for r in runs])),
                    'desvio_excesso': float(np.std([r['excesso'] for r in runs])),
                    'por_semente': runs,
                }
                for field, runs in by_field.items()
            },
        }

    return results


def speaker_recall(fits: list[dict]) -> dict[str, np.ndarray]:
    """Calcula a revocação de cada locutor na célula cruzada, por origem.

    O agregado esconde a pergunta que importa aqui: a acurácia se distribui entre os
    locutores ou se concentra em poucos? São interpretações diferentes do mesmo
    número, e apenas uma delas sustenta falar em identidade.

    Args:
        fits: Ajustes carregados.

    Returns:
        Por origem, o vetor de revocação média por locutor.
    """
    recalls: dict[str, np.ndarray] = {}
    for origin in TRACKS:
        crossed = next(t for t in TRACKS if t != origin)
        per_speaker: dict[int, list[float]] = {}
        for fit in (f for f in fits if f['origem'] == origin):
            hits: Counter = Counter()
            support: Counter = Counter()
            for recording in fit['gravacoes']:
                speaker = recording['alvo_base_zero'] + 1
                support[speaker] += 1
                if recording['predicoes_base_zero'][crossed] == recording['alvo_base_zero']:
                    hits[speaker] += 1
            for speaker, total in support.items():
                per_speaker.setdefault(speaker, []).append(hits[speaker] / total)
        recalls[origin] = np.asarray([np.mean(v) for _, v in sorted(per_speaker.items())])
    return recalls


def write_report(
    results: dict,
    recalls: dict[str, np.ndarray],
    coverage: tuple[int, int, list[str]],
    output: Path,
) -> None:
    """Grava o resultado em texto, JSON e figura.

    Args:
        results: Saída de :func:`analyse`.
        recalls: Revocação por locutor, por origem.
        coverage: ``(com metadado, total, nomes sem metadado)``.
        output: Diretório onde gravar.
    """
    output.mkdir(parents=True, exist_ok=True)
    covered, total, missing = coverage

    lines = [
        'Estrutura dos erros na travessia do microfone',
        f'Cobertura de metadados: {covered} de {total} locutores.',
        f'Sem metadado, excluídos da análise de grupo: {", ".join(missing) or "nenhum"}',
        '',
        'Erros que caem dentro do mesmo grupo do alvo, contra o acaso estrutural que',
        'preserva as frequências dos grupos alvo e das classes preditas.',
        '',
        f'{"origem":<8}{"campo":<10}{"observado":>11}{"esperado":>11}{"excesso":>10}{"desvio":>9}',
    ]
    for origin, block in results.items():
        for field, stats in block['campos'].items():
            lines.append(
                f'{origin:<8}{field:<10}{stats["observado"] * 100:>10.1f}%'
                f'{stats["esperado"] * 100:>10.1f}%'
                f'{stats["excesso"] * 100:>9.1f}%{stats["desvio_excesso"] * 100:>8.1f}'
            )

    lines += ['', 'Revocação por locutor na célula cruzada:', '']
    for origin, values in recalls.items():
        quartiles = np.percentile(values, [25, 50, 75])
        lines.append(
            f'  treino em {origin}: média {values.mean() * 100:.1f}%  '
            f'mediana {quartiles[1] * 100:.1f}%  '
            f'quartis {quartiles[0] * 100:.1f}%/{quartiles[2] * 100:.1f}%  '
            f'mín {values.min() * 100:.1f}%  máx {values.max() * 100:.1f}%'
        )
        lines.append(
            f'    nunca acertado: {(values == 0).sum()}   '
            f'abaixo de 10%: {(values < 0.10).sum()}   '
            f'acima de 90%: {(values > 0.90).sum()}'
        )

    lines += [
        '',
        'Um excesso próximo de zero indica que os erros não respeitam a fronteira do',
        'grupo, e portanto que a decisão não se organiza por ele. Um excesso alto é',
        'compatível com decisão por grupo, mas também com um identificador que confunde',
        'vozes parecidas — que tendem a estar no mesmo grupo. A medida não separa as',
        'duas hipóteses; ela apenas descarta a primeira quando o excesso é nulo.',
        '',
        'Procedência dos metadados: speaker-info.txt extraído do zip oficial do VCTK',
        '0.92. Cobre os 108 locutores do experimento. O próprio arquivo registra que',
        'p280 e p315 não possuem trilha mic2, que é a razão de a numeração ser definida',
        'sobre a interseção e de serem 108 locutores, e não 110.',
    ]

    (output / 'estrutura_erros.txt').write_text('\n'.join(lines) + '\n')
    (output / 'estrutura_erros.json').write_text(json.dumps({
        'cobertura': {'com_metadado': covered, 'total': total, 'ausentes': missing},
        'resultados': results,
        'revocacao_por_locutor': {k: v.tolist() for k, v in recalls.items()},
    }, indent=2, ensure_ascii=False))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, (origin, values) in zip(axes, recalls.items()):
        axis.hist(values * 100, bins=20, range=(0, 100), color='steelblue',
                  edgecolor='white')
        axis.axvline(values.mean() * 100, color='crimson', linestyle='--',
                     label=f'média {values.mean() * 100:.1f}%')
        axis.set(title=f'Treino em {origin}, teste na outra captura',
                 xlabel='Revocação do locutor (%)', ylabel='Locutores')
        axis.legend()
        axis.grid(True, axis='y', alpha=0.3)
    figure.suptitle('A acurácia cruzada não descreve nenhum locutor típico')
    figure.tight_layout()
    figure.savefig(output / 'estrutura_erros.png', dpi=150)
    plt.close(figure)

    logger.info('Relatório gravado em %s', output)


def main() -> int:
    """Executa a análise sobre as predições já persistidas.

    Returns:
        Código de saída: ``0`` em caso de sucesso, ``1`` em caso de erro.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--matriz', type=Path,
                        default=Path('runs/models/vctk_transfer_matrix'),
                        help='Diretório de saída da matriz de transferência.')
    parser.add_argument('--metadados', type=Path,
                        default=Path('runs/features/vctk_speaker_info.txt'),
                        help='speaker-info.txt do VCTK.')
    parser.add_argument('--manifesto', type=Path,
                        default=Path('runs/features/vctk_manifesto.json'),
                        help='Manifesto que traduz índice para nome do corpus.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                        datefmt='%H:%M:%S')

    try:
        info = read_speaker_info(args.metadados)
        fits = load_predictions(args.matriz)
    except FileNotFoundError as error:
        logger.error('%s', error)
        return 1

    names = json.loads(args.manifesto.read_text(encoding='utf-8'))['locutores']
    missing = sorted(name for name in names.values() if name not in info)
    covered = len(names) - len(missing)
    logger.info('Metadados cobrem %d de %d locutores; sem cobertura: %s',
                covered, len(names), ', '.join(missing) or 'nenhum')

    groupings = {field: index_to_group(args.manifesto, info, field) for field in GROUPINGS}
    results = analyse(fits, groupings)
    recalls = speaker_recall(fits)

    for origin, block in results.items():
        logger.info('=== treino em %s, teste em %s (acurácia %.2f%%) ===',
                    origin, block['captura_avaliada'], block['acuracia_media'] * 100)
        for field, stats in block['campos'].items():
            logger.info('  %-9s erros no mesmo grupo: %.1f%%  esperado %.1f%%  excesso %+.1f%%',
                        field, stats['observado'] * 100, stats['esperado'] * 100,
                        stats['excesso'] * 100)

    write_report(results, recalls, (covered, len(names), missing),
                 args.matriz / 'estrutura_erros')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
