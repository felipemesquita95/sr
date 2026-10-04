#!/usr/bin/env python3
"""Compara controles mistos com os controles de atividade do VCTK."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'runs/models'
SELECTION = ROOT / 'docs/vctk_mixed_probe_selection.json'
OUTPUT = ROOT / 'docs/resultado_atividade_mais_baixa_vctk.md'
ARCHITECTURES = ('cnn', 'temporal_cnn', 'attention')
GROUPS = {
    10: ('unfiltered', 'active', 'low', 'mixed'),
    20: ('active', 'mixed'),
}
LABELS = {'unfiltered': 'Sem seleção', 'active': 'Atividade',
          'low': 'Baixa atividade', 'mixed': 'Atividade + baixa atividade'}


def percent(value: float) -> str:
    return f'{value*100:.2f}'.replace('.', ',') + '%'


def fold(prefix: str, condition: str, mic: str,
         architecture: str, fold_number: int) -> tuple[dict, dict]:
    root = MODELS / f'{prefix}_{condition}_{mic}' / architecture / f'particao{fold_number}'
    division = json.loads((root / 'divisao.json').read_text())
    metrics = json.loads((root / 'metricas.json').read_text())
    if not (root / 'modelo.keras').is_file():
        raise FileNotFoundError(root / 'modelo.keras')
    if division['teste'] != metrics['num_amostras_teste']:
        raise ValueError(f'Métrica e teste incompatíveis: {root}')
    return division, metrics


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    report = ['# VCTK: efeito de combinar atividade e baixa atividade', '',
              'As condições foram obtidas dos mesmos arquivos completos, após o',
              'filtro para 8 kHz. O detector usa energia (`top_db=30`); baixa',
              'atividade não equivale necessariamente a silêncio puro.',
              'Os quadros escolhidos são ordenados no tempo, mas podem vir de',
              'trechos separados da mesma gravação.', '']

    for k, conditions in GROUPS.items():
        cohort = selection[f'coorte_{k}']
        report += [f'## Entradas de {k} quadros', '',
                   f'{cohort["pares"]:,} gravações pareadas; 108 locutores; '
                   f'mínimo de {cohort["minimo_por_grupo_de_teste"]} gravações por '
                   'locutor em cada grupo de teste.'.replace(',', '.'), '']
        if k == 10:
            report += ['O misto contém **5 quadros de atividade + 5 de baixa atividade**.',
                       'A comparação com 10 de atividade testa a substituição de',
                       'metade dos quadros, mantendo a largura fixa.', '']
        else:
            report += ['O misto contém **10 quadros de atividade + 10 de baixa atividade**.',
                       'O controle tem 20 quadros de atividade. Ambos usam a mesma',
                       'seleção de gravações, as mesmas partições e a mesma largura.', '']

        prefix = f'vctk_activity{k}'
        reference = {}
        intra = {}
        report += ['### Teste no mesmo microfone', '',
                   '| Microfone | Rede | ' + ' | '.join(LABELS[c] for c in conditions) + ' |',
                   '|---|---|' + '---:|' * len(conditions)]
        for mic in ('mic1', 'mic2'):
            for architecture in ARCHITECTURES:
                scores = {}
                for condition in conditions:
                    values = []
                    for fold_number in range(1, 6):
                        division, metrics = fold(prefix, condition, mic, architecture, fold_number)
                        if division['quadros_por_gravacao'] != k:
                            raise ValueError(f'Largura incorreta: {prefix} {condition} {mic}')
                        signature = tuple(division[key] for key in
                                          ('particao', 'treino', 'validacao', 'teste',
                                           'validation_fold_offset', 'num_folds', 'validation_seed'))
                        key = (mic, fold_number)
                        if key in reference and signature != reference[key]:
                            raise ValueError(f'Partições diferentes: {prefix} {condition} {mic}')
                        reference[key] = signature
                        values.append(metrics['acuracia'])
                    scores[condition] = mean(values)
                    intra[(condition, mic, architecture)] = scores[condition]
                report.append(f'| {mic} | {architecture} | '
                              + ' | '.join(percent(scores[c]) for c in conditions) + ' |')

        report += ['', '### Teste ao trocar de microfone', '',
                   '| Treino → teste | Rede | ' + ' | '.join(LABELS[c] for c in conditions) + ' |',
                   '|---|---|' + '---:|' * len(conditions)]
        for source, target in (('mic1', 'mic2'), ('mic2', 'mic1')):
            for architecture in ARCHITECTURES:
                scores = {}
                for condition in conditions:
                    path = MODELS / f'{prefix}_{condition}_cross_pareado' / \
                        f'{source}_para_{target}' / architecture / 'resumo.json'
                    summary = json.loads(path.read_text())
                    if abs(summary['acuracia_origem_media'] -
                           intra[(condition, source, architecture)]) > 1e-6:
                        raise ValueError(f'Teste de origem divergente: {path}')
                    scores[condition] = summary['acuracia_outro_media']
                report.append(f'| {source} → {target} | {architecture} | '
                              + ' | '.join(percent(scores[c]) for c in conditions) + ' |')
        report.append('')

    report += ['O acaso em 108 classes é 0,93%. Cada célula resume cinco partições.',
               'Os dois grupos de tamanho de entrada usam coortes ligeiramente',
               'diferentes (17.272 e 17.270 gravações); compare principalmente',
               'condições **dentro de cada grupo**. A inicialização das redes é',
               'estocástica e não foi repetida com várias sementes.', '']
    text = '\n'.join(report).replace('17,272', '17.272').replace('17,270', '17.270')
    OUTPUT.write_text(text)
    print(OUTPUT)


if __name__ == '__main__':
    main()
