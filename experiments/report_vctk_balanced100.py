#!/usr/bin/env python3
"""Publica resultados do controle pareado de atividade com 100 locutores."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'runs/models'
SELECTION = ROOT / 'docs/vctk_bal100_selection.json'
OUTPUT = ROOT / 'docs/resultado_vctk_bal100_atividade.md'
CONDITIONS = ('active20', 'low20', 'unfiltered40', 'active40', 'mixed40')
ARCHITECTURES = ('cnn', 'temporal_cnn', 'attention')
LABELS = {'active20': '20 atividade', 'low20': '20 baixa',
          'unfiltered40': '40 sem seleção', 'active40': '40 atividade',
          'mixed40': '20 atividade + 20 baixa'}


def percent(value: float) -> str:
    return f'{value*100:.2f}'.replace('.', ',') + '%'


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    if (selection['locutores'], selection['gravacoes_por_locutor']) != (100, 25):
        raise ValueError('Seleção balanceada incompatível.')
    lines = [
        '# VCTK: atividade e baixa atividade, 100 locutores balanceados', '',
        'Cada uma das 100 pessoas tem 25 gravações pareadas nos dois microfones.',
        'Em cada uma das cinco partições, por pessoa são 15 para treino, 5 para',
        'validação e 5 para teste. Todas as condições usam as mesmas gravações',
        'e os mesmos papéis. Mapeamento em `vctk_bal100_selection.json`.',
        'Baixa atividade é uma medida de energia, não silêncio anotado.', '',
        'A comparação de 20 quadros confronta atividade e baixa atividade.',
        'A comparação de 40 confronta atividade pura, mistura 20+20 e quadros',
        'uniformemente amostrados do áudio inteiro. São quadros de posições',
        'possivelmente não contíguas, ordenados pela posição temporal.', '',
    ]

    same: dict[tuple[str, str, str], float] = {}
    reference = {}
    for group in (("active20", "low20"),
                  ("unfiltered40", "active40", "mixed40")):
        lines += [f'## {20 if group[0] == "active20" else 40} quadros', '',
                  '### Teste no mesmo microfone', '',
                  '| Microfone | Rede | ' + ' | '.join(LABELS[c] for c in group) + ' |',
                  '|---|---|' + '---:|' * len(group)]
        for mic in ('mic1', 'mic2'):
            for architecture in ARCHITECTURES:
                scores = {}
                for condition in group:
                    vals = []
                    for fold in range(1, 6):
                        path = MODELS / f'vctk_bal100_{condition}_{mic}' / \
                            architecture / f'particao{fold}'
                        division = json.loads((path / 'divisao.json').read_text())
                        metrics = json.loads((path / 'metricas.json').read_text())
                        expected_frames = selection['condicoes'][condition]
                        if (division['treino'], division['validacao'], division['teste'],
                            division['quadros_por_gravacao']) != \
                                (1500, 500, 500, expected_frames):
                            raise ValueError(f'Divisão inesperada: {path}')
                        signature = (division['particao'], division['treino'],
                                     division['validacao'], division['teste'],
                                     division['validation_fold_offset'],
                                     division['validation_seed'])
                        key = (mic, fold)
                        if key in reference and reference[key] != signature:
                            raise ValueError(f'Partições divergentes: {path}')
                        reference[key] = signature
                        if not (path / 'modelo.keras').is_file():
                            raise FileNotFoundError(path / 'modelo.keras')
                        vals.append(metrics['acuracia'])
                    scores[condition] = mean(vals)
                    same[condition, mic, architecture] = scores[condition]
                lines.append(f'| {mic} | {architecture} | '
                             + ' | '.join(percent(scores[c]) for c in group) + ' |')
        lines += ['', '### Troca de microfone', '',
                  '| Treino → teste | Rede | ' +
                  ' | '.join(LABELS[c] for c in group) + ' |',
                  '|---|---|' + '---:|' * len(group)]
        for source, target in (('mic1', 'mic2'), ('mic2', 'mic1')):
            for architecture in ARCHITECTURES:
                scores = {}
                for condition in group:
                    path = MODELS / f'vctk_bal100_{condition}_cross_pareado' / \
                        f'{source}_para_{target}' / architecture / 'resumo.json'
                    summary = json.loads(path.read_text())
                    if abs(summary['acuracia_origem_media'] -
                           same[condition, source, architecture]) > 1e-6:
                        raise ValueError(f'Acurácia origem divergente: {path}')
                    scores[condition] = summary['acuracia_outro_media']
                lines.append(f'| {source} → {target} | {architecture} | '
                             + ' | '.join(percent(scores[c]) for c in group) + ' |')
        lines.append('')
    lines += ['Acaso: 1% em 100 classes. Cada célula agrega cinco partições.',
              'O experimento não testa os 108 locutores do corpus original;',
              'compare apenas condições desta coorte de 100 locutores.', '']
    OUTPUT.write_text('\n'.join(lines))
    print(OUTPUT)


if __name__ == '__main__':
    main()
