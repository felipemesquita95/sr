#!/usr/bin/env python3
"""Compara 77 e 153 quadros nas mesmas gravações e partições do VCTK."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'runs/models'
OUTPUT = ROOT / 'docs/comparacao_pareada_77_153.md'
ARCHITECTURES = ('cnn', 'temporal_cnn', 'attention')


def read_fold(mic: str, architecture: str, fold: int, frames: int) -> tuple[dict, dict]:
    variant = 'vctk77_matched153' if frames == 77 else 'vctk153'
    path = MODELS / f'{variant}_{mic}' / architecture / f'particao{fold}'
    division = json.loads((path / 'divisao.json').read_text())
    metrics = json.loads((path / 'metricas.json').read_text())
    if not (path / 'modelo.keras').is_file():
        raise FileNotFoundError(path / 'modelo.keras')
    if division['quadros_por_gravacao'] != frames:
        raise ValueError(f'Comprimento incorreto: {path}')
    return division, metrics


def main() -> None:
    with (ROOT / 'runs/features/relatorio_comprimentos_8k.csv').open(newline='') as stream:
        lengths = [int(row['quadros']) for row in csv.DictReader(stream)
                   if row['trilha'] == 'vctk8k_mic1']
    if len(lengths) != 21_523 or sum(length >= 153 for length in lengths) != 18_067:
        raise ValueError('A seleção de gravações mudou; rever os totais do relatório.')
    optimum = max(range(1, max(lengths) + 1),
                  key=lambda k: k * sum(length >= k for length in lengths))
    if optimum != 153:
        raise ValueError(f'O máximo de quadros mudou para {optimum}.')
    lines = [
        '# Comparação pareada: 77 versus 153 quadros no VCTK', '',
        'As duas durações usam as mesmas 18.067 gravações e, em cada uma das',
        'cinco partições, os mesmos conjuntos de treino, validação e teste.',
        'Todos os modelos foram treinados novamente com a duração correspondente.',
        'Ambos usam os primeiros quadros de cada gravação; não há seleção de',
        'atividade vocal neste experimento.', '',
        '| Microfone | Rede | 77 quadros | 153 quadros | Diferença (p.p.) |',
        '|---|---|---:|---:|---:|',
    ]
    for mic in ('mic1', 'mic2'):
        for architecture in ARCHITECTURES:
            scores = {77: [], 153: []}
            for fold in range(1, 6):
                (short_div, short_metric) = read_fold(mic, architecture, fold, 77)
                (long_div, long_metric) = read_fold(mic, architecture, fold, 153)
                for key in ('particao', 'treino', 'validacao', 'teste',
                            'validation_fold_offset', 'num_folds', 'validation_seed'):
                    if short_div[key] != long_div[key]:
                        raise ValueError(f'Partições diferentes: {mic} {architecture} '
                                         f'{fold} {key}')
                for frames, div, metric in ((77, short_div, short_metric),
                                            (153, long_div, long_metric)):
                    if div['teste'] != metric['num_amostras_teste']:
                        raise ValueError(f'Métrica incompatível: {mic} {architecture} '
                                         f'{fold} {frames}')
                    scores[frames].append(metric['acuracia'])
            short, long = mean(scores[77]), mean(scores[153])
            lines.append(f'| {mic} | {architecture} | {short*100:.2f}% | '
                         f'{long*100:.2f}% | {(long-short)*100:+.2f} |')
    lines += [
        '',
        '## Quantidade de quadros de entrada', '',
        '| Seleção | Gravações | Quadros por gravação | Total de quadros |',
        '|---|---:|---:|---:|',
        '| Corpus completo, 77 | 21.523 | 77 | 1.657.271 |',
        '| Seleção pareada, 77 | 18.067 | 77 | 1.391.159 |',
        '| Seleção pareada, 153 | 18.067 | 153 | 2.764.251 |',
        '',
        'Na seleção pareada, 153 entrega 1.373.092 quadros adicionais, ou',
        '98,7% mais quadros de entrada que 77. A unidade de treino é a',
        'gravação, de modo que ambas as durações continuam com 18.067 exemplos.',
        'Entre todos os cortes inteiros possíveis no corpus, 153 maximiza o',
        'produto gravações elegíveis × quadros por gravação.',
        'A acurácia dentro do mesmo microfone pode refletir voz e canal;',
        'o teste entre microfones avalia essa transferência separadamente.', '',
    ]
    OUTPUT.write_text('\n'.join(lines))
    print(OUTPUT)


if __name__ == '__main__':
    main()
