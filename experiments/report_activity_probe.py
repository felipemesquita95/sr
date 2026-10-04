#!/usr/bin/env python3
"""Consolida o experimento de atividade e baixa atividade do VCTK."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'runs/models'
SELECTION = ROOT / 'docs/vctk_activity_probe_selection.json'
OUTPUT = ROOT / 'docs/resultado_atividade_baixa_vctk.md'
CONDITIONS = ('unfiltered', 'active', 'low')
LABELS = {'unfiltered': 'Sem seleção', 'active': 'Atividade', 'low': 'Baixa atividade'}
ARCHITECTURES = ('cnn', 'temporal_cnn', 'attention')


def percent(value: float) -> str:
    return f'{value*100:.2f}'.replace('.', ',') + '%'


def fold_result(prefix: str, condition: str, mic: str,
                architecture: str, fold: int) -> tuple[dict, dict]:
    path = MODELS / f'{prefix}_{condition}_{mic}' / architecture / f'particao{fold}'
    division = json.loads((path / 'divisao.json').read_text())
    metrics = json.loads((path / 'metricas.json').read_text())
    if not (path / 'modelo.keras').is_file():
        raise FileNotFoundError(path / 'modelo.keras')
    if division['teste'] != metrics['num_amostras_teste']:
        raise ValueError(f'Métrica e partição incompatíveis: {path}')
    return division, metrics


def main() -> None:
    selection = json.loads(SELECTION.read_text())
    prefix = selection['prefix']
    k = selection['quadros']
    lines = [
        '# VCTK: quadros sem seleção, com atividade e com baixa atividade', '',
        f'Foram usadas as mesmas {selection["pares"]:,} gravações pareadas, dos '
        f'{selection["locutores"]} locutores, com {k} quadros por gravação em cada condição.',
        'As posições de atividade e baixa atividade foram escolhidas a partir do',
        'áudio original inteiro, após o filtro e a reamostragem para 8 kHz.',
        'A condição sem seleção amostra uniformemente o áudio inteiro;',
        'as outras duas amostram posições com rótulo comum aos dois microfones.',
        'Os quadros são espalhados pela gravação e concatenados em uma entrada',
        'curta; eles não formam necessariamente um trecho contínuo de fala.',
        'Há uma margem de um quadro em cada transição. Treino, validação e teste',
        'usam as mesmas gravações nas três condições e nos dois microfones.', '',
        'O detector mede energia, não uma anotação humana de fala. Baixa atividade',
        'não deve ser interpretada como silêncio garantido.', '',
        '## Teste no mesmo microfone', '',
        '| Microfone | Rede | Sem seleção | Atividade | Baixa atividade |',
        '|---|---|---:|---:|---:|',
    ]

    reference = {}
    intra = {}
    for mic in ('mic1', 'mic2'):
        for architecture in ARCHITECTURES:
            values = {}
            for condition in CONDITIONS:
                scores = []
                for fold in range(1, 6):
                    division, metrics = fold_result(prefix, condition, mic, architecture, fold)
                    if division['quadros_por_gravacao'] != k:
                        raise ValueError(f'Comprimento diferente em {condition}/{mic}/{architecture}/{fold}')
                    fingerprint = tuple(division[key] for key in
                                        ('particao', 'treino', 'validacao', 'teste',
                                         'validation_fold_offset', 'num_folds', 'validation_seed'))
                    key = (mic, fold)
                    if key in reference and fingerprint != reference[key]:
                        raise ValueError(f'Partições diferentes em {condition}/{mic}/{architecture}/{fold}')
                    reference[key] = fingerprint
                    scores.append(metrics['acuracia'])
                values[condition] = mean(scores)
                intra[(condition, mic, architecture)] = values[condition]
            lines.append(f'| {mic} | {architecture} | '
                         + ' | '.join(percent(values[c]) for c in CONDITIONS) + ' |')

    lines += ['', '## Teste ao trocar de microfone', '',
              '| Treino → teste | Rede | Sem seleção | Atividade | Baixa atividade |',
              '|---|---|---:|---:|---:|']
    for source, target in (('mic1', 'mic2'), ('mic2', 'mic1')):
        for architecture in ARCHITECTURES:
            values = []
            for condition in CONDITIONS:
                path = MODELS / f'{prefix}_{condition}_cross_pareado' / \
                    f'{source}_para_{target}' / architecture / 'resumo.json'
                summary = json.loads(path.read_text())
                origin = intra[(condition, source, architecture)]
                if abs(summary['acuracia_origem_media'] - origin) > 1e-6:
                    raise ValueError(f'Cross-mic diverge da origem: {path}')
                values.append(summary['acuracia_outro_media'])
            lines.append(f'| {source} → {target} | {architecture} | '
                         + ' | '.join(percent(value) for value in values) + ' |')

    lines += ['',
              f'Acaso em 108 classes: {percent(1/108)}. Cada célula resume cinco partições.',
              'A detecção usa o mesmo critério de energia nas três condições;',
              'a interpretação deve levar em conta classificação incorreta de fala fraca,',
              'respiração e ruído. Os modelos têm inicialização estocástica; uma execução',
              'por partição não mede toda a variabilidade de treino.', '',
              f'Arquivo de seleção: `{SELECTION.relative_to(ROOT)}`; '
              f'SHA-256 das chaves: `{selection["sha256_gravacoes"]}`.', '']
    OUTPUT.write_text('\n'.join(lines).replace(f'{selection["pares"]:,}',
                                               f'{selection["pares"]:,}'.replace(',', '.')))
    print(OUTPUT)


if __name__ == '__main__':
    main()
