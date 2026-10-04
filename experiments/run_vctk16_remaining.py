#!/usr/bin/env python3
"""Conclui x-vector, CNN temporal e CNN cepstral e monta tabela comparativa."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'output'
XVECTOR = OUTPUT / 'vctk16_xvector_results'
TEMPORAL = OUTPUT / 'vctk16_temporal_results'
CEPSTRAL = OUTPUT / 'vctk16_cnn_results'
DIRECTIONS = (('mic1', 'mic1'), ('mic1', 'mic2'),
              ('mic2', 'mic1'), ('mic2', 'mic2'))


def call(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def await_xvector() -> None:
    ready = XVECTOR / 'dynamic' / 'summary.json'
    while not ready.exists():
        print('Aguardando os cinco folds da x-vector dinâmica...', flush=True)
        time.sleep(60)


def run_model(architecture: str, mode: str, output: Path) -> None:
    call('experiments/run_vctk16_xvector.py', '--architecture', architecture,
         '--mode', mode, '--output', str(output))
    call('experiments/summarize_vctk16_xvector.py', '--mode', mode,
         '--root', str(output))


def nested_xvector() -> dict:
    """Escolhe representação pela validação de cada fold e origem, nunca pelo teste."""
    values = {f'{source}->{target}': [] for source, target in DIRECTIONS}
    choices = []
    for fold in range(1, 6):
        for source in ('mic1', 'mic2'):
            validation = {}
            for mode in ('static', 'dynamic'):
                path = XVECTOR / mode / f'fold{fold}' / source / 'validation_metrics.json'
                validation[mode] = json.loads(path.read_text())['f1_macro']
            mode = 'dynamic' if validation['dynamic'] > validation['static'] else 'static'
            choices.append({'fold': fold, 'source': source,
                            'mode': mode, 'validation_f1': validation})
            for target in ('mic1', 'mic2'):
                path = XVECTOR / mode / f'fold{fold}' / source / target / 'metrics.json'
                values[f'{source}->{target}'].append(json.loads(path.read_text()))
    summary = {
        'protocol': 'Representação escolhida separadamente em cada fold e '
                    'microfone de origem, por F1 macro na validação daquele fold.',
        'choices': choices,
        'directions': {
            direction: {
                name: {
                    'mean': float(np.mean([row[name] for row in rows])),
                    'std': float(np.std([row[name] for row in rows], ddof=1)),
                    'fold_values': [row[name] for row in rows],
                }
                for name in ('accuracy', 'precision_macro', 'recall_macro',
                             'f1_macro', 'one_vs_rest_eer',
                             'one_vs_rest_min_dcf_prior_0_01',
                             'one_vs_rest_min_dcf_prior_0_001')
            }
            for direction, rows in values.items()
        },
    }
    (XVECTOR / 'nested_representation_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


def make_report(nested: dict) -> None:
    entries = (
        ('x-vector, MFCC', XVECTOR / 'static' / 'summary.json'),
        ('x-vector, MFCC+Δ+ΔΔ', XVECTOR / 'dynamic' / 'summary.json'),
        ('x-vector, escolha por fold', None),
        ('CNN temporal, MFCC+Δ+ΔΔ', TEMPORAL / 'dynamic' / 'summary.json'),
        ('CNN cepstral, MFCC+Δ+ΔΔ', CEPSTRAL / 'dynamic' / 'summary.json'),
    )
    lines = [
        '# VCTK 16 kHz — comparação das três redes', '',
        '108 locutores; 120 gravações pareadas por locutor; 111 quadros por '
        'gravação; cinco folds 60/20/20. A divisão é por gravação e idêntica '
        'nos dois microfones. Cada modelo e seu z-score usam só o microfone de '
        'origem no treino. Teste cruzado usa o mesmo checkpoint.', '',
        'Os quadros foram escolhidos pela maior soma de RMS normalizado num '
        'intervalo contínuo de 111 quadros, comum ao par de microfones.', '',
        '| Rede | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |',
        '|---|---:|---:|---:|---:|',
    ]
    all_results = {}
    for label, path in entries:
        summary = nested if path is None else json.loads(path.read_text())
        all_results[label] = summary
        cells = []
        for source, target in DIRECTIONS:
            value = summary['directions'][f'{source}->{target}']['accuracy']
            cells.append(f"{100 * value['mean']:.2f} ± {100 * value['std']:.2f}%")
        lines.append('| ' + ' | '.join((label, *cells)) + ' |')
    for title, key in (('Precisão macro', 'precision_macro'),
                       ('Recall macro', 'recall_macro'),
                       ('F1 macro', 'f1_macro'),
                       ('EER um-contra-todos (softmax)', 'one_vs_rest_eer')):
        lines += ['', f'## {title}', '',
                  '| Rede | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |',
                  '|---|---:|---:|---:|---:|']
        for label, summary in all_results.items():
            cells = []
            for source, target in DIRECTIONS:
                value = summary['directions'][f'{source}->{target}'][key]
                cells.append(f"{100 * value['mean']:.2f} ± {100 * value['std']:.2f}%")
            lines.append('| ' + ' | '.join((label, *cells)) + ' |')
    lines += [
        '', 'As linhas com MFCC+Δ+ΔΔ usam a mesma representação para '
        'comparar arquiteturas. Essa representação foi escolhida inicialmente '
        'pela validação do fold 1 da x-vector; portanto, essa escolha global '
        'não é uma seleção aninhada independente dos cinco testes. A linha '
        '“escolha por fold” corrige isso para a x-vector: em cada fold, decide '
        'entre MFCC e MFCC+Δ+ΔΔ usando somente a validação daquele fold e '
        'microfone de origem.', '',
        'A seleção das 120 gravações mais longas por locutor favorece áudios '
        'longos. Estes números se aplicam a essa coorte, não a todas as '
        'gravações do VCTK. A janela temporal comum usa o RMS dos dois '
        'microfones; isso também faz parte do protocolo offline pareado.', '',
        'EER e minDCF por classe, quando presentes nos arquivos de métricas, '
        'foram calculados com softmax em conjunto fechado e não equivalem '
        'ao protocolo aberto de verificação por embeddings.', '',
    ]
    (OUTPUT / 'vctk16_all_models_results.md').write_text('\n'.join(lines))
    (OUTPUT / 'vctk16_all_models_results.json').write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + '\n')


def main() -> None:
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    await_xvector()
    print('x-vector dinâmica concluída; iniciando x-vector estática.', flush=True)
    run_model('xvector', 'static', XVECTOR)
    nested = nested_xvector()
    print('x-vector com escolha aninhada concluída; iniciando CNN temporal.',
          flush=True)
    run_model('temporal_cnn', 'dynamic', TEMPORAL)
    print('CNN temporal concluída; iniciando CNN cepstral.', flush=True)
    run_model('cnn', 'dynamic', CEPSTRAL)
    make_report(nested)
    print('Tabela final:', OUTPUT / 'vctk16_all_models_results.md', flush=True)


if __name__ == '__main__':
    main()
