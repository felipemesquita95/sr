#!/usr/bin/env python3
"""Compare the completed CMN and baseline runs with the same batch size."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cmn', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_cmn_gpu_b128_comparison.md'))
    args = parser.parse_args()
    baseline = json.loads((args.baseline / 'dynamic/summary.json').read_text())
    cmn = json.loads((args.cmn / 'dynamic/summary.json').read_text())
    lines = ['# CMN × referência — VCTK 16 kHz', '',
             'X-vector, MFCC+Δ+ΔΔ, lote 128; 108 locutores, 120 leituras por '
             'locutor, 111 quadros; cinco folds 60/20/20. Média ± desvio entre '
             'folds. Os folds compartilham locutores e não são independentes.', '',
             '| Treino → teste | Referência | CMN | Diferença de acurácia |',
             '|---|---:|---:|---:|']
    for direction in ('mic1->mic1', 'mic1->mic2', 'mic2->mic1', 'mic2->mic2'):
        a = baseline['directions'][direction]['accuracy']
        b = cmn['directions'][direction]['accuracy']
        delta = np.asarray(b['fold_values']) - np.asarray(a['fold_values'])
        lines.append(f"| {direction} | {100*a['mean']:.2f} ± {100*a['std']:.2f}% "
                     f"| {100*b['mean']:.2f} ± {100*b['std']:.2f}% "
                     f"| {100*delta.mean():+.2f} pp |")
    lines += ['', 'CMN subtrai a média temporal dos 30 MFCCs estáticos de cada '
              'gravação. As duas condições mantêm os deltas, folds e sementes '
              'e ajustam o z-score somente no treino de origem. O checkpoint '
              'é escolhido pela validação. Melhora de transferência não demonstra '
              'isolamento da identidade vocal dos efeitos de sessão.', '',
              f'CMN: `{args.cmn}`', f'Referência: `{args.baseline}`', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print(args.output)


if __name__ == '__main__':
    main()
