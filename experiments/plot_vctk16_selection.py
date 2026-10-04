#!/usr/bin/env python3
"""Mostra distribuição de quadros após trim e a seleção balanceada."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sr-mplconfig')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    with (ROOT / 'output/vctk16_trim_lengths.csv').open(newline='') as stream:
        all_frames = np.array([int(row['mfcc_frames_16k'])
                               for row in csv.DictReader(stream)])
    cohort = json.loads((ROOT / 'output/vctk16_cohort_120.json').read_text())
    selected = np.array([item['frames_after_trim'] for item in cohort['items']])
    k = cohort['frames_per_recording']
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    bins = np.arange(0, max(np.percentile(all_frames, 99.5),
                            np.percentile(selected, 99.5)) + 10, 10)
    ax.hist(all_frames, bins=bins, alpha=.5, label='Todos os 43.873 pares',
            color='#306da1')
    ax.hist(selected, bins=bins, alpha=.65, label='Coorte: 120 por locutor',
            color='#d28b2f')
    ax.axvline(k, ls='--', lw=2, color='#b53a35',
               label=f'K comum = {k} quadros')
    ax.set(xlabel='Quadros MFCC após o trim (16 kHz)',
           ylabel='Número de gravações',
           title='Comprimento das gravações antes da seleção da janela RMS')
    ax.legend()
    output = ROOT / 'output/vctk16_distribuicao_quadros.png'
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
