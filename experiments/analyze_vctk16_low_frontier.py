#!/usr/bin/env python3
"""Mostra o compromisso entre quadros de baixa atividade e leituras por locutor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze(source: Path, output_prefix: Path, fixed_folds: bool) -> list[dict]:
    rows = [json.loads(line) for line in source.open()]
    speakers = sorted({row['speaker'] for row in rows})
    if len(speakers) != 108:
        raise ValueError(f'Esperados 108 locutores, encontrados {len(speakers)}.')
    speaker_ids = np.array([speakers.index(row['speaker']) for row in rows])
    groups = np.array([row['fold_group'] if row['fold_group'] is not None else -1
                       for row in rows])
    low = np.array([row['low_count'] for row in rows])
    active = np.array([row['active_count'] for row in rows])
    limit = min(200, int(low.max()), int(active.max()))
    frontier = []
    for k in range(1, limit + 1):
        eligible = (low >= k) & (active >= k)
        counts = np.bincount(speaker_ids[eligible], minlength=108)
        if fixed_folds:
            if np.any(groups < 0):
                raise ValueError('Há gravações sem grupo de fold no modo fixo.')
            cells = np.bincount(5 * speaker_ids[eligible] + groups[eligible],
                                minlength=108 * 5)
            per_speaker = int(cells.min()) * 5
        else:
            per_speaker = int(counts.min() // 5) * 5
        frontier.append({
            'k_low_and_active': k,
            'eligible_pairs': int(eligible.sum()),
            'min_eligible_per_speaker': int(counts.min()),
            'recordings_per_speaker': per_speaker,
            'selected_pairs': per_speaker * 108,
            'low_frames_total': per_speaker * 108 * k,
        })
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with output_prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=frontier[0].keys())
        writer.writeheader()
        writer.writerows(frontier)
    valid = [row for row in frontier if row['recordings_per_speaker'] >= 5]
    if valid:
        best = max(valid, key=lambda row: (row['low_frames_total'],
                                           row['recordings_per_speaker'],
                                           row['k_low_and_active']))
        print('Best total frames:', best)
        print('Max K:', valid[-1])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot([r['k_low_and_active'] for r in frontier],
               [r['recordings_per_speaker'] for r in frontier])
    ax[0].set(xlabel='Quadros por gravação (K)',
              ylabel='Gravações iguais por locutor',
              title='Fronteira de balanceamento')
    ax[1].plot([r['k_low_and_active'] for r in frontier],
               [r['low_frames_total'] for r in frontier])
    ax[1].set(xlabel='Quadros por gravação (K)',
              ylabel='Total de quadros de baixa atividade',
              title='Aproveitamento do conjunto')
    for axis in ax:
        axis.grid(alpha=.2)
        if valid:
            axis.set_xlim(0, valid[-1]['k_low_and_active'] + 2)
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix('.png'), dpi=160)
    plt.close(fig)
    return frontier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output_prefix', type=Path)
    parser.add_argument('--fixed-folds', action='store_true')
    args = parser.parse_args()
    analyze(args.source, args.output_prefix, args.fixed_folds)
