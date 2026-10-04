#!/usr/bin/env python3
"""Quantifica a troca entre locutores, gravações iguais e quadros de atividade."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'docs/vctk_activity8k_counts.csv'
TABLE = ROOT / 'docs/fronteira_locutores_atividade_vctk.csv'
FIGURE = ROOT / 'docs/figuras/fronteira_locutores_atividade_vctk.png'


def main() -> None:
    by_speaker = defaultdict(list)
    with SOURCE.open(newline='') as stream:
        for row in csv.DictReader(stream):
            by_speaker[int(row['locutor'])].append((
                int(row['baixa_comum_segura']), int(row['atividade_comum_segura'])))
    if len(by_speaker) != 108:
        raise ValueError('Contagem inesperada de locutores.')
    by_speaker = {key: np.asarray(value, dtype=int)
                  for key, value in by_speaker.items()}

    def active_limits(low: int, recordings: int) -> list[tuple[int, int]]:
        limits = []
        for speaker in range(1, 109):
            candidates = by_speaker[speaker]
            active = candidates[candidates[:, 0] >= low, 1]
            limit = (int(np.partition(active, -recordings)[-recordings])
                     if len(active) >= recordings else 0)
            limits.append((limit, speaker))
        return sorted(limits, key=lambda item: (-item[0], item[1]))

    rows = []
    for omitted in range(0, 21):
        keep = 108 - omitted
        for recordings in (5, 10, 15, 20, 25, 30, 35, 40, 50, 60):
            best = None
            for low in range(1, 101):
                limits = active_limits(low, recordings)
                activity = limits[keep - 1][0]
                if activity < low:
                    continue
                best = (low, activity, limits)
            if best is None:
                continue
            low, activity, limits = best
            selected = set(s for _, s in limits[:keep])
            rows.append({
                'locutores_retirados': omitted,
                'locutores_mantidos': keep,
                'gravacoes_por_locutor': recordings,
                'gravacoes_selecionadas': keep * recordings,
                'baixa_k1_max': low,
                'atividade_k2_max': activity,
                'locutores_retirados_ids': ','.join(str(s) for s in range(1, 109)
                                                   if s not in selected),
            })

    with TABLE.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.09, top=.94, hspace=.11)
    colors = {20: '#145a8d', 30: '#27804a', 40: '#d04b30'}
    for n in (20, 30, 40):
        subset = [row for row in rows if row['gravacoes_por_locutor'] == n]
        x = [row['locutores_retirados'] for row in subset]
        y = [row['baixa_k1_max'] for row in subset]
        axes[0].plot(x, y, '-o', markersize=3, color=colors[n],
                     label=f'{n} gravações por locutor')
    axes[0].set_ylabel('K1 máximo de baixa atividade')
    axes[0].set_title('Mais quadros ao retirar os locutores com menos gravações elegíveis')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.25)

    for low, color in ((10, '#145a8d'), (20, '#d04b30')):
        vals = []
        for omitted in range(21):
            keep = 108 - omitted
            counts = []
            for speaker in range(1, 109):
                matrix = by_speaker[speaker]
                counts.append(int(np.count_nonzero(
                    (matrix[:, 0] >= low) & (matrix[:, 1] >= low))))
            raw = sorted(counts, reverse=True)[keep - 1]
            vals.append(5 * (raw // 5))
        axes[1].step(range(21), vals, where='post', color=color,
                     label=f'{low} baixa + {low} atividade')
    axes[1].set_ylabel('Máx. gravações iguais por locutor')
    axes[1].set_xlabel('Locutores retirados (dos 108 originais)')
    axes[1].set_xticks(range(0, 21, 2))
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.25)
    fig.savefig(FIGURE, bbox_inches='tight')
    plt.close(fig)
    print(TABLE)
    print(FIGURE)
    for omitted, n in ((0, 20), (1, 20), (4, 20), (6, 20),
                       (8, 30), (9, 30), (10, 40)):
        row = next(r for r in rows if r['locutores_retirados'] == omitted
                   and r['gravacoes_por_locutor'] == n)
        print(row)


if __name__ == '__main__':
    main()
