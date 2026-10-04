#!/usr/bin/env python3
"""Fronteira de quadros ativos/baixos com 108 locutores e 60/20/20 exato."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'docs/vctk_activity8k_counts.csv'
OUTPUT = ROOT / 'docs/analise_balanceamento_atividade_vctk.csv'
FIGURE = ROOT / 'docs/figuras/balanceamento_atividade_vctk.png'


def main() -> None:
    by_speaker = defaultdict(list)
    with SOURCE.open(newline='') as stream:
        for row in csv.DictReader(stream):
            by_speaker[int(row['locutor'])].append((
                int(row['baixa_comum_segura']),
                int(row['atividade_comum_segura']),
                (int(row['enunciado']) - 1) % 5,
            ))
    if len(by_speaker) != 108:
        raise ValueError(f'Esperados 108 locutores; encontrados {len(by_speaker)}.')
    by_speaker = {speaker: np.asarray(rows, dtype=int)
                  for speaker, rows in by_speaker.items()}

    def max_active(low: int, per_speaker: int) -> int:
        limits = []
        for speaker in range(1, 109):
            available = by_speaker[speaker]
            active = available[available[:, 0] >= low, 1]
            if len(active) < per_speaker:
                return 0
            limits.append(int(np.partition(active, -per_speaker)[-per_speaker]))
        return min(limits)

    rows = []
    for low in range(1, 101):
        for per_speaker in range(5, 101, 5):
            active = max_active(low, per_speaker)
            if active == 0:
                continue
            eligible = sum(int(np.count_nonzero(
                (values[:, 0] >= low) & (values[:, 1] >= active)))
                for values in by_speaker.values())
            fixed_group_min = min(int(np.count_nonzero(
                (by_speaker[s][:, 0] >= low) &
                (by_speaker[s][:, 1] >= active) &
                (by_speaker[s][:, 2] == group)))
                for s in range(1, 109) for group in range(5))
            rows.append({
                'baixa_k1': low,
                'atividade_k2_max': active,
                'gravacoes_por_locutor': per_speaker,
                'gravacoes_selecionadas': 108 * per_speaker,
                'gravacoes_elegiveis': eligible,
                'treino_por_locutor': per_speaker * 3 // 5,
                'validacao_por_locutor': per_speaker // 5,
                'teste_por_locutor': per_speaker // 5,
                'quadros_baixa_selecionados': low * 108 * per_speaker,
                'quadros_atividade_selecionados': active * 108 * per_speaker,
                'minimo_grupo_original': fixed_group_min,
            })

    with OUTPUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    k1 = np.arange(7, 29)
    max_recordings = np.array([max((r['gravacoes_por_locutor'] for r in rows
                                    if r['baixa_k1'] == k), default=0) for k in k1])
    selected_low = k1 * max_recordings * 108

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(left=.18, right=.97, bottom=.09, top=.94, hspace=.10)
    axes[0].step(k1, max_recordings, where='post', color='#145a8d', linewidth=2)
    axes[0].scatter([9, 10, 12, 15, 20, 26],
                    max_recordings[[2, 3, 5, 8, 13, 19]],
                    color='#d04b30', zorder=3)
    axes[0].set_ylabel('Gravações iguais por locutor')
    axes[0].set_title('108 locutores; K1 de baixa atividade e K2 de atividade')
    axes[0].grid(alpha=.25)
    axes[1].plot(k1, selected_low, color='#27804a', linewidth=2,
                 label='Quadros de baixa atividade no conjunto balanceado')
    axes[1].set_ylabel('Total de quadros de baixa atividade')
    axes[1].set_yticks([0, 10_000, 20_000, 30_000],
                       ['0', '10 mil', '20 mil', '30 mil'])
    axes[1].set_xlabel('K1 mínimo por gravação')
    axes[1].grid(alpha=.25)
    axes[1].legend(loc='upper right', frameon=False)
    axes[1].set_xlim(7, 28)
    axes[0].set_ylim(-1, 40)
    fig.savefig(FIGURE, bbox_inches='tight')
    plt.close(fig)

    print(f'CSV: {OUTPUT}')
    print(f'Figura: {FIGURE}')
    print('K1 | gravações/locutor | K2 máximo | elegíveis | quadros baixos selecionados')
    for low in (9, 10, 11, 12, 15, 20, 26, 27):
        options = [r for r in rows if r['baixa_k1'] == low]
        if not options:
            print(low, 'inviável')
            continue
        best = max(options, key=lambda r: r['gravacoes_por_locutor'])
        print(low, best['gravacoes_por_locutor'], best['atividade_k2_max'],
              best['gravacoes_elegiveis'], best['quadros_baixa_selecionados'])


if __name__ == '__main__':
    main()
