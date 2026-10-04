#!/usr/bin/env python3
"""Seleciona coorte pareada para atividade, baixa atividade e mistura."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def evenly(indices: list[int], count: int) -> list[int]:
    if len(indices) < count:
        raise ValueError('Quadros insuficientes.')
    if count == 0:
        return []
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)[positions].tolist()


def select(candidates: Path, output: Path, k: int,
           per_speaker: int | None = None, seed: int = 42) -> dict:
    rows = [json.loads(line) for line in candidates.open()]
    by_speaker = defaultdict(list)
    for row in rows:
        if row['low_count'] >= k and row['active_count'] >= k:
            by_speaker[row['speaker']].append(row)
    speakers = sorted({row['speaker'] for row in rows})
    if len(speakers) != 108:
        raise ValueError('O inventário não cobre os 108 locutores.')
    limit = min(len(by_speaker[speaker]) for speaker in speakers)
    n = per_speaker if per_speaker is not None else limit // 5 * 5
    if n < 5 or n % 5 or n > limit:
        raise ValueError(f'Gravações por locutor inviáveis: N={n}, limite={limit}.')
    selected = []
    for speaker in speakers:
        available = sorted(by_speaker[speaker], key=lambda row: row['utterance'])
        random.Random(f'{seed}:{speaker}').shuffle(available)
        for index, row in enumerate(available[:n]):
            low = evenly(row['low'], k)
            active = evenly(row['active'], k)
            low_half = k // 2
            mixed = sorted(evenly(row['low'], low_half)
                           + evenly(row['active'], k - low_half))
            selected.append({
                'speaker': speaker, 'utterance': row['utterance'],
                'fold_group': index // (n // 5),
                'frames_full': row['frames_full'],
                'low_count': row['low_count'],
                'active_count': row['active_count'],
                'indices': {'low': low, 'active': active, 'mixed': mixed},
            })
    result = {
        'source': str(candidates), 'speaker_count': 108,
        'pairs_per_speaker': n, 'selected_pairs': len(selected),
        'frames_per_recording': k, 'seed': seed,
        'conditions': ['low', 'active', 'mixed'],
        'selection': f'Pares com ao menos {k} quadros comuns seguros de cada '
                     'tipo; amostragem determinística igual por locutor.',
        'folds': 'Cinco grupos de mesmo tamanho por locutor; treino 3 grupos, '
                 'validação 1 e teste 1. Mesmo grupo para todas as condições e microfones.',
        'low_definition': 'RMS em quadros 32 ms/16 ms após decimação 48→16 kHz; '
                          'abaixo de -30 dB do pico de cada microfone nos dois canais, '
                          'com mesmo rótulo nos quadros vizinhos.',
        'feature_representation': '30 MFCC estáticos; deltas omitidos porque os '
                                  'quadros selecionados podem ser não contíguos.',
        'items': selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'K={k}, N={n}, pares={len(selected)}, mínimos por locutor={limit}',
          flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path,
                        default=Path('output/vctk16_low_activity_all_candidates.jsonl'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_low_cohort.json'))
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--per-speaker', type=int)
    args = parser.parse_args()
    select(args.candidates, args.output, args.k, args.per_speaker)
