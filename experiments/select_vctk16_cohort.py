#!/usr/bin/env python3
"""Seleciona 120 pares por locutor e o K comum após a contagem completa."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def select(lengths_csv: Path, expected_pairs: int, output: Path,
           per_speaker: int = 120, seed: int = 42) -> dict:
    with lengths_csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_pairs:
        raise ValueError(f'Contagem incompleta: {len(rows)} de {expected_pairs} pares.')
    keys = [(row['speaker'], row['utterance']) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Há pares duplicados no inventário.')
    if per_speaker % 5:
        raise ValueError('A quantidade por locutor deve ser múltipla de 5.')
    by_speaker = defaultdict(list)
    for row in rows:
        row['mfcc_frames_16k'] = int(row['mfcc_frames_16k'])
        by_speaker[row['speaker']].append(row)
    if len(by_speaker) != 108:
        raise ValueError(f'Esperados 108 locutores, encontrados {len(by_speaker)}.')
    if any(len(group) < per_speaker for group in by_speaker.values()):
        raise ValueError('Algum locutor tem menos gravações que o N pedido.')

    selected = []
    for speaker in sorted(by_speaker):
        group = sorted(by_speaker[speaker],
                       key=lambda row: (-row['mfcc_frames_16k'], row['utterance']))
        chosen = group[:per_speaker]
        random.Random(f'{seed}:{speaker}').shuffle(chosen)
        for index, row in enumerate(chosen):
            selected.append({
                'speaker': speaker, 'utterance': row['utterance'],
                'frames_after_trim': row['mfcc_frames_16k'],
                'fold_group': index // (per_speaker // 5),
                'trim_start_sample': int(row['start_sample']),
                'trim_end_sample': int(row['end_sample']),
            })

    k = min(row['frames_after_trim'] for row in selected)
    absolute_min = min(rows, key=lambda row: row['mfcc_frames_16k'])
    limiting = [row for row in selected if row['frames_after_trim'] == k]
    manifest = {
        'source': str(lengths_csv), 'source_pair_count': len(rows),
        'speaker_count': len(by_speaker), 'pairs_per_speaker': per_speaker,
        'selected_pairs': len(selected), 'frames_per_recording': k,
        'absolute_min_all_pairs': {
            'speaker': absolute_min['speaker'],
            'utterance': absolute_min['utterance'],
            'frames': absolute_min['mfcc_frames_16k'],
        },
        'limiting_selected_pairs': limiting,
        'selection': 'N pares mais longos após trim por locutor; '
                     'empate por identificador do enunciado',
        'folds': '5 grupos iguais por locutor, embaralhados com semente fixa; '
                 'teste=f, validacao=(f+1)%5, treino=restantes',
        'seed': seed, 'items': selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(f'{len(rows)} pares, {len(by_speaker)} locutores; '
          f'{len(selected)} selecionados; K={k}; '
          f'mínimo absoluto={absolute_min["mfcc_frames_16k"]}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=Path,
                        default=Path('output/vctk16_trim_lengths.csv'))
    parser.add_argument('--expected-pairs', type=int, default=43873)
    parser.add_argument('--per-speaker', type=int, default=120)
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_cohort_120.json'))
    args = parser.parse_args()
    select(args.lengths, args.expected_pairs, args.output, args.per_speaker)
