#!/usr/bin/env python3
"""Registra a duração em quadros de cada gravação processada a 8 kHz."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TRACKS = {'brsd': 400, 'vctk8k_mic1': 21_523, 'vctk8k_mic2': 21_523}
HOP_SECONDS = 128 / 8000


def collect(root: Path, names: list[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    summary: dict = {}
    for track in names:
        files = sorted((root / track).glob('*/*/mfccs.npy'),
                       key=lambda p: (int(p.parent.parent.name), int(p.parent.name)))
        if len(files) != TRACKS[track]:
            raise RuntimeError(f'{track}: {len(files)} arquivos; esperado {TRACKS[track]}')
        track_rows = []
        for file in files:
            matrix = np.load(file, mmap_mode='r', allow_pickle=False)
            if matrix.ndim != 2 or matrix.shape[0] != 40 or matrix.shape[1] < 1:
                raise RuntimeError(f'MFCC inválido: {file} {matrix.shape}')
            frames = int(matrix.shape[1])
            track_rows.append({
                'corpus': 'brsd' if track == 'brsd' else 'vctk',
                'trilha': track,
                'locutor': int(file.parent.parent.name),
                'enunciado': int(file.parent.name),
                'quadros': frames,
                'duracao_aprox_s': round(frames * HOP_SECONDS, 3),
            })
        rows.extend(track_rows)
        shortest = min(track_rows, key=lambda row: row['quadros'])
        longest = max(track_rows, key=lambda row: row['quadros'])
        summary[track] = {
            'gravacoes': len(track_rows),
            'media_quadros': round(sum(row['quadros'] for row in track_rows) / len(track_rows), 2),
            'menor': shortest,
            'maior': longest,
        }
    if 'vctk8k_mic1' in summary and 'vctk8k_mic2' in summary:
        vctk_rows = [row for row in rows if row['corpus'] == 'vctk']
        summary['vctk_combinado'] = {
            'gravacoes': len(vctk_rows),
            'media_quadros': round(sum(row['quadros'] for row in vctk_rows) / len(vctk_rows), 2),
            'menor': min(vctk_rows, key=lambda row: row['quadros']),
            'maior': max(vctk_rows, key=lambda row: row['quadros']),
        }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('runs/features'))
    parser.add_argument('--tracks', default=','.join(TRACKS))
    args = parser.parse_args()
    names = [name.strip() for name in args.tracks.split(',')]
    if any(name not in TRACKS for name in names):
        parser.error(f'Trilhas válidas: {", ".join(TRACKS)}')
    rows, summary = collect(args.root, names)
    csv_path = args.root / 'relatorio_comprimentos_8k.csv'
    json_path = args.root / 'relatorio_comprimentos_8k.json'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
