#!/usr/bin/env python3
"""Extrai MFCCs estáticos pareados para três condições de atividade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import decimate

from process_vctk_pair_corrected import features_from_reduced


AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/'
             'VCTK-Corpus-0.92/wav48_silence_trimmed')


def build(cohort_path: Path, output: Path, limit: int = 0) -> None:
    cohort = json.loads(cohort_path.read_text())
    k = cohort['frames_per_recording']
    items = cohort['items'][:limit] if limit else cohort['items']
    for number, item in enumerate(items, 1):
        needs = {}
        for condition in cohort['conditions']:
            folder = output / condition / item['speaker'] / item['utterance']
            for mic in ('mic1', 'mic2'):
                path = folder / f'{mic}.npz'
                if path.exists():
                    with np.load(path) as archive:
                        if archive['mfcc'].shape == (30, k):
                            continue
                needs[(condition, mic)] = path
        if not needs:
            continue
        for mic in ('mic1', 'mic2'):
            if not any(key[1] == mic for key in needs):
                continue
            path = AUDIO / item['speaker'] / (
                f"{item['speaker']}_{item['utterance']}_{mic}.flac")
            audio, rate = sf.read(path, dtype='float64')
            if rate != 48000 or audio.ndim != 1:
                raise ValueError(f'Áudio inesperado: {path}')
            reduced = decimate(audio, q=3, n=8, ftype='iir', zero_phase=True)
            static, _, _ = features_from_reduced(reduced, 16000, 30,
                                                  compute_derivatives=False)
            if static.shape[1] != item['frames_full']:
                raise ValueError(f'Grade de quadros mudou: {path}: '
                                 f'{static.shape[1]} != {item["frames_full"]}')
            for condition in cohort['conditions']:
                target = needs.get((condition, mic))
                if target is None:
                    continue
                indices = np.asarray(item['indices'][condition], dtype=np.int64)
                matrix = static[:, indices]
                if matrix.shape != (30, k):
                    raise ValueError(f'Matriz incorreta: {target}: {matrix.shape}')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.stem + '.tmp.npz')
                np.savez_compressed(temporary, mfcc=matrix)
                temporary.replace(target)
        if number % 100 == 0:
            print(f'{number}/{len(items)} pares extraídos', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_low_cohort.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_low_features'))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    build(args.cohort, args.output, args.limit)
