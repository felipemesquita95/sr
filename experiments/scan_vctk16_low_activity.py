#!/usr/bin/env python3
"""Conta quadros de baixa atividade no áudio completo das leituras da coorte."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import decimate


ROOT = Path(__file__).resolve().parent.parent
AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/'
             'wav48_silence_trimmed')


def frame_rms(audio: np.ndarray, frame: int = 512, hop: int = 256) -> np.ndarray:
    starts = np.arange(0, len(audio) - frame + 1, hop)
    squares = np.square(audio, dtype=np.float64)
    summed = np.r_[0.0, np.cumsum(squares)]
    return np.sqrt((summed[starts + frame] - summed[starts]) / frame)


def erode_one(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    safe[1:] &= mask[:-1]
    safe[:-1] &= mask[1:]
    return safe


def scan_pair(item: dict, top_db: float = 30.0) -> dict:
    masks = []
    audio_lengths = []
    for mic in ('mic1', 'mic2'):
        path = AUDIO / item['speaker'] / (
            f"{item['speaker']}_{item['utterance']}_{mic}.flac")
        audio, rate = sf.read(path, dtype='float32')
        if rate != 48000 or audio.ndim != 1:
            raise ValueError(f'Áudio inesperado: {path}')
        audio_lengths.append(len(audio))
        reduced = decimate(audio, q=3, n=8, ftype='iir', zero_phase=True)
        rms = frame_rms(reduced)
        peak = float(rms.max())
        active = rms >= peak * 10 ** (-top_db / 20) if peak else np.zeros_like(rms, bool)
        masks.append(active)
    if len(set(audio_lengths)) != 1 or len(masks[0]) != len(masks[1]):
        raise ValueError(f'Microfones desalinhados: {item["speaker"]}_{item["utterance"]}')
    active = np.flatnonzero(erode_one(masks[0] & masks[1]))
    low = np.flatnonzero(erode_one(~masks[0] & ~masks[1]))
    return {
        'speaker': item['speaker'], 'utterance': item['utterance'],
        'fold_group': item.get('fold_group'), 'frames_full': len(masks[0]),
        'active': active.tolist(), 'low': low.tolist(),
        'active_count': len(active), 'low_count': len(low),
        'top_db_relative_to_each_mic_peak': top_db,
        'source': 'full original wav48_silence_trimmed audio, decimated to 16 kHz',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=ROOT / 'output/vctk16_cohort_120.json')
    parser.add_argument('--all-pairs', action='store_true',
                        help='Varre todos os pares do inventário, sem folds pré-atribuídos.')
    parser.add_argument('--lengths', type=Path,
                        default=ROOT / 'output/vctk16_trim_lengths.csv')
    parser.add_argument('--reuse-jsonl', type=Path,
                        help='Reaproveita pares já medidos no mesmo detector.')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'output/vctk16_low_activity_candidates.jsonl')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.all_pairs:
        with args.lengths.open(newline='') as stream:
            items = [{'speaker': row['speaker'], 'utterance': row['utterance']}
                     for row in csv.DictReader(stream)]
    else:
        cohort = json.loads(args.cohort.read_text())
        items = cohort['items']
    items = items[:args.limit] if args.limit else items
    prior = {}
    if args.reuse_jsonl:
        with args.reuse_jsonl.open() as stream:
            for line in stream:
                row = json.loads(line)
                prior[(row['speaker'], row['utterance'])] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    with temporary.open('w') as stream:
        for index, item in enumerate(items, 1):
            key = (item['speaker'], item['utterance'])
            row = prior.get(key)
            if row is None:
                row = scan_pair(item)
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            if index % 500 == 0:
                stream.flush()
                print(f'{index}/{len(items)} pares verificados', flush=True)
    temporary.replace(args.output)
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
