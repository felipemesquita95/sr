#!/usr/bin/env python3
"""Measure speech frames after the existing soft trim, without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import decimate
from silero_vad import get_speech_timestamps, load_silero_vad

from process_vctk_pair_corrected import soft_crop


AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')


def scan_one(item: dict, model) -> dict:
    result = {'speaker': item['speaker'], 'utterance': item['utterance'],
              'fold_group': item['fold_group'], 'mics': {}}
    start, end = item['trim_start_sample'], item['trim_end_sample']
    for mic in ('mic1', 'mic2'):
        path = AUDIO / item['speaker'] / f"{item['speaker']}_{item['utterance']}_{mic}.flac"
        original, rate = sf.read(path, dtype='float32')
        if rate != 48000 or original.ndim != 1:
            raise ValueError(f'Unexpected input: {path}')
        reduced = decimate(soft_crop(original, start, end, rate),
                           q=3, n=8, ftype='iir', zero_phase=True)
        timestamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(reduced, dtype=np.float32)), model,
            sampling_rate=16000, threshold=.5, min_speech_duration_ms=250,
            min_silence_duration_ms=100, speech_pad_ms=30)
        frame_starts = np.arange(0, len(reduced) - 512 + 1, 256)
        centers = frame_starts + 256
        speech = np.zeros(len(centers), dtype=bool)
        for segment in timestamps:
            speech |= (centers >= segment['start']) & (centers < segment['end'])
        transitions = np.diff(np.r_[False, speech, False].astype(np.int8))
        runs = (np.flatnonzero(transitions == -1) -
                np.flatnonzero(transitions == 1))
        result['mics'][mic] = {
            'frames_total': int(len(speech)),
            'frames_speech': int(speech.sum()),
            'longest_contiguous_speech': int(runs.max()) if len(runs) else 0,
            'speech_segments_samples': timestamps,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_cohort_120.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_silero/scan.jsonl'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--stop-index', type=int, default=0)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    items = cohort['items'][:args.limit] if args.limit else cohort['items']
    items = items[args.start_index:args.stop_index or None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        with args.output.open() as stream:
            for line in stream:
                row = json.loads(line)
                done.add((row['speaker'], row['utterance']))
    torch.set_num_threads(1)
    model = load_silero_vad()
    with args.output.open('a') as stream:
        for index, item in enumerate(items, 1):
            if (item['speaker'], item['utterance']) in done:
                continue
            row = scan_one(item, model)
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            if index % 100 == 0:
                stream.flush()
                print(f'{index}/{len(items)} pairs scanned', flush=True)
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
