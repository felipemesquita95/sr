#!/usr/bin/env python3
"""Extract MFCC from Silero-retained speech in a separate feature directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import decimate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from process_vctk_pair_corrected import features_from_reduced, soft_crop
from sr.preprocessing.window_selection import frame_rms


AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')


def best_window(audio: np.ndarray, k: int) -> tuple[int, int]:
    rms = frame_rms(audio, 512, 256)
    if len(rms) < k:
        raise ValueError(f'Only {len(rms)} MFCC frames for K={k}')
    cumulative = np.r_[0.0, np.cumsum(rms / max(float(rms.max()), 1e-12))]
    start = int(np.argmax(cumulative[k:] - cumulative[:-k]))
    return start, start + k


def build_one(item: dict, scan: dict, root: Path, k: int,
              condition: str = 'vad') -> None:
    folder = root / item['speaker'] / item['utterance']
    manifest = folder / 'manifest.json'
    if manifest.exists() and all((folder / f'{mic}.npz').exists()
                                 for mic in ('mic1', 'mic2')):
        previous = json.loads(manifest.read_text())
        if (previous.get('frames_per_recording') == k
                and previous.get('condition') == condition):
            return
    folder.mkdir(parents=True, exist_ok=True)
    info = {'speaker': item['speaker'], 'utterance': item['utterance'],
            'condition': condition,
            'frames_per_recording': k, 'feature': 'MFCC30+delta+delta_delta',
            'voice_activity_detector': 'Silero VAD 6.2.3' if condition == 'vad' else None,
            'microphone_timestamps': 'independent',
            'trim_start_sample': item['trim_start_sample'],
            'trim_end_sample': item['trim_end_sample'], 'mics': {}}
    for mic in ('mic1', 'mic2'):
        path = AUDIO / item['speaker'] / f"{item['speaker']}_{item['utterance']}_{mic}.flac"
        original, sr = sf.read(path, dtype='float32')
        if sr != 48000 or original.ndim != 1:
            raise ValueError(f'Unexpected audio: {path}')
        cropped = soft_crop(original, item['trim_start_sample'],
                            item['trim_end_sample'], sr)
        reduced = decimate(cropped, q=3, n=8, ftype='iir', zero_phase=True)
        segments = (scan['mics'][mic]['speech_segments_samples']
                    if condition == 'vad' else [])
        speech = (np.concatenate([reduced[part['start']:part['end']]
                                  for part in segments])
                  if condition == 'vad' else reduced)
        start, end = best_window(speech, k)
        static, _, _ = features_from_reduced(speech, 16000, 30,
                                             compute_derivatives=False)
        static = static[:, start:end]
        delta = librosa.feature.delta(static, width=9).astype(np.float32)
        delta2 = librosa.feature.delta(static, width=9, order=2).astype(np.float32)
        np.savez_compressed(folder / f'{mic}.npz', mfcc=static,
                            delta=delta, delta_delta=delta2)
        info['mics'][mic] = {'speech_segments': segments,
                             'speech_samples': len(speech),
                             'selected_window_start_frame': start,
                             'selected_window_end_frame': end}
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_silero/cohort.json'))
    parser.add_argument('--scan', type=Path,
                        default=Path('output/vctk16_silero/scan.jsonl'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_silero/features'))
    parser.add_argument('--condition', choices=('vad', 'baseline'), default='vad')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--stop-index', type=int, default=0)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    scans = {(row['speaker'], row['utterance']): row
             for row in (json.loads(line) for line in args.scan.open())}
    items = cohort['items'][:args.limit] if args.limit else cohort['items']
    items = items[args.start_index:args.stop_index or None]
    for index, item in enumerate(items, 1):
        build_one(item, scans[(item['speaker'], item['utterance'])],
                  args.output, cohort['frames_per_recording'], args.condition)
        if index % 100 == 0:
            print(f'{index}/{len(items)} pairs extracted', flush=True)


if __name__ == '__main__':
    main()
