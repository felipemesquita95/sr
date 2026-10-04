#!/usr/bin/env python3
"""Pick a balanced, fold-preserving subset with enough detected speech."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def speech_samples(row: dict, mic: str) -> int:
    return sum(segment['end'] - segment['start']
               for segment in row['mics'][mic]['speech_segments_samples'])


def select(cohort_path: Path, scan_path: Path, output: Path,
           k: int, per_speaker: int | None = None) -> dict:
    cohort = json.loads(cohort_path.read_text())
    scans = {(row['speaker'], row['utterance']): row
             for row in (json.loads(line) for line in scan_path.open())}
    items = cohort['items']
    if len(scans) != len(items):
        raise ValueError(f'Incomplete Silero scan: {len(scans)}/{len(items)}')
    by_group = defaultdict(list)
    required_samples = (k - 1) * 256 + 512
    for item in items:
        row = scans[(item['speaker'], item['utterance'])]
        if all(speech_samples(row, mic) >= required_samples
               for mic in ('mic1', 'mic2')):
            by_group[(item['speaker'], item['fold_group'])].append(item)
    speakers = sorted({item['speaker'] for item in items})
    limit = min(len(by_group[(speaker, fold)])
                for speaker in speakers for fold in range(5))
    n = per_speaker // 5 if per_speaker else limit
    if n < 1 or n > limit or (per_speaker and per_speaker % 5):
        raise ValueError(f'Unavailable count: requested {per_speaker}, '
                         f'limit {limit * 5} per speaker')
    chosen = []
    for speaker in speakers:
        for fold in range(5):
            options = list(by_group[(speaker, fold)])
            random.Random(f"{cohort['seed']}:{speaker}:{fold}:{k}").shuffle(options)
            chosen.extend(options[:n])
    result = {
        **{key: value for key, value in cohort.items() if key != 'items'},
        'source_cohort': str(cohort_path), 'silero_scan': str(scan_path),
        'items': chosen, 'selected_pairs': len(chosen),
        'pairs_per_speaker': 5 * n, 'frames_per_recording': k,
        'selection': 'Same source 60/20/20 fold groups; equal count per '
                     'speaker per group; each microphone independently has '
                     'enough Silero-detected speech for K MFCC frames.',
        'silero': {'version': '6.2.3', 'threshold': .5,
                   'min_speech_duration_ms': 250,
                   'min_silence_duration_ms': 100,
                   'speech_pad_ms': 30},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'K={k}, N={5*n}/speaker, total={len(chosen)}', flush=True)
    return result


def frontier(cohort_path: Path, scan_path: Path) -> None:
    cohort = json.loads(cohort_path.read_text())
    scans = {(row['speaker'], row['utterance']): row
             for row in (json.loads(line) for line in scan_path.open())}
    if len(scans) != len(cohort['items']):
        raise ValueError(f'Incomplete scan: {len(scans)}/{len(cohort["items"])}')
    speakers = sorted({item['speaker'] for item in cohort['items']})
    results = []
    for k in range(20, 301):
        by_group = defaultdict(int)
        required = (k - 1) * 256 + 512
        for item in cohort['items']:
            row = scans[(item['speaker'], item['utterance'])]
            if all(speech_samples(row, mic) >= required
                   for mic in ('mic1', 'mic2')):
                by_group[(item['speaker'], item['fold_group'])] += 1
        n = min(by_group[(speaker, fold)]
                for speaker in speakers for fold in range(5))
        results.append((k, n * 5, k * n * 5))
    for k, n, total_frames in results:
        if k not in (200, 153, 120, 111, 100, 90, 80, 70, 60, 50, 40, 30, 20):
            continue
        print(f'K={k:3} N={n:3} per speaker, '
              f'{n*len(speakers):5} total recordings, '
              f'{total_frames:5} frames/speaker')
    best = max(results, key=lambda row: row[2])
    print(f'Maximum K*N: K={best[0]}, N={best[1]}, '
          f'{best[2]} frames/speaker')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_cohort_120.json'))
    parser.add_argument('--scan', type=Path,
                        default=Path('output/vctk16_silero/scan.jsonl'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_silero/cohort.json'))
    parser.add_argument('--k', type=int)
    parser.add_argument('--per-speaker', type=int)
    args = parser.parse_args()
    if args.k:
        select(args.cohort, args.scan, args.output, args.k,
               args.per_speaker)
    else:
        frontier(args.cohort, args.scan)


if __name__ == '__main__':
    main()
