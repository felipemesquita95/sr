#!/usr/bin/env python3
"""Mede todos os pares VCTK após trim, antes de fixar o K comum."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf

from process_vctk_pair_corrected import common_bounds


def scan(manifest: Path, audio_root: Path, destination: Path,
         limit: int = 0, all_files: bool = False) -> None:
    catalog = json.loads(manifest.read_text())['enunciados']
    if all_files:
        pairs = []
        for speaker in sorted(catalog):
            directory = audio_root / speaker
            for mic1 in sorted(directory.glob(f'{speaker}_*_mic1.flac')):
                mic2 = mic1.with_name(mic1.name.replace('_mic1.flac', '_mic2.flac'))
                if mic2.exists():
                    utterance = mic1.name[len(speaker) + 1:-len('_mic1.flac')]
                    pairs.append((speaker, utterance))
    else:
        pairs = [(speaker, utterance) for speaker, mapping in sorted(catalog.items())
                 for utterance in mapping.values()]
    if limit:
        pairs = pairs[:limit]
    destination.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if destination.exists():
        with destination.open(newline='') as stream:
            done = {(row['speaker'], row['utterance'])
                    for row in csv.DictReader(stream)}
    fields = ['speaker', 'utterance', 'source_frames', 'start_sample',
              'end_sample', 'trimmed_frames_48k', 'samples_16k',
              'mfcc_frames_16k', 'active_rms_frames', 'total_rms_frames']
    with destination.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not done:
            writer.writeheader()
        for index, (speaker, utterance) in enumerate(pairs, 1):
            if (speaker, utterance) in done:
                continue
            base = audio_root / speaker / f'{speaker}_{utterance}'
            mic1, sr1 = sf.read(str(base) + '_mic1.flac', dtype='float64')
            mic2, sr2 = sf.read(str(base) + '_mic2.flac', dtype='float64')
            if sr1 != 48000 or sr2 != 48000:
                raise ValueError(f'Taxa inesperada: {speaker}_{utterance}')
            start, end, activity = common_bounds(mic1, mic2, sr1)
            remaining = end - start
            samples_16k = (remaining + 2) // 3
            mfcc_frames = max(0, 1 + (samples_16k - 512) // 256)
            writer.writerow({
                'speaker': speaker, 'utterance': utterance,
                'source_frames': len(mic1), 'start_sample': start,
                'end_sample': end, 'trimmed_frames_48k': remaining,
                'samples_16k': samples_16k, 'mfcc_frames_16k': mfcc_frames,
                'active_rms_frames': activity['active_frames'],
                'total_rms_frames': activity['total_frames'],
            })
            if index % 100 == 0:
                stream.flush()
                print(f'{index}/{len(pairs)} pares medidos', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path('runs/features/vctk_manifesto.json'))
    parser.add_argument('--audio-root', type=Path,
                        default=Path('/media/lsmsqt/HDD/datasets/vctk/'
                                     'VCTK-Corpus-0.92/wav48_silence_trimmed'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_trim_lengths.csv'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--all-files', action='store_true',
                        help='Inclui todos os pares dos 108 locutores, além do manifesto antigo.')
    args = parser.parse_args()
    scan(args.manifest, args.audio_root, args.output, args.limit, args.all_files)
