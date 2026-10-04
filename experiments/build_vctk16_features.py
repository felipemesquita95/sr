#!/usr/bin/env python3
"""Extrai as duas trilhas da coorte balanceada na janela RMS comum de K quadros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_vctk_pair_corrected import process_pair


def build(cohort_path: Path, audio_root: Path, output_root: Path,
          n_mfcc: int, limit: int = 0, reference: Path | None = None) -> None:
    cohort = json.loads(cohort_path.read_text())
    k = cohort['frames_per_recording']
    items = cohort['items'][:limit] if limit else cohort['items']
    for index, item in enumerate(items, 1):
        speaker, utterance = item['speaker'], item['utterance']
        base = audio_root / speaker / f'{speaker}_{utterance}'
        output = output_root / speaker / utterance
        manifest_file = output / 'manifest.json'
        if manifest_file.exists() and (output / 'mic1.npz').exists() \
                and (output / 'mic2.npz').exists():
            previous = json.loads(manifest_file.read_text())
            if (previous['frame_counts'] == {'mic1': k, 'mic2': k}
                    and previous['n_mfcc'] == n_mfcc
                    and previous.get('delta_scope') == 'selected_window_only'
                    and previous['trim_start_sample'] == item['trim_start_sample']
                    and previous['trim_end_sample'] == item['trim_end_sample']):
                continue
        actual = process_pair(Path(str(base) + '_mic1.flac'),
                              Path(str(base) + '_mic2.flac'), output,
                              target_sr=16000, n_mfcc=n_mfcc,
                              threshold_db=-30, margin_ms=100,
                              fade_ms=8, end_margin_ms=250,
                              k_frames=k)
        if (actual['trim_start_sample'] != item['trim_start_sample']
                or actual['trim_end_sample'] != item['trim_end_sample']):
            raise ValueError(f'Trim mudou após a seleção: {speaker}_{utterance}')
        if actual['frame_counts'] != {'mic1': k, 'mic2': k}:
            raise ValueError(f'Quantidade de quadros incorreta: {speaker}_{utterance}')
        if reference:
            prior = json.loads((reference / speaker / utterance / 'manifest.json').read_text())
            keys = ('trim_start_sample', 'trim_end_sample',
                    'selected_window_start_frame', 'selected_window_end_frame')
            if any(actual[key] != prior[key] for key in keys):
                raise ValueError(f'Janela mudou em relação à referência: {speaker}_{utterance}')
        if index % 100 == 0:
            print(f'{index}/{len(items)} pares extraídos', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_cohort_120.json'))
    parser.add_argument('--audio-root', type=Path,
                        default=Path('/media/lsmsqt/HDD/datasets/vctk/'
                                     'VCTK-Corpus-0.92/wav48_silence_trimmed'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_corrected_features'))
    parser.add_argument('--n-mfcc', type=int, default=30)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    build(args.cohort, args.audio_root, args.output, args.n_mfcc, args.limit, args.reference)
