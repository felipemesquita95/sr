#!/usr/bin/env python3
"""40-MFCC VCTK extraction with exactly the 30-MFCC cohort/windows."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('baseline', 'rasta', 'cmvn_logmel'), required=True)
    parser.add_argument('--cohort', type=Path, default=Path('output/vctk16_cohort_120.json'))
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.kind == 'baseline':
        from build_vctk16_features import build
        audio = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
        build(args.cohort, audio, args.output, 40, args.limit, args.reference)
        cohort = json.loads(args.cohort.read_text())
        count = min(args.limit, len(cohort['items'])) if args.limit else len(cohort['items'])
        (args.output/'protocol.json').write_text(json.dumps(dict(
            name='MFCC40-matched-to-MFCC30', n_mfcc=40, target_sr=16000,
            frames_per_recording=cohort['frames_per_recording'],
            reference=str(args.reference), cohort=str(args.cohort), pairs_processed=count,
            delta_scope='selected_window_only'), indent=2)+'\n')
    else:
        from build_vctk16_rasta_features import build
        if args.kind == 'rasta':
            build(args.cohort, args.reference, args.output, args.limit, 40)
        else:
            import torch
            from silero_vad import load_silero_vad
            from logmel_speech_normalization import PROTOCOL, make_extractor
            torch.set_num_threads(1)
            build(args.cohort, args.reference, args.output, args.limit, 40,
                  feature_extractor=make_extractor(load_silero_vad()),
                  representation={**PROTOCOL, 'n_mfcc': 40})


if __name__ == '__main__':
    main()
