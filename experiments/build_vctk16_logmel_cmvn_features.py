#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from silero_vad import load_silero_vad
from build_vctk16_rasta_features import build
from logmel_speech_normalization import PROTOCOL, make_extractor

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', type=Path, default=Path('output/vctk16_cohort_120.json'))
    p.add_argument('--baseline', type=Path, default=Path('output/vctk16_corrected_features'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--n-mfcc', type=int, choices=(30, 40), default=30)
    a = p.parse_args()
    torch.set_num_threads(1)
    build(a.cohort, a.baseline, a.output, a.limit, a.n_mfcc,
          feature_extractor=make_extractor(load_silero_vad()),
          representation={**PROTOCOL, 'n_mfcc': a.n_mfcc})
