#!/usr/bin/env python3
"""Evaluate an interrupted x-vector run using its best validation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
import numpy as np
from keras.models import load_model

from run_vctk16_xvector import load_features, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-mic', choices=('mic1', 'mic2'), required=True)
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    arrays, labels, groups = load_features(cohort, args.features, 'dynamic')
    test_idx = np.flatnonzero(groups == args.fold - 1)
    source = args.source_mic
    output = args.output / 'dynamic' / f'fold{args.fold}' / source
    checkpoint = output / 'best_epoch.keras'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    with np.load(output / 'normalization.npz') as normalization:
        mean, std = normalization['mean'], normalization['std']
    model = load_model(checkpoint)
    results = {}
    for target in ('mic1', 'mic2'):
        x = (arrays[target][test_idx].copy() - mean) / std
        results[target] = metrics(model, x, labels[test_idx],
                                  output / target, 108)
        print(f'{source}->{target}: accuracy={results[target]["accuracy"]:.4f} '
              f'F1={results[target]["f1_macro"]:.4f}', flush=True)
    (output / 'checkpoint_evaluation.json').write_text(json.dumps({
        'checkpoint': str(checkpoint),
        'best_epoch': json.loads((output / 'best_epoch.json').read_text()),
        'status': 'interrupted_before_early_stopping',
        'metrics': {target: {'accuracy': result['accuracy'],
                             'f1_macro': result['f1_macro']}
                    for target, result in results.items()},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
