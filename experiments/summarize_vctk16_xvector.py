#!/usr/bin/env python3
"""Resume os cinco folds da x-vector sem misturar validação e teste."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root: Path, mode: str, cohort_path: Path | None = None) -> dict:
    metrics = (
        'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
        'one_vs_rest_eer', 'one_vs_rest_min_dcf_prior_0_01',
        'one_vs_rest_min_dcf_prior_0_001',
    )
    cohort = json.loads(cohort_path.read_text()) if cohort_path else None
    microphones = cohort.get('microphones', ('mic1', 'mic2')) if cohort else ('mic1', 'mic2')
    directions = {}
    for source in microphones:
        for target in microphones:
            records = []
            for fold in range(1, 6):
                path = root / mode / f'fold{fold}' / source / target / 'metrics.json'
                records.append(json.loads(path.read_text()))
            directions[f'{source}->{target}'] = {
                name: {
                    'mean': float(np.mean([row[name] for row in records])),
                    'std': float(np.std([row[name] for row in records], ddof=1)),
                    'fold_values': [row[name] for row in records],
                }
                for name in metrics
            }
    cohort = json.loads(cohort_path.read_text()) if cohort_path else None
    protocol = ('108 locutores, 120 leituras/locutor, 111 quadros/leitura; '
                '5 folds por leitura, 60/20/20; teste com o mesmo modelo '
                'treinado apenas no microfone de origem.')
    if cohort:
        protocol = (f"{cohort['speaker_count']} locutores, "
                    f"{cohort['pairs_per_speaker']} leituras/locutor, "
                    f"{cohort['frames_per_recording']} quadros/leitura; "
                    '5 folds por leitura, 60/20/20; teste com o mesmo modelo '
                    'treinado apenas no microfone de origem.')
        if cohort.get('dataset') == 'brsd':
            protocol = (f"BRSD, {cohort['speaker_count']} locutores, 5 textos/locutor; "
                        f"{cohort['frames_per_recording']} quadros de fala; "
                        'cinco folds leave-one-text-out, validação por locutor com semente 42; '
                        'Silero e mesmas partições entre condições. Um aparelho por locutor.')
    summary = {
        'mode': mode,
        'protocol': protocol,
        'directions': directions,
    }
    output = root / mode / 'summary.json'
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('output/vctk16_xvector_results'))
    parser.add_argument('--mode', choices=('static', 'dynamic'), required=True)
    parser.add_argument('--cohort', type=Path)
    args = parser.parse_args()
    result = summarize(args.root, args.mode, args.cohort)
    for direction, values in result['directions'].items():
        print(direction, 'accuracy', f"{values['accuracy']['mean']:.4f}",
              'F1 macro', f"{values['f1_macro']['mean']:.4f}")
