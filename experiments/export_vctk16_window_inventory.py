#!/usr/bin/env python3
"""Exporta os limites e a área RMS de cada janela selecionada no VCTK."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def export(cohort_path: Path, features_root: Path, output_path: Path) -> None:
    cohort = json.loads(cohort_path.read_text())
    fields = (
        'speaker', 'utterance', 'fold_group', 'frames_after_trim',
        'trim_start_sample_48k', 'trim_end_sample_48k',
        'window_start_frame', 'window_end_frame_exclusive',
        'window_area_normalized_rms',
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in cohort['items']:
            manifest_path = (features_root / item['speaker'] /
                             item['utterance'] / 'manifest.json')
            manifest = json.loads(manifest_path.read_text())
            writer.writerow({
                'speaker': item['speaker'],
                'utterance': item['utterance'],
                'fold_group': item['fold_group'],
                'frames_after_trim': item['frames_after_trim'],
                'trim_start_sample_48k': manifest['trim_start_sample'],
                'trim_end_sample_48k': manifest['trim_end_sample'],
                'window_start_frame': manifest['selected_window_start_frame'],
                'window_end_frame_exclusive': manifest['selected_window_end_frame'],
                'window_area_normalized_rms':
                    manifest['selected_window_area_normalized_rms'],
            })


if __name__ == '__main__':
    export(Path('output/vctk16_cohort_120.json'),
           Path('output/vctk16_corrected_features'),
           Path('output/vctk16_windows.csv'))
