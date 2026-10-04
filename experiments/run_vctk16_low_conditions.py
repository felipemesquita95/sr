#!/usr/bin/env python3
"""Treina as três redes em atividade, baixa atividade e mistura pareadas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ARCHITECTURES = ('xvector', 'temporal_cnn', 'cnn')
CONDITIONS = ('low', 'active', 'mixed')
DIRECTIONS = ('mic1->mic1', 'mic1->mic2', 'mic2->mic1', 'mic2->mic2')


def run(cohort_path: Path, features_root: Path, result_root: Path) -> None:
    project = Path(__file__).resolve().parent.parent
    cohort = json.loads(cohort_path.read_text())
    for architecture in ARCHITECTURES:
        for condition in CONDITIONS:
            output = result_root / condition / architecture
            feature = features_root / condition
            print(f'Treinando {architecture}, condição {condition}', flush=True)
            subprocess.run([
                sys.executable, 'experiments/run_vctk16_xvector.py',
                '--architecture', architecture, '--mode', 'static',
                '--cohort', str(cohort_path), '--features', str(feature),
                '--output', str(output),
            ], cwd=project, check=True)
            subprocess.run([
                sys.executable, 'experiments/summarize_vctk16_xvector.py',
                '--mode', 'static', '--cohort', str(cohort_path),
                '--root', str(output),
            ], cwd=project, check=True)
    summaries = {
        condition: {
            architecture: json.loads((result_root / condition / architecture /
                                      'static/summary.json').read_text())
            for architecture in ARCHITECTURES
        }
        for condition in CONDITIONS
    }
    report = [
        '# VCTK 16 kHz — atividade e baixa atividade', '',
        f"{cohort['speaker_count']} locutores; {cohort['pairs_per_speaker']} "
        f"gravações por locutor; {cohort['frames_per_recording']} quadros por "
        'gravação; cinco folds 60/20/20. Mesmas gravações, quadros e folds nas '
        'três condições e nos dois microfones.', '',
        'Baixa atividade é uma medida de energia relativa ao pico de cada '
        'microfone, não silêncio anotado. Os quadros vêm do áudio completo '
        '`wav48_silence_trimmed` a 16 kHz; podem não ser contíguos. '
        'Por isso usamos 30 MFCC estáticos, sem delta/delta-delta. '
        'O controle não é diretamente comparável ao experimento principal '
        'com 111 quadros contínuos após trim.', '',
    ]
    for metric, title in (('accuracy', 'Acurácia'), ('f1_macro', 'F1 macro'),
                          ('precision_macro', 'Precisão macro'),
                          ('recall_macro', 'Recall macro')):
        report += [f'## {title}', '',
                   '| Rede | Condição | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |',
                   '|---|---|---:|---:|---:|---:|']
        for architecture in ARCHITECTURES:
            for condition in CONDITIONS:
                values = summaries[condition][architecture]['directions']
                cells = [
                    f"{100 * values[direction][metric]['mean']:.2f} ± "
                    f"{100 * values[direction][metric]['std']:.2f}%"
                    for direction in DIRECTIONS
                ]
                report.append('| ' + ' | '.join((architecture, condition, *cells)) + ' |')
        report.append('')
    report += [
        'O teste cruzado usa o mesmo checkpoint da origem; normalização e '
        'validação usam apenas o microfone de treino. O conjunto foi escolhido '
        'por disponibilidade de quadros nas duas condições, igualmente para '
        'todos os locutores.', '',
    ]
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / 'report.md').write_text('\n'.join(report))
    (result_root / 'summary.json').write_text(
        json.dumps({'cohort': cohort_path.as_posix(), 'summaries': summaries},
                   ensure_ascii=False, indent=2) + '\n')
    print('Relatório:', result_root / 'report.md', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_low_cohort.json'))
    parser.add_argument('--features', type=Path,
                        default=Path('output/vctk16_low_features'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_low_results'))
    args = parser.parse_args()
    run(args.cohort.resolve(), args.features.resolve(), args.output.resolve())
