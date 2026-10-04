#!/usr/bin/env python3
"""Compara deltas só na validação e conclui os cinco folds da x-vector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'output/vctk16_xvector_results'


def command(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def validation_f1(mode: str) -> dict[str, float]:
    results = {}
    for source in ('mic1', 'mic2'):
        path = RESULTS / mode / 'fold1' / source / 'validation_metrics.json'
        results[source] = json.loads(path.read_text())['f1_macro']
    return results


def main() -> None:
    static = validation_f1('static')
    command('experiments/run_vctk16_xvector.py', '--mode', 'dynamic',
            '--max-folds', '1', '--validation-only')
    dynamic = validation_f1('dynamic')
    static_mean = sum(static.values()) / 2
    dynamic_mean = sum(dynamic.values()) / 2
    chosen = 'dynamic' if dynamic_mean > static_mean else 'static'
    selection = {
        'criterion': 'mean macro F1 on fold 1 validation, mic1 and mic2 sources',
        'static': {'per_source': static, 'mean': static_mean},
        'dynamic': {'per_source': dynamic, 'mean': dynamic_mean},
        'chosen': chosen,
        'test_seen_before_selection': False,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'representation_selection.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + '\n')
    print('Representação escolhida:', chosen, selection, flush=True)
    command('experiments/run_vctk16_xvector.py', '--mode', chosen)
    command('experiments/summarize_vctk16_xvector.py', '--mode', chosen)


if __name__ == '__main__':
    main()
