#!/usr/bin/env python3
"""Run the remaining channel-normalization experiments in sequence on GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / 'output/vctk16_cohort_120.json'
FEATURES = ROOT / 'output/vctk16_corrected_features'
STORAGE = Path('/media/lsmsqt/HDD/sr_project')
RASTA_FEATURES = STORAGE / 'vctk16_rasta_features'
STATE = ROOT / 'output/vctk16_channel_suite_status.json'
REPORT = ROOT / 'output/vctk16_channel_suite_comparison.md'
ARCHITECTURES = ('xvector', 'cnn', 'temporal_cnn')


def result_root(architecture: str, condition: str) -> Path:
    if architecture == 'xvector':
        return STORAGE / f'{condition}_vctk16_gpu_b128'
    return STORAGE / f'{condition}_vctk16_{architecture}_gpu_b128'


def plan() -> list[dict]:
    stages = []
    for architecture in ('cnn', 'temporal_cnn'):
        for condition in ('cmn', 'baseline'):
            stages.append({'name': f'{architecture}/{condition}',
                           'architecture': architecture, 'condition': condition})
    stages.append({'name': 'prepare/RASTA', 'condition': 'prepare'})
    for architecture in ARCHITECTURES:
        stages.append({'name': f'{architecture}/rasta',
                       'architecture': architecture, 'condition': 'rasta'})
    return stages


def report() -> None:
    lines = ['# VCTK 16 kHz — referência, CMN e RASTA', '',
             '108 locutores × 120 leituras pareadas; 111 quadros; cinco folds '
             '60/20/20; MFCC30+Δ+ΔΔ; lote 128. Média ± desvio entre folds. '
             'Uma célula só é preenchida após os cinco folds concluídos.', '',
             '| Rede | Condição | mic1→mic1 | mic1→mic2 | mic2→mic1 | mic2→mic2 |',
             '|---|---|---:|---:|---:|---:|']
    for architecture in ARCHITECTURES:
        for condition in ('baseline', 'cmn', 'rasta'):
            root = result_root(architecture, condition) / 'dynamic'
            summary = root / 'summary.json'
            data = json.loads(summary.read_text()) if summary.exists() else None
            cells = []
            for direction in ('mic1->mic1', 'mic1->mic2', 'mic2->mic1', 'mic2->mic2'):
                if data is None:
                    cells.append('pendente')
                else:
                    value = data['directions'][direction]['accuracy']
                    cells.append(f"{100*value['mean']:.2f} ± {100*value['std']:.2f}%")
            lines.append('| ' + ' | '.join((architecture, condition, *cells)) + ' |')
    lines += ['', 'RASTA é aplicado no log-mel do áudio recortado completo, antes '
              'da DCT. Os mesmos índices de janela da referência são mantidos. '
              'Deltas são recalculados somente nos 111 quadros selecionados. '
              'RASTA não é combinado com CMN nesta ablação.', '',
              'Todos os modelos usam apenas treino de origem para o z-score '
              'e validação de origem para escolher o checkpoint. Folds compartilham '
              'locutores; melhora cruzada não demonstra isolamento da sessão.', '']
    REPORT.write_text('\n'.join(lines))


def save(state: dict) -> None:
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATE)


def call(arguments: list[str]) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def execute_stage(stage: dict) -> None:
    condition = stage['condition']
    if condition == 'prepare':
        call(['experiments/build_vctk16_rasta_features.py', '--cohort', str(COHORT),
              '--baseline', str(FEATURES), '--output', str(RASTA_FEATURES)])
        return
    architecture = stage['architecture']
    output = result_root(architecture, condition)
    feature_root = RASTA_FEATURES if condition == 'rasta' else FEATURES
    call(['experiments/run_vctk16_xvector.py', '--cohort', str(COHORT),
          '--features', str(feature_root), '--output', str(output),
          '--architecture', architecture, '--mode', 'dynamic',
          '--cmn-alpha', '1' if condition == 'cmn' else '0',
          '--batch-size', '128', '--require-gpu'])
    call(['experiments/summarize_vctk16_xvector.py', '--root', str(output),
          '--cohort', str(COHORT), '--mode', 'dynamic'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stages = plan()
    if args.dry_run:
        print(json.dumps(stages, indent=2))
        return
    os.environ['KERAS_BACKEND'] = 'torch'
    os.environ['KERAS_TORCH_DEVICE'] = 'cuda'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/matplotlib')
    for name in ('HSA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        os.environ.pop(name, None)
    # Lock only the suite's orchestrator; no concurrent GPU workers are spawned.
    import fcntl
    lock = (ROOT / 'tmp/vctk16_channel_suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not COHORT.exists() or not FEATURES.exists() or not STORAGE.exists():
        raise RuntimeError('Dataset, features or HDD unavailable')
    state = {'status': 'starting', 'stages': stages,
             'started_at': datetime.now(timezone.utc).isoformat()}
    save(state)
    report()
    try:
        call(['-u', '-c', 'import torch; '
              'assert torch.cuda.is_available(), "GPU unavailable; no CPU fallback"; '
              'print("GPU:", torch.cuda.get_device_name(0), '
              'torch.ones(1, device="cuda").item(), flush=True)'])
    except subprocess.CalledProcessError:
        state['status'] = 'blocked_no_gpu'
        save(state)
        raise
    state['status'] = 'running'
    preparation_failed = False
    for stage in stages:
        if stage['condition'] == 'rasta' and preparation_failed:
            stage['status'] = 'blocked_by_feature_preparation'
            save(state)
            continue
        stage['status'] = 'running'
        state['current_stage'] = stage['name']
        save(state)
        print(f"\nStarting stage: {stage['name']}", flush=True)
        try:
            execute_stage(stage)
        except Exception as error:
            stage['status'] = 'failed'
            stage['error'] = str(error)
            preparation_failed |= stage['condition'] == 'prepare'
            print(f"Failed stage {stage['name']}: {error}", flush=True)
        else:
            stage['status'] = 'complete'
            print(f"Completed stage: {stage['name']}", flush=True)
        save(state)
        report()
    state['status'] = ('complete' if all(s.get('status') == 'complete' for s in stages)
                       else 'finished_with_errors')
    state['current_stage'] = None
    save(state)
    print('Suite:', state['status'], REPORT, flush=True)
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
