#!/usr/bin/env python3
"""Isolated normalization ablations, CNN and temporal CNN, 30 MFCCs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from run_vctk16_channel_suite import ROOT, COHORT, FEATURES, RASTA_FEATURES, STORAGE, result_root

STATE = ROOT / 'output/vctk16_isolated_suite_status.json'
REPORT = ROOT / 'output/vctk16_isolated_suite_comparison.md'
DIRECTIONS = ('mic1->mic1', 'mic1->mic2', 'mic2->mic1', 'mic2->mic2')

def plan():
    return [dict(name=f'{architecture}/{condition}', architecture=architecture,
                 condition=condition, status='pending')
            for condition in ('zscore', 'cmn', 'cmvn', 'rasta')
            for architecture in ('cnn', 'temporal_cnn')]

def output_root(architecture, condition):
    if condition == 'zscore':
        return result_root(architecture, 'baseline')
    return STORAGE / f'isolated_{condition}_vctk16_{architecture}_gpu_b128'

def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATE)

def report():
    lines = ['# VCTK 16 kHz — normalizações separadas', '',
             '30 MFCCs + Δ + ΔΔ; 111 quadros; cinco folds 60/20/20; GPU, lote 128.',
             'Z-score apenas na condição zscore. CMN, CMVN e RASTA sem z-score global.',
             'Z-score reaproveita os baselines concluídos com o mesmo protocolo.', '',
             '| Rede | Condição | mic1→mic1 | mic1→mic2 | mic2→mic1 | mic2→mic2 |',
             '|---|---|---:|---:|---:|---:|']
    for stage in plan():
        summary = output_root(stage['architecture'], stage['condition']) / 'dynamic/summary.json'
        data = json.loads(summary.read_text()) if summary.exists() else None
        cells = []
        for direction in DIRECTIONS:
            value = data['directions'][direction]['accuracy'] if data else None
            cells.append(f"{100*value['mean']:.2f} ± {100*value['std']:.2f}%" if value else 'pendente')
        lines.append('| ' + ' | '.join([stage['architecture'], stage['condition'], *cells]) + ' |')
    lines += ['', 'CMN/CMVN por janela de gravação; CMVN escala Δ/ΔΔ pelo desvio dos estáticos.',
              'RASTA no log-mel antes da DCT, com as janelas pareadas existentes.',
              'X-vector, combinações e 40 MFCCs adiados por solicitação do usuário.', '']
    REPORT.write_text('\n'.join(lines))

def call(arguments):
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stages = plan()
    if args.dry_run:
        print(json.dumps(stages, indent=2))
        return
    lock = (ROOT / 'tmp/vctk16_channel_suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not os.path.ismount('/media/lsmsqt/HDD'):
        raise RuntimeError('HDD not mounted; refusing to start')
    os.environ.update(KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda',
                      OMP_NUM_THREADS='4', MPLCONFIGDIR=str(ROOT / 'tmp/matplotlib'))
    for name in ('HSA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        os.environ.pop(name, None)
    state = dict(status='running', stages=stages, current_stage=None,
                 started_at=datetime.now(timezone.utc).isoformat())
    save(state)
    report()
    for stage in stages:
        state['current_stage'] = stage['name']
        stage['status'] = 'running'
        save(state)
        architecture, condition = stage['architecture'], stage['condition']
        output = output_root(architecture, condition)
        features = RASTA_FEATURES if condition == 'rasta' else FEATURES
        arguments = ['experiments/run_vctk16_xvector.py', '--cohort', str(COHORT),
                     '--features', str(features), '--output', str(output),
                     '--architecture', architecture, '--mode', 'dynamic',
                     '--batch-size', '128', '--require-gpu', '--n-mfcc', '30']
        if condition != 'zscore':
            arguments.append('--no-global-zscore')
        if condition == 'cmn':
            arguments += ['--cmn-alpha', '1']
        elif condition == 'cmvn':
            arguments.append('--cmvn')
        print('Starting stage:', stage['name'], flush=True)
        try:
            call(arguments)
            call(['experiments/summarize_vctk16_xvector.py', '--root', str(output),
                  '--cohort', str(COHORT), '--mode', 'dynamic'])
        except Exception as error:
            stage.update(status='failed', error=str(error))
            state['status'] = 'finished_with_errors'
            save(state)
            report()
            raise
        stage['status'] = 'complete'
        save(state)
        report()
    state.update(status='complete', current_stage=None)
    save(state)

if __name__ == '__main__':
    main()
