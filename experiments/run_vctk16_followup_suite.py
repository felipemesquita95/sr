#!/usr/bin/env python3
"""Wait for baseline/CMN/RASTA, then run CMN+RASTA and CMVN at 30 MFCCs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from run_vctk16_channel_suite import (
    ROOT, COHORT, FEATURES, RASTA_FEATURES, ARCHITECTURES, result_root, STORAGE,
)

STATE = ROOT / 'output/vctk16_followup_suite_status.json'
REPORT = ROOT / 'output/vctk16_followup_suite_comparison.md'
PREREQUISITE = ROOT / 'output/vctk16_channel_suite_status.json'
N_MFCC = 30


def output_root(architecture: str, condition: str) -> Path:
    root = result_root(architecture, condition)
    return root if N_MFCC == 30 else root.with_name(root.name.replace('vctk16_', 'vctk40_'))


def plan(n_mfcc: int = 30) -> list[dict]:
    stages = []
    if n_mfcc == 40:
        stages = [dict(name='prepare/MFCC40', condition='prepare_baseline', status='pending'),
                  dict(name='prepare/RASTA40', condition='prepare_rasta', status='pending')]
    conditions = ('cmn_rasta', 'cmvn') if n_mfcc == 30 else (
        'baseline', 'cmn', 'rasta', 'cmn_rasta', 'cmvn')
    return stages + [dict(name=f'{architecture}/{condition}', architecture=architecture,
                 condition=condition, status='pending')
            for condition in conditions
            for architecture in ARCHITECTURES]


def save(state: dict) -> None:
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATE)


def report() -> None:
    lines = ['# VCTK 16 kHz — comparação com CMN+RASTA e CMVN', '',
             f'{N_MFCC} MFCCs + Δ + ΔΔ, 111 quadros, lote 128; mesma coorte, folds e '
             'sementes. Acurácia média ± desvio nos cinco folds.', '',
             '| Rede | Condição | mic1→mic1 | mic1→mic2 | mic2→mic1 | mic2→mic2 |',
             '|---|---|---:|---:|---:|---:|']
    for architecture in ARCHITECTURES:
        for condition in ('baseline', 'cmn', 'rasta', 'cmn_rasta', 'cmvn'):
            summary = output_root(architecture, condition) / 'dynamic/summary.json'
            data = json.loads(summary.read_text()) if summary.exists() else None
            cells = []
            for direction in ('mic1->mic1', 'mic1->mic2', 'mic2->mic1', 'mic2->mic2'):
                value = data['directions'][direction]['accuracy'] if data else None
                cells.append(f"{100*value['mean']:.2f} ± {100*value['std']:.2f}%"
                             if value else 'pendente')
            lines.append('| ' + ' | '.join((architecture, condition, *cells)) + ' |')
    lines += ['', 'CMN+RASTA: características RASTA existentes, depois subtração '
              f'da média dos {N_MFCC} MFCCs estáticos na janela; deltas mantidos.', '',
              'CMVN: características baseline; média e desvio de cada MFCC '
              'estático calculados somente na janela da própria gravação. '
              'Estáticos centrados e divididos pelo desvio (piso 1e-8); Δ e ΔΔ '
              'divididos pelo mesmo desvio estático, equivalente à transformação '
              'dos deltas após CMVN. Não há combinação CMVN+RASTA.', '',
              'O z-score global continua estimado somente no treino de origem. '
              'Checkpoint escolhido na validação de origem. Folds compartilham '
              'locutores; resultados não isolam efeitos de sessão.', '']
    REPORT.write_text('\n'.join(lines))


def call(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    global STATE, REPORT, PREREQUISITE, FEATURES, RASTA_FEATURES, N_MFCC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--n-mfcc', type=int, choices=(30, 40), default=30)
    args = parser.parse_args()
    N_MFCC = args.n_mfcc
    if N_MFCC == 40:
        STATE = ROOT / 'output/vctk40_suite_status.json'
        REPORT = ROOT / 'output/vctk40_suite_comparison.md'
        PREREQUISITE = ROOT / 'output/vctk16_followup_suite_status.json'
        FEATURES = STORAGE / 'vctk40_corrected_features'
        RASTA_FEATURES = STORAGE / 'vctk40_rasta_features'
    stages = plan(N_MFCC)
    if args.dry_run:
        print(json.dumps(stages, indent=2))
        return
    (ROOT / 'tmp').mkdir(exist_ok=True)
    own_lock = (ROOT / f'tmp/vctk{N_MFCC}_followup_suite.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='waiting_for_current_suite', stages=stages,
                 started_at=datetime.now(timezone.utc).isoformat(), current_stage=None)
    save(state)
    report()
    print('Aguardando conclusão de', PREREQUISITE, flush=True)
    gpu_lock = (ROOT / 'tmp/vctk16_channel_suite.lock').open('a')
    while True:
        if not PREREQUISITE.exists():
            time.sleep(30)
            continue
        prerequisite = json.loads(PREREQUISITE.read_text())
        if prerequisite['status'] in ('finished_with_errors', 'blocked_no_gpu'):
            state['status'] = 'blocked_by_current_suite'
            save(state)
            raise RuntimeError('Current suite did not complete successfully')
        if prerequisite['status'] == 'complete':
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                if json.loads(PREREQUISITE.read_text())['status'] == 'complete':
                    break
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        time.sleep(30)
    os.environ.update(KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda',
                      OMP_NUM_THREADS='4', MPLCONFIGDIR=str(ROOT / 'tmp/matplotlib'))
    for name in ('HSA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        os.environ.pop(name, None)
    state['status'] = 'running'
    save(state)
    preparation_failed = False
    for stage in stages:
        condition = stage['condition']
        if preparation_failed:
            stage['status'] = 'blocked_by_feature_preparation'
            save(state)
            continue
        stage['status'] = 'running'
        state['current_stage'] = stage['name']
        save(state)
        print('Starting stage:', stage['name'], flush=True)
        try:
            if condition == 'prepare_baseline':
                call(['experiments/build_vctk16_features.py', '--cohort', str(COHORT),
                      '--output', str(FEATURES), '--n-mfcc', str(N_MFCC),
                      '--reference', str(ROOT / 'output/vctk16_corrected_features')])
            elif condition == 'prepare_rasta':
                call(['experiments/build_vctk16_rasta_features.py', '--cohort', str(COHORT),
                      '--baseline', str(FEATURES), '--output', str(RASTA_FEATURES),
                      '--n-mfcc', str(N_MFCC)])
            else:
                output = output_root(stage['architecture'], condition)
                features = RASTA_FEATURES if condition in ('rasta', 'cmn_rasta') else FEATURES
                command = ['experiments/run_vctk16_xvector.py', '--cohort', str(COHORT),
                           '--features', str(features), '--output', str(output),
                           '--architecture', stage['architecture'], '--mode', 'dynamic',
                           '--batch-size', '128', '--require-gpu', '--n-mfcc', str(N_MFCC)]
                if condition in ('cmn', 'cmn_rasta'):
                    command += ['--cmn-alpha', '1']
                elif condition == 'cmvn':
                    command += ['--cmvn']
                call(command)
                call(['experiments/summarize_vctk16_xvector.py', '--root', str(output),
                      '--cohort', str(COHORT), '--mode', 'dynamic'])
        except Exception as error:
            stage.update(status='failed', error=str(error))
            preparation_failed = condition.startswith('prepare_')
            print('Failed stage:', stage['name'], error, flush=True)
        else:
            stage['status'] = 'complete'
            print('Completed stage:', stage['name'], flush=True)
        save(state)
        report()
    state.update(current_stage=None, status=('complete' if all(
        stage['status'] == 'complete' for stage in stages) else 'finished_with_errors'))
    save(state)
    print('Followup suite:', state['status'], flush=True)
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
