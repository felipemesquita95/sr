#!/usr/bin/env python3
"""VCTK speech log-mel CMVN, then BRSD Silero isolated normalization suite."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
STORAGE = Path('/media/lsmsqt/HDD/sr_project')
STATE = ROOT/'output/normalization_remaining_status.json'
REPORT = ROOT/'output/normalization_remaining_comparison.md'
VCTK_FEATURES = STORAGE/'vctk16_speech_logmel_cmvn_features'
BRSD_FEATURES = STORAGE/'brsd_silero_isolated_features'

def plan():
    stages = [dict(name='vctk/prepare_logmel_cmvn', dataset='vctk', condition='prepare',
                   feature_root=str(VCTK_FEATURES), expected_count=12960)]
    stages += [dict(name=f'vctk/{a}/cmvn_logmel', dataset='vctk', architecture=a,
                   condition='cmvn_logmel', feature_root=str(VCTK_FEATURES),
                   result_root=str(STORAGE/f'isolated_logmel_cmvn_vctk16_{a}_gpu_b128'))
               for a in ('cnn', 'temporal_cnn')]
    stages += [dict(name='brsd/prepare_silero', dataset='brsd', condition='prepare',
                    feature_root=str(BRSD_FEATURES), expected_count=400)]
    stages += [dict(name=f'brsd/{a}/{c}', dataset='brsd', architecture=a, condition=c,
                   feature_root=str(BRSD_FEATURES/c),
                   result_root=str(STORAGE/f'isolated_silero_{c}_brsd_{a}_gpu_b128'))
               for c in ('zscore', 'cmn', 'cmvn', 'rasta', 'cmvn_logmel')
               for a in ('cnn', 'temporal_cnn')]
    return [dict(**stage, status='pending') for stage in stages]

def save(state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temp = STATE.with_suffix('.tmp.json')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    temp.replace(STATE)

def report():
    lines = ['# Normalizações restantes — VCTK e BRSD', '',
             'CNN e CNN temporal; GPU obrigatória, lote 128, cinco folds.',
             'VCTK: 16 kHz, 30 MFCCs, mesmas janelas de 111 quadros.',
             'BRSD: 16 kHz, 30 MFCCs + Δ + ΔΔ, Silero, 80 locutores × 5 textos.',
             'Z-score somente na referência BRSD zscore; outros tratamentos separados.', '',
             '| Corpus | Rede | Condição | Direção | Acurácia média ± desvio |',
             '|---|---|---|---|---:|']
    for stage in plan():
        if 'architecture' not in stage:
            continue
        summary = Path(stage['result_root'])/'dynamic'/'summary.json'
        if not summary.exists():
            lines.append(f"| {stage['dataset']} | {stage['architecture']} | {stage['condition']} | — | pendente |")
            continue
        for direction, values in json.loads(summary.read_text())['directions'].items():
            value = values['accuracy']
            lines.append(f"| {stage['dataset']} | {stage['architecture']} | {stage['condition']} | {direction} | {100*value['mean']:.2f} ± {100*value['std']:.2f}% |")
    lines += ['', 'BRSD possui um aparelho por locutor; resultados não demonstram eliminação do confundimento de canal.', '']
    REPORT.write_text('\n'.join(lines))

def valid_stage(stage):
    if stage['condition'] == 'prepare':
        p = Path(stage['feature_root'])/'protocol.json'
        return p.exists() and json.loads(p.read_text())['pairs_processed'] == stage['expected_count']
    root = Path(stage['result_root'])/'dynamic'
    summary = root/'summary.json'
    if not summary.exists():
        return False
    channels = ('audio',) if stage['dataset'] == 'brsd' else ('mic1', 'mic2')
    data = json.loads(summary.read_text())
    if set(data['directions']) != {f'{a}->{b}' for a in channels for b in channels}:
        return False
    for values in data['directions'].values():
        if len(values['accuracy']['fold_values']) != 5:
            return False
    for fold in range(1, 6):
        for channel in channels:
            folder = root/f'fold{fold}'/channel
            completed_file = folder/'completed.json'
            if not completed_file.exists() or not (folder/'modelo.keras').exists():
                return False
            completed = json.loads(completed_file.read_text())
            if (completed.get('global_zscore', True) != (stage['condition'] == 'zscore')
                    or completed.get('cmn_alpha', 0) != (1 if stage['condition'] == 'cmn' else 0)
                    or completed.get('cmvn', False) != (stage['condition'] == 'cmvn')):
                return False
    return True

def command(stage):
    if stage['condition'] == 'prepare':
        script = ('experiments/build_vctk16_logmel_cmvn_features.py' if stage['dataset'] == 'vctk'
                  else 'experiments/build_brsd_silero_conditions.py')
        return [script, '--output', stage['feature_root']]
    brsd = stage['dataset'] == 'brsd'
    cohort = BRSD_FEATURES/'cohort.json' if brsd else ROOT/'output/vctk16_cohort_120.json'
    args = ['experiments/run_vctk16_xvector.py', '--cohort', str(cohort),
            '--features', stage['feature_root'], '--output', stage['result_root'],
            '--architecture', stage['architecture'], '--mode', 'dynamic',
            '--batch-size', '128', '--require-gpu', '--n-mfcc', '30']
    if brsd:
        args += ['--source-mic', 'audio', '--epochs', '1000', '--patience', '30']
    if stage['condition'] != 'zscore':
        args += ['--no-global-zscore']
    if stage['condition'] == 'cmn':
        args += ['--cmn-alpha', '1']
    elif stage['condition'] == 'cmvn':
        args += ['--cmvn']
    return args

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    stages = plan()
    if args.dry_run:
        print(json.dumps([dict(stage=stage, command=command(stage)) for stage in stages], indent=2))
        return
    state = dict(status='waiting_for_isolated_suite', current_stage=None, stages=stages,
                 started_at=datetime.now(timezone.utc).isoformat())
    save(state)
    while True:
        previous = json.loads((ROOT/'output/vctk16_isolated_suite_status.json').read_text())
        if previous['status'] == 'complete':
            from watch_vctk_suites import suite_valid
            if not suite_valid(previous, 30):
                raise RuntimeError('Previous isolated suite failed validation')
            break
        if previous['status'] == 'finished_with_errors':
            raise RuntimeError('Previous isolated suite has errors')
        time.sleep(30)
    lock = (ROOT/'tmp/vctk16_channel_suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda', OMP_NUM_THREADS='4',
                      MPLCONFIGDIR=str(ROOT/'tmp/matplotlib'))
    for name in ('HSA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
        os.environ.pop(name, None)
    state['status'] = 'running'
    for stage in stages:
        if not os.path.ismount('/media/lsmsqt/HDD') or not STORAGE.is_dir():
            raise RuntimeError('HDD unavailable')
        if min(shutil.disk_usage(p).free for p in (ROOT, STORAGE)) < 2*2**30:
            raise RuntimeError('Less than 2 GiB free')
        state['current_stage'] = stage['name']
        stage['status'] = 'running'
        save(state)
        report()
        print('Starting stage:', stage['name'], flush=True)
        try:
            if not valid_stage(stage):
                subprocess.run([sys.executable, *command(stage)], cwd=ROOT, check=True)
                if stage['condition'] != 'prepare':
                    cohort = BRSD_FEATURES/'cohort.json' if stage['dataset'] == 'brsd' else ROOT/'output/vctk16_cohort_120.json'
                    subprocess.run([sys.executable, 'experiments/summarize_vctk16_xvector.py',
                        '--root', stage['result_root'], '--cohort', str(cohort), '--mode',
                        'dynamic'],
                        cwd=ROOT, check=True)
            if not valid_stage(stage):
                raise RuntimeError('Stage output failed validation')
        except Exception as error:
            stage.update(status='failed', error=str(error))
            state.update(status='finished_with_errors')
            save(state)
            raise
        stage['status'] = 'complete'
        save(state)
        report()
    state.update(status='complete', current_stage=None)
    save(state)

if __name__ == '__main__':
    main()
