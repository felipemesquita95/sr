#!/usr/bin/env python3
"""Check authorized queues and resume inactive services every 30 minutes."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
import os

ROOT = Path(__file__).resolve().parent.parent
STORAGE = Path('/media/lsmsqt/HDD/sr_project')
WATCH = ROOT / 'output/vctk_watchdog_status.json'
QUEUES = (
    ('sr-vctk16-channel-suite', 'vctk16_channel_suite_status.json',
     'experiments/run_vctk16_channel_suite.py', []),
    ('sr-vctk16-followup-suite', 'vctk16_followup_suite_status.json',
     'experiments/run_vctk16_followup_suite.py', []),
    ('sr-vctk40-suite', 'vctk40_suite_status.json',
     'experiments/run_vctk16_followup_suite.py', ['--n-mfcc', '40']),
)
ISOLATED_REQUEST = ROOT / 'output/vctk16_isolated_request.json'
if ISOLATED_REQUEST.exists():
    QUEUES = (('sr-vctk16-isolated-suite', 'vctk16_isolated_suite_status.json',
               'experiments/run_vctk16_isolated_suite.py', []),)
REMAINING_REQUEST = ROOT / 'output/normalization_remaining_request.json'
if REMAINING_REQUEST.exists():
    QUEUES = (('sr-normalization-remaining', 'normalization_remaining_status.json',
               'experiments/run_normalization_remaining.py', []),)
NORMALIZATION40_REQUEST = ROOT / 'output/normalization40_request.json'
if NORMALIZATION40_REQUEST.exists():
    QUEUES = (('sr-normalization40', 'normalization40_status.json',
               'experiments/run_normalization40.py', []),)


def suite_valid(state: dict, n_mfcc: int) -> bool:
    if state.get('status') != 'complete':
        return False
    for stage in state['stages']:
        if stage.get('status') != 'complete':
            return False
        if 'dataset' in stage and ('result_root' in stage or 'expected_count' in stage):
            if stage.get('n_mfcc') == 40:
                from run_normalization40 import valid_stage
            else:
                from run_normalization_remaining import valid_stage
            if not valid_stage(stage):
                return False
            continue
        architecture = stage.get('architecture')
        if not architecture:
            feature = STORAGE / ('vctk16_rasta_features' if n_mfcc == 30 else
                                 'vctk40_rasta_features')
            if stage['condition'] in ('prepare', 'prepare_rasta'):
                protocol = json.loads((feature / 'protocol.json').read_text())
                if protocol['pairs_processed'] != 12960 or protocol['n_mfcc'] != n_mfcc:
                    return False
            continue
        middle = '' if architecture == 'xvector' else architecture + '_'
        if ISOLATED_REQUEST.exists():
            from run_vctk16_isolated_suite import output_root
            result = output_root(architecture, stage['condition']) / 'dynamic'
        else:
            result = STORAGE / f"{stage['condition']}_vctk{16 if n_mfcc == 30 else 40}_{middle}gpu_b128/dynamic"
        summary = json.loads((result / 'summary.json').read_text())
        for direction in ('mic1->mic1', 'mic1->mic2', 'mic2->mic1', 'mic2->mic2'):
            if len(summary['directions'][direction]['accuracy']['fold_values']) != 5:
                return False
        for fold in range(1, 6):
            for source in ('mic1', 'mic2'):
                d = result / f'fold{fold}' / source
                completed = json.loads((d / 'completed.json').read_text())
                if ISOLATED_REQUEST.exists():
                    expected_zscore = stage['condition'] == 'zscore'
                    if completed.get('global_zscore', True) != expected_zscore:
                        return False
                    if completed.get('cmn_alpha', 0) != (1 if stage['condition'] == 'cmn' else 0):
                        return False
                    if completed.get('cmvn', False) != (stage['condition'] == 'cmvn'):
                        return False
                    if completed.get('n_mfcc', 30) != 30:
                        return False
                if not (d / 'modelo.keras').is_file():
                    return False
    return True


def main() -> None:
    previous = json.loads(WATCH.read_text()) if WATCH.exists() else {}
    state = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                 status='checking', queues=[], retries=previous.get('retries', {}))
    try:
        if not os.path.ismount('/media/lsmsqt/HDD'):
            subprocess.run(['udisksctl', 'mount', '-b', '/dev/sda3',
                            '--filesystem-type', 'ntfs'], check=True, timeout=30)
        if not STORAGE.is_dir():
            raise RuntimeError('HDD storage unavailable; no workers started')
        state['free_gib'] = {str(p): round(shutil.disk_usage(p).free / 2**30, 2)
                             for p in (ROOT, STORAGE)}
        if min(state['free_gib'].values()) < 2:
            raise RuntimeError('Less than 2 GiB free; need space before restarting')
        all_complete = True
        for unit, filename, script, arguments in QUEUES:
            path = ROOT / 'output' / filename
            queue = json.loads(path.read_text()) if path.exists() else {}
            active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet',
                                     unit + '.service']).returncode == 0
            complete = suite_valid(queue, 40 if arguments else 30)
            state['queues'].append(dict(unit=unit, active=active,
                                        status=queue.get('status', 'not_started'),
                                        current_stage=queue.get('current_stage'),
                                        validated_complete=complete))
            if complete:
                state['retries'][unit] = 0
                continue
            all_complete = False
            if active:
                state['retries'][unit] = 0
                continue
            retries = state['retries'].get(unit, 0)
            if retries >= 3:
                raise RuntimeError(f'{unit}: three unsuccessful restart attempts; inspect logs')
            subprocess.run(['systemctl', '--user', 'reset-failed', unit + '.service'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['systemd-run', '--user', '--unit=' + unit, '--collect',
                            '--working-directory=' + str(ROOT),
                            str(ROOT / '.venv/bin/python'), '-u', str(ROOT / script),
                            *arguments], check=True)
            state['retries'][unit] = retries + 1
            state['queues'][-1]['action'] = 'restarted'
        state['status'] = 'complete' if all_complete else 'monitoring'
    except Exception as error:
        state.update(status='blocked', error=str(error))
    temporary = WATCH.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(WATCH)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if state['status'] == 'complete':
        subprocess.run(['systemctl', '--user', 'disable', '--now', 'sr-vctk-watchdog.timer'],
                       check=False)


if __name__ == '__main__':
    main()
