#!/usr/bin/env python3
"""40 MFCCs, VCTK/BRSD, CNN/temporal CNN; isolated normalizations, no X-vector."""
import json
from pathlib import Path
import run_normalization_remaining as base

ROOT, STORAGE = base.ROOT, base.STORAGE
VCTK_FEATURES = STORAGE/'vctk40_isolated_features'
BRSD_FEATURES = STORAGE/'brsd40_silero_isolated_features'
CONDITIONS = ('zscore', 'cmn', 'cmvn', 'rasta', 'cmvn_logmel')
ARCHITECTURES = ('cnn', 'temporal_cnn')
_valid = base.valid_stage
_command = base.command
_report = base.report


def plan():
    stages = [dict(name=f'vctk/prepare_{kind}', dataset='vctk', condition='prepare',
                   kind=kind, feature_root=str(VCTK_FEATURES/kind), expected_count=12960)
              for kind in ('baseline', 'rasta', 'cmvn_logmel')]
    stages += [dict(name=f'vctk/{architecture}/{condition}', dataset='vctk',
                    architecture=architecture, condition=condition,
                    feature_root=str(VCTK_FEATURES/(condition if condition in ('rasta', 'cmvn_logmel') else 'baseline')),
                    result_root=str(STORAGE/f'isolated_{condition}_vctk40_{architecture}_gpu_b128'))
               for condition in CONDITIONS for architecture in ARCHITECTURES]
    stages += [dict(name='brsd/prepare_silero40', dataset='brsd', condition='prepare',
                    feature_root=str(BRSD_FEATURES), expected_count=400)]
    stages += [dict(name=f'brsd/{architecture}/{condition}', dataset='brsd',
                    architecture=architecture, condition=condition,
                    feature_root=str(BRSD_FEATURES/condition),
                    result_root=str(STORAGE/f'isolated_silero_{condition}_brsd40_{architecture}_gpu_b128'))
               for condition in CONDITIONS for architecture in ARCHITECTURES]
    return [dict(**stage, n_mfcc=40, status='pending') for stage in stages]


def valid_stage(stage):
    if not _valid(stage):
        return False
    if stage['condition'] == 'prepare':
        data = json.loads((Path(stage['feature_root'])/'protocol.json').read_text())
        return data.get('protocol', data).get('n_mfcc') == 40
    root = Path(stage['result_root'])/'dynamic'
    channels = ('audio',) if stage['dataset'] == 'brsd' else ('mic1', 'mic2')
    for fold in range(1, 6):
        for channel in channels:
            completed = json.loads((root/f'fold{fold}'/channel/'completed.json').read_text())
            if completed.get('n_mfcc') != 40 or completed.get('architecture') != stage['architecture']:
                return False
    return True


def command(stage):
    if stage['condition'] == 'prepare':
        if stage['dataset'] == 'brsd':
            return ['experiments/build_brsd_silero_conditions.py', '--output',
                    stage['feature_root'], '--n-mfcc', '40', '--reference',
                    str(STORAGE/'brsd_silero_isolated_features')]
        reference = (ROOT/'output/vctk16_corrected_features' if stage['kind'] == 'baseline'
                     else VCTK_FEATURES/'baseline')
        return ['experiments/build_vctk40_conditions.py', '--kind', stage['kind'],
                '--reference', str(reference), '--output', stage['feature_root']]
    args = _command(stage)
    args[args.index('--n-mfcc')+1] = '40'
    return args


def report():
    _report()
    text = base.REPORT.read_text().replace('Normalizações restantes', 'Normalizações com 40 MFCCs')
    base.REPORT.write_text(text.replace('30 MFCCs', '40 MFCCs'))


def main():
    base.STATE = ROOT/'output/normalization40_status.json'
    base.REPORT = ROOT/'output/normalization40_comparison.md'
    base.VCTK_FEATURES = VCTK_FEATURES
    base.BRSD_FEATURES = BRSD_FEATURES
    base.plan, base.command, base.valid_stage, base.report = plan, command, valid_stage, report
    base.main()


if __name__ == '__main__':
    main()
