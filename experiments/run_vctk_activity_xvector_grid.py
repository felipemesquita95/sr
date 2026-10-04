#!/usr/bin/env python3
"""X-vector on the completed 100-frame Silero VAD activity grid, 40 MFCCs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    100: Path('/media/lsmsqt/HDD/sr_project/vctk100_activity_20260929'),
}
NORMS = ('zscore', 'cmn')
MICS = ('mic1', 'mic2')
METRICS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'one_vs_rest_eer')


def save(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_sources() -> dict:
    manifests = {}
    vad = None
    for frames, directory in SOURCES.items():
        protocol = json.loads((directory / 'protocol.json').read_text())
        cohort = json.loads((directory / 'cohort.json').read_text())
        completed = json.loads((directory / 'features/completed.json').read_text())
        status = json.loads((directory / 'status.json').read_text())
        assert status['status'] == 'complete'
        assert protocol['silero_version'] == '6.2.3'
        assert protocol['frames'] == cohort['frames_per_recording'] == frames
        assert 40 in protocol['mfcc_counts']
        assert 'activity' in protocol['classes']
        assert set(MICS) == {'mic1', 'mic2'}
        assert len(cohort['items']) == 6480
        if vad is None:
            vad = protocol['vad']
        else:
            assert protocol['vad'] == vad
        for mic in MICS:
            assert (directory / f'features/baseline_activity_{mic}.npy').is_file()
        manifests[str(frames)] = dict(source=str(directory),
                                      manifest_sha256=cohort['manifest_sha256'],
                                      recordings=len(cohort['items']),
                                      feature_completion=completed)
    return dict(dataset='VCTK', silero_vad_version='6.2.3', vad=vad,
                train_class='activity', test_class='activity',
                frames=list(SOURCES), n_mfcc=40, feature_channels=120,
                normalizations=list(NORMS), architecture='xvector',
                microphones=list(MICS), folds=5, split='60/20/20 per speaker',
                seed=42, epochs=150, patience=15, batch_size=128,
                learning_rate=.001, gpu_required=True,
                zscore='global mean/std fitted on source training fold only',
                cmn='subtract static MFCC mean per selected recording; no global z-score',
                delta_scope='existing selected Silero speech window',
                models_total=20, evaluations_total=40,
                sources=manifests,
                comparison_note='Same 100-frame Silero cohort and folds as CNN and temporal CNN.')


def stages():
    for frames in SOURCES:
        for norm in NORMS:
            for fold in range(1, 6):
                for mic in MICS:
                    yield dict(frames=frames, normalization=norm, fold=fold, train_mic=mic)


def stage_name(stage):
    return f"{stage['frames']}_{stage['normalization']}_fold{stage['fold']}_{stage['train_mic']}"


def load_array(source: Path, mic: str, frames: int, norm: str):
    import numpy as np
    original = np.load(source / f'features/baseline_activity_{mic}.npy', mmap_mode='r')
    assert original.shape[1:] == (3, 40, frames)
    matrix = np.asarray(original).reshape(len(original), 120, frames).copy()
    if norm == 'cmn':
        matrix[:, :40] -= matrix[:, :40].mean(axis=2, keepdims=True)
    assert np.isfinite(matrix).all()
    return matrix


def evaluate(model, x, y, folder: Path):
    import numpy as np
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve

    with torch.no_grad():
        scores = np.concatenate([model(x[i:i + 128], training=False).detach().cpu().numpy()
                                 for i in range(0, len(x), 128)])
    assert np.isfinite(scores).all()
    predicted = scores.argmax(axis=1)
    report = classification_report(y, predicted, labels=np.arange(scores.shape[1]),
                                   output_dict=True, zero_division=0)
    binary = (np.arange(scores.shape[1])[None, :] == y[:, None]).ravel()
    fpr, tpr, _ = roc_curve(binary, scores.ravel(), drop_intermediate=False)
    fnr = 1 - tpr
    cross = int(np.argmin(np.abs(fpr - fnr)))
    result = dict(n_recordings=len(y), accuracy=float(np.mean(predicted == y)),
                  precision_macro=float(report['macro avg']['precision']),
                  recall_macro=float(report['macro avg']['recall']),
                  f1_macro=float(report['macro avg']['f1-score']),
                  one_vs_rest_eer=float((fpr[cross] + fnr[cross]) / 2))
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'metrics.json', result)
    save(folder / 'classification_report.json', report)
    np.savez_compressed(folder / 'predictions.npz', true=y, predicted=predicted, scores=scores)
    np.save(folder / 'confusion.npy', confusion_matrix(y, predicted,
                                                       labels=np.arange(scores.shape[1])))
    return result


def train_one(output: Path, stage: dict, smoke: bool) -> None:
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    os.environ.setdefault('KERAS_TORCH_DEVICE', 'cuda')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/sr-vctk-activity-xvector-mpl')
    import numpy as np
    import torch
    from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.models import load_model
    from keras.utils import set_random_seed
    from vctk16_normalization import global_statistics

    sys.path.insert(0, str(ROOT / 'src'))
    from sr.models import build_model

    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; refusing CPU training')
    torch.set_num_threads(4)
    frames = stage['frames']
    norm = stage['normalization']
    source = SOURCES[frames]
    cohort = json.loads((source / 'cohort.json').read_text())
    labels = np.array([item['label'] for item in cohort['items']], dtype=np.int64)
    groups = np.array([item['fold_group'] for item in cohort['items']], dtype=np.int8)
    fold = stage['fold'] - 1
    train = np.flatnonzero((groups != fold) & (groups != (fold + 1) % 5))
    validation = np.flatnonzero(groups == (fold + 1) % 5)
    test = np.flatnonzero(groups == fold)
    assert len(train) + len(validation) + len(test) == len(labels)
    assert len(validation) == len(test) == len(labels) // 5
    assert len(train) == 3 * len(test)
    assert set(train).isdisjoint(validation) and set(train).isdisjoint(test)
    if smoke:
        train, validation, test = train[:128], validation[:64], test[:64]
    model_dir = output / ('smoke' if smoke else 'models') / stage_name(stage)
    model_dir.mkdir(parents=True, exist_ok=True)
    array = load_array(source, stage['train_mic'], frames, norm)
    mean, std = global_statistics(array[train], norm == 'zscore')
    train_x = (array[train] - mean) / std
    val_x = (array[validation] - mean) / std
    np.savez(model_dir / 'normalization.npz', mean=mean, std=std)
    save(model_dir / 'input_protocol.json', dict(stage=stage,
         manifest_sha256=cohort['manifest_sha256'], silero_vad_version='6.2.3',
         input_shape=[120, frames], global_zscore=norm == 'zscore',
         cmn_per_recording=norm == 'cmn', gpu=torch.cuda.get_device_name(0),
         train=train.tolist(), validation=validation.tolist(), test=test.tolist()))
    set_random_seed(cohort['seed'] + fold)
    model = build_model('xvector', input_shape=(120, frames), num_classes=108,
                        learning_rate=.001)
    checkpoint = model_dir / 'best.keras'
    initial = 0
    previous_best = None
    if checkpoint.exists() and (model_dir / 'history.csv').exists() and not smoke:
        model = load_model(checkpoint)
        with (model_dir / 'history.csv').open() as stream:
            previous = list(csv.DictReader(stream))
        if previous:
            best = max(previous, key=lambda row: float(row['val_accuracy']))
            initial = int(best['epoch']) + 1
            previous_best = float(best['val_accuracy'])
    callbacks = [ModelCheckpoint(checkpoint, monitor='val_accuracy', mode='max',
                                 save_best_only=True,
                                 initial_value_threshold=previous_best),
                 CSVLogger(model_dir / 'history.csv', append=initial > 0),
                 ReduceLROnPlateau(monitor='val_accuracy', mode='max', factor=.2,
                                   patience=15, min_lr=1e-6),
                 EarlyStopping(monitor='val_accuracy', mode='max', patience=15,
                               restore_best_weights=True)]
    started = time.monotonic()
    model.fit(train_x, labels[train], validation_data=(val_x, labels[validation]),
              initial_epoch=initial, epochs=1 if smoke else 150, batch_size=128,
              callbacks=callbacks, verbose=2)
    model = load_model(checkpoint)
    evaluations = []
    for target_mic in MICS:
        target = load_array(source, target_mic, frames, norm)
        x = (target[test] - mean) / std
        metrics = evaluate(model, x, labels[test], model_dir / f'activity_{target_mic}')
        evaluations.append(dict(**stage, architecture='xvector', n_mfcc=40,
                                train_class='activity', test_class='activity',
                                test_mic=target_mic, **metrics))
        del target, x
    save(model_dir / 'evaluations.json', evaluations)
    if not smoke:
        save(model_dir / 'completed.json', dict(stage=stage,
             manifest_sha256=cohort['manifest_sha256'], evaluations=2,
             elapsed_seconds=time.monotonic() - started,
             eer_note='Closed-set softmax one-versus-rest diagnostic'))


def summarize(output: Path) -> None:
    import numpy as np
    rows = []
    for stage in stages():
        folder = output / 'models' / stage_name(stage)
        if (folder / 'completed.json').exists():
            rows.extend(json.loads((folder / 'evaluations.json').read_text()))
    if not rows:
        return
    fields = ['frames', 'n_mfcc', 'normalization', 'architecture', 'train_class',
              'test_class', 'train_mic', 'test_mic', 'fold']
    metric_fields = ['n_recordings', *METRICS]
    temp = output / 'resultados_folds.tmp.csv'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + metric_fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output / 'resultados_folds.csv')
    grouped = defaultdict(list)
    group_fields = fields[:-1]
    for row in rows:
        grouped[tuple(row[key] for key in group_fields)].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        record = dict(zip(group_fields, key), folds_completed=len(group), folds_expected=5)
        for metric in METRICS:
            values = [row[metric] for row in group]
            record[metric + '_mean'] = float(np.mean(values))
            record[metric + '_std'] = float(np.std(values))
        summaries.append(record)
    temp = output / 'resultados_resumo.tmp.csv'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    temp.replace(output / 'resultados_resumo.csv')


def run_suite(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plan = validate_sources()
    existing = output / 'protocol.json'
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise RuntimeError('Existing output belongs to a different protocol')
    save(existing, plan)
    env = dict(os.environ, KERAS_BACKEND='torch', KERAS_TORCH_DEVICE='cuda',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               MPLCONFIGDIR='/tmp/sr-vctk-activity-xvector-mpl')
    lock = (ROOT / 'tmp/vctk16_channel_suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (output / 'smoke/completed.json').exists():
        smoke_stage = dict(frames=100, normalization='zscore', fold=1, train_mic='mic1')
        save(output / 'status.json', dict(status='smoke', models_completed=0, models_total=20,
                                          current_model=stage_name(smoke_stage)))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(output),
                        '--stage', json.dumps(smoke_stage), '--smoke'],
                       env=env, check=True, stdout=(output / 'smoke.log').open('a'),
                       stderr=subprocess.STDOUT)
        save(output / 'smoke/completed.json', dict(stage=smoke_stage, status='passed'))
    for index, stage in enumerate(stages(), 1):
        folder = output / 'models' / stage_name(stage)
        if not (folder / 'completed.json').exists():
            save(output / 'status.json', dict(status='training', models_completed=index - 1,
                 models_total=20, current_model=stage_name(stage),
                 updated_at=datetime.now(timezone.utc).isoformat()))
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'execution.log').open('a') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), str(output),
                                '--stage', json.dumps(stage)], env=env, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        summarize(output)
        save(output / 'status.json', dict(status='training', models_completed=index,
             models_total=20, current_model=None,
             updated_at=datetime.now(timezone.utc).isoformat()))
        print(f'{index}/20 models completed: {stage_name(stage)}', flush=True)
    save(output / 'status.json', dict(status='complete', models_completed=20,
         models_total=20, current_model=None,
         updated_at=datetime.now(timezone.utc).isoformat()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage:
        train_one(output, json.loads(args.stage), args.smoke)
    elif args.launch:
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'suite_execution.log').open('a') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(output)],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        save(output / 'execution.json', dict(pid=child.pid, started_at=datetime.now(timezone.utc).isoformat()))
        print(f'Suite PID: {child.pid}', flush=True)
    else:
        try:
            run_suite(output)
        except Exception as error:
            state_file = output / 'status.json'
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            state.update(status='failed', error=repr(error),
                         updated_at=datetime.now(timezone.utc).isoformat())
            save(state_file, state)
            raise


if __name__ == '__main__':
    main()
