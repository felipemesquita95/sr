#!/usr/bin/env python3
"""Treina uma arquitetura nos cinco folds e testa nos dois microfones por gravação."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
import numpy as np
from vctk16_normalization import apply_cmvn, CMVN_PROTOCOL, global_statistics
from keras.callbacks import Callback
from keras import backend as keras_backend
from keras.models import load_model
from keras.utils import set_random_seed
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from sr.config import Settings  # noqa: E402
from sr.features import DataSplit  # noqa: E402
from sr.training import TrainingSubsystem  # noqa: E402
from sr.models import build_model  # noqa: E402


class BestEpochCheckpoint(Callback):
    """Atomically keep the best validation checkpoint for interrupted runs."""

    def __init__(self, output: Path) -> None:
        super().__init__()
        self.output = output
        self.state_file = output / 'best_epoch.json'
        self.best = (float(json.loads(self.state_file.read_text())['val_accuracy'])
                     if self.state_file.exists() else -float('inf'))

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        value = float((logs or {}).get('val_accuracy', -float('inf')))
        if value <= self.best:
            return
        temporary = self.output / 'best_epoch.tmp.keras'
        final = self.output / 'best_epoch.keras'
        self.model.save(temporary)
        temporary.replace(final)
        state = self.output / 'best_epoch.json.tmp'
        state.write_text(json.dumps({'epoch': epoch + 1,
                                     'val_accuracy': value}) + '\n')
        state.replace(self.state_file)
        self.best = value


class CheckpointedTrainingSubsystem(TrainingSubsystem):
    """Resume from the best completed epoch after an interrupted local run."""

    def train(self, architecture, split, output, fold=1, *, seed=None):
        output.mkdir(parents=True, exist_ok=True)
        if seed is not None:
            set_random_seed(seed)
        checkpoint = output / 'best_epoch.keras'
        state_file = output / 'best_epoch.json'
        if checkpoint.exists() and state_file.exists():
            state = json.loads(state_file.read_text())
            initial_epoch = int(state['epoch'])
            model = load_model(checkpoint)
            print(f'Resuming {architecture} from best epoch {initial_epoch}',
                  flush=True)
            for name in ('progresso.json', 'progresso.png'):
                prior = output / name
                if prior.exists():
                    prior.replace(output / name.replace('.', '_before_resume.', 1))
        else:
            initial_epoch = 0
            model = build_model(
                architecture, input_shape=split.input_shape,
                num_classes=self.settings.num_speakers,
                learning_rate=self.settings.learning_rate)
            for name in ('progresso.json', 'progresso.png'):
                prior = output / name
                if prior.exists():
                    prior.replace(output / name.replace('.', '_interrupted.', 1))
        callbacks = self._callbacks(architecture, output, fold)
        callbacks.append(BestEpochCheckpoint(output))
        history = model.fit(
            split.train_x, split.train_y,
            validation_data=(split.validation_x, split.validation_y),
            initial_epoch=initial_epoch,
            epochs=self.settings.epochs,
            batch_size=self.settings.batch_size,
            callbacks=callbacks, verbose=2)
        best_model = load_model(checkpoint)
        best_model.save(output / 'modelo.keras')
        return best_model, history


def apply_cepstral_mean_subtraction(matrix: np.ndarray, alpha: float,
                                  n_mfcc: int = 30) -> np.ndarray:
    """Subtract each recording's static MFCC mean; deltas are unchanged.

    An additive constant has zero temporal derivative. Keeping the stored deltas
    therefore matches recomputing them after mean subtraction, while avoiding a
    second feature extraction from audio.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('cmn_alpha must be between 0 and 1')
    if alpha == 0.0:
        return matrix
    result = matrix.copy()
    result[:n_mfcc] -= alpha * result[:n_mfcc].mean(axis=1, keepdims=True)
    return result


def load_features(cohort: dict, root: Path, mode: str,
                  cmn_alpha: float = 0.0,
                  cmvn: bool = False, n_mfcc: int = 30) -> tuple[dict, np.ndarray, np.ndarray]:
    items = cohort['items']
    speakers = sorted({item['speaker'] for item in items})
    labels = np.array([speakers.index(item['speaker']) for item in items],
                      dtype=np.int64)
    groups = np.array([item['fold_group'] for item in items], dtype=np.int8)
    k = cohort['frames_per_recording']
    channels = n_mfcc if mode == 'static' else 3*n_mfcc
    arrays = {mic: np.empty((len(items), channels, k), dtype=np.float32)
              for mic in cohort.get('microphones', ('mic1', 'mic2'))}
    for index, item in enumerate(items):
        directory = root / item['speaker'] / item['utterance']
        for mic in arrays:
            with np.load(directory / f'{mic}.npz') as feature:
                static = feature['mfcc']
                if mode == 'static':
                    matrix = static
                else:
                    matrix = np.concatenate((static, feature['delta'],
                                             feature['delta_delta']), axis=0)
            if cohort.get('dataset') == 'brsd' and matrix.shape[1] >= k:
                matrix = matrix[:, :k]
            if matrix.shape != (channels, k):
                raise ValueError(f'Forma incorreta em {directory}/{mic}: {matrix.shape}')
            arrays[mic][index] = (apply_cmvn(matrix, n_mfcc) if cmvn else
                                 apply_cepstral_mean_subtraction(matrix, cmn_alpha, n_mfcc))
        if (index + 1) % 1000 == 0:
            print(f'{index + 1}/{len(items)} pares carregados', flush=True)
    return arrays, labels, groups


def metrics(model, x: np.ndarray, y: np.ndarray, output: Path,
            speakers: int) -> dict:
    scores = model.predict(x, verbose=0)
    predicted = np.argmax(scores, axis=1)
    binary = (np.arange(speakers)[None, :] == y[:, None]).ravel()
    fpr, tpr, _ = roc_curve(binary, scores.ravel(), drop_intermediate=False)
    fnr = 1 - tpr
    crossing = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[crossing] + fnr[crossing]) / 2)

    def min_dcf(prior: float) -> float:
        cost = prior * fnr + (1 - prior) * fpr
        return float(np.min(cost) / min(prior, 1 - prior))

    report = classification_report(y, predicted, labels=np.arange(speakers),
                                   output_dict=True, zero_division=0)
    result = {
        'n_recordings': len(y), 'accuracy': float(np.mean(predicted == y)),
        'precision_macro': float(report['macro avg']['precision']),
        'recall_macro': float(report['macro avg']['recall']),
        'f1_macro': float(report['macro avg']['f1-score']),
        'precision_weighted': float(report['weighted avg']['precision']),
        'recall_weighted': float(report['weighted avg']['recall']),
        'f1_weighted': float(report['weighted avg']['f1-score']),
        'one_vs_rest_eer': eer,
        'one_vs_rest_min_dcf_prior_0_01': min_dcf(.01),
        'one_vs_rest_min_dcf_prior_0_001': min_dcf(.001),
        'verification_protocol_note':
            'Scores softmax de cada classe em pares reivindicacao/gravação '
            'do conjunto fechado; não comparar numericamente com EER/minDCF '
            'de embeddings e provas abertas da literatura.',
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (output / 'classification_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    np.savez_compressed(output / 'predictions.npz',
                        true=y, predicted=predicted, scores=scores)
    np.save(output / 'confusion.npy',
            confusion_matrix(y, predicted, labels=np.arange(speakers)))
    return result


def run(cohort_path: Path, features_root: Path, output_root: Path,
        mode: str, max_folds: int, validation_only: bool,
        architecture: str = 'xvector',
        sources: tuple[str, ...] = ('mic1', 'mic2'),
        cmn_alpha: float = 0.0, batch_size: int = 64,
        require_gpu: bool = False, cmvn: bool = False, n_mfcc: int = 30,
        global_zscore: bool = True, epochs: int = 150, patience: int = 15) -> None:
    if not 0.0 <= cmn_alpha <= 1.0:
        raise ValueError('cmn_alpha must be between 0 and 1')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if cmvn and cmn_alpha:
        raise ValueError('CMVN and --cmn-alpha are separate conditions')
    if n_mfcc not in (30, 40):
        raise ValueError('Only the authorized 30/40 MFCC protocols are supported')
    if require_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required; refusing to train on CPU')
        probe = torch.ones(1, device='cuda')
        print('GPU:', torch.cuda.get_device_name(0), probe.item(), flush=True)
        if os.environ.get('KERAS_TORCH_DEVICE', 'cuda') == 'cpu':
            raise RuntimeError('KERAS_TORCH_DEVICE=cpu conflicts with --require-gpu')
    cohort = json.loads(cohort_path.read_text())
    num_speakers = len({item['speaker'] for item in cohort['items']})
    arrays, labels, groups = load_features(cohort, features_root, mode,
                                          cmn_alpha, cmvn, n_mfcc)
    settings = Settings(num_speakers=num_speakers, architectures=(architecture,),
                        epochs=epochs, batch_size=batch_size, learning_rate=0.001,
                        early_stopping_patience=patience)
    trainer = CheckpointedTrainingSubsystem(settings)
    total_folds = min(max_folds, 5)
    expected_test = len(cohort['items']) // 5
    if len(cohort['items']) != expected_test * 5:
        raise ValueError('A quantidade de gravações precisa ser múltipla de 5.')
    for fold in range(total_folds):
        test_idx = np.flatnonzero(groups == fold)
        val_idx = np.flatnonzero(groups == (fold + 1) % 5)
        train_idx = np.flatnonzero((groups != fold) & (groups != (fold + 1) % 5))
        if 'partitions' in cohort:
            partition = cohort['partitions'][str(fold + 1)]
            train_idx, val_idx, test_idx = [np.asarray(partition[name], dtype=int)
                                          for name in ('train', 'validation', 'test')]
            if (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx)
                    or set(val_idx) & set(test_idx)
                    or set(np.r_[train_idx, val_idx, test_idx]) != set(range(len(labels)))):
                raise ValueError('Invalid explicit partition')
        if not (len(train_idx) == 3 * expected_test
                and len(val_idx) == len(test_idx) == expected_test):
            raise ValueError('Partição diferente de 60/20/20 no manifesto.')
        for source in sources:
            output = output_root / mode / f'fold{fold + 1}' / source
            previous = output / 'completed.json'
            protocol = {'global_zscore': global_zscore, 'cmn_alpha': cmn_alpha,
                        'cmvn': cmvn, 'n_mfcc': n_mfcc,
                        'features_root': str(features_root)}
            if cohort.get('dataset') == 'brsd':
                protocol.update(epochs=epochs, patience=patience, num_speakers=num_speakers)
            protocol_file = output / 'input_protocol.json'
            if protocol_file.exists():
                if json.loads(protocol_file.read_text()) != protocol:
                    raise ValueError(f'{output}: input normalization protocol mismatch')
            elif not global_zscore and output.exists() and any(output.iterdir()):
                raise ValueError(f'{output}: cannot reuse untagged checkpoints without z-score')
            output.mkdir(parents=True, exist_ok=True)
            protocol_file.write_text(json.dumps(protocol, indent=2) + '\n')
            if previous.exists():
                recorded = json.loads(previous.read_text())
                if recorded.get('global_zscore', True) != global_zscore:
                    raise ValueError(f'{output}: global z-score mismatch')
                if recorded.get('architecture', 'xvector') != architecture:
                    raise ValueError(f'{output} belongs to another architecture')
                recorded_alpha = recorded.get('cmn_alpha', 0.0)
                if recorded_alpha != cmn_alpha:
                    raise ValueError(f'{output} belongs to CMN alpha={recorded_alpha}; '
                                     f'requested alpha={cmn_alpha}')
                if recorded.get('cmvn', False) != cmvn:
                    raise ValueError(f'{output} belongs to another CMVN condition')
                if cmvn and recorded.get('cmvn_protocol') != CMVN_PROTOCOL:
                    raise ValueError(f'{output} belongs to another CMVN protocol')
                if recorded.get('n_mfcc', 30) != n_mfcc:
                    raise ValueError(f'{output} belongs to another MFCC count')
                recorded_batch = recorded.get('batch_size', 64)
                if recorded_batch != batch_size:
                    raise ValueError(f'{output} belongs to batch={recorded_batch}; '
                                     f'requested batch={batch_size}')
            finished = output / ('validation_metrics.json' if validation_only
                                 else 'completed.json')
            if finished.exists() and (output / 'modelo.keras').exists():
                print(f'Retomando: {output} já concluído', flush=True)
                continue
            train_x = arrays[source][train_idx].copy()
            val_x = arrays[source][val_idx].copy()
            test_x = arrays[source][test_idx].copy()
            mean, std = global_statistics(train_x, global_zscore)
            output.mkdir(parents=True, exist_ok=True)
            np.savez(output / 'normalization.npz', mean=mean, std=std)
            for tensor in (train_x, val_x, test_x):
                tensor -= mean
                tensor /= std
            split = DataSplit(train_x, labels[train_idx], val_x,
                              labels[val_idx], test_x, labels[test_idx],
                              mean, std)
            print(f'Treinando {architecture}, {mode}, fold {fold + 1}, origem {source}: '
                  f'{split.describe()}', flush=True)
            prior_model = output / 'modelo.keras'
            prior_validation = output / 'validation_metrics.json'
            if not validation_only and prior_model.exists() and prior_validation.exists():
                print(f'Reutilizando modelo escolhido na validação: {prior_model}',
                      flush=True)
                model = load_model(prior_model)
            else:
                model, _ = trainer.train(architecture, split, output,
                                         fold=fold + 1,
                                         seed=cohort['seed'] + fold)
                validation_result = metrics(model, val_x, labels[val_idx],
                                            output / 'validation', num_speakers)
                prior_validation.write_text(
                    json.dumps(validation_result, indent=2) + '\n')
            if not validation_only:
                for target in arrays:
                    target_x = (test_x if target == source
                                else arrays[target][test_idx].copy())
                    if target != source:
                        target_x -= mean
                        target_x /= std
                    result = metrics(model, target_x, labels[test_idx],
                                     output / target, num_speakers)
                    print(f'{architecture} {mode} fold{fold + 1} {source}->{target}: '
                          f'acurácia={result["accuracy"]:.4f}, '
                          f'F1={result["f1_macro"]:.4f}', flush=True)
                finished.write_text(json.dumps({
                    'mode': mode, 'architecture': architecture,
                    'fold': fold + 1, 'source': source,
                    'cohort': str(cohort_path),
                    'cmn_alpha': cmn_alpha,
                    'global_zscore': global_zscore,
                    'cmvn': cmvn,
                    'cmvn_protocol': CMVN_PROTOCOL if cmvn else None,
                    'n_mfcc': n_mfcc,
                    'batch_size': batch_size,
                    'features_root': str(features_root),
                    'frames_per_recording': cohort['frames_per_recording'],
                }, indent=2) + '\n')
            del model, split, train_x, val_x, test_x
            keras_backend.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_cohort_120.json'))
    parser.add_argument('--features', type=Path,
                        default=Path('output/vctk16_corrected_features'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_xvector_results'))
    parser.add_argument('--mode', choices=('static', 'dynamic'), required=True)
    parser.add_argument('--architecture',
                        choices=('xvector', 'temporal_cnn', 'cnn'),
                        default='xvector')
    parser.add_argument('--max-folds', type=int, default=5)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--source-mic', choices=('mic1', 'mic2', 'both', 'audio'),
                        default='both')
    parser.add_argument('--cmn-alpha', type=float, default=0.0,
                        help='Fraction of each recording static MFCC mean to subtract')
    parser.add_argument('--cmvn', action='store_true',
                        help='Per-recording static MFCC mean/std normalization; '
                             'scale deltas by the same static std')
    parser.add_argument('--no-global-zscore', action='store_true',
                        help='Disable training-set standardization for isolated ablations')
    parser.add_argument('--n-mfcc', type=int, choices=(30, 40), default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--require-gpu', action='store_true',
                        help='Fail instead of falling back to CPU')
    args = parser.parse_args()
    if args.no_global_zscore and args.output == Path('output/vctk16_xvector_results'):
        parser.error('Use --output with a new directory without global z-score')
    if args.cmvn and args.cmn_alpha:
        parser.error('--cmvn cannot be combined with --cmn-alpha')
    if args.cmvn and args.output == Path('output/vctk16_xvector_results'):
        parser.error('Use --output with a new directory for CMVN')
    if args.n_mfcc != 30 and args.output == Path('output/vctk16_xvector_results'):
        parser.error('Use --output with a new directory for 40 MFCCs')
    if args.cmn_alpha and args.output == Path('output/vctk16_xvector_results'):
        parser.error('Use --output with a new directory for a CMN condition')
    if args.batch_size != 64 and args.output == Path('output/vctk16_xvector_results'):
        parser.error('Use --output with a new directory when changing batch size')
    logging.basicConfig(level=logging.INFO)
    run(args.cohort, args.features, args.output,
        args.mode, args.max_folds, args.validation_only,
        args.architecture,
        ('mic1', 'mic2') if args.source_mic == 'both' else (args.source_mic,),
        args.cmn_alpha, args.batch_size, args.require_gpu, args.cmvn, args.n_mfcc,
        not args.no_global_zscore, args.epochs, args.patience)
