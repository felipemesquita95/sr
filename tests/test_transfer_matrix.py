"""Propriedades da comparação pareada: muda a captura, preserva-se o experimento."""

from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from sr.config import Settings
from sr.features import FeatureAdjustmentSubsystem
from sr.features.adjustment import EPSILON, PairedPartition
from transfer_matrix import (MICROPHONES, SPLIT_SEED, TRAINING_SEEDS,
                             evaluate_origin, write_report)


@pytest.fixture
def paired_features():
    rng = np.random.default_rng(51)
    return tuple({(s, u): (rng.normal(size=(4, 3 + u % 5)) + offset).astype(np.float32)
                  for s in range(1, 4) for u in range(1, 31)} for offset in (0, 20))


@pytest.fixture
def adjustment():
    return FeatureAdjustmentSubsystem(Settings(num_speakers=3, num_utterances=30,
                                               max_frames_cap=5))


def partition_for(adjustment, features, seed=SPLIT_SEED):
    return adjustment.paired_partition(*features, seed=seed)


@pytest.mark.parametrize('seed', [42, 73, 101])
def test_partition_is_disjoint_exhaustive_and_identical_between_microphones(
        adjustment, paired_features, seed):
    partition = partition_for(adjustment, paired_features, seed)
    assert partition == partition_for(adjustment, paired_features[::-1], seed)
    groups = [set(partition.train), set(partition.validation), set(partition.test)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == set(paired_features[0])
    for speaker in range(1, 4):
        assert [sum(key[0] == speaker for key in group) for group in groups] == [14, 10, 6]
    for source, target in (paired_features, paired_features[::-1]):
        split = adjustment.prepare_paired_microphone(source, target, partition)
        assert split.partition is partition
        np.testing.assert_array_equal(split.source.test_y, split.target_test_y)
        for i, key in enumerate(partition.test):
            for features, tensor in ((source, split.source.test_x), (target, split.target_test_x)):
                expected = (adjustment.pad_or_truncate(features[key], 5)
                            - split.source.normalization_mean[0]) / split.source.normalization_std[0]
                np.testing.assert_allclose(tensor[i], expected)


def test_split_seed_changes_membership_only_preserving_support(adjustment, paired_features):
    first = partition_for(adjustment, paired_features, 42)
    assert first == partition_for(adjustment, paired_features, 42)
    second = partition_for(adjustment, paired_features, 43)
    for role in ('train', 'validation', 'test'):
        a, b = getattr(first, role), getattr(second, role)
        assert a != b
        assert len(a) == len(b)
        assert [sum(k[0] == s for k in a) for s in range(1, 4)] == [
            sum(k[0] == s for k in b) for s in range(1, 4)]
    other = FeatureAdjustmentSubsystem(replace(adjustment.settings, validation_seed=999))
    assert first == partition_for(other, paired_features)


def test_missing_capture_excludes_pair_from_every_role(adjustment, paired_features):
    first, second = paired_features
    second = dict(second)
    del second[(1, 1)]
    partition = partition_for(adjustment, (first, second))
    assert (1, 1) not in partition.train + partition.validation + partition.test
    assert set(partition.train + partition.validation + partition.test) == first.keys() & second.keys()
    assert partition == partition_for(adjustment, (dict(reversed(list(second.items()))), first))


@pytest.mark.parametrize('source_index', [0, 1])
def test_all_cells_use_only_origin_training_statistics(adjustment, paired_features, source_index):
    source, target = paired_features[source_index], paired_features[1 - source_index]
    partition = partition_for(adjustment, paired_features)
    paired = adjustment.prepare_paired_microphone(source, target, partition)
    raw = np.stack([adjustment.pad_or_truncate(source[key], 5) for key in partition.train])
    mean = raw.mean(axis=(0, 2), keepdims=True)
    std = raw.std(axis=(0, 2), keepdims=True) + EPSILON
    np.testing.assert_array_equal(paired.source.normalization_mean, mean)
    np.testing.assert_array_equal(paired.source.normalization_std, std)
    np.testing.assert_allclose(paired.source.train_x, (raw - mean) / std)
    # Alterar validação, teste e toda a outra captura não pode afetar o treino.
    modified_source = dict(source)
    for key in partition.validation + partition.test:
        modified_source[key] = np.full((4, 100), 10000, dtype=np.float32)
    modified_target = {key: value * 1000 for key, value in target.items()}
    changed = adjustment.prepare_paired_microphone(modified_source, modified_target, partition)
    np.testing.assert_array_equal(changed.source.train_x, paired.source.train_x)
    np.testing.assert_array_equal(changed.source.normalization_mean, mean)
    np.testing.assert_array_equal(changed.source.normalization_std, std)
    assert changed.source.input_shape == (4, 5)
    inverse = adjustment.prepare_paired_microphone(target, source, partition)
    assert not np.allclose(inverse.source.normalization_mean, mean)


def test_short_source_training_alone_sets_length(adjustment, paired_features):
    partition = partition_for(adjustment, paired_features)
    source, target = paired_features
    source = dict(source)
    for key in partition.train:
        source[key] = source[key][:, :2]
    paired = adjustment.prepare_paired_microphone(source, target, partition)
    assert paired.source.input_shape == (4, 2)
    assert paired.target_test_x.shape[2] == 2


def test_invalid_partition_and_missing_pairs_fail(adjustment, paired_features):
    with pytest.raises(ValueError, match='disjuntos'):
        PairedPartition(((1, 1),), ((1, 2),), ((1, 1),), 42)
    with pytest.raises(ValueError, match='duplicado'):
        PairedPartition(((1, 1), (1, 1)), ((1, 2),), ((1, 3),), 42)
    partition = partition_for(adjustment, paired_features)
    first, second = paired_features
    second = dict(second)
    del second[partition.test[0]]
    with pytest.raises(ValueError, match='duas trilhas'):
        adjustment.prepare_paired_microphone(first, second, partition)
    with pytest.raises(ValueError, match='suficientes'):
        adjustment.paired_partition(first, second, seed=42, validation_per_speaker=25)


def test_training_seeds_never_change_partition_and_each_checkpoint_is_evaluated_twice(
        adjustment, paired_features, tmp_path):
    partition = partition_for(adjustment, paired_features)
    fits, predictions = [], []

    class InstrumentModel:
        def __init__(self, split):
            self.split = split

        def predict(self, tensor, verbose=0):
            predictions.append((self, tensor))
            labels = self.split.test_y.copy()
            if tensor is not self.split.test_x:
                labels = (labels + 1) % 3
            return np.eye(3)[labels]

        def to_json(self):
            return '{}'

    class InstrumentTrainer:
        def train(self, architecture, split, output, *, seed):
            fits.append((architecture, split, output, seed))
            return InstrumentModel(split), SimpleNamespace(history={})

    runs = []
    for index, source in enumerate(MICROPHONES):
        paired = adjustment.prepare_paired_microphone(
            paired_features[index], paired_features[1 - index], partition)
        runs.extend(evaluate_origin(InstrumentTrainer(), paired, source, tmp_path, 3, 'cnn'))
    assert len(fits) == 6
    assert [fit[3] for fit in fits] == list(TRAINING_SEEDS) * 2
    for start in (0, 3):
        assert all(fit[0] == 'cnn' for fit in fits[start:start + 3])
        assert all(fit[1] is fits[start][1] for fit in fits[start:start + 3])
    assert len(predictions) == 12
    for i in range(0, 12, 2):
        assert predictions[i][0] is predictions[i + 1][0]
    assert len({id(predictions[i][0]) for i in range(0, 12, 2)}) == 6
    for run, (_, split, directory, seed) in zip(runs, fits):
        assert run['perda_acuracia_pp'] == 100
        assert run['perda_por_locutor_pp'] == [100] * 3
        saved = json.loads((directory / 'predicoes.json').read_text())
        assert saved['semente'] == seed
        assert [(row['locutor'], row['enunciado']) for row in saved['gravacoes']] == list(partition.test)
        with np.load(directory / 'normalizacao.npz') as normalization:
            np.testing.assert_array_equal(normalization['mean'], split.normalization_mean)
            np.testing.assert_array_equal(normalization['std'], split.normalization_std)
    write_report(runs, tmp_path, 3)
    report = json.loads((tmp_path / 'matriz_transferencia.json').read_text())
    assert len(report['ajustes']) == 6
    for source in MICROPHONES:
        for target in MICROPHONES:
            assert report['matriz'][source][target]['accuracy']['media'] == float(source == target)
        assert report['perdas_pareadas'][source]['media_pp'] == 100
    assert 'suporte' in (tmp_path / 'matriz_transferencia.txt').read_text()
    assert (tmp_path / 'matriz_transferencia.png').read_bytes().startswith(b'\x89PNG')


@pytest.mark.parametrize('seed', [None, 17])
def test_trainer_applies_optional_seed_before_build_and_keeps_origin_validation(
        monkeypatch, tmp_path, adjustment, paired_features, seed):
    import sr.training.trainer as trainer

    partition = partition_for(adjustment, paired_features)
    split = adjustment.prepare_paired_microphone(*paired_features, partition).source
    events = []

    class Model:
        def fit(self, x, y, **kwargs):
            assert x is split.train_x and y is split.train_y
            assert kwargs['validation_data'][0] is split.validation_x
            assert kwargs['validation_data'][1] is split.validation_y
            events.append('fit')
            return object()

        def save(self, path):
            events.append('save')

    def build(*args, **kwargs):
        events.append('build')
        return Model()

    monkeypatch.setattr(trainer, 'set_random_seed', lambda value: events.append(('seed', value)))
    monkeypatch.setattr(trainer, 'build_model', build)
    monkeypatch.setattr(trainer.TrainingSubsystem, '_callbacks', lambda *args: [])
    trainer.TrainingSubsystem(adjustment.settings).train('cnn', split, tmp_path, seed=seed)
    assert events == ([] if seed is None else [('seed', seed)]) + ['build', 'fit', 'save']


def test_execution_persists_fixed_design_and_refuses_to_mix_artifacts(
        monkeypatch, tmp_path, paired_features):
    import transfer_matrix

    paths = (tmp_path / 'mic1', tmp_path / 'mic2')
    settings = Settings(num_speakers=3, num_utterances=30, num_mfccs=4, max_frames_cap=300,
                        architectures=('cnn',), features_path_train=paths[0], features_path_test=paths[1])
    features = tuple({key: np.tile(value, (1, 100)) for key, value in data.items()}
                     for data in paired_features)
    monkeypatch.setattr(FeatureAdjustmentSubsystem, 'load_features',
                        lambda self, path: features[paths.index(path)])
    manifesto = tmp_path / 'manifesto.json'
    manifesto.write_text(json.dumps({
        'locutores': {str(s): f'p{s}' for s in range(1, 4)},
        'enunciados': {f'p{s}': {str(u): str(u) for u in range(1, 31)} for s in range(1, 4)}}))
    calls = []

    def evaluate(training, paired, source, output, num_speakers, architecture):
        calls.append((source, paired.partition))
        assert (output / 'configuracao.json').is_file()
        assert (output / 'divisao.json').is_file()
        return []

    monkeypatch.setattr(transfer_matrix, 'evaluate_origin', evaluate)
    monkeypatch.setattr(transfer_matrix, 'write_report', lambda *args: None)
    output = tmp_path / 'resultado'
    transfer_matrix.run_matrix(settings, output, manifesto)
    assert [call[0] for call in calls] == ['mic1', 'mic2']
    assert calls[0][1] is calls[1][1]
    config = json.loads((output / 'configuracao.json').read_text())
    assert config['sementes_treino'] == list(TRAINING_SEEDS)
    assert config['semente_divisao'] == SPLIT_SEED
    assert config['settings']['architectures'] == ['cnn']
    assert config['backend'] == 'torch'
    assert config['codigo_sha256'] and config['git_commit'] and config['versoes']
    division = json.loads((output / 'divisao.json').read_text())
    assert division['test'] == [list(key) for key in calls[0][1].test]
    inputs = json.loads((output / 'features.json').read_text())
    assert len(inputs['mic1']) == len(inputs['mic2']) == 90
    assert all(row['pareado'] and len(row['sha256']) == 64 for row in inputs['mic1'])
    assert (output / 'manifesto_vctk.json').read_bytes() == manifesto.read_bytes()
    with pytest.raises(ValueError, match='saída deve estar vazia'):
        transfer_matrix.run_matrix(settings, output, manifesto)
