"""Propriedades da referência linear sobre os tensores da matriz pareada."""

import json

import numpy as np
import pytest

from sr.config import Settings
from sr.features import FeatureAdjustmentSubsystem
from static_reference import (MICROPHONES, REGULARIZATION_GRID, evaluate_origin,
                              run, select_regularization, static_features)


def test_static_features_are_temporal_statistics_without_changing_coefficients():
    tensor = np.array([[[1., 3., 5.], [2., 2., 2.]], [[4., 4., 4.], [1., 2., 3.]]])
    values = static_features(tensor)
    assert values.shape == (2, 4)
    np.testing.assert_allclose(values[:, :2], tensor.mean(axis=2))
    np.testing.assert_allclose(values[:, 2:], tensor.std(axis=2))
    with pytest.raises(ValueError, match='tridimensional'):
        static_features(tensor[0])


def test_regularization_selection_uses_validation_and_keeps_predefined_grid(monkeypatch):
    import static_reference

    calls = []

    class Classifier:
        def __init__(self, regularization):
            self.regularization = regularization
            self.n_iter_ = np.array([3])

        def fit(self, values, labels):
            calls.append(('fit', self.regularization, values.copy(), labels.copy()))
            return self

        def predict(self, values):
            calls.append(('predict', self.regularization, values.copy()))
            return np.full(len(values), int(self.regularization == 1.0))

    monkeypatch.setattr(static_reference, 'build_classifier', lambda c: Classifier(c))
    train_x, train_y = np.array([[10.]]), np.array([0])
    validation_x, validation_y = np.array([[20.], [21.]]), np.array([1, 1])
    selected, conditions, _ = select_regularization(train_x, train_y, validation_x, validation_y)
    assert selected == 1.0
    assert [condition['C'] for condition in conditions] == list(REGULARIZATION_GRID)
    assert all(call[2][0, 0] == 20. for call in calls if call[0] == 'predict')


def test_origin_records_each_regularization_and_evaluates_same_fit_twice(monkeypatch):
    import static_reference

    settings = Settings(num_speakers=2, num_utterances=20, num_mfccs=2, max_frames_cap=3)
    adjustment = FeatureAdjustmentSubsystem(settings)
    features = tuple({(speaker, utterance): np.full((2, 3), speaker + offset, dtype=float)
                      for speaker in range(1, 3) for utterance in range(1, 21)}
                     for offset in (0, 10))
    partition = adjustment.paired_partition(*features, seed=42, validation_per_speaker=2)
    paired = adjustment.prepare_paired_microphone(*features, partition)
    models, evaluated = [], []

    class Classifier:
        n_iter_ = np.array([1])

        def fit(self, values, labels):
            models.append(self)
            return self

        def predict(self, values):
            return np.zeros(len(values), dtype=int)

        def predict_proba(self, values):
            evaluated.append(self)
            return np.tile([1., 0.], (len(values), 1))

    monkeypatch.setattr(static_reference, 'build_classifier', lambda c: Classifier())
    run, _, _, _ = evaluate_origin(paired, 'mic1', 2)
    assert run['dimensao_entrada'] == 4
    assert [row['C'] for row in run['condicoes_regularizacao']] == list(REGULARIZATION_GRID)
    assert len(models) == len(REGULARIZATION_GRID)
    assert len(evaluated) == 2
    assert evaluated[0] is evaluated[1] is models[0]


def test_execution_records_design_normalization_and_predictions(tmp_path, monkeypatch):
    paths = (tmp_path / 'mic1', tmp_path / 'mic2')
    settings = Settings(num_speakers=3, num_utterances=30, num_mfccs=40, max_frames_cap=300,
                        features_path_train=paths[0], features_path_test=paths[1])
    adjustment = FeatureAdjustmentSubsystem(settings)
    rng = np.random.default_rng(9)
    features = tuple({(speaker, utterance): rng.normal(
        loc=speaker + offset, size=(40, 300)).astype(np.float32)
                       for speaker in range(1, 4) for utterance in range(1, 31)}
                     for offset in (0, 10))
    partition = adjustment.paired_partition(*features, seed=42)
    matrix = tmp_path / 'matriz'
    matrix.mkdir()
    (matrix / 'divisao.json').write_text(json.dumps({
        'train': partition.train, 'validation': partition.validation,
        'test': partition.test, 'seed': partition.seed}))
    (matrix / 'matriz_transferencia.json').write_text(json.dumps({'ajustes': [
        {'origem': source, 'celulas': {target: {'accuracy': 0. for _ in range(1)}
                                       for target in MICROPHONES}}
        for source in MICROPHONES for _ in range(3)]}))
    for index, source in enumerate(MICROPHONES):
        split = adjustment.prepare_paired_microphone(features[index], features[1 - index], partition)
        directory = matrix / source / 'semente17'
        directory.mkdir(parents=True)
        np.savez(directory / 'normalizacao.npz', mean=split.source.normalization_mean,
                 std=split.source.normalization_std)
    monkeypatch.setattr(FeatureAdjustmentSubsystem, 'load_features',
                        lambda _, path: features[paths.index(path)])
    output = tmp_path / 'saida'
    run(settings, output, matrix)
    assert json.loads((output / 'divisao.json').read_text())['test'] == [list(key) for key in partition.test]
    configuration = json.loads((output / 'configuracao.json').read_text())
    assert configuration['grade_regularizacao'] == list(REGULARIZATION_GRID)
    for source in MICROPHONES:
        metricas = json.loads((output / source / 'metricas.json').read_text())
        predictions = json.loads((output / source / 'predicoes.json').read_text())['gravacoes']
        assert len(metricas['condicoes_regularizacao']) == len(REGULARIZATION_GRID)
        assert len(predictions) == len(partition.test)
        assert all(set(row['predicoes_base_zero']) == set(MICROPHONES) for row in predictions)
        with np.load(output / source / 'normalizacao.npz') as saved:
            assert saved['num_frames'] == 300
            assert saved['mean'].shape == (1, 40, 1)
