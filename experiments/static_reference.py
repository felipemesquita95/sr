#!/usr/bin/env python3
"""Compara uma referência linear estática à distribuição da CNN pareada.

Cada gravação é o vetor formado pela média e pelo desvio ao longo dos 300
quadros dos mesmos MFCCs normalizados que alimentaram a matriz de transferência.
Não lê assinaturas de 48 kHz nem áudio. A divisão vem do artefato da matriz e as
estatísticas de normalização reconstruídas são conferidas contra as persistidas.

A grade de regularização ``(0.1, 1.0, 10.0)`` foi definida antes da execução. Em
cada origem, a acurácia da validação daquela origem escolhe um único valor de C;
o teste não participa dessa escolha. Empates mantêm o primeiro C da grade.

Uso::

    KERAS_BACKEND=torch .venv/bin/python experiments/static_reference.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from sr.config import Settings, load_settings  # noqa: E402
from sr.evaluation import evaluate_model  # noqa: E402
from sr.features import FeatureAdjustmentSubsystem  # noqa: E402
from sr.features.adjustment import PairedPartition  # noqa: E402
from transfer_matrix import MICROPHONES, SPLIT_SEED  # noqa: E402

REGULARIZATION_GRID = (0.1, 1.0, 10.0)
CLASSIFIER_SEED = 42
SOLVER = 'lbfgs'
MAX_ITERATIONS = 1000


def write_json(path: Path, payload: object) -> None:
    """Persiste números NumPy e índices de gravações como JSON legível."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str,
                               allow_nan=False) + '\n', encoding='utf-8')


def load_partition(path: Path) -> PairedPartition:
    """Lê a divisão já executada pela matriz, sem sortear outra."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    partition = PairedPartition(
        tuple(tuple(key) for key in payload['train']),
        tuple(tuple(key) for key in payload['validation']),
        tuple(tuple(key) for key in payload['test']),
        payload['seed'])
    if partition.seed != SPLIT_SEED:
        raise ValueError(f'A divisão da matriz deveria usar a semente {SPLIT_SEED}.')
    return partition


def static_features(tensor: np.ndarray) -> np.ndarray:
    """Resume cada tensor normalizado por média e desvio temporal.

    Args:
        tensor: Lote de forma ``(gravações, coeficientes, quadros)``.

    Returns:
        Vetores de forma ``(gravações, 2 * coeficientes)``.
    """
    if tensor.ndim != 3:
        raise ValueError('Os MFCCs pareados devem formar um tensor tridimensional.')
    return np.concatenate((tensor.mean(axis=2), tensor.std(axis=2)), axis=1)


class ProbabilityAdapter:
    """Adapta ``predict_proba`` do scikit-learn à interface do avaliador comum."""

    def __init__(self, classifier: LogisticRegression) -> None:
        self.classifier = classifier

    def predict(self, values: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Retorna probabilidades sem usar a opção de verbosidade da rede."""
        del verbose
        return self.classifier.predict_proba(values)


def build_classifier(regularization: float) -> LogisticRegression:
    """Constrói a única família linear e hiperparâmetros fixados do instrumento."""
    return LogisticRegression(
        C=regularization, solver=SOLVER, max_iter=MAX_ITERATIONS,
        random_state=CLASSIFIER_SEED)


def select_regularization(train_x: np.ndarray, train_y: np.ndarray,
                          validation_x: np.ndarray, validation_y: np.ndarray) -> tuple[float, list[dict], LogisticRegression]:
    """Escolhe C apenas pela validação da origem, na ordem da grade pré-definida."""
    conditions, selected_classifier = [], None
    for regularization in REGULARIZATION_GRID:
        classifier = build_classifier(regularization).fit(train_x, train_y)
        conditions.append({
            'C': regularization,
            'acuracia_validacao': float(np.mean(classifier.predict(validation_x) == validation_y)),
            'iteracoes': classifier.n_iter_.tolist(),
        })
        if selected_classifier is None or conditions[-1]['acuracia_validacao'] > max(
                row['acuracia_validacao'] for row in conditions[:-1]):
            selected_classifier = classifier
    selected = max(conditions, key=lambda condition: condition['acuracia_validacao'])
    return float(selected['C']), conditions, selected_classifier


def evaluate_origin(paired, source: str, num_speakers: int) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Ajusta uma origem e mede o mesmo classificador nas duas capturas de teste."""
    source_split = paired.source
    train_x = static_features(source_split.train_x)
    validation_x = static_features(source_split.validation_x)
    test_x = {source: static_features(source_split.test_x),
              next(mic for mic in MICROPHONES if mic != source): static_features(paired.target_test_x)}
    selected_c, conditions, classifier = select_regularization(
        train_x, source_split.train_y, validation_x, source_split.validation_y)
    results = {mic: evaluate_model(ProbabilityAdapter(classifier), values, source_split.test_y, num_speakers)
               for mic, values in test_x.items()}
    run = {
        'origem': source,
        'C_selecionado': selected_c,
        'condicoes_regularizacao': conditions,
        'tamanhos': {'treino': len(source_split.train_y), 'validacao': len(source_split.validation_y),
                     'teste': len(source_split.test_y)},
        'dimensao_entrada': int(train_x.shape[1]),
        'celulas': {mic: {name: float(getattr(result, name)) for name in
                          ('accuracy', 'precision', 'recall', 'f1')} for mic, result in results.items()},
        'suporte_por_locutor': np.bincount(source_split.test_y, minlength=num_speakers).tolist(),
        'perda_acuracia_pp': (results[source].accuracy - results[
            next(mic for mic in MICROPHONES if mic != source)].accuracy) * 100,
        'perda_por_locutor_pp': ((results[source].per_class_accuracy() - results[
            next(mic for mic in MICROPHONES if mic != source)].per_class_accuracy()) * 100).tolist(),
    }
    return run, {'mean': source_split.normalization_mean, 'std': source_split.normalization_std}, \
        results[source].predictions, results[next(mic for mic in MICROPHONES if mic != source)].predictions


def cnn_distribution(matrix_path: Path) -> dict:
    """Lê todas as sementes CNN da matriz, sem eleger uma delas."""
    payload = json.loads((matrix_path / 'matriz_transferencia.json').read_text(encoding='utf-8'))
    runs = payload['ajustes']
    if len(runs) != 6 or {run['origem'] for run in runs} != set(MICROPHONES):
        raise ValueError('A matriz de referência deve conter os seis ajustes CNN.')
    return {
        source: {target: [float(run['celulas'][target]['accuracy']) for run in runs
                           if run['origem'] == source] for target in MICROPHONES}
        for source in MICROPHONES
    }


def write_report(output: Path, runs: list[dict], cnn: dict, num_speakers: int) -> None:
    """Grava a comparação numérica completa, sem resumir a CNN à melhor semente."""
    write_json(output / 'referencia_estatica.json', {
        'acaso': 1 / num_speakers,
        'grade_regularizacao': list(REGULARIZATION_GRID),
        'semente_classificador': CLASSIFIER_SEED,
        'ajustes': runs,
        'cnn_por_semente': cnn,
    })
    lines = ['Referência estática — regressão logística sobre MFCCs pareados',
             f'{num_speakers} locutores; acaso {100 / num_speakers:.2f}%',
             f'Grade C pré-definida: {REGULARIZATION_GRID}; escolha: validação da origem.',
             'Cada vetor tem média e desvio temporal dos 40 coeficientes: 80 números.',
             'A CNN é mostrada por todas as sementes; nenhuma foi escolhida como melhor.', '']
    for run in runs:
        source = run['origem']
        lines.append(f'{source}; C selecionado={run["C_selecionado"]}; '
                     f'validação={[row["acuracia_validacao"] for row in run["condicoes_regularizacao"]]}')
        for target in MICROPHONES:
            values = cnn[source][target]
            lines.append(f'  teste {target}: estática {run["celulas"][target]["accuracy"] * 100:.2f}%; '
                         f'CNN sementes {[round(value * 100, 2) for value in values]}%')
        lines.append(f'  perda intra - cross: {run["perda_acuracia_pp"]:.2f} pp')
    (output / 'referencia_estatica.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(settings: Settings, output: Path, matrix_path: Path, origins: tuple[str, ...] = MICROPHONES) -> None:
    """Executa a referência usando exclusivamente a população e o desenho da matriz."""
    if settings.max_frames_cap != 300 or settings.num_mfccs != 40:
        raise ValueError('A referência exige exatamente 40 coeficientes e teto de 300 quadros.')
    if settings.permute_labels:
        raise ValueError('A referência exige os rótulos verdadeiros.')
    if not set(origins) <= set(MICROPHONES) or not origins:
        raise ValueError('As origens devem ser mic1 e/ou mic2.')
    existing = output.exists() and any(output.iterdir())
    if existing and not (output / 'configuracao.json').is_file():
        raise ValueError(f'A saída existente não é uma execução retomável: {output}')
    partition = load_partition(matrix_path / 'divisao.json')
    cnn = cnn_distribution(matrix_path)
    adjustment = FeatureAdjustmentSubsystem(settings)
    paths = (settings.features_path_train, settings.features_path_test)
    if any(path is None for path in paths) or paths[0].resolve() == paths[1].resolve():
        raise ValueError('Declare diretórios distintos para mic1 e mic2, nessa ordem.')
    features = {mic: adjustment.load_features(path) for mic, path in zip(MICROPHONES, paths)}
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        'settings': asdict(settings), 'matriz_origem': str(matrix_path.resolve()),
        'semente_divisao': partition.seed, 'semente_classificador': CLASSIFIER_SEED,
        'grade_regularizacao': list(REGULARIZATION_GRID), 'solver': SOLVER,
        'max_iteracoes': MAX_ITERATIONS, 'python': platform.python_version(),
        'versoes': {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn')},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'codigo_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(), ROOT / 'experiments/transfer_matrix.py')},
    }
    if existing:
        previous = json.loads((output / 'configuracao.json').read_text(encoding='utf-8'))
        if previous['matriz_origem'] != configuration['matriz_origem'] or previous['grade_regularizacao'] != configuration['grade_regularizacao']:
            raise ValueError('A saída existente foi iniciada com outro desenho experimental.')
    else:
        write_json(output / 'divisao.json', asdict(partition))
        write_json(output / 'configuracao.json', configuration)
    for source in origins:
        index = MICROPHONES.index(source)
        if (output / source).exists():
            raise ValueError(f'A origem {source} já foi executada nesta saída.')
        paired = adjustment.prepare_paired_microphone(features[source], features[MICROPHONES[1 - index]], partition)
        persisted = np.load(matrix_path / source / 'semente17' / 'normalizacao.npz')
        if not (np.array_equal(paired.source.normalization_mean, persisted['mean']) and
                np.array_equal(paired.source.normalization_std, persisted['std'])):
            raise ValueError('A normalização reconstruída não coincide com a matriz executada.')
        run_data, normalization, source_predictions, target_predictions = evaluate_origin(
            paired, source, settings.num_speakers)
        directory = output / source
        directory.mkdir()
        np.savez(directory / 'normalizacao.npz', **normalization, num_frames=paired.source.input_shape[1])
        target = MICROPHONES[1 - index]
        write_json(directory / 'predicoes.json', {'origem': source, 'gravacoes': [
            {'locutor': key[0], 'enunciado': key[1], 'alvo_base_zero': int(paired.source.test_y[i]),
             'predicoes_base_zero': {source: int(source_predictions[i]), target: int(target_predictions[i])}}
            for i, key in enumerate(partition.test)]})
        write_json(directory / 'metricas.json', run_data)
    if all((output / source / 'metricas.json').is_file() for source in MICROPHONES):
        runs = [json.loads((output / source / 'metricas.json').read_text(encoding='utf-8'))
                for source in MICROPHONES]
        write_report(output, runs, cnn, settings.num_speakers)


def main() -> int:
    """Carrega a configuração da referência e mantém a matriz CNN intacta."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', '-c', default=ROOT / 'configs/vctk_static_reference.env')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--matriz', type=Path, default=ROOT / 'runs/models/vctk_transfer_matrix')
    parser.add_argument('--origem', choices=MICROPHONES,
                        help='Executa uma origem; a outra pode retomar a mesma saída.')
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        run(settings, args.output or settings.models_path, args.matriz,
            (args.origem,) if args.origem else MICROPHONES)
    except (OSError, ValueError, KeyError) as error:
        print(f'Erro: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
