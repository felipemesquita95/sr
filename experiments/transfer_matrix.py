#!/usr/bin/env python3
"""Quanto custa trocar de microfone quando treino e teste continuam comparáveis?

Subtrair o cross-microfone do intra convencional mistura a troca de captura com
outras quantidades de treino e teste e outra regra de validação. Essa diferença
não estima a penalidade de transferência. A referência necessária é o mesmo
checkpoint reconhecendo as duas capturas das mesmas gravações inéditas.

Aqui a divisão por enunciado é única: 20% dos pares de cada locutor para teste
(arredondados para cima), dez para validação e o restante para treino. As três
sementes variam a inicialização e o treino, nunca a divisão. Cada semente produz
uma CNN por origem; a validação permanece nessa origem. Cada linha da matriz
compara o mesmo modelo e a mesma normalização de treino nos dois microfones.

Uma perda estável nas duas direções quantifica a transferência condicionada a
esta divisão. Assimetria exige declarar a direção; uma referência intra também
baixa reduz o que se pode atribuir à troca na diferença entre protocolos antigos.
A medida inclui os efeitos do microfone no pré-processamento, inclusive no VAD:
não separa voz de sessão nem estima uma porcentagem de informação de canal.

Uso (somente MFCCs persistidos; nenhum acesso ao áudio)::

    KERAS_BACKEND=torch .venv/bin/python experiments/transfer_matrix.py

A saída contém matriz em TXT/JSON/PNG, diferenças pareadas por locutor, divisão,
configuração efetiva, procedência dos tensores, predições e seis checkpoints.
O desvio entre sementes é descritivo, condicionado a um único teste compartilhado;
não é intervalo de confiança nem variação entre partições independentes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from sr.config import Settings, load_settings  # noqa: E402
from sr.features import FeatureAdjustmentSubsystem  # noqa: E402

SPLIT_SEED = 42
TRAINING_SEEDS = (17, 29, 43)
TEST_FRACTION = 0.2
VALIDATION_PER_SPEAKER = 10
MICROPHONES = ('mic1', 'mic2')
logger = logging.getLogger('transfer_matrix')


def write_json(path: Path, payload: object) -> None:
    """Persiste estruturas numéricas e caminhos sem perder os índices das gravações."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str,
                               allow_nan=False) + '\n', encoding='utf-8')


def summarize(runs: list[dict]) -> dict:
    """Agrega sementes depois de calcular cada diferença dentro do mesmo ajuste."""
    matrix, losses = {}, {}
    for source in MICROPHONES:
        selected = [run for run in runs if run['origem'] == source]
        matrix[source] = {}
        for target in MICROPHONES:
            matrix[source][target] = {}
            for metric in ('accuracy', 'precision', 'recall', 'f1'):
                values = [run['celulas'][target][metric] for run in selected]
                matrix[source][target][metric] = {
                    'media': float(np.mean(values)), 'desvio': float(np.std(values)),
                    'por_semente': values,
                }
        values = np.asarray([run['perda_por_locutor_pp'] for run in selected])
        global_values = [run['perda_acuracia_pp'] for run in selected]
        losses[source] = {
            'media_pp': float(np.mean(global_values)),
            'desvio_pp': float(np.std(global_values)),
            'por_semente_pp': global_values,
            'por_locutor': [
                {'locutor': i + 1, 'media_pp': float(np.mean(column)),
                 'desvio_pp': float(np.std(column)), 'por_semente_pp': column.tolist()}
                for i, column in enumerate(values.T)
            ],
        }
    return {'matriz': matrix, 'perdas_pareadas': losses}


def write_report(runs: list[dict], output: Path, num_speakers: int) -> None:
    """Mostra as quatro células e a perda intra menos cross, inclusive por locutor."""
    summary = summarize(runs)
    write_json(output / 'matriz_transferencia.json', {
        'sementes_treino': TRAINING_SEEDS, 'semente_divisao': SPLIT_SEED,
        'acaso': 1 / num_speakers, 'unidade_perda': 'pontos percentuais; intra - cross',
        'desvio': 'populacional entre sementes (ddof=0), condicionado à divisão fixa',
        'ajustes': runs, **summary,
    })
    lines = ['Matriz de transferência — CNN, divisão fixa por enunciado',
             f'{num_speakers} locutores; acaso {100 / num_speakers:.2f}%',
             f'Semente da divisão: {SPLIT_SEED}; sementes de treino: {TRAINING_SEEDS}',
             'Linhas: origem do treino; colunas: captura de teste (mic1, mic2).',
             'Acurácia em %, média ± desvio populacional entre sementes.', '']
    for source in MICROPHONES:
        cells = [summary['matriz'][source][target]['accuracy'] for target in MICROPHONES]
        lines.append(source + '  ' + '  '.join(
            f"{cell['media'] * 100:.2f} ± {cell['desvio'] * 100:.2f}" for cell in cells))
    lines += ['', 'Matrizes por semente (mic1, mic2):']
    for seed in TRAINING_SEEDS:
        for run in runs:
            if run['semente'] == seed:
                lines.append(f"{seed} {run['origem']}  " + '  '.join(
                    f"{run['celulas'][target]['accuracy'] * 100:.2f}%" for target in MICROPHONES))
                lines.append(f"  Tamanhos por captura: {run['tamanhos']}")
    lines += ['', 'Perda pareada = intra - cross, em pontos percentuais; positiva = queda.',
              'Cada diferença usa o mesmo checkpoint, normalização e enunciados de teste.']
    for source in MICROPHONES:
        loss = summary['perdas_pareadas'][source]
        lines.append(f"{source}: {loss['media_pp']:.2f} ± {loss['desvio_pp']:.2f} pp; "
                     f"por semente: {loss['por_semente_pp']}")
        lines.append('locutor  suporte  média_pp  desvio_pp  perdas nas três sementes')
        support = next(run['suporte_por_locutor'] for run in runs if run['origem'] == source)
        for row in loss['por_locutor']:
            lines.append(f"{row['locutor']:>7}  {support[row['locutor'] - 1]:>7}  "
                         f"{row['media_pp']:>8.2f}  {row['desvio_pp']:>9.2f}  "
                         + '  '.join(f'{v:.2f}' for v in row['por_semente_pp']))
    lines += ['', 'A dispersão entre sementes não é um intervalo de confiança.',
              'A medida é condicionada à divisão e inclui efeitos do pré-processamento.',
              'Não separa voz de sessão nem mede uma porcentagem de informação de canal.']
    (output / 'matriz_transferencia.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    matrix = np.array([[summary['matriz'][s][t]['accuracy']['media'] * 100
                        for t in MICROPHONES] for s in MICROPHONES])
    axes[0].imshow(matrix, vmin=0, vmax=100, cmap='Blues')
    for i, source in enumerate(MICROPHONES):
        for j, target in enumerate(MICROPHONES):
            cell = summary['matriz'][source][target]['accuracy']
            axes[0].text(j, i, f"{matrix[i, j]:.2f}%\n± {cell['desvio'] * 100:.2f}",
                         ha='center', va='center', color='white' if matrix[i, j] > 60 else 'black')
    axes[0].set(xticks=[0, 1], xticklabels=MICROPHONES, yticks=[0, 1],
                yticklabels=MICROPHONES, xlabel='Teste', ylabel='Treino', title='Acurácia: média ± desvio')
    for axis, source in zip(axes[1:], MICROPHONES):
        rows = summary['perdas_pareadas'][source]['por_locutor']
        for i, seed in enumerate(TRAINING_SEEDS):
            axis.scatter([r['locutor'] for r in rows],
                         [r['por_semente_pp'][i] for r in rows], s=9, alpha=0.5, label=str(seed))
        axis.plot([r['locutor'] for r in rows], [r['media_pp'] for r in rows],
                  color='black', linewidth=0.8, label='média')
        axis.axhline(0, color='gray', linestyle=':')
        axis.set(xlabel='Locutor (índice)', ylabel='Perda intra − cross (pp)',
                 title=f'Treino em {source}', ylim=(-100, 100))
        axis.legend(title='Semente', fontsize=8)
    figure.suptitle('Mesmas gravações inéditas; validação e normalização na origem')
    figure.tight_layout()
    figure.savefig(output / 'matriz_transferencia.png', dpi=150)
    plt.close(figure)


def evaluate_origin(training, paired, source: str, output: Path, num_speakers: int) -> list[dict]:
    """Faz três ajustes de uma origem, avaliando cada checkpoint nas duas capturas."""
    from keras import backend
    from sr.evaluation import evaluate_model

    split = paired.source
    target = next(mic for mic in MICROPHONES if mic != source)
    runs = []
    for seed in TRAINING_SEEDS:
        backend.clear_session()
        directory = output / source / f'semente{seed}'
        directory.mkdir(parents=True)
        np.savez(directory / 'normalizacao.npz', mean=split.normalization_mean,
                 std=split.normalization_std, num_frames=split.input_shape[1])
        write_json(directory / 'ajuste.json', {'origem': source, 'validacao': source,
                   'semente_treino': seed, 'semente_divisao': paired.partition.seed,
                   'arquitetura': 'cnn', 'input_shape': split.input_shape})
        model, history = training.train('cnn', split, directory, seed=seed)
        (directory / 'modelo.json').write_text(model.to_json(), encoding='utf-8')
        write_json(directory / 'historico.json', history.history)
        results = {
            source: evaluate_model(model, split.test_x, split.test_y, num_speakers),
            target: evaluate_model(model, paired.target_test_x, paired.target_test_y, num_speakers),
        }
        rows = [
            {'locutor': key[0], 'enunciado': key[1], 'alvo_base_zero': int(split.test_y[i]),
             'predicoes_base_zero': {mic: int(results[mic].predictions[i]) for mic in MICROPHONES}}
            for i, key in enumerate(paired.partition.test)
        ]
        write_json(directory / 'predicoes.json', {'origem': source, 'semente': seed, 'gravacoes': rows})
        run = {
            'origem': source, 'semente': seed, 'checkpoint': str(directory.relative_to(output) / 'modelo.keras'),
            'tamanhos': {'treino': len(split.train_y), 'validacao': len(split.validation_y),
                         'teste': len(split.test_y)},
            'celulas': {mic: {name: getattr(result, name) for name in
                             ('accuracy', 'precision', 'recall', 'f1')} for mic, result in results.items()},
            'suporte_por_locutor': np.bincount(split.test_y, minlength=num_speakers).tolist(),
            'perda_acuracia_pp': (results[source].accuracy - results[target].accuracy) * 100,
            'perda_por_locutor_pp': ((results[source].per_class_accuracy()
                                     - results[target].per_class_accuracy()) * 100).tolist(),
        }
        write_json(directory / 'metricas.json', run)
        runs.append(run)
        del model, history
        gc.collect()
    return runs


def run_matrix(settings: Settings, output: Path, manifesto: Path) -> None:
    """Executa a medida em uma saída nova, registrando o desenho antes dos ajustes."""
    import keras
    import torch
    from sr.training import TrainingSubsystem

    if keras.backend.backend() != 'torch':
        raise ValueError('Este instrumento exige KERAS_BACKEND=torch.')
    if settings.architectures != ('cnn',) or settings.permute_labels:
        raise ValueError('A matriz exige ARCHITECTURES=cnn e rótulos verdadeiros.')
    paths = (settings.features_path_train, settings.features_path_test)
    if any(path is None for path in paths) or paths[0].resolve() == paths[1].resolve():
        raise ValueError('Declare diretórios distintos para mic1 e mic2, nessa ordem.')
    if settings.max_frames_cap != 300:
        raise ValueError('A matriz usa o comprimento comum de 300 quadros.')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'A saída deve estar vazia para não misturar execuções: {output}')
    manifest = json.loads(manifesto.read_text())
    adjustment = FeatureAdjustmentSubsystem(settings)
    features = {mic: adjustment.load_features(path) for mic, path in zip(MICROPHONES, paths)}
    partition = adjustment.paired_partition(features['mic1'], features['mic2'], seed=SPLIT_SEED,
                                          test_fraction=TEST_FRACTION,
                                          validation_per_speaker=VALIDATION_PER_SPEAKER)
    # A CNN deve ter a mesma forma nas duas direções; só o treino define elegibilidade.
    for mic in MICROPHONES:
        if max(features[mic][key].shape[1] for key in partition.train) < 300:
            raise ValueError(f'O treino de {mic} não atinge os 300 quadros comuns.')
        for (speaker, utterance), matrix in features[mic].items():
            if matrix.ndim != 2 or matrix.shape[0] != settings.num_mfccs or matrix.shape[1] == 0:
                raise ValueError(f'MFCC inválido em {mic}, locutor {speaker}, enunciado {utterance}.')
            original = manifest['locutores'][str(speaker)]
            manifest['enunciados'][original][str(utterance)]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifesto, output / 'manifesto_vctk.json')
    write_json(output / 'divisao.json', asdict(partition))
    # Hash dos tensores carregados: identifica os dados realmente consumidos, sem reler áudio.
    write_json(output / 'features.json', {
        mic: [{'locutor': key[0], 'enunciado': key[1], 'shape': value.shape,
               'dtype': str(value.dtype), 'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
               'pareado': key in features[MICROPHONES[1 - MICROPHONES.index(mic)]]}
              for key, value in sorted(data.items())] for mic, data in features.items()
    })
    write_json(output / 'configuracao.json', {
        'settings': asdict(settings), 'semente_divisao': SPLIT_SEED,
        'sementes_treino': TRAINING_SEEDS, 'fracao_teste': TEST_FRACTION,
        'validacao_por_locutor': VALIDATION_PER_SPEAKER,
        'trilhas': {mic: str(path.resolve()) for mic, path in zip(MICROPHONES, paths)},
        'python': platform.python_version(), 'backend': keras.backend.backend(),
        'versoes': {name: importlib.metadata.version(name) for name in
                    ('keras', 'torch', 'numpy', 'scikit-learn', 'matplotlib')},
        'cuda': torch.version.cuda, 'rocm': torch.version.hip,
        'cudnn': torch.backends.cudnn.version(),
        'plataforma': platform.platform(),
        'dispositivos': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'cuda_disponivel': torch.cuda.is_available(),
        'algoritmos_deterministicos': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        'codigo_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__).resolve(), *sorted((ROOT / 'src/sr').rglob('*.py'))]},
    })
    runs = []
    for source in MICROPHONES:
        target = next(mic for mic in MICROPHONES if mic != source)
        paired = adjustment.prepare_paired_microphone(features[source], features[target], partition)
        runs.extend(evaluate_origin(TrainingSubsystem(settings), paired, source, output,
                                    settings.num_speakers))
        del paired
        gc.collect()
    write_report(runs, output, settings.num_speakers)


def main() -> int:
    """Carrega o perfil exclusivo da matriz e mantém os resultados antigos intactos."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', '-c', default=os.environ.get(
        'SR_CONFIG', str(ROOT / 'configs/vctk_transfer_matrix.env')))
    parser.add_argument('--output', type=Path, help='Diretório vazio para os artefatos.')
    parser.add_argument('--manifesto', type=Path, default=ROOT / 'runs/features/vctk_manifesto.json')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    try:
        settings = load_settings(args.config)
        run_matrix(settings, args.output or settings.models_path, args.manifesto)
    except (OSError, ValueError, KeyError) as error:
        logger.error('%s', error)
        return 1
    return 0


if __name__ == '__main__':
    os.environ.setdefault('KERAS_BACKEND', 'torch')
    raise SystemExit(main())
