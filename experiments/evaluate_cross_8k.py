#!/usr/bin/env python3
"""Avalia checkpoints intra-microfone na outra captura do mesmo teste."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from sr.config import load_settings  # noqa: E402
from sr.evaluation import evaluate_model  # noqa: E402
from sr.features import FeatureAdjustmentSubsystem  # noqa: E402
from sr.features.adjustment import PairedPartition  # noqa: E402
from sr.models import ARCHITECTURES  # noqa: E402, F401 (registra camadas Keras)


def partition_for_fold(adjustment, first, second, fold):
    """Reconstrói os papéis dos treinos originais, usando só pares presentes."""
    settings = adjustment.settings
    if first.keys() != second.keys():
        raise ValueError('Mic1 e mic2 não têm a mesma seleção de gravações.')
    buckets = {'treino': [], 'validacao': [], 'teste': []}
    rng = np.random.default_rng(settings.validation_seed)
    for speaker in range(1, settings.num_speakers + 1):
        available = sorted(u for s, u in first if s == speaker)
        if not available:
            raise ValueError(f'Locutor {speaker} sem gravações.')
        role_indices = (list(range(1, settings.num_utterances + 1))
                        if settings.min_frames and settings.validation_fold_offset
                        else available)
        roles = adjustment.fold_roles(role_indices, fold, rng)
        for utterance in available:
            buckets[roles[utterance]].append((speaker, utterance))
    return PairedPartition(tuple(buckets['treino']), tuple(buckets['validacao']),
                           tuple(buckets['teste']), settings.validation_seed)


def evaluate_variant(prefix: str) -> None:
    from keras import backend
    from keras.models import load_model

    configs = {
        mic: load_settings(ROOT / 'configs' /
                           ('vctk8k.env' if prefix == 'vctk8k' and mic == 'mic1'
                            else f'{prefix}_{mic}.env'))
        for mic in ('mic1', 'mic2')
    }
    adjustments = {mic: FeatureAdjustmentSubsystem(configs[mic]) for mic in configs}
    features = {mic: adjustments[mic].load_features() for mic in configs}
    output_root = ROOT / 'runs/models' / f'{prefix}_cross_pareado'
    for source, target in (('mic1', 'mic2'), ('mic2', 'mic1')):
        settings = configs[source]
        adjustment = adjustments[source]
        for fold in range(1, settings.num_folds + 1):
            partition = partition_for_fold(adjustment, features[source], features[target], fold)
            paired = adjustment.prepare_paired_microphone(
                features[source], features[target], partition)
            digest = hashlib.sha256(json.dumps(partition.test).encode()).hexdigest()
            for architecture in settings.architectures:
                origin_dir = settings.models_path / architecture / f'particao{fold}'
                division = json.loads((origin_dir / 'divisao.json').read_text())
                previous = json.loads((origin_dir / 'metricas.json').read_text())
                if (division['teste'] != len(partition.test) or
                        division['treino'] != len(partition.train) or
                        division['validacao'] != len(partition.validation) or
                        division['quadros_por_gravacao'] != paired.source.test_x.shape[-1]):
                    raise ValueError(f'Divisão reconstruída diverge do treino: {origin_dir}')
                model = load_model(origin_dir / 'modelo.keras', compile=False)
                same = evaluate_model(model, paired.source.test_x,
                                      paired.source.test_y, settings.num_speakers)
                other = evaluate_model(model, paired.target_test_x,
                                       paired.target_test_y, settings.num_speakers)
                if abs(same.accuracy - previous['acuracia']) > 1e-6:
                    raise ValueError(f'Avaliação na origem diverge: {origin_dir}')
                output = output_root / f'{source}_para_{target}' / architecture
                output.mkdir(parents=True, exist_ok=True)
                (output / f'particao{fold}.json').write_text(json.dumps({
                    'origem': source, 'destino': target, 'arquitetura': architecture,
                    'particao': fold, 'quadros': int(paired.source.test_x.shape[-1]),
                    'treino': len(partition.train), 'validacao': len(partition.validation),
                    'teste_pares': len(partition.test), 'sha256_chaves_teste': digest,
                    'acuracia_origem': same.accuracy,
                    'acuracia_outro_microfone': other.accuracy,
                    'queda_pontos_percentuais': (same.accuracy - other.accuracy) * 100,
                    'f1_origem': same.f1, 'f1_outro_microfone': other.f1,
                    'checkpoint': str(origin_dir / 'modelo.keras'),
                }, indent=2, ensure_ascii=False) + '\n')
                print(f'{prefix} {source}→{target} {architecture} fold {fold}: '
                      f'{same.accuracy*100:.2f}% → {other.accuracy*100:.2f}%', flush=True)
                del model
                backend.clear_session()
            del paired
            gc.collect()

        for architecture in settings.architectures:
            output = output_root / f'{source}_para_{target}' / architecture
            rows = [json.loads((output / f'particao{fold}.json').read_text())
                    for fold in range(1, settings.num_folds + 1)]
            summary = {
                'arquitetura': architecture, 'origem': source, 'destino': target,
                'particoes': settings.num_folds,
                'acuracia_origem_media': float(np.mean([r['acuracia_origem'] for r in rows])),
                'acuracia_outro_media': float(np.mean([r['acuracia_outro_microfone'] for r in rows])),
                'queda_media_pp': float(np.mean([r['queda_pontos_percentuais'] for r in rows])),
                'queda_desvio_pp': float(np.std([r['queda_pontos_percentuais'] for r in rows])),
            }
            (output / 'resumo.json').write_text(json.dumps(summary, indent=2,
                                                     ensure_ascii=False) + '\n')
    print(f'Cross mic pareado concluído: {prefix}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True,
                        help='Prefixo dos perfis pareados, como vctk153 ou vctk_activity20_active')
    args = parser.parse_args()
    evaluate_variant(args.variant)


if __name__ == '__main__':
    main()
