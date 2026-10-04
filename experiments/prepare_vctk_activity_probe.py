#!/usr/bin/env python3
"""Separa quadros de atividade e baixa atividade no áudio VCTK inteiro."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from sr.preprocessing import signal  # noqa: E402

AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
MANIFEST = ROOT / 'runs/features/vctk_manifesto.json'
FEATURES = ROOT / 'runs/features'
CANDIDATES = ROOT / 'docs/vctk_activity8k_candidates.jsonl'
COUNTS = ROOT / 'docs/vctk_activity8k_counts.csv'
SUMMARY = ROOT / 'docs/vctk_activity_probe_selection.json'
FRAME = 256
HOP = 128
TOP_DB = 30


def erode_one(mask: np.ndarray) -> np.ndarray:
    """Exige o mesmo rótulo nos quadros vizinhos, evitando fronteiras."""
    result = mask.copy()
    result[1:] &= mask[:-1]
    result[:-1] &= mask[1:]
    return result


def activity_mask(path: Path, n_frames: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype='float32', always_2d=False)
    if rate != 48_000:
        raise ValueError(f'Taxa inesperada: {path}: {rate}')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    filtered = signal.antialias_filter(audio, rate, 8_000)
    narrow = signal.resample(filtered, rate, 8_000)
    intervals = librosa.effects.split(narrow, top_db=TOP_DB,
                                      frame_length=FRAME, hop_length=HOP)
    centers = np.arange(n_frames) * HOP
    mask = np.zeros(n_frames, dtype=bool)
    for start, end in intervals:
        mask |= (centers >= start) & (centers < end)
    return mask


def scan() -> None:
    manifest = json.loads(MANIFEST.read_text())
    candidate_tmp = CANDIDATES.with_suffix('.jsonl.tmp')
    count_tmp = COUNTS.with_suffix('.csv.tmp')
    with candidate_tmp.open('w') as candidate_stream, count_tmp.open('w', newline='') as count_stream:
        writer = csv.DictWriter(count_stream, fieldnames=[
            'locutor', 'enunciado', 'arquivo', 'quadros', 'atividade_comum_segura',
            'baixa_comum_segura'])
        writer.writeheader()
        done = 0
        for speaker_key, speaker in manifest['locutores'].items():
            for utterance_key, identifier in manifest['enunciados'][speaker].items():
                n = []
                masks = []
                for mic in ('mic1', 'mic2'):
                    feature = FEATURES / f'vctk8k_{mic}' / speaker_key / utterance_key / 'mfccs.npy'
                    matrix = np.load(feature, mmap_mode='r')
                    n.append(int(matrix.shape[1]))
                    path = AUDIO / speaker / f'{speaker}_{identifier}_{mic}.flac'
                    masks.append(activity_mask(path, n[-1]))
                if n[0] != n[1]:
                    raise ValueError(f'Microfones com números de quadros diferentes: {speaker}_{identifier}')
                active = np.flatnonzero(erode_one(masks[0] & masks[1])).tolist()
                low = np.flatnonzero(erode_one(~masks[0] & ~masks[1])).tolist()
                record = {'locutor': int(speaker_key), 'enunciado': int(utterance_key),
                          'arquivo': f'{speaker}_{identifier}', 'quadros': n[0],
                          'active': active, 'low': low}
                candidate_stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                writer.writerow({'locutor': speaker_key, 'enunciado': utterance_key,
                                 'arquivo': record['arquivo'], 'quadros': n[0],
                                 'atividade_comum_segura': len(active),
                                 'baixa_comum_segura': len(low)})
                done += 1
            candidate_stream.flush()
            count_stream.flush()
            if int(speaker_key) % 10 == 0:
                print(f'VAD 8 kHz: {speaker_key}/108 locutores; {done} pares', flush=True)
    candidate_tmp.replace(CANDIDATES)
    count_tmp.replace(COUNTS)
    print(f'VAD 8 kHz completo: {done} pares', flush=True)


def select_even(candidates: list[int], k: int) -> np.ndarray:
    if len(candidates) < k:
        raise ValueError('Quadros insuficientes para o corte.')
    positions = np.linspace(0, len(candidates) - 1, k, dtype=np.int64)
    return np.asarray(candidates, dtype=np.int64)[positions]


def config_text(condition: str, mic: str, k: int, prefix: str) -> str:
    return f'''# Diagnóstico pareado no áudio completo: {condition}, {mic}, {k} quadros.
EXPERIMENT_NAME={prefix}-{condition}-{mic}
DATASET_FORMAT=vctk
VCTK_ROOT={AUDIO}
VCTK_MIC={mic}
VCTK_MICS=mic1,mic2
NUM_SPEAKERS=108
NUM_UTTERANCES=200
SOURCE_SAMPLING_RATE=48000
TARGET_SAMPLING_RATE=8000
ENABLE_VAD=false
VAD_TOP_DB=30
PRE_EMPHASIS_COEF=0.97
NUM_MFCCS=40
FRAME_SIZE=256
NUM_FOLDS=5
VALIDATION_SEED=42
VALIDATION_FOLD_OFFSET=1
MIN_FRAMES={k}
MAX_FRAMES_CAP={k}
ARCHITECTURES=cnn,temporal_cnn,attention
EPOCHS=1000
BATCH_SIZE=64
LEARNING_RATE=0.001
EARLY_STOPPING_PATIENCE=30
FEATURES_PATH={FEATURES / f'{prefix}_{condition}_{mic}'}
MODELS_PATH={ROOT / 'runs/models' / f'{prefix}_{condition}_{mic}'}
ENABLE_PLOTS=false
NUM_PLOT_EXAMPLES=4
'''


def build() -> None:
    if not CANDIDATES.is_file():
        raise FileNotFoundError(CANDIDATES)
    records = [json.loads(line) for line in CANDIDATES.open()]
    if len(records) != 21_523:
        raise ValueError(f'Esperados 21.523 pares, encontrados {len(records)}.')

    # Preserva todos os locutores com ao menos três gravações em cada um dos
    # cinco grupos de teste. O filtro em 8 kHz e a margem de fronteira tornam
    # inviáveis os 20 quadros estimados no detector exploratório de 48 kHz.
    for k in range(20, 0, -1):
        eligible = [r for r in records if len(r['active']) >= k and len(r['low']) >= k]
        by_speaker = Counter(r['locutor'] for r in eligible)
        by_group = Counter((r['locutor'], (r['enunciado'] - 1) % 5) for r in eligible)
        min_group = min(by_group.get((s, g), 0) for s in range(1, 109) for g in range(5))
        if min_group >= 3:
            break
    if k < 10:
        raise ValueError(f'O detector 8 kHz deixou apenas {k} quadros por condição.')
    prefix = f'vctk_activity{k}'
    print(f'Seleção: {k} quadros por condição, {len(eligible)} pares, '
          f'mínimo {min(by_speaker.values())} por locutor e '
          f'{min_group} em cada grupo de teste.', flush=True)

    for position, record in enumerate(eligible, 1):
        speaker = record['locutor']
        utterance = record['enunciado']
        n = record['quadros']
        indices = {
            'unfiltered': select_even(list(range(n)), k),
            'active': select_even(record['active'], k),
            'low': select_even(record['low'], k),
        }
        for mic in ('mic1', 'mic2'):
            original = FEATURES / f'vctk8k_{mic}' / str(speaker) / str(utterance) / 'mfccs.npy'
            matrix = np.load(original, mmap_mode='r')
            if matrix.shape != (40, n):
                raise ValueError(f'MFCC inesperado: {original}: {matrix.shape}')
            for condition, selected in indices.items():
                output = FEATURES / f'{prefix}_{condition}_{mic}' / str(speaker) / str(utterance) / 'mfccs.npy'
                if output.is_file():
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix('.tmp.npy')
                np.save(temporary, np.asarray(matrix[:, selected], dtype=np.float32))
                temporary.replace(output)
        if position % 1000 == 0:
            print(f'Features VAD: {position}/{len(eligible)} pares', flush=True)

    for condition in ('unfiltered', 'active', 'low'):
        for mic in ('mic1', 'mic2'):
            (ROOT / 'configs' / f'{prefix}_{condition}_{mic}.env').write_text(
                config_text(condition, mic, k, prefix))
    summary = {'prefix': prefix, 'quadros': k, 'pares': len(eligible),
               'locutores': 108, 'minimo_por_locutor': min(by_speaker.values()),
               'minimo_por_grupo_de_teste': min_group,
               'sha256_gravacoes': hashlib.sha256(json.dumps(
                   [(r['locutor'], r['enunciado']) for r in eligible],
                   separators=(',', ':')).encode()).hexdigest(),
               'metodo': 'librosa.effects.split em 8 kHz, top_db=30, janela 256, salto 128; '
                         'interseção dos microfones; margem de um quadro em cada fronteira',
               'condicoes': ['unfiltered', 'active', 'low']}
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(SUMMARY, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('scan', 'build'))
    stage = parser.parse_args().stage
    if stage == 'scan':
        scan()
    else:
        build()


if __name__ == '__main__':
    main()
