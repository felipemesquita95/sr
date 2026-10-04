#!/usr/bin/env python3
"""Conta atividade e baixa atividade em cada gravação VCTK completa."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
MANIFEST = ROOT / 'runs/features/vctk_manifesto.json'
LENGTHS = ROOT / 'runs/features/relatorio_comprimentos_8k.csv'
OUTPUT = ROOT / 'docs/vad_audio_completo_vctk.csv'
SUMMARY = ROOT / 'docs/vad_audio_completo_vctk.json'
HOP_SAMPLES_48K = 768  # 16 ms


def active_mask(path: Path, frames: int) -> np.ndarray:
    audio, rate = sf.read(path, dtype='float32', always_2d=False)
    if rate != 48_000:
        raise ValueError(f'Taxa inesperada em {path}: {rate}')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    intervals = librosa.effects.split(audio, top_db=30)
    centers = np.arange(frames) * HOP_SAMPLES_48K
    mask = np.zeros(frames, dtype=bool)
    for start, end in intervals:
        mask |= (centers >= start) & (centers < end)
    return mask


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    lengths = {}
    with LENGTHS.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['corpus'] == 'vctk':
                lengths[(row['trilha'].rsplit('_', 1)[-1],
                         int(row['locutor']), int(row['enunciado']))] = int(row['quadros'])

    fields = ['locutor', 'enunciado', 'arquivo', 'quadros_mfcc',
              'voz_mic1', 'baixa_mic1', 'voz_mic2', 'baixa_mic2',
              'voz_comum', 'baixa_comum', 'discordantes']
    rows = []
    with OUTPUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for speaker_index, speaker in manifest['locutores'].items():
            speaker_number = int(speaker_index)
            for utterance_index, identifier in manifest['enunciados'][speaker].items():
                utterance_number = int(utterance_index)
                n1 = lengths[('mic1', speaker_number, utterance_number)]
                n2 = lengths[('mic2', speaker_number, utterance_number)]
                if n1 != n2:
                    raise ValueError(f'Comprimentos diferentes: {speaker} {identifier}')
                masks = {}
                for mic in ('mic1', 'mic2'):
                    path = AUDIO / speaker / f'{speaker}_{identifier}_{mic}.flac'
                    masks[mic] = active_mask(path, n1)
                first, second = masks['mic1'], masks['mic2']
                row = {
                    'locutor': speaker_index, 'enunciado': utterance_index,
                    'arquivo': f'{speaker}_{identifier}', 'quadros_mfcc': n1,
                    'voz_mic1': int(first.sum()), 'baixa_mic1': int((~first).sum()),
                    'voz_mic2': int(second.sum()), 'baixa_mic2': int((~second).sum()),
                    'voz_comum': int((first & second).sum()),
                    'baixa_comum': int((~first & ~second).sum()),
                    'discordantes': int((first ^ second).sum()),
                }
                writer.writerow(row)
                rows.append(row)
            stream.flush()
            if speaker_number % 10 == 0:
                print(f'{speaker_number}/108 locutores; {len(rows)} pares', flush=True)

    summary = {'pares': len(rows), 'limiar_vad_db': 30,
               'descricao': 'VAD em cada FLAC completo; classificação no centro do quadro MFCC.'}
    for field in ('quadros_mfcc', 'voz_mic1', 'baixa_mic1', 'voz_mic2',
                  'baixa_mic2', 'voz_comum', 'baixa_comum', 'discordantes'):
        values = np.array([r[field] for r in rows])
        summary[field] = {
            'min': int(values.min()), 'q1': float(np.quantile(values, .25)),
            'mediana': float(np.median(values)), 'q3': float(np.quantile(values, .75)),
            'max': int(values.max()), 'sem_quadros': int((values == 0).sum()),
        }
    for k in (1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150):
        selected = [r for r in rows if min(r['voz_comum'], r['baixa_comum']) >= k]
        by_speaker = Counter(int(r['locutor']) for r in selected)
        summary[f'elegiveis_{k}'] = {
            'pares': len(selected),
            'locutores': sum(by_speaker.get(s, 0) > 0 for s in range(1, 109)),
            'minimo_por_locutor': min(by_speaker.get(s, 0) for s in range(1, 109)),
        }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(OUTPUT)
    print(SUMMARY)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
