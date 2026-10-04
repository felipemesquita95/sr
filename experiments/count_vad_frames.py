#!/usr/bin/env python3
"""Conta fala detectada e baixa atividade nos primeiros 77/153 quadros VCTK."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
MANIFEST = ROOT / 'runs/features/vctk_manifesto.json'
LENGTHS = ROOT / 'runs/features/relatorio_comprimentos_8k.csv'
OUTPUT = ROOT / 'docs/vad_quadros_vctk.csv'
SUMMARY = ROOT / 'docs/vad_quadros_vctk.json'
HOP_SAMPLES_48K = 768  # 128 amostras a 8 kHz = 16 ms
MAX_FRAMES = 153


def active_mask(path: Path, frames: int) -> np.ndarray:
    needed = (frames - 1) * HOP_SAMPLES_48K + 2048
    audio, rate = sf.read(path, frames=needed, dtype='float32', always_2d=False)
    if rate != 48_000:
        raise ValueError(f'Taxa inesperada: {path}: {rate}')
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
              'voz77_mic1', 'baixa77_mic1', 'voz77_mic2', 'baixa77_mic2',
              'voz153_mic1', 'baixa153_mic1', 'voz153_mic2', 'baixa153_mic2',
              'voz77_comum', 'baixa77_comum', 'voz153_comum', 'baixa153_comum']
    rows = []
    for speaker_index, speaker in manifest['locutores'].items():
        speaker_number = int(speaker_index)
        for utterance_index, identifier in manifest['enunciados'][speaker].items():
            utterance_number = int(utterance_index)
            n1 = lengths[('mic1', speaker_number, utterance_number)]
            n2 = lengths[('mic2', speaker_number, utterance_number)]
            if n1 != n2:
                raise ValueError(f'Comprimentos diferentes: {speaker} {identifier}')
            count = min(n1, MAX_FRAMES)
            masks = {}
            for mic in ('mic1', 'mic2'):
                path = AUDIO / speaker / f'{speaker}_{identifier}_{mic}.flac'
                masks[mic] = active_mask(path, count)
            row = {'locutor': speaker_index, 'enunciado': utterance_index,
                   'arquivo': f'{speaker}_{identifier}', 'quadros_mfcc': n1}
            for mic in ('mic1', 'mic2'):
                for limit in (77, 153):
                    sample = masks[mic][:min(count, limit)]
                    row[f'voz{limit}_{mic}'] = int(sample.sum())
                    row[f'baixa{limit}_{mic}'] = int(len(sample) - sample.sum())
            for limit in (77, 153):
                first = masks['mic1'][:min(count, limit)]
                second = masks['mic2'][:min(count, limit)]
                row[f'voz{limit}_comum'] = int((first & second).sum())
                row[f'baixa{limit}_comum'] = int((~first & ~second).sum())
            rows.append(row)
        if speaker_number % 10 == 0:
            print(f'{speaker_number}/108 locutores; {len(rows)} pares', flush=True)

    with OUTPUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {'pares': len(rows), 'limiar_vad_db': 30,
               'janela_vad_amostras_48k': 2048, 'salto_mfcc_ms': 16,
               'descricao': 'VAD nos primeiros 153 quadros; presença no centro do quadro MFCC.'}
    for field in ('baixa77_mic1', 'baixa77_mic2', 'baixa153_mic1',
                  'baixa153_mic2', 'baixa77_comum', 'voz77_comum',
                  'baixa153_comum', 'voz153_comum'):
        values = np.array([r[field] for r in rows])
        summary[field] = {'min': int(values.min()), 'q1': float(np.quantile(values, .25)),
                          'mediana': float(np.median(values)),
                          'q3': float(np.quantile(values, .75)),
                          'max': int(values.max()), 'sem_quadros': int((values == 0).sum())}

    # Uma gravação só é utilizável no contraste pareado se tiver K quadros de
    # cada condição nos mesmos instantes de ambos os microfones.
    for minimum_length in (77, 153):
        eligible = [r for r in rows if r['quadros_mfcc'] >= minimum_length]
        counts = {}
        for k in (1, 5, 10, 20, 30, 40, 50):
            selected = [r for r in eligible if r[f'voz{minimum_length}_comum'] >= k
                        and r[f'baixa{minimum_length}_comum'] >= k]
            by_speaker = Counter(int(r['locutor']) for r in selected)
            counts[str(k)] = {'pares': len(selected),
                              'locutores': sum(by_speaker.get(s, 0) > 0 for s in range(1, 109)),
                              'minimo_por_locutor': min(by_speaker.get(s, 0)
                                                        for s in range(1, 109)),
                              'locutores_com_20_ou_mais': sum(by_speaker.get(s, 0) >= 20
                                                              for s in range(1, 109))}
        summary[f'elegiveis_base_{minimum_length}'] = counts
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(OUTPUT)
    print(SUMMARY)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
