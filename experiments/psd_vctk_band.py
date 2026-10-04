#!/usr/bin/env python3
"""Mede a distribuição de potência do VCTK original antes da redução para 8 kHz."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'runs/features/vctk_manifesto.json'
AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
OUTPUT = ROOT / 'docs/figuras/psd_vctk_8k.png'
REPORT = ROOT / 'docs/psd_vctk_8k.json'
SEED = 42
PER_SPEAKER = 3
NFFT = 4096
ACTIVE_DB = 30


def analyze(path: Path):
    audio, rate = sf.read(path, dtype='float32', always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 48_000:
        raise ValueError(f'Taxa inesperada em {path}: {rate}')
    freq, _, psd_frames = spectrogram(
        audio, fs=rate, window='hann', nperseg=NFFT, noverlap=NFFT // 2,
        detrend='constant', scaling='density', mode='psd',
    )
    df = freq[1] - freq[0]
    frame_power = psd_frames.sum(axis=0) * df
    active = frame_power >= frame_power.max() * 10 ** (-ACTIVE_DB / 10)
    psd = psd_frames[:, active].mean(axis=1)
    full_psd = psd_frames.mean(axis=1)
    total = psd.sum()
    fractions = {
        'abaixo_3k6': float(psd[freq < 3600].sum() / total),
        'abaixo_4k': float(psd[freq < 4000].sum() / total),
        'abaixo_4k_sinal_inteiro': float(full_psd[freq < 4000].sum() / full_psd.sum()),
        '4k_a_8k': float(psd[(freq >= 4000) & (freq < 8000)].sum() / total),
        '8k_a_24k': float(psd[freq >= 8000].sum() / total),
    }
    return freq, psd / (total * df), fractions, int(active.sum()), len(active)


def main():
    manifest = json.loads(MANIFEST.read_text())
    rng = np.random.default_rng(SEED)
    records = {'mic1': [], 'mic2': []}
    curves = {'mic1': [], 'mic2': []}
    frequency = None
    for speaker in manifest['locutores'].values():
        utterances = manifest['enunciados'][speaker]
        choices = rng.choice(sorted(utterances, key=int), size=PER_SPEAKER, replace=False)
        for index in choices:
            identifier = utterances[str(index)]
            for mic in ('mic1', 'mic2'):
                path = AUDIO / speaker / f'{speaker}_{identifier}_{mic}.flac'
                frequency, curve, fractions, active, total = analyze(path)
                records[mic].append({
                    'speaker': speaker, 'utterance': str(index), 'file': str(path),
                    'active_frames': active, 'total_frames': total, **fractions,
                })
                curves[mic].append(curve)

    summary = {}
    for mic, rows in records.items():
        summary[mic] = {
            'recordings': len(rows),
            'speakers': len({row['speaker'] for row in rows}),
            'bands': {},
        }
        for band in ('abaixo_3k6', 'abaixo_4k', 'abaixo_4k_sinal_inteiro',
                     '4k_a_8k', '8k_a_24k'):
            values = np.array([row[band] for row in rows])
            summary[mic]['bands'][band] = {
                'median': float(np.median(values)),
                'q1': float(np.quantile(values, .25)),
                'q3': float(np.quantile(values, .75)),
                'min': float(values.min()),
                'max': float(values.max()),
            }

    REPORT.write_text(json.dumps({
        'method': {'sample': '3 paired recordings per speaker, seed 42',
                   'sampling_rate': 48000, 'nperseg': NFFT,
                   'activity': 'spectral frame power within 30 dB of recording peak'},
        'summary': summary, 'records': records,
    }, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for mic, color in (('mic1', '#2367a1'), ('mic2', '#d47132')):
        median_psd = np.median(np.stack(curves[mic]), axis=0)
        axes[0].plot(frequency / 1000, 10 * np.log10(np.maximum(median_psd, 1e-12)),
                     label=mic, color=color, lw=1)
        cumulative = np.cumsum(median_psd) / median_psd.sum()
        axes[1].plot(frequency / 1000, 100 * cumulative, label=mic, color=color, lw=1.5)
    for ax in axes:
        ax.axvline(3.6, color='black', ls='--', lw=.9, label='corte 3,6 kHz')
        ax.axvline(4, color='gray', ls=':', lw=.9, label='Nyquist 8 kHz')
        ax.set_xlim(0, 12)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('PSD normalizada (dB/Hz)')
    axes[1].set_ylabel('Potência acumulada (%)')
    axes[1].set_xlabel('Frequência (kHz)')
    fig.suptitle('VCTK original: 3 gravações por locutor, quadros ativos')
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)
    print(REPORT)
    print(OUTPUT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
