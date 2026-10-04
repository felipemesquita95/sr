#!/usr/bin/env python3
"""Gera áudio e figura para inspecionar o corte de um par VCTK."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import decimate

from process_vctk_pair_corrected import common_bounds, rms_frames, soft_crop
from sr.preprocessing.window_selection import best_common_rms_window


def preview(mic1: Path, mic2: Path, output: Path,
            k_frames: int | None = None) -> None:
    first, sr = sf.read(mic1, dtype='float64')
    second, other_sr = sf.read(mic2, dtype='float64')
    if sr != 48000 or other_sr != sr or first.ndim != 1 or second.ndim != 1:
        raise ValueError('Esperados dois canais mono de 48 kHz.')
    start, end, _ = common_bounds(first, second, sr)
    trimmed = soft_crop(first, start, end, sr)
    output.mkdir(parents=True, exist_ok=True)
    sf.write(output / 'original_mic1.wav', first, sr)
    sf.write(output / 'recortado_mic1.wav', trimmed, sr)
    reduced1 = decimate(trimmed, 3, n=8, ftype='iir', zero_phase=True)
    reduced2 = decimate(soft_crop(second, start, end, sr), 3, n=8,
                        ftype='iir', zero_phase=True)
    sf.write(output / 'recortado_mic1_16khz.wav', reduced1, 16000)
    selected_span = None
    if k_frames is not None:
        window_start, window_end, area = best_common_rms_window(
            reduced1, reduced2, k_frames)
        left = window_start * 256
        right = (window_end - 1) * 256 + 512
        selected_span = (left / 16000, right / 16000)
        sf.write(output / 'janela_selecionada_mic1_16khz.wav',
                 reduced1[left:right], 16000)
        print(f'Janela RMS: quadros {window_start}:{window_end}, '
              f'tempo {selected_span[0]:.3f}–{selected_span[1]:.3f} s '
              f'após trim, área normalizada {area:.3f}')

    frame, hop = round(0.032 * sr), round(0.016 * sr)
    e1, e2 = rms_frames(first, frame, hop), rms_frames(second, frame, hop)
    db1 = 20 * np.log10(np.maximum(e1, 1e-10))
    db2 = 20 * np.log10(np.maximum(e2, 1e-10))
    threshold1 = 20 * np.log10(max(float(e1.max()), 1e-10)) - 30
    threshold2 = 20 * np.log10(max(float(e2.max()), 1e-10)) - 30
    original_t = np.arange(len(first)) / sr
    trim_t = np.arange(len(trimmed)) / sr
    rms_t = np.arange(len(e1)) * hop / sr + frame / (2 * sr)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    axes[0].plot(original_t, first, lw=0.35, color='#255b8b')
    axes[0].axvspan(start / sr, end / sr, color='#53a653', alpha=0.12,
                    label='Trecho mantido')
    for boundary in (start / sr, end / sr):
        axes[0].axvline(boundary, color='#bd4238', ls='--', lw=1)
    axes[0].set(title='Arquivo original wav48_silence_trimmed — mic1',
                ylabel='Amplitude', xlim=(0, len(first) / sr))
    axes[0].legend(loc='upper right')

    axes[1].plot(rms_t, db1, label='RMS mic1', lw=1)
    axes[1].plot(rms_t, db2, label='RMS mic2', lw=1, alpha=0.75)
    axes[1].axhline(threshold1, color='#255b8b', ls=':', label='Limiar mic1')
    axes[1].axhline(threshold2, color='#e08a33', ls=':', label='Limiar mic2')
    for boundary in (start / sr, end / sr):
        axes[1].axvline(boundary, color='#bd4238', ls='--', lw=1)
    axes[1].set(ylabel='RMS (dBFS)', xlim=(0, len(first) / sr))
    axes[1].legend(loc='lower right', ncol=2)

    axes[2].plot(trim_t, trimmed, lw=0.35, color='#357c44')
    if selected_span is not None:
        axes[2].axvspan(*selected_span, color='#eea42b', alpha=0.16,
                        label=f'Janela escolhida: {k_frames} quadros')
        axes[2].legend(loc='upper right')
    axes[2].set(title='Após novo corte suave, tempo reiniciado em zero',
                ylabel='Amplitude', xlabel='Tempo (s)', xlim=(0, len(trimmed) / sr))

    fig.savefig(output / 'comparacao_trim.png', dpi=170)
    plt.close(fig)
    print(f'Original: {len(first)/sr:.3f} s; mantido: {len(trimmed)/sr:.3f} s; '
          f'retirado: {start/sr:.3f} s no início, {(len(first)-end)/sr:.3f} s no fim')
    print(output / 'comparacao_trim.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mic1', type=Path)
    parser.add_argument('mic2', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--k-frames', type=int)
    args = parser.parse_args()
    preview(args.mic1, args.mic2, args.output, args.k_frames)
