#!/usr/bin/env python3
"""Plot an audited BRSD Silero/RMS frame selection used by the 16 kHz runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/lsmsqt/HDD/sr_project/brsd_silero_isolated_features')
SOURCE40 = Path('/media/lsmsqt/HDD/sr_project/brsd40_silero_isolated_features')
DEST = ROOT / 'docs/figuras/protocolo_silero_brsd/01_selecao_111_quadros.png'


def main() -> None:
    first = json.loads((SOURCE / 'audit/1.json').read_text())
    second = json.loads((SOURCE40 / 'audit/1.json').read_text())
    for key in ('trim_start_sample', 'trim_end_sample',
                'selected_window_start_frame', 'selected_window_end_frame'):
        assert first[key] == second[key]
    signal, sr = sf.read(first['input'], dtype='float32')
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = signal[first['trim_start_sample']:first['trim_end_sample']]
    frame_a = first['selected_window_start_frame']
    frame_b = first['selected_window_end_frame']
    assert frame_b - frame_a == 111
    start = frame_a * .016
    end = (frame_b - 1) * .016 + .032
    t = np.arange(signal.size) / sr
    stride = max(1, sr // 1000)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for ax, zoom in zip(axes, (False, True)):
        ax.plot(t[::stride], signal[::stride], color='#17334c', lw=.55,
                rasterized=True)
        for j, segment in enumerate(first['silero_segments']):
            ax.axvspan(segment['start'] / 16000,
                       segment['end'] / 16000, color='#75b8aa', alpha=.28,
                       label='Fala detectada pelo Silero' if j == 0 else None)
        ax.axvspan(start, end, color='#df7835', alpha=.53,
                   label='111 quadros escolhidos por RMS')
        ax.set_ylabel('Amplitude')
        ax.grid(alpha=.16)
        if zoom:
            ax.set_xlim(max(0, start - 1.2), min(t[-1], end + 1.2))
            ax.set_title('Detalhe da janela: 1,792 s de fala continua')
        else:
            ax.set_xlim(0, t[-1])
            ax.set_title('BRSD 1.wav: gravacao apos recorte de bordas')
    axes[0].legend(loc='upper right', frameon=False, ncol=2)
    axes[1].set_xlabel('Tempo apos recorte de bordas (s)')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST, dpi=175, facecolor='white')
    plt.close(fig)
    print(DEST)
    print(f'window={start:.3f}..{end:.3f}s; sr={sr}; original_frames={first["original_frames"]}')


if __name__ == '__main__':
    main()
