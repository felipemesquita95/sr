#!/usr/bin/env python3
"""Pranchas de auditoria: gravação selecionada mais curta de cada locutor."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/sr-mplconfig')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import decimate  # noqa: E402

from process_vctk_pair_corrected import soft_crop  # noqa: E402
from sr.preprocessing.window_selection import best_common_rms_window  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/'
             'VCTK-Corpus-0.92/wav48_silence_trimmed')


def main() -> None:
    cohort = json.loads((ROOT / 'output/vctk16_cohort_120.json').read_text())
    by_speaker = {}
    for item in cohort['items']:
        speaker = item['speaker']
        if (speaker not in by_speaker
                or item['frames_after_trim'] < by_speaker[speaker]['frames_after_trim']):
            by_speaker[speaker] = item
    chosen = [by_speaker[s] for s in sorted(by_speaker)]
    output = ROOT / 'output/vctk16_auditoria_locutores'
    output.mkdir(parents=True, exist_ok=True)
    k = cohort['frames_per_recording']
    for page in range((len(chosen) + 5) // 6):
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
        for ax, item in zip(axes.flat, chosen[page * 6:(page + 1) * 6]):
            speaker, utterance = item['speaker'], item['utterance']
            base = AUDIO / speaker / f'{speaker}_{utterance}'
            mic1, sr = sf.read(str(base) + '_mic1.flac', dtype='float64')
            mic2, _ = sf.read(str(base) + '_mic2.flac', dtype='float64')
            left, right = item['trim_start_sample'], item['trim_end_sample']
            reduced1 = decimate(soft_crop(mic1, left, right, sr),
                                3, n=8, ftype='iir', zero_phase=True)
            reduced2 = decimate(soft_crop(mic2, left, right, sr),
                                3, n=8, ftype='iir', zero_phase=True)
            window_start, window_end, _ = best_common_rms_window(
                reduced1, reduced2, k)
            t = np.arange(len(mic1)) / sr
            ax.plot(t, mic1, lw=.3, color='#336e98')
            ax.axvspan(left / sr, right / sr, color='#55a35e', alpha=.12)
            selected_start = left / sr + window_start * .016
            selected_end = left / sr + (window_end - 1) * .016 + .032
            ax.axvspan(selected_start, selected_end, color='#df992d', alpha=.2)
            ax.axvline(left / sr, color='#b63e38', ls='--', lw=.8)
            ax.axvline(right / sr, color='#b63e38', ls='--', lw=.8)
            ax.set(title=f'{speaker}_{utterance}: {item["frames_after_trim"]} '
                         f'quadros após trim', xlim=(0, len(mic1) / sr),
                   xlabel='Tempo (s)', ylabel='Amplitude')
        for ax in axes.flat[len(chosen[page * 6:(page + 1) * 6]):]:
            ax.axis('off')
        fig.suptitle('Verde: após trim; laranja: janela RMS de 111 quadros. '
                     'Exemplo mais curto selecionado por locutor.')
        destination = output / f'pagina_{page + 1:02d}.png'
        fig.savefig(destination, dpi=140)
        plt.close(fig)
        print(destination, flush=True)


if __name__ == '__main__':
    main()
