#!/usr/bin/env python3
"""Plot one source VCTK pair with Silero speech decisions and our soft trim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import decimate
from silero_vad import get_speech_timestamps, load_silero_vad


AUDIO = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')


def detect_full(audio: np.ndarray, model) -> list[dict]:
    reduced = decimate(audio, q=3, n=8, ftype='iir', zero_phase=True)
    return get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(reduced, dtype=np.float32)),
        model, sampling_rate=16000, threshold=.5,
        min_speech_duration_ms=250, min_silence_duration_ms=100,
        speech_pad_ms=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speaker', default='p225')
    parser.add_argument('--utterance', default='241')
    parser.add_argument('--cohort', type=Path,
                        default=Path('output/vctk16_cohort_120.json'))
    parser.add_argument('--scan', type=Path,
                        default=Path('output/vctk16_silero/scan_all.jsonl'))
    parser.add_argument('--output', type=Path,
                        default=Path('output/vctk16_silero/example_p225_241_vad_original.png'))
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    item = next(x for x in cohort['items'] if x['speaker'] == args.speaker
                and x['utterance'] == args.utterance)
    row = next(x for x in map(json.loads, args.scan.open())
               if x['speaker'] == args.speaker and x['utterance'] == args.utterance)
    start = item['trim_start_sample'] / 48000
    end = item['trim_end_sample'] / 48000
    torch.set_num_threads(1)
    model = load_silero_vad()
    fig, axes = plt.subplots(3, 1, figsize=(13, 7.2), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1.3]})
    full_results = {}
    duration = None
    for ax, mic, color in zip(axes[:2], ('mic1', 'mic2'), ('#198754', '#1976a3')):
        path = AUDIO / args.speaker / f'{args.speaker}_{args.utterance}_{mic}.flac'
        audio, sr = sf.read(path, dtype='float32')
        if sr != 48000:
            raise ValueError(path)
        duration = len(audio) / sr
        segments = detect_full(audio, model)
        full_results[mic] = segments
        ax.plot(np.arange(0, len(audio), 8) / sr, audio[::8],
                color='#25334a', linewidth=.45, alpha=.8, rasterized=True)
        for seg in segments:
            ax.axvspan(seg['start'] / 16000, seg['end'] / 16000,
                       color=color, alpha=.2, linewidth=0)
        ax.axvline(start, color='#b75a25', linestyle='--', linewidth=1.25)
        ax.axvline(end, color='#b75a25', linestyle='--', linewidth=1.25)
        ax.set_ylabel(f'{mic}\namplitude')
        ax.grid(axis='y', alpha=.2)
    decision_ax = axes[2]
    for y, mic, color in ((2, 'mic1', '#198754'), (1, 'mic2', '#1976a3')):
        for segment in full_results[mic]:
            decision_ax.broken_barh(
                [(segment['start'] / 16000,
                  (segment['end'] - segment['start']) / 16000)],
                (y - .28, .56), facecolors=color)
        for segment in row['mics'][mic]['speech_segments_samples']:
            decision_ax.broken_barh(
                [(start + segment['start'] / 16000,
                  (segment['end'] - segment['start']) / 16000)],
                (y - .11, .22), facecolors='#ffd449', edgecolors='#685b15',
                linewidth=.4)
    decision_ax.axvspan(start, end, color='#b75a25', alpha=.08)
    decision_ax.axvline(start, color='#b75a25', linestyle='--', linewidth=1.25)
    decision_ax.axvline(end, color='#b75a25', linestyle='--', linewidth=1.25)
    decision_ax.set_yticks([1, 2], ['mic2', 'mic1'])
    decision_ax.set_ylim(.45, 2.55)
    decision_ax.set_xlim(0, duration)
    decision_ax.set_ylabel('Decisão\nde fala')
    decision_ax.set_xlabel('Tempo no arquivo de entrada (s)')
    decision_ax.grid(axis='x', alpha=.2)
    fig.legend(handles=[
        Patch(facecolor='#198754', alpha=.3, label='Silero no arquivo inteiro (mic1)'),
        Patch(facecolor='#1976a3', alpha=.3, label='Silero no arquivo inteiro (mic2)'),
        Patch(facecolor='#ffd449', label='Silero após nosso trim: fala usada'),
        Line2D([0], [0], color='#b75a25', linestyle='--',
               label='Limites do nosso trim'),
    ], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.015))
    fig.suptitle(f'VCTK {args.speaker}_{args.utterance}: Silero VAD nos dois microfones',
                 fontsize=15, fontweight='bold')
    fig.text(.5, .924,
             'Entrada wav48_silence_trimmed a 48 kHz; Silero recebe versão decimada a 16 kHz. '
             'Sem fala detectada ≠ silêncio comprovado.',
             ha='center', fontsize=9, color='#4a5260')
    fig.tight_layout(rect=(0, .075, 1, .91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches='tight')
    print(args.output)
    print(json.dumps({mic: {'full_segments': len(full_results[mic]),
                            'trimmed_segments': len(row['mics'][mic]['speech_segments_samples']),
                            'speech_frames_after_trim': row['mics'][mic]['frames_speech'],
                            'total_frames_after_trim': row['mics'][mic]['frames_total']}
                      for mic in ('mic1', 'mic2')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
