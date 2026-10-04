#!/usr/bin/env python3
"""Processa um par VCTK com limites de corte compartilhados entre microfones.

Entrada: dois FLACs de wav48_silence_trimmed, correspondentes à mesma leitura.
Saída: arrays MFCC, delta e delta-delta e um manifesto de processamento.
Este script não seleciona gravações, folds ou K; essas decisões são posteriores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import decimate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from sr.preprocessing.window_selection import best_common_rms_window  # noqa: E402


def rms_frames(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(audio) < frame:
        return np.array([np.sqrt(np.mean(np.square(audio, dtype=np.float64)))])
    starts = np.arange(0, len(audio) - frame + 1, hop)
    energy = np.square(audio, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(energy)))
    return np.sqrt((cumulative[starts + frame] - cumulative[starts]) / frame)


def common_bounds(
    first: np.ndarray,
    second: np.ndarray,
    sr: int,
    threshold_db: float = -30.0,
    margin_ms: float = 100.0,
    end_margin_ms: float = 250.0,
    min_active_frames: int = 5,
) -> tuple[int, int, dict]:
    """Retém a união da atividade dos dois canais, inclusive pausas internas."""
    if len(first) != len(second):
        raise ValueError('O par de microfones deve ter o mesmo número de amostras.')
    frame, hop = round(0.032 * sr), round(0.016 * sr)
    energies = [rms_frames(channel, frame, hop) for channel in (first, second)]
    active = np.zeros(len(energies[0]), dtype=bool)
    for energy in energies:
        peak = float(energy.max())
        if peak > 0:
            active |= energy >= peak * 10 ** (threshold_db / 20)
    transitions = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    sustained = np.zeros_like(active)
    for run_start, run_end in zip(starts, ends):
        if run_end - run_start >= min_active_frames:
            sustained[run_start:run_end] = True
    if sustained.any():
        active = sustained
    if not active.any():
        return 0, len(first), {'active_frames': 0, 'total_frames': len(active)}
    indexes = np.flatnonzero(active)
    start_margin = round(sr * margin_ms / 1000)
    end_margin = round(sr * end_margin_ms / 1000)
    start = max(0, int(indexes[0]) * hop - start_margin)
    end = min(len(first), int(indexes[-1]) * hop + frame + end_margin)
    return start, end, {'active_frames': int(active.sum()),
                        'total_frames': len(active),
                        'min_active_frames': min_active_frames}


def soft_crop(audio: np.ndarray, start: int, end: int, sr: int,
              fade_ms: float = 8.0) -> np.ndarray:
    cropped = audio[start:end].copy()
    fade = min(round(sr * fade_ms / 1000), len(cropped) // 2)
    if fade and start > 0:
        cropped[:fade] *= np.linspace(0, 1, fade, endpoint=True)
    if fade and end < len(audio):
        cropped[-fade:] *= np.linspace(1, 0, fade, endpoint=True)
    return cropped


def features(audio: np.ndarray, source_sr: int, target_sr: int,
             n_mfcc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_sr != 48000 or target_sr not in (8000, 16000):
        raise ValueError('O protocolo exige entrada 48 kHz e saída 8 ou 16 kHz.')
    reduced = decimate(audio, q=source_sr // target_sr, n=8,
                       ftype='iir', zero_phase=True)
    return features_from_reduced(reduced, target_sr, n_mfcc)


def features_from_reduced(reduced: np.ndarray, target_sr: int,
                          n_mfcc: int, compute_derivatives: bool = True
                          ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    emphasized = np.empty_like(reduced)
    emphasized[0] = reduced[0]
    emphasized[1:] = reduced[1:] - 0.97 * reduced[:-1]
    frame, hop = round(0.032 * target_sr), round(0.016 * target_sr)
    if len(emphasized) < frame:
        raise ValueError('Trecho curto demais para uma janela completa de 32 ms.')
    static = librosa.feature.mfcc(
        y=emphasized, sr=target_sr, n_mfcc=n_mfcc,
        n_fft=frame, hop_length=hop, window='hamming', center=False,
    ).astype(np.float32)
    if not compute_derivatives:
        return static, None, None
    if static.shape[1] < 9:
        raise ValueError('São necessários pelo menos 9 quadros para os deltas.')
    delta = librosa.feature.delta(static, width=9).astype(np.float32)
    delta2 = librosa.feature.delta(static, width=9, order=2).astype(np.float32)
    return static, delta, delta2


def process_pair(mic1: Path, mic2: Path, output: Path, target_sr: int,
                 n_mfcc: int, threshold_db: float, margin_ms: float,
                 fade_ms: float, end_margin_ms: float = 250.0,
                 k_frames: int | None = None) -> dict:
    audio1, sr1 = sf.read(mic1, dtype='float64')
    audio2, sr2 = sf.read(mic2, dtype='float64')
    if sr1 != 48000 or sr2 != 48000 or audio1.ndim != 1 or audio2.ndim != 1:
        raise ValueError('São esperados dois sinais mono de 48 kHz.')
    start, end, activity = common_bounds(audio1, audio2, sr1,
                                         threshold_db, margin_ms, end_margin_ms)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    reduced = [decimate(soft_crop(audio, start, end, sr1, fade_ms),
                        q=sr1 // target_sr, n=8, ftype='iir', zero_phase=True)
               for audio in (audio1, audio2)]
    if k_frames is not None:
        frame = round(0.032 * target_sr)
        hop = round(0.016 * target_sr)
        window_start, window_end, area = best_common_rms_window(
            reduced[0], reduced[1], k_frames, frame, hop)
    else:
        window_start, window_end, area = 0, None, None
    for name, signal in zip(('mic1', 'mic2'), reduced):
        static, _, _ = features_from_reduced(signal, target_sr, n_mfcc,
                                             compute_derivatives=False)
        static = static[:, window_start:window_end]
        if static.shape[1] < 9:
            raise ValueError('A janela selecionada precisa de pelo menos 9 quadros.')
        delta = librosa.feature.delta(static, width=9).astype(np.float32)
        delta2 = librosa.feature.delta(static, width=9, order=2).astype(np.float32)
        np.savez_compressed(output / f'{name}.npz',
                            mfcc=static, delta=delta, delta_delta=delta2)
        counts[name] = int(static.shape[1])
    manifest = {
        'inputs': {'mic1': str(mic1), 'mic2': str(mic2)},
        'source_sr': sr1, 'target_sr': target_sr,
        'trim_start_sample': start, 'trim_end_sample': end,
        'trim_start_seconds': start / sr1, 'trim_end_seconds': end / sr1,
        'threshold_db_relative_to_each_mic_peak': threshold_db,
        'start_margin_ms': margin_ms, 'end_margin_ms': end_margin_ms,
        'fade_ms': fade_ms,
        'activity': activity, 'n_mfcc': n_mfcc,
        'frame_ms': 32, 'hop_ms': 16, 'window': 'hamming',
        'stft_center': False, 'pre_emphasis': 0.97,
        'decimate': {'filter': 'Chebyshev I', 'order': 8,
                     'zero_phase': True, 'q': sr1 // target_sr},
        'delta_width_frames': 9, 'frame_counts': counts,
        'delta_scope': 'selected_window_only',
        'selected_window_start_frame': window_start,
        'selected_window_end_frame': window_end,
        'selected_window_area_normalized_rms': area,
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mic1', type=Path)
    parser.add_argument('mic2', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--target-sr', type=int, choices=(8000, 16000), default=8000)
    parser.add_argument('--n-mfcc', type=int, default=30)
    parser.add_argument('--threshold-db', type=float, default=-30)
    parser.add_argument('--margin-ms', type=float, default=100)
    parser.add_argument('--end-margin-ms', type=float, default=250)
    parser.add_argument('--fade-ms', type=float, default=8)
    parser.add_argument('--k-frames', type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(process_pair(args.mic1, args.mic2, args.output,
                                  args.target_sr, args.n_mfcc,
                                  args.threshold_db, args.margin_ms,
                                  args.fade_ms, args.end_margin_ms,
                                  args.k_frames),
                     indent=2, ensure_ascii=False))
