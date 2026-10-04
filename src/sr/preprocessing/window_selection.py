"""Seleciona uma janela contínua comum aos dois microfones pela área RMS."""

from __future__ import annotations

import numpy as np


def frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """RMS de quadros completos, na mesma grade temporal usada pelo MFCC."""
    if len(audio) < frame:
        return np.empty(0, dtype=np.float64)
    starts = np.arange(0, len(audio) - frame + 1, hop)
    cumulative = np.r_[0.0, np.cumsum(np.square(audio, dtype=np.float64))]
    return np.sqrt((cumulative[starts + frame] - cumulative[starts]) / frame)


def best_common_rms_window(mic1: np.ndarray, mic2: np.ndarray,
                           k_frames: int, frame: int = 512,
                           hop: int = 256) -> tuple[int, int, float]:
    """Retorna [início, fim) de K quadros com maior área RMS normalizada.

    Os dois microfones usam a mesma janela. Cada curva RMS é dividida pelo seu
    próprio pico para que a diferença de ganho entre canais não escolha o trecho.
    Empates ficam com a primeira janela. Não seleciona colunas não contíguas.
    """
    if k_frames <= 0:
        raise ValueError('k_frames deve ser positivo.')
    rms1 = frame_rms(mic1, frame, hop)
    rms2 = frame_rms(mic2, frame, hop)
    if len(rms1) != len(rms2):
        raise ValueError('Os dois sinais precisam compartilhar a grade de quadros.')
    if k_frames > len(rms1):
        raise ValueError('A gravação contém menos quadros do que K.')
    score = (rms1 / max(float(rms1.max()), 1e-12)
             + rms2 / max(float(rms2.max()), 1e-12)) / 2
    cumulative = np.r_[0.0, np.cumsum(score)]
    areas = cumulative[k_frames:] - cumulative[:-k_frames]
    start = int(np.argmax(areas))
    return start, start + k_frames, float(areas[start])
