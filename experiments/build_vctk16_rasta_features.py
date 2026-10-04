#!/usr/bin/env python3
"""RASTA-MFCC on full trimmed audio, retaining the existing paired windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import decimate, freqz, lfilter, lfilter_zi

from process_vctk_pair_corrected import soft_crop


B = np.array([.2, .1, 0., -.1, -.2], dtype=np.float64)
A = np.array([1., -.94], dtype=np.float64)
PROTOCOL = {
    'name': 'RASTA-MFCC', 'version': 1,
    'numerator': B.tolist(), 'denominator': A.tolist(),
    'domain': 'power_to_db mel spectrogram, before DCT',
    'initialization': 'steady state scaled by first log-mel frame',
    'scope': 'full trimmed utterance, before retaining existing window',
    'hop_ms': 16, 'frame_ms': 32, 'n_mels': 128,
    'n_mfcc': 30, 'dct_type': 2, 'dct_norm': 'ortho',
    'top_db': 80, 'center': False, 'window': 'hamming',
    'pre_emphasis': .97, 'fade_ms': 8,
    'source_sr': 48000, 'target_sr': 16000,
    'decimate': {'q': 3, 'order': 8, 'filter': 'Chebyshev I', 'zero_phase': True},
    'delay_compensation': False, 'delta_width_frames': 9,
    'delta_scope': 'selected_window_only', 'cmn': False,
    'source': 'https://labrosa.ee.columbia.edu/~dpwe/papers/HermM94-rasta.pdf',
}


def rasta_filter(log_mel: np.ndarray) -> np.ndarray:
    if log_mel.ndim != 2 or log_mel.shape[1] < 5:
        raise ValueError('RASTA requires a frequency-by-time matrix with >=5 frames')
    if not np.isfinite(log_mel).all():
        raise ValueError('Non-finite input to RASTA')
    # Initialize from a constant extension of the first frame. Constant spectral
    # offsets then cancel without a zero-padding onset transient.
    initial = lfilter_zi(B, A)[None, :] * log_mel[:, :1]
    filtered, _ = lfilter(B, A, log_mel.astype(np.float64), axis=1, zi=initial)
    return filtered


def extract(reduced: np.ndarray, start: int, end: int,
            n_mfcc: int = 30) -> tuple[np.ndarray, ...]:
    emphasized = np.empty_like(reduced)
    emphasized[0] = reduced[0]
    emphasized[1:] = reduced[1:] - .97 * reduced[:-1]
    mel = librosa.feature.melspectrogram(
        y=emphasized, sr=16000, n_fft=512, hop_length=256,
        window='hamming', center=False, n_mels=128, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=80.0)
    static = dct(rasta_filter(log_mel), type=2, axis=0, norm='ortho')[:n_mfcc]
    static = static[:, start:end].astype(np.float32)
    if static.shape != (n_mfcc, end - start) or end - start < 9:
        raise ValueError('Selected RASTA window is unavailable')
    delta = librosa.feature.delta(static, width=9).astype(np.float32)
    delta2 = librosa.feature.delta(static, width=9, order=2).astype(np.float32)
    return static, delta, delta2


def build(cohort_path: Path, baseline: Path, output: Path, limit: int = 0,
          n_mfcc: int = 30, feature_extractor=None, representation=None) -> None:
    protocol = PROTOCOL if representation is None else representation
    extractor = extract if feature_extractor is None else feature_extractor
    if output.resolve() == baseline.resolve():
        raise ValueError('RASTA features must use a separate directory')
    cohort = json.loads(cohort_path.read_text())
    items = cohort['items'][:limit] if limit else cohort['items']
    k = cohort['frames_per_recording']
    for index, item in enumerate(items, 1):
        relative = Path(item['speaker']) / item['utterance']
        source = json.loads((baseline / relative / 'manifest.json').read_text())
        start, end = (source['selected_window_start_frame'],
                      source['selected_window_end_frame'])
        if (end - start != k or source['target_sr'] != 16000
                or source['trim_start_sample'] != item['trim_start_sample']
                or source['trim_end_sample'] != item['trim_end_sample']):
            raise ValueError(f'Baseline protocol mismatch: {relative}')
        if source['n_mfcc'] != n_mfcc:
            raise ValueError('Baseline MFCC count differs from requested RASTA count')
        expected = {**source, 'representation': {**protocol, 'n_mfcc': n_mfcc}}
        directory = output / relative
        manifest = directory / 'manifest.json'
        done = manifest.exists() and json.loads(manifest.read_text()) == expected
        if done:
            for mic in ('mic1', 'mic2'):
                p = directory / f'{mic}.npz'
                if not p.exists():
                    done = False
                    break
                with np.load(p) as z:
                    if any(z[key].shape != (n_mfcc, k) or not np.isfinite(z[key]).all()
                           for key in ('mfcc', 'delta', 'delta_delta')):
                        done = False
        if not done:
            directory.mkdir(parents=True, exist_ok=True)
            for mic in ('mic1', 'mic2'):
                signal, sr = sf.read(source['inputs'][mic], dtype='float64')
                if sr != 48000 or signal.ndim != 1:
                    raise ValueError(f'Invalid audio: {source["inputs"][mic]}')
                cropped = soft_crop(signal, source['trim_start_sample'],
                                    source['trim_end_sample'], sr, source['fade_ms'])
                reduced = decimate(cropped, q=3, n=8, ftype='iir', zero_phase=True)
                static, delta, delta2 = extractor(reduced, start, end, n_mfcc)
                temporary = directory / f'{mic}.tmp.npz'
                np.savez_compressed(temporary, mfcc=static,
                                    delta=delta, delta_delta=delta2)
                temporary.replace(directory / f'{mic}.npz')
            temporary_manifest = directory / 'manifest.tmp.json'
            temporary_manifest.write_text(json.dumps(expected, indent=2) + '\n')
            temporary_manifest.replace(manifest)
        if index % 100 == 0 or index == len(items):
            print(f"{protocol['name']} features: {index}/{len(items)} paired recordings", flush=True)
    if representation is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'protocol.json').write_text(json.dumps({
            **protocol, 'n_mfcc': n_mfcc, 'cohort': str(cohort_path),
            'pairs_processed': len(items)}, indent=2) + '\n')
        return
    omega, response = freqz(B, A, worN=4096, fs=1/.016)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.json').write_text(json.dumps({
        **protocol, 'n_mfcc': n_mfcc, 'cohort': str(cohort_path), 'pairs_processed': len(items),
        'modulation_peak_hz': float(omega[np.argmax(np.abs(response))]),
        'coefficient_note': 'Canonical coefficients retained at a 16 ms hop; '
                            'physical modulation response differs from a 10 ms hop.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=Path('output/vctk16_cohort_120.json'))
    parser.add_argument('--baseline', type=Path, default=Path('output/vctk16_corrected_features'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--n-mfcc', type=int, choices=(30, 40), default=30)
    args = parser.parse_args()
    build(args.cohort, args.baseline, args.output, args.limit, args.n_mfcc)
