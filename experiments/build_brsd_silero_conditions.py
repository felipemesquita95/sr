#!/usr/bin/env python3
"""BRSD speech-only matched features, 16 kHz, 30 MFCCs, Silero 6.2.3."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.fft import dct
from scipy.signal import decimate, resample_poly
from silero_vad import load_silero_vad
from logmel_speech_normalization import (SILERO, log_mel_spectrum,
    speech_segments, speech_mask, normalize_log_mel, derivatives)
from build_vctk16_rasta_features import rasta_filter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from process_vctk_pair_corrected import common_bounds, soft_crop
from sr.preprocessing.window_selection import best_common_rms_window

RATE = 16000
N_MFCC = 30
CONDITIONS = ('zscore', 'cmn', 'cmvn', 'rasta', 'cmvn_logmel')
PROTOCOL = dict(dataset='brsd', version=3, source_sr=48000, target_sr=RATE,
    n_mfcc=N_MFCC, n_mels=128, frame_ms=32, hop_ms=16, pre_emphasis=.97,
    filter='decimate internal Chebyshev I order 8, zero phase; no separate Butterworth',
    resampling='integer rate: decimate; non-integer native rate: resample_poly with its own antialias filter',
    window='hamming', center=False, training_features='MFCC30+delta+delta_delta, last executed VCTK protocol',
    trim=dict(threshold_db=-30, min_active_frames=5, start_margin_ms=100, end_margin_ms=250, fade_ms=8),
    silero_version='6.2.3', silero=SILERO, speech_frame_rule='center in speech interval',
    speech_scope='full softly trimmed recording', no_speech_fallback='stop and report recording',
    short_segments='require a continuous Silero speech run of at least 111 frames for selection',
    derivatives='recomputed only inside the selected continuous 111-frame window; no concatenation',
    alignment='same fixed 111-frame continuous RMS window in all conditions, inside Silero speech',
    cmvn_logmel='per-band mean/std on Silero speech frames of trimmed recording before DCT, floor 1e-8',
    cmn_cmvn='per retained aligned MFCC window in training loader',
    rasta='full original log-mel before speech-frame selection and DCT',
    global_zscore='only zscore condition; estimated on train fold')

def contiguous_speech_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1),
                                           np.flatnonzero(edges == -1)) if b-a >= 9]

def reduce_audio(signal, source_rate, target_rate=RATE):
    if source_rate == target_rate:
        return signal.copy()
    if source_rate % target_rate == 0:
        return decimate(signal, q=source_rate//target_rate, n=8,
                        ftype='iir', zero_phase=True)
    # BRSD files 106–110 are 44.1 kHz: no integer decimation factor exists.
    from math import gcd
    divisor = gcd(source_rate, target_rate)
    return resample_poly(signal, target_rate//divisor, source_rate//divisor)

def mono_audio(signal):
    """Match the original librosa.load mono behavior; preserve mono inputs."""
    if signal.ndim == 1:
        return signal
    if signal.ndim == 2:
        return signal.mean(axis=1)
    raise ValueError('Invalid audio dimensions')

def speech_features(static, runs):
    parts = [derivatives(static[:, a:b]) for a, b in runs]
    if not parts:
        raise ValueError('No speech segment with at least 9 frames')
    return tuple(np.concatenate([p[i] for p in parts], axis=1) for i in range(3))

def partitions(items):
    # Reproduce BRSD reference random held-out validation reading per speaker.
    results = {}
    for fold in range(5):
        rng = np.random.default_rng(42)
        result = dict(train=[], validation=[], test=[])
        for speaker in range(1, 81):
            remaining = [u for u in range(1, 6) if u != fold+1]
            validation = int(rng.choice(remaining))
            for utterance in range(1, 6):
                index = (speaker-1)*5 + utterance-1
                name = 'test' if utterance == fold+1 else 'validation' if utterance == validation else 'train'
                result[name].append(index)
        results[str(fold+1)] = result
    return results

def build(audio, output, limit=0, n_mfcc=30, reference=None):
    protocol = dict(PROTOCOL)
    if n_mfcc != 30:
        protocol.update(n_mfcc=n_mfcc,
                        training_features=f'MFCC{n_mfcc}+delta+delta_delta, last executed VCTK protocol')
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output/'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()).get('protocol') != protocol:
        raise ValueError('BRSD feature protocol mismatch')
    torch.set_num_threads(1)
    model = load_silero_vad() if reference is None else None
    items, counts = [], []
    total = min(limit, 400) if limit else 400
    for number in range(1, total+1):
        speaker, utterance = str((number-1)//5+1), str((number-1)%5+1)
        item = dict(speaker=speaker, utterance=utterance, fold_group=int(utterance)-1)
        items.append(item)
        reference_audit = (json.loads((reference/'audit'/f'{number}.json').read_text())
                           if reference else None)
        audit_file = output/'audit'/f'{number}.json'
        paths = [output/condition/speaker/utterance/'audio.npz' for condition in CONDITIONS]
        if audit_file.exists() and all(path.exists() for path in paths):
            audit = json.loads(audit_file.read_text())
            if audit.get('protocol') != protocol:
                raise ValueError('Cached recording protocol mismatch')
            counts.append(audit['retained_frames'])
            continue
        signal, sr = sf.read(audio/f'{number}.wav', dtype='float64')
        source_channels = 1 if signal.ndim == 1 else signal.shape[1]
        signal = mono_audio(signal)
        trim_start, trim_end, activity = common_bounds(signal, signal, sr,
            threshold_db=-30, margin_ms=100, end_margin_ms=250, min_active_frames=5)
        cropped = soft_crop(signal, trim_start, trim_end, sr, fade_ms=8)
        reduced = reduce_audio(cropped, sr)
        segments = (reference_audit['silero_segments'] if reference_audit else
                    speech_segments(reduced, RATE, model))
        logmel = log_mel_spectrum(reduced, RATE)
        mask = speech_mask(segments, logmel.shape[1], RATE)
        runs = contiguous_speech_runs(mask)
        candidates = []
        for a, b in runs:
            if b-a < 111:
                continue
            run_audio = reduced[a*256:(b-1)*256+512]
            first, last, area = best_common_rms_window(run_audio, run_audio, 111, 512, 256)
            candidates.append((area, a+first, a+last))
        if not candidates:
            raise ValueError(f'BRSD recording {number}: no continuous Silero speech window of 111 frames; inspect before proceeding')
        area, first, last = max(candidates, key=lambda row: row[0])
        if reference_audit:
            actual = (trim_start, trim_end, first, last, sr, len(mask))
            expected = tuple(reference_audit[key] for key in (
                'trim_start_sample', 'trim_end_sample', 'selected_window_start_frame',
                'selected_window_end_frame', 'source_sr', 'original_frames'))
            if actual != expected:
                raise ValueError(f'BRSD recording {number}: window differs from 30 MFCC reference')
        baseline = dct(logmel, type=2, axis=0, norm='ortho')[:n_mfcc]
        rasta = dct(rasta_filter(logmel), type=2, axis=0, norm='ortho')[:n_mfcc]
        cmvn_logmel = dct(normalize_log_mel(logmel, mask), type=2, axis=0, norm='ortho')[:n_mfcc]
        extracted = {name: derivatives(matrix[:, first:last]) for name, matrix in
                     [('baseline', baseline), ('rasta', rasta), ('cmvn_logmel', cmvn_logmel)]}
        for condition, path in zip(CONDITIONS, paths):
            key = condition if condition in ('rasta', 'cmvn_logmel') else 'baseline'
            static, delta, delta2 = extracted[key]
            if not all(np.isfinite(x).all() for x in (static, delta, delta2)):
                raise ValueError('Non-finite BRSD features')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name('audio.tmp.npz')
            np.savez_compressed(temp, mfcc=static, delta=delta, delta_delta=delta2)
            temp.replace(path)
        count = int(extracted['baseline'][0].shape[1])
        counts.append(count)
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = audit_file.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(dict(protocol=protocol, input=str(audio/f'{number}.wav'),
            source_sr=sr, source_channels=source_channels,
            mono_conversion='channel mean, matching original librosa mono load',
            trim_start_sample=trim_start, trim_end_sample=trim_end,
            activity=activity, selected_window_start_frame=first, selected_window_end_frame=last,
            selected_window_area_normalized_rms=area,
            silero_segments=segments, retained_frame_runs=runs,
            retained_frames=count, original_frames=len(mask)), indent=2)+'\n')
        temporary.replace(audit_file)
        print(f'BRSD Silero: {number}/{total}, retained {count} frames', flush=True)
    cohort = dict(dataset='brsd', items=items, speaker_count=len({i['speaker'] for i in items}),
                  pairs_per_speaker=5, frames_per_recording=min(counts),
                  seed=42, microphones=['audio'], protocol=protocol)
    if not limit:
        cohort['partitions'] = partitions(items)
    (output/'cohort.json').write_text(json.dumps(cohort, indent=2)+'\n')
    protocol_file.write_text(json.dumps(dict(protocol=protocol, pairs_processed=total,
        frames_per_recording=min(counts)), indent=2)+'\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', type=Path, default=Path('/media/lsmsqt/HDD/datasets/brsd/utterances'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--n-mfcc', type=int, choices=(30, 40), default=30)
    p.add_argument('--reference', type=Path,
                   help='Reuse Silero segments and verify windows against a previous extraction')
    a = p.parse_args()
    build(a.audio, a.output, a.limit, a.n_mfcc, a.reference)
