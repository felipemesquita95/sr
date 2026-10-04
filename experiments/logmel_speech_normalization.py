"""Speech-only per-band log-mel CMVN, before DCT; no corpus statistics."""
import librosa
import numpy as np
from scipy.fft import dct

SILERO = dict(threshold=.5, min_speech_duration_ms=250,
              min_silence_duration_ms=100, speech_pad_ms=30)
PROTOCOL = dict(name='speech-logmel-CMVN-before-DCT', version=1,
                domain='power_to_db mel spectrogram, before DCT',
                normalization_scope='full trimmed utterance speech frames only',
                stats='independent mean/std for each of 128 log-mel bands',
                speech_frame_rule='frame center inside Silero speech interval',
                no_speech_fallback='all utterance frames, warned and counted',
                std_floor=1e-8, silero_version='6.2.3', silero=SILERO,
                hop_ms=16, frame_ms=32, n_mels=128, n_mfcc=30,
                dct_type=2, dct_norm='ortho', top_db=80, center=False,
                window='hamming', pre_emphasis=.97, delta_width_frames=9,
                delta_scope='selected_window_only', global_zscore=False)

def log_mel_spectrum(reduced, rate):
    emphasized = np.empty_like(reduced)
    emphasized[0] = reduced[0]
    emphasized[1:] = reduced[1:] - .97 * reduced[:-1]
    mel = librosa.feature.melspectrogram(y=emphasized, sr=rate,
        n_fft=round(.032*rate), hop_length=round(.016*rate),
        window='hamming', center=False, n_mels=128, power=2.)
    return librosa.power_to_db(mel, ref=1., amin=1e-10, top_db=80.)

def speech_mask(segments, frames, rate):
    centers = np.arange(frames)*round(.016*rate) + round(.032*rate)//2
    mask = np.zeros(frames, dtype=bool)
    for segment in segments:
        mask |= (centers >= segment['start']) & (centers < segment['end'])
    return mask

def normalize_log_mel(matrix, mask):
    if matrix.ndim != 2 or mask.shape != (matrix.shape[1],):
        raise ValueError('Invalid speech mask shape')
    if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError('Invalid log-mel input')
    reference = matrix[:, mask] if mask.any() else matrix
    mean = reference.mean(axis=1, keepdims=True)
    std = np.maximum(reference.std(axis=1, keepdims=True), 1e-8)
    return (matrix-mean)/std

def derivatives(static):
    if static.shape[1] < 9:
        raise ValueError('At least nine contiguous frames required')
    return (static.astype(np.float32),
            librosa.feature.delta(static, width=9).astype(np.float32),
            librosa.feature.delta(static, width=9, order=2).astype(np.float32))

def speech_segments(signal, rate, model):
    import torch
    from silero_vad import get_speech_timestamps
    return get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(signal, dtype=np.float32)),
                                 model, sampling_rate=rate, **SILERO)

def make_extractor(model):
    def extract(reduced, start, end, n_mfcc=30):
        logmel = log_mel_spectrum(reduced, 16000)
        mask = speech_mask(speech_segments(reduced, 16000, model), logmel.shape[1], 16000)
        if not mask.any():
            print('WARNING: Silero found no speech; using full utterance CMVN statistics', flush=True)
        normalized = normalize_log_mel(logmel, mask)
        static = dct(normalized, type=2, axis=0, norm='ortho')[:n_mfcc, start:end]
        if static.shape != (n_mfcc, end-start):
            raise ValueError('Selected window missing')
        return derivatives(static)
    return extract
