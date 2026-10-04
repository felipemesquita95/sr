import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from build_brsd_silero_conditions import partitions, contiguous_speech_runs, speech_features, PROTOCOL, reduce_audio, mono_audio
from run_normalization_remaining import plan, command

def test_brsd_partitions_match_original_reference_and_keep_test_text_out():
    items = [dict(speaker=str(s), utterance=str(u)) for s in range(1, 81) for u in range(1, 6)]
    for fold, split in partitions(items).items():
        assert [len(split[k]) for k in ('train', 'validation', 'test')] == [240, 80, 80]
        assert not set(split['train']) & set(split['validation'])
        assert not set(split['test']) & set(split['train']+split['validation'])
        assert {items[i]['utterance'] for i in split['test']} == {fold}
        assert fold not in {items[i]['utterance'] for i in split['train']+split['validation']}
        rng = np.random.default_rng(42)
        for speaker in range(1, 81):
            expected = int(rng.choice([u for u in range(1, 6) if str(u) != fold]))
            assert (speaker-1)*5+expected-1 in split['validation']

def test_deltas_do_not_cross_gaps():
    static = np.r_[np.zeros(12), np.ones(12)*100][None, :]
    _, delta, delta2 = speech_features(static, [(0, 12), (12, 24)])
    np.testing.assert_allclose(delta, 0, atol=1e-10)
    np.testing.assert_allclose(delta2, 0, atol=1e-10)
    assert contiguous_speech_runs(np.r_[np.ones(8, bool), False, np.ones(9, bool)]) == [(9, 18)]

def test_remaining_plan_preserves_corpus_processing_and_excludes_xvector():
    assert PROTOCOL['window'] == 'hamming' and PROTOCOL['center'] is False
    assert PROTOCOL['target_sr'] == 16000 and PROTOCOL['n_mfcc'] == 30
    for stage in plan():
        if 'architecture' not in stage:
            continue
        args = command(stage)
        assert stage['architecture'] in ('cnn', 'temporal_cnn')
        assert '--no-global-zscore' in args or stage['condition'] == 'zscore'
        assert args[args.index('--mode')+1] == 'dynamic'

def test_rate_reduction_is_exactly_one_decimate_without_extra_filter():
    from scipy.signal import decimate
    signal = np.random.default_rng(42).normal(size=48000)
    np.testing.assert_array_equal(reduce_audio(signal, 48000, 16000),
        decimate(signal, q=3, n=8, ftype='iir', zero_phase=True))
    np.testing.assert_array_equal(reduce_audio(signal, 48000, 8000),
        decimate(signal, q=6, n=8, ftype='iir', zero_phase=True))

def test_noninteger_native_rate_uses_single_polyphase_resampling():
    from scipy.signal import resample_poly
    signal = np.random.default_rng(42).normal(size=44100)
    np.testing.assert_array_equal(reduce_audio(signal, 44100, 8000),
                                  resample_poly(signal, 80, 441))

def test_stereo_is_mixed_like_original_loader_and_mono_is_preserved():
    import librosa
    signal = np.random.default_rng(42).normal(size=(16000, 2))
    np.testing.assert_array_equal(mono_audio(signal), librosa.to_mono(signal.T))
    original = signal[:, 0]
    assert mono_audio(original) is original
