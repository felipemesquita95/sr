#!/usr/bin/env python3
"""Figures from the actual Silero VCTK manifests and feature arrays."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import soundfile as sf
from scipy.signal import decimate, welch

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('/media/lsmsqt/HDD/sr_project')
OUT = ROOT / 'docs/figuras/protocolo_silero_vctk'
RUNS = {
    40: STORE / 'vctk40_activity_nonactivity_20260929',
    80: STORE / 'vctk80_activity_20260929_112427',
    100: STORE / 'vctk100_activity_20260929',
}
SPEAKER, UTTERANCE = 'p225', '013'
NAVY, TEAL, ORANGE, PURPLE = '#17334c', '#087d86', '#d27737', '#70598c'


def item_for(frames: int) -> tuple[int, dict]:
    cohort = json.loads((RUNS[frames] / 'cohort.json').read_text())
    for index, item in enumerate(cohort['items']):
        if (item['speaker'], item['utterance']) == (SPEAKER, UTTERANCE):
            return index, item
    raise ValueError('Illustrative recording absent from cohort')


def audit_rows() -> dict:
    path = ROOT / 'output/disponibilidade_frames_vad200_20260928_2310/segments_counts.jsonl'
    rows = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get('speaker'), row.get('utterance')) == (SPEAKER, UTTERANCE):
                rows[row['group'].split()[-1]] = row
    assert set(rows) == {'mic1', 'mic2'}
    return rows


def audio(item: dict) -> tuple[dict, dict]:
    raw, reduced = {}, {}
    for mic in ('mic1', 'mic2'):
        values, rate = sf.read(item['paths'][mic], dtype='float64')
        assert rate == 48000 and values.ndim == 1
        raw[mic] = values
        reduced[mic] = decimate(values, q=3, n=8, ftype='iir', zero_phase=True)
        assert len(reduced[mic]) == item['source_samples_16k']
    return raw, reduced


def fmt_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=175, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def audio_chain(raw, reduced, item):
    signal = reduced['mic1']
    segment = item['windows']['activity']
    start, stop = segment['segment_start'], segment['segment_end']
    selected = signal[start:stop]
    pre = np.empty_like(selected)
    pre[0] = selected[0]
    pre[1:] = selected[1:] - .97 * selected[:-1]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True,
                             constrained_layout=True)
    for ax, values, rate, label, color in [
        (axes[0], raw['mic1'], 48000, 'FLAC de entrada · mic1 · 48 kHz', NAVY),
        (axes[1], signal, 16000, 'Após decimação antialias · 16 kHz', TEAL),
        (axes[2], pre, 16000, 'Pré-ênfase 0,97 no segmento comum de fala', ORANGE),
    ]:
        stride = max(1, len(values) // 6000)
        offset = start/16000 if values is pre else 0
        ax.plot(offset + np.arange(0, len(values), stride) / rate, values[::stride],
                color=color, lw=.55)
        ax.set_ylabel('Amplitude')
        ax.set_title(label, loc='left', fontsize=10, fontweight='bold')
        fmt_axis(ax)
    axes[-1].set_xlabel('Tempo no arquivo (s)')
    axes[-1].set_xlim(0, len(signal) / 16000)
    save(fig, '01_audio_cadeia.png')


def spectrum(raw, reduced, item):
    original = raw['mic1']
    transformed = reduced['mic1']
    segment = item['windows']['activity']
    selected = transformed[segment['segment_start']:segment['segment_end']]
    pre = np.empty_like(selected)
    pre[0] = selected[0]
    pre[1:] = selected[1:] - .97*selected[:-1]
    f48, p48 = welch(original, fs=48000, nperseg=4096)
    f16, p16 = welch(transformed, fs=16000, nperseg=2048)
    fb, pb = welch(selected, fs=16000, nperseg=2048)
    fa, pa = welch(pre, fs=16000, nperseg=2048)
    db = lambda value, reference: 10*np.log10(np.maximum(value, 1e-18)/reference)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), constrained_layout=True)
    axes[0].plot(f48/1000, db(p48, p48.max()), color=NAVY, lw=1,
                 label='Arquivo de entrada, 48 kHz')
    axes[0].plot(f16/1000, db(p16, p48.max()), color=TEAL, lw=1,
                 label='Após decimação, 16 kHz')
    axes[0].axvline(8, color=ORANGE, ls='--', lw=1, label='Nyquist em 16 kHz')
    axes[0].set(xlim=(0,24), ylim=(-110,10), ylabel='PSD relativa (dB)',
                title='Espectro do arquivo completo')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(fb/1000, db(pb, pb.max()), color=TEAL, lw=1,
                 label='Segmento comum, antes da pré-ênfase')
    axes[1].plot(fa/1000, db(pa, pb.max()), color=ORANGE, lw=1,
                 label='Após pré-ênfase 0,97')
    axes[1].set(xlim=(0,8), ylim=(-95,18), xlabel='Frequência (kHz)',
                ylabel='PSD relativa (dB)', title='Segmento usado para extrair MFCCs')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.title.set_fontweight('bold')
        fmt_axis(ax)
    save(fig, '01b_espectro.png')


def vad_selection(reduced, audit, items):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.5), sharex=True,
                             gridspec_kw={'height_ratios': [1.4, 1.4, 1.15]},
                             constrained_layout=True)
    end = len(reduced['mic1']) / 16000
    for ax, mic in zip(axes[:2], ('mic1', 'mic2')):
        signal = reduced[mic]
        stride = max(1, len(signal) // 5500)
        ax.plot(np.arange(0, len(signal), stride) / 16000, signal[::stride],
                color=NAVY, lw=.6)
        for a, b in audit[mic]['activity_segments_samples']:
            ax.axvspan(a/16000, b/16000, color=TEAL, alpha=.18)
        ax.set_title(f'{mic}: Silero VAD · verde = fala detectada',
                     loc='left', fontsize=10, fontweight='bold')
        ax.set_ylabel('Amplitude')
        fmt_axis(ax)
    ax = axes[2]
    bands = [('40 não fala', items[40]['windows']['non_activity'], ORANGE),
             ('40 fala', items[40]['windows']['activity'], '#2899a0'),
             ('80 fala', items[80]['windows']['activity'], TEAL),
             ('100 fala', items[100]['windows']['activity'], PURPLE)]
    for index, (name, window, color) in enumerate(bands):
        a, b = window['window_start']/16000, window['window_end']/16000
        ax.broken_barh([(a, b-a)], (index-.28, .56), facecolors=color)
        ax.text(b+.04, index, f'{name}: {a:.3f}–{b:.3f} s',
                va='center', fontsize=8, color=NAVY)
    ax.set(ylim=(-.6, 3.6), yticks=[], xlim=(0, end), xlabel='Tempo no mesmo arquivo (s)')
    ax.set_title('Janelas centrais selecionadas · instantes idênticos nos dois microfones',
                 loc='left', fontsize=10, fontweight='bold')
    fmt_axis(ax)
    save(fig, '02_vad_janelas.png')


def features(item, index):
    import librosa
    from scipy.fft import dct
    from build_vctk16_rasta_features import rasta_filter

    segment = item['windows']['activity']
    raw, rate = sf.read(item['paths']['mic1'], dtype='float64')
    assert rate == 48000
    reduced = decimate(raw, q=3, n=8, ftype='iir', zero_phase=True)
    signal = np.asarray(reduced[segment['segment_start']:segment['segment_end']],
                        dtype=np.float32)
    pre = np.empty_like(signal)
    pre[0] = signal[0]
    pre[1:] = signal[1:] - .97*signal[:-1]
    mel = librosa.feature.melspectrogram(y=pre, sr=16000, n_fft=512,
                                         hop_length=256, window='hamming',
                                         center=False, n_mels=128, power=2.)
    logmel = librosa.power_to_db(mel, ref=1., amin=1e-10, top_db=80.)
    first = segment['first_frame']
    expected = dct(logmel, type=2, axis=0, norm='ortho')[:40, first:first+100]
    original = np.load(RUNS[100] / 'features/baseline_activity_mic1.npy', mmap_mode='r')
    tensor = np.array(original[index])
    np.testing.assert_allclose(tensor[0], expected, rtol=1e-4, atol=2e-4)
    rasta = np.load(RUNS[100] / 'features/rasta_activity_mic1.npy', mmap_mode='r')
    rasta_static = np.array(rasta[index, 0])
    expected_rasta = dct(rasta_filter(logmel), type=2, axis=0, norm='ortho')[:40, first:first+100]
    np.testing.assert_allclose(rasta_static, expected_rasta, rtol=1e-4, atol=2e-4)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), constrained_layout=True)
    panels = [
        (logmel[:, first:first+100], 'Log-mel · 128 bandas'),
        (tensor[0], 'MFCCs estáticos · 40 coeficientes'),
        (tensor[1], 'Δ · variação temporal'),
        (tensor[2], 'ΔΔ · segunda variação'),
    ]
    for ax, (matrix, title) in zip(axes.flat, panels):
        ax.imshow(matrix, origin='lower', aspect='auto', cmap='magma')
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
        ax.set_xlabel('Quadro da janela de 100')
        ax.set_ylabel('Banda' if matrix.shape[0] == 128 else 'Coeficiente')
    save(fig, '03_mel_mfcc_deltas.png')

    # The same example after each isolated condition.
    static = tensor[0]
    mean, std = static.mean(axis=1, keepdims=True), np.maximum(static.std(axis=1, keepdims=True), 1e-8)
    stats_path = RUNS[100] / 'models/1_40_zscore_temporal_cnn_activity_mic1/normalization.npz'
    with np.load(stats_path) as stats:
        global_mean, global_std = stats['mean'], stats['std']
    assert global_mean.shape == global_std.shape == (1, 120, 1)
    states = [
        ('MFCC bruto', static),
        ('Z-score · treino mic1/fold1', (static-global_mean[0, :40])/global_std[0, :40]),
        ('CMN · esta janela', static-mean),
        ('CMVN · esta janela', (static-mean)/std),
        ('RASTA · antes da DCT', rasta_static),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.8), constrained_layout=True)
    for ax, (title, matrix) in zip(axes.flat, states):
        bound = np.percentile(np.abs(matrix), 98)
        ax.imshow(matrix, origin='lower', aspect='auto', cmap='RdBu_r',
                  vmin=-bound, vmax=bound)
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
        ax.set_xlabel('Quadro')
        ax.set_ylabel('MFCC')
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05, .85, 'Mesma gravação e janela.\nEscala de cor independente\nem cada painel.',
                       transform=axes.flat[-1].transAxes, va='top',
                       fontsize=10, color=NAVY)
    save(fig, '05_normalizacoes_exemplo.png')


def flow():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis('off')
    boxes = [
        (.06, 2.28, 1.3, .66, 'Áudio\n48 kHz', NAVY),
        (1.63, 2.28, 1.32, .66, '16 kHz\n+ VAD', TEAL),
        (3.22, 2.28, 1.4, .66, 'Segmento\ncomum', TEAL),
        (4.88, 2.28, 1.18, .66, 'Log-mel', NAVY),
        (6.32, 2.28, 1.23, .66, 'DCT / MFCC', NAVY),
        (7.82, 2.28, 1.35, .66, 'Janela\ncentral', TEAL),
        (9.43, 2.28, 1.36, .66, 'Δ + ΔΔ', NAVY),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.06',
                                    facecolor=color, edgecolor='none'))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                fontsize=9.5, color='white', fontweight='bold')
    for x0, x1 in [(1.36, 1.63), (2.95, 3.22), (4.62, 4.88),
                   (6.06, 6.32), (7.55, 7.82), (9.17, 9.43)]:
        ax.annotate('', xy=(x1-.03, 2.61), xytext=(x0+.02, 2.61),
                    arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.5))
    ax.annotate('RASTA: filtra o log-mel no segmento contínuo\nantes da DCT e da janela final',
                xy=(5.5, 2.25), xytext=(4.5, 1.25), fontsize=10,
                color=PURPLE, ha='center',
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.2))
    ax.annotate('Z-score: média/desvio do treino de origem\nCMN: média da própria janela\nCMVN: média/desvio da própria janela',
                xy=(10.1, 2.25), xytext=(8.7, .52), fontsize=10,
                color=ORANGE, ha='center',
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))
    save(fig, '04_pontos_normalizacao.png')


def results():
    with (RUNS[40] / 'resultados_resumo.csv').open() as stream:
        forty = list(csv.DictReader(stream))
    with (RUNS[100] / 'resultados_resumo.csv').open() as stream:
        hundred = list(csv.DictReader(stream))
    with (ROOT / 'output/vctk100_xvector_silero_20260929/results/resultados_resumo.csv').open() as stream:
        xvector = list(csv.DictReader(stream))

    # Fixed network/feature/normalization to isolate the class comparison.
    classes = [('activity', 'activity', 'Fala → fala'),
               ('activity', 'non_activity', 'Fala → não fala'),
               ('non_activity', 'activity', 'Não fala → fala'),
               ('non_activity', 'non_activity', 'Não fala → não fala')]
    dirs = [('mic1', 'mic1'), ('mic1', 'mic2'), ('mic2', 'mic1'), ('mic2', 'mic2')]
    matrix = np.empty((4, 4))
    for i, (train_class, test_class, _) in enumerate(classes):
        for j, (source, target) in enumerate(dirs):
            row = [r for r in forty if r['architecture']=='temporal_cnn'
                   and r['normalization']=='zscore' and r['n_mfcc']=='40'
                   and r['train_class']==train_class and r['test_class']==test_class
                   and r['train_mic']==source and r['test_mic']==target]
            assert len(row)==1
            matrix[i,j] = 100*float(row[0]['accuracy_mean'])
    fig, ax = plt.subplots(figsize=(9.7, 3.5), constrained_layout=True)
    ax.imshow(matrix, cmap='YlGnBu', vmin=0, vmax=100)
    ax.set_xticks(range(4), ['1→1','1→2','2→1','2→2'])
    ax.set_yticks(range(4), [x[2] for x in classes])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{matrix[i,j]:.1f}%', ha='center', va='center',
                    fontsize=10, color='white' if matrix[i,j]>65 else NAVY,
                    fontweight='bold')
    ax.set_title('40 quadros · CNN temporal · 40 MFCCs · z-score',
                 loc='left', fontweight='bold', color=NAVY)
    save(fig, '06_atividade_resultados.png')

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True,
                             constrained_layout=True)
    directions = [('mic1', 'mic2', 'Treino mic1 → teste mic2'),
                  ('mic2', 'mic1', 'Treino mic2 → teste mic1')]
    conditions = [('temporal_cnn', 'zscore', hundred, 'Temporal / Z'),
                  ('temporal_cnn', 'cmn', hundred, 'Temporal / CMN'),
                  ('temporal_cnn', 'cmvn', hundred, 'Temporal / CMVN'),
                  ('temporal_cnn', 'rasta', hundred, 'Temporal / RASTA'),
                  ('xvector', 'zscore', xvector, 'X-vector / Z'),
                  ('xvector', 'cmn', xvector, 'X-vector / CMN')]
    for ax, (source, target, title) in zip(axes, directions):
        values=[]
        for arch,norm,rows,_ in conditions:
            row=[r for r in rows if r['architecture']==arch and r['normalization']==norm
                 and r['n_mfcc']=='40' and r['train_mic']==source and r['test_mic']==target
                 and r['train_class']==r['test_class']=='activity']
            assert len(row)==1
            values.append(100*float(row[0]['accuracy_mean']))
        ax.barh(range(len(values)), values,
                color=[NAVY,TEAL,'#78aab0',PURPLE,ORANGE,'#e7a16d'])
        ax.set_yticks(range(len(values)), [r[3] for r in conditions], fontsize=8)
        ax.set_xlim(0,100)
        ax.invert_yaxis()
        ax.set_title(title, loc='left', fontsize=9, fontweight='bold')
        ax.set_xlabel('Acurácia média (%)')
        fmt_axis(ax)
    save(fig, '07_normalizacoes_cross_100.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    items = {frames: item_for(frames)[1] for frames in RUNS}
    index, selected = item_for(100)
    assert all(item['paths'] == selected['paths'] for item in items.values())
    audit = audit_rows()
    raw, reduced = audio(selected)
    audio_chain(raw, reduced, selected)
    spectrum(raw, reduced, selected)
    vad_selection(reduced, audit, items)
    features(selected, index)
    flow()
    results()
    manifest = {
        'example': f'{SPEAKER}_{UTTERANCE}', 'cohort_index_100': index,
        'source_files': selected['paths'], 'audio_samples_16k': len(reduced['mic1']),
        'silero_segments': audit, 'windows': {str(k): v['windows'] for k,v in items.items()},
        'figures': sorted(p.name for p in OUT.glob('*.png')),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
