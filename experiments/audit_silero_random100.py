#!/usr/bin/env python3
"""Auditoria CPU do VAD em 100 BRSD e 100 pares VCTK, sem alterar treinos."""
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from math import gcd
import html

import numpy as np
import soundfile as sf
import torch
from scipy.signal import decimate, resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps

ROOT = Path('/home/lsmsqt/Documents/sr')
STORAGE = Path('/media/lsmsqt/HDD/sr_project')
OUT = STORAGE / 'auditoria_silero_100_20260928'
VCTK = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
BRSD = Path('/media/lsmsqt/HDD/datasets/brsd/utterances')
RATE = 16000
PARAMS = dict(threshold=.5, min_speech_duration_ms=250,
              min_silence_duration_ms=100, speech_pad_ms=30)


def reduce(signal, rate):
    if rate == RATE:
        return signal.copy()
    if rate % RATE == 0:
        return decimate(signal, rate // RATE, n=8, ftype='iir', zero_phase=True)
    factor = gcd(rate, RATE)
    return resample_poly(signal, RATE // factor, rate // factor)


def atomic_json(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in
            zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def process(path, dataset, key, channel, model):
    folder = OUT / dataset / key / channel
    folder.mkdir(parents=True, exist_ok=True)
    raw, source_rate = sf.read(path, dtype='float64')
    channels = 1 if raw.ndim == 1 else raw.shape[1]
    mono = raw if raw.ndim == 1 else raw.mean(axis=1)
    audio = reduce(mono, source_rate)
    assert audio.ndim == 1 and len(audio) and np.isfinite(audio).all()
    speech = get_speech_timestamps(torch.from_numpy(np.asarray(audio, dtype=np.float32)),
                                   model, sampling_rate=RATE, **PARAMS)
    mask = np.zeros(len(audio), dtype=bool)
    for region in speech:
        start, end = int(region['start']), int(region['end'])
        assert 0 <= start < end <= len(audio)
        mask[start:end] = True
    nonspeech = intervals(~mask)
    # These WAVs are listening previews, not contiguous training examples.
    for name, values in [('original_16k.wav', audio),
                         ('atividade_preview.wav', audio[mask]),
                         ('nao_atividade_preview.wav', audio[~mask])]:
        if len(values):
            sf.write(folder / name, values.astype(np.float32), RATE, subtype='FLOAT')
    np.savez_compressed(folder / 'mascara.npz', speech_mask=mask, sample_rate=RATE)
    records = []
    for label, regions in [('atividade', intervals(mask)), ('nao_atividade', nonspeech)]:
        offset = 0
        for index, (start, end) in enumerate(regions):
            records.append(dict(condition=label, segment=index, start_sample=start,
                                end_sample=end, start_seconds=start / RATE,
                                end_seconds=end / RATE,
                                preview_start_sample=offset, preview_end_sample=offset + end - start))
            offset += end - start
    with (folder / 'segmentos.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    info = dict(dataset=dataset, key=key, channel=channel, input=str(path),
                source_rate=source_rate, source_channels=channels,
                sample_rate=RATE, samples=len(audio), duration_seconds=len(audio) / RATE,
                activity_seconds=int(mask.sum()) / RATE,
                non_activity_seconds=int((~mask).sum()) / RATE,
                activity_fraction=float(mask.mean()), speech_segments=len(intervals(mask)),
                non_speech_segments=len(nonspeech),
                longest_activity_seconds=max((b-a for a, b in intervals(mask)), default=0) / RATE,
                longest_non_activity_seconds=max((b-a for a, b in nonspeech), default=0) / RATE,
                folder=str(folder.relative_to(OUT)), silero=PARAMS,
                scope='full source recording; no extra RMS trim, fade, pre-emphasis or feature selection',
                preview='separate original intervals concatenated in time order for listening only',
                segments=records)
    assert np.isclose(info['duration_seconds'], info['activity_seconds'] + info['non_activity_seconds'])
    atomic_json(folder / 'audit.json', info)
    return info, mask, audio


def svg(audio, mask):
    n = len(audio)
    points = np.linspace(0, n-1, min(n, 900), dtype=int)
    scale = max(float(np.max(np.abs(audio))), 1e-8)
    trace = ' '.join(f'{i*900/(len(points)-1):.1f},{45-audio[p]/scale*39:.1f}'
                     for i, p in enumerate(points))
    spans = ''.join(f'<rect x="{a/n*900:.2f}" y="0" width="{(b-a)/n*900:.2f}" height="90" fill="#b8e5d1"/>'
                    for a, b in intervals(mask))
    return f'<svg viewBox="0 0 900 90" role="img" aria-label="Verde: atividade Silero; fundo claro: não atividade">{spans}<polyline points="{trace}" fill="none" stroke="#213d58" stroke-width=".8"/></svg>'


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    brsd = [BRSD / f'{number}.wav' for number in range(1, 401)]
    # The full paired source inventory avoids 40k slow metadata round-trips on NTFS.
    with (ROOT / 'output/vctk16_trim_lengths.csv').open() as stream:
        catalogue = list(csv.DictReader(stream))
    pairs = sorted((VCTK / row['speaker'] / f'{row["speaker"]}_{row["utterance"]}_mic1.flac',
                    VCTK / row['speaker'] / f'{row["speaker"]}_{row["utterance"]}_mic2.flac')
                   for row in catalogue)
    assert len(brsd) == 400 and len(pairs) >= 100
    chosen_brsd = random.Random(42).sample(brsd, 100)
    chosen_pairs = random.Random(42).sample(pairs, 100)
    assert all(p.is_file() for p in chosen_brsd)
    assert all(a.is_file() and b.is_file() for a,b in chosen_pairs)
    atomic_json(OUT / 'selection.json', dict(seed=42, sampling='uniform without replacement; independent RNG per corpus',
                universe_brsd=len(brsd), universe_vctk_pairs=len(pairs),
                inventory=str(ROOT / 'output/vctk16_trim_lengths.csv'),
                brsd=[str(p) for p in chosen_brsd], vctk=[[str(a), str(b)] for a,b in chosen_pairs]))
    atomic_json(OUT / 'protocol.json', dict(silero_version='6.2.3', silero=PARAMS, seed=42,
                sample_rate=RATE, brsd_files=100, vctk_pairs=100, tracks=300,
                vad_device='cpu', extra_trim=False, pre_emphasis=False,
                resampling='integer: decimate n8 IIR zero_phase; noninteger: resample_poly',
                source_vctk='wav48_silence_trimmed; supplied corpus already trimmed by its authors',
                interpretation='non activity is the complement of padded Silero speech intervals; not guaranteed silence',
                purpose='audio/VAD audit, not MFCC extraction or a training run'))
    model = load_silero_vad().cpu().eval()
    infos, cards, pair_rows = [], [], []
    jobs = [('brsd', path.stem, [('audio', path)]) for path in chosen_brsd]
    jobs += [('vctk', a.name.removesuffix('_mic1.flac'), [('mic1', a), ('mic2', b)]) for a,b in chosen_pairs]
    for count, (dataset, key, tracks) in enumerate(jobs, 1):
        masks = []
        for channel, path in tracks:
            info, mask, audio = process(path, dataset, key, channel, model)
            infos.append(info)
            masks.append(mask)
            links = []
            for label, name, duration in [('Original', 'original_16k.wav', info['duration_seconds']),
                         ('Atividade', 'atividade_preview.wav', info['activity_seconds']),
                         ('Não atividade', 'nao_atividade_preview.wav', info['non_activity_seconds'])]:
                player = (f'<audio controls preload="none" src="{info["folder"]}/{name}"></audio>'
                          if duration > 0 else '<span>Nenhum trecho detectado</span>')
                links.append(f'<div><b>{label} ({duration:.2f} s)</b>{player}</div>')
            cards.append(f'<article data-group="{dataset}/{channel}"><h2>{dataset.upper()} · {html.escape(key)} · {channel}</h2>'
                         f'<p>Atividade: {info["activity_fraction"]*100:.1f}% · {info["speech_segments"]} intervalos de fala · '
                         f'<a href="{info["folder"]}/segmentos.csv">Intervalos CSV</a></p>{svg(audio,mask)}'
                         f'<div class="players">{"".join(links)}</div></article>')
        if dataset == 'vctk':
            assert masks[0].shape == masks[1].shape, f'Pair length mismatch: {key}'
            pair_rows.append(dict(key=key, agreement_fraction=float((masks[0] == masks[1]).mean()),
                    both_activity_seconds=int((masks[0] & masks[1]).sum()) / RATE,
                    both_non_activity_seconds=int((~masks[0] & ~masks[1]).sum()) / RATE,
                    disagreement_seconds=int((masks[0] != masks[1]).sum()) / RATE))
        atomic_json(OUT / 'progress.json', dict(status='running', completed_units=count,
                    total_units=200, completed_tracks=len(infos), total_tracks=300,
                    updated_at=datetime.now(timezone.utc).isoformat()))
        print(f'{count}/200 ({len(infos)}/300 trilhas): {dataset}/{key}', flush=True)
    columns = [key for key in infos[0] if key not in ('segments', 'silero')]
    with (OUT / 'resumo_amostras.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(infos)
    with (OUT / 'comparacao_pares.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    groups = {}
    for group in ('brsd/audio', 'vctk/mic1', 'vctk/mic2'):
        rows = [i for i in infos if f'{i["dataset"]}/{i["channel"]}' == group]
        assert len(rows) == 100
        groups[group] = dict(count=len(rows), duration_seconds=sum(r['duration_seconds'] for r in rows),
                            activity_seconds=sum(r['activity_seconds'] for r in rows),
                            non_activity_seconds=sum(r['non_activity_seconds'] for r in rows),
                            zero_activity=sum(r['activity_seconds'] == 0 for r in rows),
                            zero_non_activity=sum(r['non_activity_seconds'] == 0 for r in rows),
                            activity_run_at_least_1_792=sum(r['longest_activity_seconds'] >= 1.792 for r in rows),
                            non_activity_run_at_least_1_792=sum(r['longest_non_activity_seconds'] >= 1.792 for r in rows))
    atomic_json(OUT / 'summary.json', dict(groups=groups, completed_at=datetime.now(timezone.utc).isoformat(),
                mean_pair_agreement=float(np.mean([r['agreement_fraction'] for r in pair_rows]))))
    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Silero · 100 amostras por corpus</title>
<style>body{max-width:1080px;margin:32px auto;padding:0 20px;font-family:system-ui;color:#18344a;background:#f4f6f8}article{background:white;border-radius:12px;padding:20px;margin:20px 0}h2{font-size:18px}svg{width:100%;background:#fff0dc}.players{display:flex;gap:20px;flex-wrap:wrap}.players div{flex:1;min-width:250px}audio{display:block;width:100%;margin-top:8px}select{font-size:16px;padding:10px}pre{white-space:pre-wrap}</style>
<h1>Silero: atividade e não atividade</h1><p>100 arquivos BRSD e 100 pares VCTK (300 trilhas). Sorteio uniforme de todos os pares disponíveis, semente 42. Áudio completo convertido a 16 kHz, sem novo trim por RMS. Verde: atividade; fundo claro: não atividade. Silero aplicado independentemente a cada microfone.</p>
<p>Os players de atividade e não atividade concatenam os intervalos apenas para escuta. Cada intervalo mantém seus limites no CSV; essas prévias não são exemplos contínuos de treino. Não atividade não significa silêncio comprovado. O VCTK de origem já é silence_trimmed.</p>
<p><a href="resumo_amostras.csv">Resumo das 300 trilhas</a> · <a href="comparacao_pares.csv">Concordância mic1/mic2</a> · <a href="selection.json">Sorteio</a> · <a href="protocol.json">Protocolo</a></p>
<select id="filter"><option value="all">Todas as trilhas</option><option value="brsd/audio">BRSD</option><option value="vctk/mic1">VCTK mic1</option><option value="vctk/mic2">VCTK mic2</option></select>'''
    page += ''.join(cards) + '''<script>document.getElementById('filter').onchange=function(){document.querySelectorAll('article').forEach(a=>a.hidden=this.value!=='all'&&a.dataset.group!==this.value)}</script></html>'''
    (OUT / 'index.html').write_text(page)
    atomic_json(OUT / 'progress.json', dict(status='complete', completed_units=200,
                total_units=200, completed_tracks=300, total_tracks=300,
                updated_at=datetime.now(timezone.utc).isoformat()))
    print('COMPLETE', OUT, flush=True)


if __name__ == '__main__':
    main()
