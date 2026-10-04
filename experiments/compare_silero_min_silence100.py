#!/usr/bin/env python3
"""Controlled 100-recording audit. No training or feature extraction."""
import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from audit_silero_random100 import reduce, intervals

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = Path('/media/lsmsqt/HDD/sr_project/auditoria_silero_100_20260928/selection.json')
RATE = 16000
CONDITIONS = (100, 200, 300)
FIXED = dict(sampling_rate=RATE, threshold=0.50, neg_threshold=0.35,
             min_speech_duration_ms=250, speech_pad_ms=30,
             max_speech_duration_s=float('inf'), return_seconds=False,
             time_resolution=1, visualize_probs=False, window_size_samples=512,
             min_silence_at_max_speech=98, use_max_poss_sil_at_max_speech=True)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chart(raw, rate, conditions):
    """Original-rate min/max envelope; all rows use the same seconds-to-x map."""
    duration = len(raw) / rate
    left, width = 65, 1100
    peak = max(float(np.max(np.abs(raw))), 1e-8)
    edges = np.linspace(0, len(raw), width + 1, dtype=int)
    envelope = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if b > a:
            envelope.append(f'M{left+i},{110-raw[a:b].max()/peak*55:.3f}V{110-raw[a:b].min()/peak*55:.3f}')
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1220 590" role="img" aria-label="Forma de onda original e decisões 100, 200 e 300 ms no mesmo eixo temporal">',
             '<rect width="1220" height="590" fill="white"/>',
             '<g font-family="sans-serif" font-size="16" fill="#18344a">',
             f'<text x="65" y="26">Forma de onda original · {rate} Hz · amplitude relativa ao pico</text>']
    for i in range(11):
        x = left + width * i / 10
        parts.append(f'<path d="M{x},40V550" stroke="#dbe2ea"/><text x="{x}" y="570" text-anchor="middle">{duration*i/10:.2f}</text>')
    parts.append(f'<path d="{"".join(envelope)}" stroke="#203c59" stroke-width="1"/>')
    for row, ms in enumerate(CONDITIONS):
        top = 200 + row * 115
        parts.append(f'<text x="65" y="{top}">min_silence_duration_ms = {ms}</text>')
        parts.append(f'<text x="50" y="{top+33}" text-anchor="end" fill="#14804a">+1</text><text x="50" y="{top+88}" text-anchor="end" fill="#d72c3e">−1</text>')
        regions = sorted(conditions[ms]['segments'], key=lambda s: s['start_sample'])
        for j, s in enumerate(regions):
            active = s['condition'] == 'atividade'
            color = '#14804a' if active else '#d72c3e'
            y = top + (28 if active else 83)
            a = left + s['start_sample'] / RATE / duration * width
            b = left + s['end_sample'] / RATE / duration * width
            parts.append(f'<rect x="{a:.4f}" y="{top+12}" width="{b-a:.4f}" height="80" fill="{color}" opacity=".08"/><path d="M{a:.4f},{y}H{b:.4f}" stroke="{color}" stroke-width="3"/>')
            if j:
                parts.append(f'<path d="M{a:.4f},{top+28}V{top+83}" stroke="#64748b" stroke-width=".7"/>')
    parts.append('<text x="615" y="588" text-anchor="middle">Tempo no arquivo original (s)</text></g></svg>')
    return ''.join(parts)


PAGE = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Silero · comparação de silêncio mínimo</title>
<style>body{margin:0;font-family:system-ui;color:#18344a;background:#eff3f7}main{max-width:1500px;margin:12px auto;padding:0 18px}h1{font-size:23px;margin:8px 0}h2{font-size:20px;margin:8px 0}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}button{padding:8px 16px;background:white;border:1px solid #a9b9c9;border-radius:6px;font-size:16px;cursor:pointer}.panel{background:white;border-radius:12px;padding:12px;margin-top:10px}.layout{display:grid;grid-template-columns:minmax(0,3fr) minmax(300px,1fr);gap:16px}img{width:100%;height:auto}.stats{display:grid;grid-template-rows:1fr 1fr 1fr;padding-top:14%}.stat{font-size:14px;border-left:4px solid #14804a;padding:8px 12px}.stat h3{margin:0 0 6px}audio{width:100%;height:28px}details{font-size:13px}p{margin:7px 0}.muted{font-size:13px;color:#556b7f}a{color:#185a93}@media(max-width:850px){.layout{grid-template-columns:1fr}.stats{padding:0;grid-template-columns:repeat(3,1fr);grid-template-rows:auto}.stat{font-size:12px}}</style>
<main><h1>Silero 6.2.3 · silêncio mínimo: 100 / 200 / 300 ms</h1>
<div class="controls"><button id="previous">← Anterior</button><button id="next">Próxima →</button><button id="pause">Pausar</button><b id="count"></b><span id="timer"></span></div>
<div class="panel"><h2 id="title"></h2><div class="layout"><img id="chart" alt="Forma de onda original e três decisões Silero no mesmo eixo temporal"><div id="stats" class="stats"></div></div></div>
<p class="muted">16 kHz · entrada 0,50 · saída 0,35 · fala mínima 250 ms · margem 30 ms. +1 verde: atividade; −1 vermelho: não atividade. Sem pré-ênfase ou novo recorte RMS. Áudios separados concatenam os intervalos, em ordem, somente para auditoria. Intervalos vazios produzem WAVs com zero quadros. Forma de onda na taxa original; decisões e áudios a 16 kHz.</p>
<p><a href="resumo_global.csv">Resumo global</a> · <a href="resumo_amostras.csv">Resumo por amostra</a> · <a href="selection.json">Lista e seed</a> · <a href="protocol.json">Protocolo</a> · <a id="audit">Intervalos da amostra</a></p>
<script>const items=DATA, INTERVAL=120000;let index=0,playing=true,deadline=Date.now()+INTERVAL;const el=id=>document.getElementById(id);function reset(){deadline=Date.now()+INTERVAL;tick()}function show(){const s=items[index];el('count').textContent=`Amostra ${index+1}/${items.length}`;el('title').textContent=s.name;el('chart').src=s.folder+'/comparacao.svg';el('audit').href=s.folder+'/audit.json';el('stats').innerHTML=s.conditions.map(c=>`<div class="stat"><h3>${c.min_silence_duration_ms} ms</h3><p>${c.activity_segments} segmentos · ${c.activity_percent.toFixed(2)}% atividade</p><p>Atividade: ${c.activity_seconds.toFixed(3)} s<br>Não atividade: ${c.non_activity_seconds.toFixed(3)} s</p><p>Segmentos de atividade:<br>Média: ${c.mean_segment_seconds.toFixed(3)} s<br>Menor: ${c.min_segment_seconds.toFixed(3)} s<br>Maior: ${c.max_segment_seconds.toFixed(3)} s</p><details><summary>Áudios e intervalos</summary>Atividade<audio controls preload="none" src="${s.folder}/atividade_${c.min_silence_duration_ms}ms.wav"></audio>Não atividade<audio controls preload="none" src="${s.folder}/nao_atividade_${c.min_silence_duration_ms}ms.wav"></audio><a href="${s.folder}/segmentos_${c.min_silence_duration_ms}ms.csv">Intervalos CSV</a></details></div>`).join('');}function step(d){index=(index+d+items.length)%items.length;show();reset()}function tick(){if(playing&&Date.now()>=deadline){step(1);return}let seconds=Math.max(0,Math.ceil((deadline-Date.now())/1000));el('timer').textContent=playing?`Próxima em ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')} · loop de 100 amostras`:'Troca automática pausada'}el('previous').onclick=()=>step(-1);el('next').onclick=()=>step(1);el('pause').onclick=()=>{playing=!playing;el('pause').textContent=playing?'Pausar':'Continuar';reset()};document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1)});show();setInterval(tick,250);tick();window.auditViewer={getState:()=>({index,count:items.length,playing,interval:INTERVAL,deadline}),step};</script></main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['vctk-mic1', 'vctk-mic2', 'brsd'], default='vctk-mic1')
    args = parser.parse_args()
    assert importlib.metadata.version('silero-vad') == '6.2.3'
    selection = json.loads(PREVIOUS.read_text())
    files = selection['brsd'] if args.group == 'brsd' else [pair[0 if args.group == 'vctk-mic1' else 1] for pair in selection['vctk']]
    assert len(files) == len(set(files)) == 100
    assert all(Path(p).is_file() for p in files)
    out = ROOT / 'output' / f'auditoria_min_silence_{args.group}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(exist_ok=False)
    write_json(out/'selection.json', dict(seed=selection['seed'], sampling=selection['sampling'],
        group=args.group, count=100, files=files, reused_selection=str(PREVIOUS),
        source_selection_sha256=hashlib.sha256(PREVIOUS.read_bytes()).hexdigest(),
        universe=selection['universe_brsd' if args.group == 'brsd' else 'universe_vctk_pairs']))
    (out/'selected_files.txt').write_text('\n'.join(files)+'\n')
    protocol = dict(silero_version='6.2.3', variable='min_silence_duration_ms', conditions=list(CONDITIONS),
        fixed={k: ('infinity' if k=='max_speech_duration_s' else v) for k,v in FIXED.items()},
        resampling='same as prior audit: integer decimate n8 IIR zero_phase; otherwise resample_poly',
        pre_emphasis=False, extra_rms_trim=False, features=False, training=False, device='cpu',
        non_activity='exact complement of padded Silero activity intervals',
        automatic_interval_ms=120000, loop=True, output=str(out))
    write_json(out/'protocol.json', protocol)
    torch.set_num_threads(1)
    model = load_silero_vad().cpu().eval()
    rows, items = [], []
    for index, source in enumerate(files, 1):
        raw, rate = sf.read(source, dtype='float64')
        mono = raw if raw.ndim==1 else raw.mean(axis=1)
        audio = np.asarray(reduce(mono, rate), dtype=np.float32)
        assert audio.ndim==1 and len(audio)>0 and np.isfinite(audio).all()
        folder = out / 'amostras' / f'{index:03d}_{Path(source).stem}'
        folder.mkdir(parents=True)
        sf.write(folder/'original_16k.wav', audio, RATE, subtype='FLOAT')
        tensor = torch.from_numpy(audio)
        results, masks = {}, {}
        for ms in CONDITIONS:
            model.reset_states()
            speech = get_speech_timestamps(tensor, model, min_silence_duration_ms=ms, **FIXED)
            mask = np.zeros(len(audio), dtype=bool)
            for s in speech:
                a,b = int(s['start']),int(s['end'])
                assert 0<=a<b<=len(audio)
                mask[a:b] = True
            regions = intervals(mask)
            lengths = np.array([b-a for a,b in regions], dtype=float)/RATE
            stats = dict(sample=index, name=Path(source).name, min_silence_duration_ms=ms,
                activity_segments=len(regions), duration_seconds=len(audio)/RATE,
                activity_seconds=int(mask.sum())/RATE, non_activity_seconds=int((~mask).sum())/RATE,
                activity_percent=float(mask.mean())*100,
                mean_segment_seconds=float(lengths.mean()) if len(lengths) else 0,
                min_segment_seconds=float(lengths.min()) if len(lengths) else 0,
                max_segment_seconds=float(lengths.max()) if len(lengths) else 0)
            records = []
            for label, decision in [('atividade',mask),('nao_atividade',~mask)]:
                sf.write(folder/f'{label}_{ms}ms.wav',audio[decision],RATE,subtype='FLOAT')
                offset=0
                for j,(a,b) in enumerate(intervals(decision)):
                    records.append(dict(condition=label, segment=j, start_sample=a,end_sample=b,
                        start_seconds=a/RATE,end_seconds=b/RATE,preview_start_sample=offset,
                        preview_end_sample=offset+b-a))
                    offset+=b-a
                info = sf.info(folder/f'{label}_{ms}ms.wav')
                assert info.frames==int(decision.sum()) and info.samplerate==RATE
            assert np.isclose(stats['activity_seconds']+stats['non_activity_seconds'],stats['duration_seconds'])
            assert sum(s['end_sample']-s['start_sample'] for s in records)==len(audio)
            write_csv(folder/f'segmentos_{ms}ms.csv', records)
            rows.append(stats)
            results[ms] = dict(**stats, segments=records, silero_segments=speech)
            masks[f'activity_{ms}ms']=mask
        np.savez_compressed(folder/'mascaras.npz',sample_rate=RATE,**masks)
        write_json(folder/'audit.json',dict(source=source, source_rate=rate, source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),conditions=results))
        (folder/'comparacao.svg').write_text(chart(mono,rate,results))
        items.append(dict(name=Path(source).name,folder=str(folder.relative_to(out)),
                          conditions=[{k:v for k,v in results[ms].items() if k not in ('segments','silero_segments')} for ms in CONDITIONS]))
        print(f'{index}/100 · {Path(source).name}',flush=True)
    global_rows=[]
    for ms in CONDITIONS:
        group=[r for r in rows if r['min_silence_duration_ms']==ms]
        global_rows.append(dict(min_silence_duration_ms=ms, recordings=len(group),
            mean_activity_segments=float(np.mean([r['activity_segments'] for r in group])),
            median_activity_segments=float(np.median([r['activity_segments'] for r in group])),
            mean_activity_seconds=float(np.mean([r['activity_seconds'] for r in group])),
            mean_non_activity_seconds=float(np.mean([r['non_activity_seconds'] for r in group])),
            mean_activity_percent=float(np.mean([r['activity_percent'] for r in group]))))
    write_csv(out/'resumo_amostras.csv',rows)
    write_csv(out/'resumo_global.csv',global_rows)
    write_json(out/'resumo_global.json',global_rows)
    write_json(out/'items.json',items)
    data=json.dumps(items,ensure_ascii=False).replace('</','<\\/')
    (out/'index.html').write_text(PAGE.replace('DATA',data))
    write_json(out/'verification.json',dict(status='complete',recordings=100,conditions=3,
        separate_wavs=len(list(out.glob('amostras/*/*ms.wav'))),charts=len(list(out.glob('amostras/*/comparacao.svg'))),
        unique_sources=len(set(files)),wav_frame_counts_verified=True,full_partition_verified=True))
    (out/'README.md').write_text(f'# Auditoria de silêncio mínimo\n\nVisualizador: {out}/index.html\n\nGráficos, máscaras, intervalos e áudios: {out}/amostras/\n\nLista: {out}/selection.json e selected_files.txt\n\nResumo global: {out}/resumo_global.csv\n\nResumo por amostra: {out}/resumo_amostras.csv\n\nSeed: {selection["seed"]}. Seleção reutilizada: {PREVIOUS}. Grupo: {args.group}.\n\nNão atividade é o complemento dos intervalos com margem de fala. Os áudios separados concatenam intervalos somente para escuta; os CSVs registram os limites e offsets. O VCTK fornecido já foi recortado pelos autores; nenhum novo recorte foi feito. WAVs vazios são mantidos com zero quadros.\n')
    print('COMPLETE '+str(out),flush=True)
    print(json.dumps(global_rows,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
