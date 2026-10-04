#!/usr/bin/env python3
"""Viewer of original-rate waveform and aligned Silero decisions."""
import html
import json
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf

OUT = Path('/media/lsmsqt/HDD/sr_project/auditoria_silero_100_20260928')


def chart(info):
    fingerprint = hashlib.sha256(json.dumps([info['input'], info['segments']], sort_keys=True).encode()).hexdigest()[:16]
    cached = OUT / info['folder'] / f'original_silero_{fingerprint}.svg'
    if cached.is_file():
        return cached.read_text()
    raw, rate = sf.read(info['input'], dtype='float64')
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    n = len(raw)
    peak = max(float(np.max(np.abs(raw))), 1e-8)
    left, width = 65, 1000
    duration = n / rate
    envelope = []
    borders = np.linspace(0, n, 1001, dtype=int)
    for i, (a, b) in enumerate(zip(borders[:-1], borders[1:])):
        part = raw[a:b]
        if len(part):
            envelope.append(f'M{left+i:.1f},{155-part.max()/peak*105:.2f}V{155-part.min()/peak*105:.2f}')
    ticks = ''.join(f'<line x1="{left+i*width/5}" y1="40" x2="{left+i*width/5}" y2="410" stroke="#dbe2ea"/>'
                    f'<text x="{left+i*width/5}" y="438" text-anchor="middle">{duration*i/5:.2f}</text>'
                    for i in range(6))
    decisions = []
    for interval in info['segments']:
        active = interval['condition'] == 'atividade'
        y = 315 if active else 395
        color = '#14804a' if active else '#d72c3e'
        a = left + interval['start_seconds'] / duration * width
        b = left + interval['end_seconds'] / duration * width
        decisions.append(f'<path d="M{a:.3f},{y}H{b:.3f}" stroke="{color}" stroke-width="4"/>'
                         f'<rect x="{a:.3f}" y="295" width="{max(0,b-a):.3f}" height="120" fill="{color}" opacity=".07"/>')
    transitions = sorted(set(i['end_seconds'] for i in info['segments'] if i['end_seconds'] < duration - 1/16000))
    vertical = ''.join(f'<line x1="{left+t/duration*width:.3f}" x2="{left+t/duration*width:.3f}" y1="315" y2="395" stroke="#64748b" stroke-width=".6"/>' for t in transitions)
    result = (f'<svg viewBox="0 0 1115 470" role="img" aria-label="Áudio original e decisão Silero alinhados no tempo">'
            f'<rect x="65" y="40" width="1000" height="230" fill="#f8fafc"/>{ticks}'
            f'<text x="65" y="25" font-weight="bold">Original · {rate} Hz · amplitude relativa ao pico</text>'
            '<line x1="65" y1="155" x2="1065" y2="155" stroke="#b8c4d0"/>'
            f'<path d="{"".join(envelope)}" stroke="#203c59" stroke-width="1"/>'
            '<text x="65" y="290" font-weight="bold">Silero · decisão após conversão a 16 kHz</text>'
            '<text x="48" y="321" text-anchor="end" fill="#14804a">+1</text>'
            '<text x="48" y="401" text-anchor="end" fill="#d72c3e">−1</text>'
            '<line x1="65" y1="355" x2="1065" y2="355" stroke="#aebcca" stroke-dasharray="4 4"/>'
            f'{"".join(decisions)}{vertical}<text x="565" y="463" text-anchor="middle">Tempo no arquivo original (s)</text></svg>')
    cached.write_text(result)
    return result


def main():
    infos = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/*/*/audit.json'))]
    if not infos:
        print('No completed samples yet', flush=True)
        return
    buckets = [[i for i in infos if f'{i["dataset"]}/{i["channel"]}' == group]
               for group in ('brsd/audio', 'vctk/mic1', 'vctk/mic2')]
    infos = [bucket[index] for index in range(max(map(len, buckets)))
             for bucket in buckets if index < len(bucket)]
    items = [dict(group=f'{i["dataset"]}/{i["channel"]}',
                  title=f'{i["dataset"].upper()} · {i["key"]} · {i["channel"]}',
                  subtitle=f'Atividade: {i["activity_seconds"]:.2f} s · Não atividade: {i["non_activity_seconds"]:.2f} s · {i["activity_fraction"]*100:.1f}% de atividade',
                  chart=chart(i), audio=f'{i["folder"]}/original_16k.wav',
                  audit=f'{i["folder"]}/audit.json') for i in infos]
    data = json.dumps(items, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Silero · original e atividade</title>
<style>body{font-family:system-ui;background:#f1f5f9;color:#18344a;margin:0}main{max-width:1200px;margin:20px auto;padding:0 24px}.panel{background:white;padding:24px;border-radius:16px}h1{font-size:26px;margin-bottom:8px}h2{font-size:22px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}button,select{padding:10px 15px;font-size:16px;border:1px solid #b9c6d2;border-radius:7px;background:white;color:#18344a}button{cursor:pointer}svg{width:100%;max-height:57vh}svg text{font-size:15px;font-family:system-ui}audio{width:100%;max-width:550px}p{line-height:1.5}.green{color:#14804a}.red{color:#d72c3e}.muted{color:#596e80;font-size:14px}</style>
<main><h1>Silero: original e atividade no mesmo tempo</h1><div class="controls"><button id="previous">← Anterior</button><button id="pause">Pausar</button><button id="next">Próxima →</button><select id="filter"><option value="all">BRSD + VCTK</option><option value="brsd/audio">BRSD</option><option value="vctk/mic1">VCTK mic1</option><option value="vctk/mic2">VCTK mic2</option></select><span id="count"></span><span>Troca automática: 2 s</span></div>
<section class="panel"><h2 id="title"></h2><p id="subtitle"></p><div id="chart"></div><p><b class="green">+1: atividade</b> · <b class="red">−1: não atividade</b></p><audio id="audio" controls preload="none"></audio><p><a id="audit">Auditoria da amostra</a> · <a href="index.html">Lista e áudios separados</a></p></section>
<p class="muted">A forma de onda é do arquivo original na taxa nativa. O player reproduz a versão a 16 kHz. Sem novo recorte por RMS ou pré-ênfase. Não atividade é o complemento da detecção Silero, incluindo a margem de fala de 30 ms; não garante silêncio. VCTK de origem: wav48_silence_trimmed. Pause a troca para escutar.</p>
<script>const items=DATA;let filtered=items,index=0,playing=true;const audio=document.getElementById('audio');function show(){const x=filtered[index];document.getElementById('title').textContent=x.title;document.getElementById('subtitle').textContent=x.subtitle;document.getElementById('chart').innerHTML=x.chart;document.getElementById('count').textContent=`${index+1} / ${filtered.length} trilhas`;audio.pause();audio.src=x.audio;document.getElementById('audit').href=x.audit;}function step(d){index=(index+d+filtered.length)%filtered.length;show()}document.getElementById('previous').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);document.getElementById('pause').onclick=()=>{playing=!playing;document.getElementById('pause').textContent=playing?'Pausar':'Continuar'};document.getElementById('filter').onchange=function(){filtered=items.filter(x=>this.value==='all'||x.group===this.value);index=0;show()};audio.onplay=()=>{playing=false;document.getElementById('pause').textContent='Continuar'};setInterval(()=>{if(playing)step(1)},2000);show();</script></main></html>'''.replace('DATA', data)
    if len(infos) < 300:
        page = page.replace('<h1>Silero: original e atividade no mesmo tempo</h1>',
            f'<h1>Silero: original e atividade no mesmo tempo</h1><p>Varredura em andamento: {len(infos)}/300 trilhas disponíveis. A página recebe novas amostras automaticamente.</p>')
        page = page.replace('</main>', '<script>setInterval(()=>{if(playing)location.reload()},30000)</script></main>')
    page = page.replace('index=0,playing=true',
                        "index=Math.min(Number(sessionStorage.getItem('sileroIndex')||0),items.length-1),playing=true")
    page = page.replace('function show(){const x=',
                        "function show(){sessionStorage.setItem('sileroIndex',index);const x=")
    page = page.replace('setInterval(()=>{if(playing)step(1)},2000);show();',
        "document.querySelectorAll('#filter option').forEach(o=>{o.disabled=o.value!=='all'&&!items.some(x=>x.group===o.value)});setInterval(()=>{if(playing)step(1)},2000);show();")
    temporary = OUT / 'visualizacao.tmp.html'
    temporary.write_text(page)
    temporary.replace(OUT / 'visualizacao.html')
    print(f'Viewer: {len(items)} tracks, {OUT / "visualizacao.html"}', flush=True)


if __name__ == '__main__':
    main()
