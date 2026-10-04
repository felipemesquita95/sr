#!/usr/bin/env python3
"""Availability tables and plots from segment counts, without feature extraction."""
import argparse
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/sr-frame-availability-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

N_VALUES=[0,10,20,30,40,50,75,100,125,150,175,200,250,300,400,500,600,750,
          1000,1250,1500,1750,2000,2500,3000,4000,5000]
M_VALUES=[1,5,10,20]
CONDITIONS=['atividade','nao_atividade','conjunta']
PERCENTILES=[0,5,10,25,50,75,90,95,100]
P_NAMES=['minimo','P5','P10','P25','mediana','P75','P90','P95','maximo']
LABELS={'atividade':'Atividade','nao_atividade':'Não atividade','conjunta':'Condição conjunta'}


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def quantiles(values):
    return dict(zip(P_NAMES,[float(x) for x in np.percentile(values,PERCENTILES)])) if len(values) else dict.fromkeys(P_NAMES,0)


def counts_at_threshold(records,speakers,n,condition,continuous=False):
    field='_longest_segment_frames' if continuous else '_frames'
    def value(r):
        a,b=r['activity'+field],r['non_activity'+field]
        return a if condition=='atividade' else b if condition=='nao_atividade' else min(a,b)
    selected=[r for r in records if value(r)>=n]
    by=Counter(r['speaker'] for r in selected)
    return selected,{s:by[s] for s in speakers}


def group_summary(group,records,base_count,base_speakers,continuous=False):
    rows,distributions=[],[]
    for condition in CONDITIONS:
        last={m:(float('inf'),float('inf')) for m in M_VALUES}
        for n in N_VALUES:
            selected,by=counts_at_threshold(records,base_speakers,n,condition,continuous)
            for speaker,count in by.items():
                distributions.append(dict(group=group,condition=condition,N=n,speaker=speaker,eligible_recordings=count))
            for m in M_VALUES:
                valid=[count for count in by.values() if count>=m]
                s_count,r_count=len(valid),sum(valid)
                assert s_count<=last[m][0] and r_count<=last[m][1]
                last[m]=(s_count,r_count)
                row=dict(group=group,unit='pair' if group=='VCTK paired' else 'recording',condition=condition,N=n,minimum_recordings_per_speaker=m,
                    base_recordings=base_count,base_speakers=len(base_speakers),processed_recordings=len(records),
                    eligible_recordings=len(selected),speakers_with_any=sum(c>0 for c in by.values()),
                    valid_speakers=s_count,valid_recordings=r_count,
                    eligible_base_percent=100*len(selected)/base_count,
                    retained_base_percent=100*r_count/base_count,
                    retained_speakers_percent=100*s_count/len(base_speakers),
                    mean_recordings_per_valid_speaker=float(np.mean(valid)) if valid else 0)
                row.update({'all_speakers_'+k:v for k,v in quantiles(list(by.values())).items()})
                row.update({'valid_speakers_'+k:v for k,v in quantiles(valid).items()})
                rows.append(row)
    return rows,distributions


def knees(rows):
    """Mark large observed drops; never chooses an input size."""
    results=[]
    for g in sorted({r['group'] for r in rows}):
        for c in CONDITIONS:
            for m in M_VALUES:
                selected=[r for r in rows if r['group']==g and r['condition']==c and r['minimum_recordings_per_speaker']==m and r['N']>0]
                candidates=[]
                for a,b in zip(selected,selected[1:]):
                    drop=a['valid_speakers']-b['valid_speakers']
                    if drop>0:
                        candidates.append(dict(group=g,condition=c,minimum_recordings_per_speaker=m,
                            from_N=a['N'],to_N=b['N'],speakers_before=a['valid_speakers'],speakers_after=b['valid_speakers'],
                            speakers_lost=drop,base_speakers_lost_percent=100*drop/a['base_speakers'],
                            slope_per_log2_N=drop/np.log2(b['N']/a['N']),
                            recordings_before=a['valid_recordings'],recordings_after=b['valid_recordings']))
                top=sorted(candidates,key=lambda r:r['slope_per_log2_N'],reverse=True)[:3]
                results.extend(top)
    return results


def plot_group(group,rows,knee_rows,out,continuous=False):
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,constrained_layout=True)
    colors=['#1f77b4','#16804a','#d97917','#9854b0']
    reference_m=5 if group=='BRSD' else 10
    unit='pares' if group=='VCTK paired' else 'gravações'
    for col,c in enumerate(CONDITIONS):
        for m,color in zip(M_VALUES,colors):
            subset=[r for r in rows if r['group']==group and r['condition']==c and r['minimum_recordings_per_speaker']==m and r['N']>0]
            x=[r['N'] for r in subset]
            for row,field in enumerate(['valid_speakers','valid_recordings','retained_base_percent']):
                axes[row,col].plot(x,[r[field] for r in subset],color=color,marker='.',markersize=4,label=f'≥ {m} {unit}/locutor')
                axes[row,col].set_xscale('log')
                axes[row,col].grid(True,alpha=.25)
                axes[row,col].set_ylim(bottom=0)
                if row==2:
                    axes[row,col].set_ylim(0,100)
        bands=[k for k in knee_rows if k['group']==group and k['condition']==c and k['minimum_recordings_per_speaker']==reference_m]
        for k in bands:
            for row in range(3):
                axes[row,col].axvspan(k['from_N'],k['to_N'],color='#ffbd59',alpha=.15)
        axes[0,col].set_title(LABELS[c])
        axes[2,col].set_xlabel('N mínimo de frames por gravação (escala log)')
        axes[2,col].set_xticks([10,50,100,200,500,1000,2000,5000],labels=['10','50','100','200','500','1000','2000','5000'],rotation=35)
        axes[0,col].legend(fontsize=8)
    for ax,label in zip(axes[:,0],['Locutores válidos','Gravações válidas' if group!='VCTK paired' else 'Pares válidos','Base retida (%)']):
        ax.set_ylabel(label)
    title=f'{group} · frames em '+('um único segmento contínuo' if continuous else 'segmentos contínuos separados (soma)')
    fig.suptitle(title+f'\nFaixas laranja: maiores quedas de locutores por duplicação de N, para ≥ {reference_m} {unit}/locutor; não são recomendações.',fontsize=13)
    stem=group.replace(' ','_')+('_continuous' if continuous else '')
    fig.savefig(out/f'{stem}.png',dpi=150)
    fig.savefig(out/f'{stem}.svg')
    plt.close(fig)
    return f'{stem}.png'


def table(rows,columns):
    result='<table><thead><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in columns)+'</tr></thead><tbody>'
    for r in rows:
        result+='<tr>'+''.join(f'<td>{r[k]:.2f}</td>' if isinstance(r[k],float) else f'<td>{html.escape(str(r[k]))}</td>' for k in columns)+'</tr>'
    return result+'</tbody></table>'


def md_table(rows,columns):
    def fmt(value):
        return f'{value:.2f}' if isinstance(value,float) else str(value)
    return ('| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+
            ''.join('| '+' | '.join(fmt(r[k]) for k in columns)+' |\n' for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    args=p.parse_args()
    root=args.results.resolve()
    inventory=json.loads((root/'inventory.json').read_text())
    rows=[json.loads(line) for line in (root/'segments_counts.jsonl').read_text().splitlines()]
    assert len({r['path'] for r in rows})==len(rows)
    failures=[r for r in rows if 'error' in r]
    measured=[r for r in rows if 'error' not in r]
    complete=len(rows)==len(inventory) and not failures
    inventory_paths={i['path'] for i in inventory}
    assert {r['path'] for r in rows}.issubset(inventory_paths)
    if complete:
        assert {r['path'] for r in measured}==inventory_paths
    out=root/f'report_{"complete" if complete else "partial"}_{len(measured)}'
    out.mkdir(exist_ok=True)
    scalar=[{k:v for k,v in r.items() if not k.endswith('_segments_samples')} for r in measured]
    save_csv(out/'counts_per_recording.csv',scalar)
    if failures:
        save_csv(out/'errors.csv',failures)
    summaries,distributions,continuous_rows,percentiles=[],[],[],[]
    groups=sorted({i['group'] for i in inventory})
    bases={}
    records_by={}
    for g in groups:
        base=[i for i in inventory if i['group']==g]
        records=[r for r in measured if r['group']==g]
        if not records:
            continue
        bases[g]=(len(base),sorted({i['speaker'] for i in base}))
        records_by[g]=records
        summary,dist=group_summary(g,records,*bases[g])
        summaries.extend(summary)
        distributions.extend(dist)
        continuous_rows.extend(group_summary(g,records,*bases[g],continuous=True)[0])
        for condition,key in [('atividade','activity_frames'),('nao_atividade','non_activity_frames')]:
            percentiles.append(dict(group=g,condition=condition,recordings=len(records),**quantiles([r[key] for r in records])))
    pairs={}
    for r in measured:
        if r['group'].startswith('VCTK '):
            pairs.setdefault((r['speaker'],r['utterance']),{})[r['group']]=r
    inventory_pairs={}
    for i in inventory:
        if i['group'].startswith('VCTK '):
            inventory_pairs.setdefault((i['speaker'],i['utterance']),set()).add(i['group'])
    base_pairs=[key for key,mics in inventory_pairs.items() if len(mics)==2]
    paired=[]
    pair_detail=[]
    for key,mics in sorted(pairs.items()):
        if len(mics)!=2:
            continue
        a,b=mics['VCTK mic1'],mics['VCTK mic2']
        row=dict(group='VCTK paired',speaker=key[0],utterance=key[1])
        detail=dict(speaker=key[0],utterance=key[1],mic1_path=a['path'],mic2_path=b['path'])
        for condition in ('activity','non_activity'):
            for suffix in ('_frames','_longest_segment_frames'):
                field=condition+suffix
                row[field]=min(a[field],b[field])
                detail['mic1_'+field]=a[field]
                detail['mic2_'+field]=b[field]
        paired.append(row)
        pair_detail.append(detail)
    if paired:
        if complete:
            assert len(paired)==len(base_pairs)
        save_csv(out/'paired_per_recording.csv',pair_detail)
        g='VCTK paired'
        bases[g]=(len(base_pairs),sorted({key[0] for key in base_pairs}))
        records_by[g]=paired
        summary,dist=group_summary(g,paired,*bases[g])
        summaries.extend(summary)
        distributions.extend(dist)
        continuous_rows.extend(group_summary(g,paired,*bases[g],continuous=True)[0])
    save_csv(out/'availability.csv',summaries)
    save_csv(out/'recordings_per_speaker.csv',distributions)
    save_csv(out/'availability_single_continuous_segment.csv',continuous_rows)
    save_csv(out/'frame_percentiles.csv',percentiles)
    knee_rows=knees(summaries)
    if knee_rows:
        save_csv(out/'possible_knee_intervals.csv',knee_rows)
    continuous_knees=knees(continuous_rows)
    plots=[(g,plot_group(g,summaries,knee_rows,out)) for g in records_by]
    continuous_plots=[(g,plot_group(g,continuous_rows,continuous_knees,out,True)) for g in records_by]
    baseline=[r for r in summaries if r['N']==0 and r['condition']=='atividade' and r['minimum_recordings_per_speaker']==1]
    overview=[dict(group=r['group'],unit=r['unit'],source_recordings=r['base_recordings'],source_speakers=r['base_speakers'],processed=r['processed_recordings']) for r in baseline]
    save_csv(out/'base_overview.csv',overview)
    meta=dict(status='complete' if complete else 'partial',source_recordings=len(inventory),processed=len(measured),
              errors=len(failures),groups=overview,N_values=N_VALUES,minimum_recordings_per_speaker=M_VALUES,
              percentiles_method='numpy linear interpolation',paired_rule='same speaker and utterance; each microphone independently satisfies N, not necessarily same timestamps',
              knee_method='three largest finite drops of valid speaker counts per log2 N interval for each minimum recordings criterion; exploratory only',
              decision=None,training=False,features=False)
    (out/'report_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    required={50,100,200,300,500,750,1000,1500,2000,2500,3000,4000,5000}
    assert required.issubset(N_VALUES)
    for r in measured:
        assert r['activity_frames']==sum(max(0,1+(b-a-512)//256) for a,b in r['activity_segments_samples'])
        assert r['non_activity_frames']==sum(max(0,1+(b-a-512)//256) for a,b in r['non_activity_segments_samples'])
        assert np.isclose(r['activity_seconds']+r['non_activity_seconds'],r['duration_seconds'])
        partition=sorted(r['activity_segments_samples']+r['non_activity_segments_samples'])
        assert partition[0][0]==0 and partition[-1][1]==r['samples_16k']
        assert all(0<=a<b<=r['samples_16k'] for a,b in partition)
        assert all(a[1]==b[0] for a,b in zip(partition,partition[1:]))
    (out/'verification.json').write_text(json.dumps(dict(counting_rechecked=len(measured),required_N_present=True,
        monotonic_survival_verified=True,full_duration_partition_verified=True,
        exact_source_inventory_coverage_verified=complete,paired_inventory_coverage_verified=complete,
        gap_and_overlap_free_sample_partition_verified=True,complete=complete),indent=2)+'\n')
    status='ANÁLISE COMPLETA' if complete else 'RESULTADOS PARCIAIS — não usar como análise final da base'
    page=f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Disponibilidade de frames · Silero 200 ms</title>
<style>body{{font-family:system-ui;color:#18344a;background:#f4f6f8;max-width:1500px;margin:25px auto;padding:0 20px}}section{{background:white;padding:20px;border-radius:12px;margin:18px 0}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border:1px solid #d4dee7;text-align:right}}th:first-child,td:first-child{{text-align:left}}a{{color:#185a93}}.warning{{color:#994511;font-weight:bold}}p{{line-height:1.5}}</style>
<h1>Disponibilidade real de frames · Silero 6.2.3 · silêncio mínimo 200 ms</h1><p class="warning">{status}</p>
<p>Áudio a 16 kHz; entrada 0,50; saída 0,35; fala mínima 250 ms; margem 30 ms. Janela 512 amostras, salto 256. Cada segmento contínuo tem seu próprio início de enquadramento; somente janelas completas são contadas. Os frames de segmentos separados são somados sem criar frames nas emendas. Sem pré-ênfase, novo recorte RMS, padding, duplicação ou treinamento. VCTK de origem: wav48_silence_trimmed, já recortado pelos autores.</p>
<section><h2>Universos e cobertura</h2>{table(overview,list(overview[0]))}<p>A porcentagem retida usa todos os arquivos de origem do respectivo grupo como denominador. “Gravações elegíveis” satisfazem N; “gravações válidas” também pertencem a locutores com pelo menos M gravações elegíveis. Pares contam como uma unidade, embora tenham dois arquivos.</p></section>
<section><h2>Percentis de frames por gravação</h2>{table(percentiles,list(percentiles[0]))}<p>Percentis com interpolação linear, incluindo gravações com zero frames.</p></section>
<p><a href="counts_per_recording.csv">Contagens por gravação</a> · <a href="availability.csv">Disponibilidade: todos os N e M</a> · <a href="recordings_per_speaker.csv">Distribuição exata por locutor (inclui zeros)</a> · <a href="frame_percentiles.csv">Percentis</a> · <a href="possible_knee_intervals.csv">Intervalos de quedas mais fortes</a> · <a href="../inventory.csv">Inventário exato</a> · <a href="../protocol.json">Protocolo</a></p>'''
    for g,file in plots:
        page+=f'<section><h2>{g}</h2><img src="{file}" alt="Curvas de disponibilidade de {g}"></section>'
    page+='<section><h2>Como ler as faixas laranja</h2><p>São intervalos observados com maiores quedas de locutores por duplicação de N, no critério de 5 gravações para BRSD e 10 para VCTK. Os CSVs também registram os critérios 1, 5, 10 e 20. Servem para inspecionar possíveis joelhos, sem estabelecer um N final. A escala logarítmica torna visíveis os valores pequenos e grandes; N=0 é a referência sem exigência de frames nas tabelas.</p><p>BRSD tem apenas cinco gravações por locutor; exigir 10 ou 20 naturalmente elimina todos. A análise pareada exige que os dois microfones satisfaçam N de forma independente; não exige atividade nos mesmos instantes.</p></section>'
    page+='<section><h2>Disponibilidade em um único segmento contínuo</h2><p>Controle adicional: se a futura entrada precisar de N frames consecutivos, use estas curvas. A contagem principal soma frames reais de vários segmentos e não garante uma entrada contínua de N frames.</p><a href="availability_single_continuous_segment.csv">Tabela do critério de segmento único</a></section>'
    for g,file in continuous_plots:
        page+=f'<section><h2>{g} · segmento único</h2><img src="{file}" alt="Disponibilidade contínua de {g}"></section>'
    page+='</html>'
    (out/'index.html').write_text(page)
    md=f'# Disponibilidade de frames — {status}\n\nSilero 6.2.3, entrada 0,50, saída 0,35, min speech 250 ms, min silence 200 ms, pad 30 ms, 16 kHz.\n\nContagem por segmento: max(0,1+floor((L-512)/256)). Sem janelas parciais ou emendas.\n\nRelatório visual: [index.html](index.html). Gráficos exportados em PNG e SVG.\n\nTabelas:\n\n- [Contagens por gravação](counts_per_recording.csv)\n- [Disponibilidade: todos os N e M](availability.csv)\n- [Distribuição exata por locutor](recordings_per_speaker.csv), incluindo zeros\n- [Percentis](frame_percentiles.csv)\n- [Pares mic1/mic2](paired_per_recording.csv)\n- [Disponibilidade em um segmento único](availability_single_continuous_segment.csv)\n- [Intervalos exploratórios de joelho](possible_knee_intervals.csv)\n\nNenhum tamanho foi escolhido.\n\nUniverso:\n\n'+md_table(overview,list(overview[0]))+'\n\nPercentis:\n\n'+md_table(percentiles,list(percentiles[0]))+'\n'
    md+='\nMínimo de gravações por locutor (M): 1, 5, 10 e 20. O BRSD tem apenas cinco gravações por locutor.\n\n“Gravações elegíveis” satisfazem N; “gravações válidas” também pertencem a locutores com pelo menos M gravações elegíveis. A porcentagem retida considera todos os arquivos de origem do respectivo grupo.\n\nA análise pareada exige N em ambos os microfones independentemente; não exige alinhamento temporal das decisões.\n'
    for g,file in plots:
        md+=f'\n## {g}\n\n![Curvas de disponibilidade de {g}]({file})\n'
    md+='\n## Intervalos de quedas fortes\n\nAs faixas laranja indicam as maiores quedas de locutores por duplicação de N: M=5 no BRSD e M=10 no VCTK. São marcadores exploratórios, sem definir o tamanho final.\n\n'
    representative=[r for r in knee_rows if r['minimum_recordings_per_speaker']==(5 if r['group']=='BRSD' else 10)]
    if representative:
        md+=md_table(representative,['group','condition','minimum_recordings_per_speaker','from_N','to_N','speakers_before','speakers_after','speakers_lost'])
    md+='\n## Frames consecutivos em um único segmento\n\nSomar frames de segmentos diferentes não garante uma entrada contínua de N frames. Estas curvas mostram a disponibilidade se N precisar caber em um segmento só.\n'
    for g,file in continuous_plots:
        md+=f'\n### {g}\n\n![Disponibilidade em segmento único de {g}]({file})\n'
    (out/'README.md').write_text(md)
    print('REPORT '+str(out),flush=True)
    print(json.dumps(meta,ensure_ascii=False),flush=True)
    print(json.dumps(percentiles,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
