#!/usr/bin/env python3
"""Count continuous-segment frames only; never computes acoustic features."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from audit_silero_random100 import reduce

ROOT=Path(__file__).resolve().parents[1]
BRSD=Path('/media/lsmsqt/HDD/datasets/brsd/utterances')
VCTK=Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
PARAMS=dict(sampling_rate=16000,threshold=.50,neg_threshold=.35,
            min_speech_duration_ms=250,min_silence_duration_ms=200,speech_pad_ms=30)
WINDOW,HOP=512,256
MODEL=None


def json_write(path,value):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def frames(length):
    return max(0,1+(length-WINDOW)//HOP)


def merge_segments(speech,n):
    result=[]
    for s in speech:
        a,b=int(s['start']),int(s['end'])
        assert 0<=a<b<=n
        if result and a<=result[-1][1]:
            result[-1][1]=max(result[-1][1],b)
        else:
            result.append([a,b])
    return result


def complement(active,n):
    result=[]
    start=0
    for a,b in active:
        if a>start:
            result.append([start,a])
        start=b
    if start<n:
        result.append([start,n])
    return result


def summarize(item,active,n,rate,channels,source_frames,elapsed,origin):
    inactive=complement(active,n)
    row=dict(item,source_rate=rate,source_channels=channels,source_samples=source_frames,
             samples_16k=n,duration_seconds=n/16000,elapsed_seconds=elapsed,origin=origin)
    for label,segments in [('activity',active),('non_activity',inactive)]:
        lengths=[b-a for a,b in segments]
        counts=[frames(x) for x in lengths]
        row.update({f'{label}_frames':sum(counts),f'{label}_seconds':sum(lengths)/16000,
                    f'{label}_segments':len(segments),f'{label}_longest_segment_frames':max(counts,default=0),
                    f'{label}_segments_with_frames':sum(c>0 for c in counts),
                    f'{label}_segments_samples':segments})
    assert sum(b-a for a,b in active+inactive)==n
    assert row['activity_frames']+row['non_activity_frames']<=frames(n)
    return row


def initialize():
    global MODEL
    torch.set_num_threads(1)
    MODEL=load_silero_vad().cpu().eval()


def scan(item):
    start=time.monotonic()
    try:
        raw,rate=sf.read(item['path'],dtype='float64')
        channels=1 if raw.ndim==1 else raw.shape[1]
        mono=raw if raw.ndim==1 else raw.mean(axis=1)
        audio=np.asarray(reduce(mono,rate),dtype=np.float32)
        assert len(audio)>0 and np.isfinite(audio).all()
        MODEL.reset_states()
        with torch.inference_mode():
            speech=get_speech_timestamps(torch.from_numpy(audio),MODEL,**PARAMS)
        return summarize(item,merge_segments(speech,len(audio)),len(audio),rate,channels,len(raw),
                         time.monotonic()-start,'new VAD200 inference')
    except Exception as error:
        return dict(item,error=repr(error),elapsed_seconds=time.monotonic()-start)


def inventory():
    items=[]
    brsd=sorted(BRSD.glob('*.wav'),key=lambda p:int(p.stem))
    assert [int(p.stem) for p in brsd]==list(range(1,401))
    for p in brsd:
        number=int(p.stem)
        items.append(dict(group='BRSD',speaker=str((number-1)//5+1),utterance=str((number-1)%5+1),path=str(p)))
    for entry in sorted(os.scandir(VCTK),key=lambda e:e.name):
        if not entry.is_dir():
            continue
        for f in sorted(os.scandir(entry.path),key=lambda e:e.name):
            if not f.name.endswith(('_mic1.flac','_mic2.flac')):
                continue
            speaker,utterance,mic=f.name.removesuffix('.flac').rsplit('_',2)
            assert speaker==entry.name
            items.append(dict(group=f'VCTK {mic}',speaker=speaker,utterance=utterance,path=f.path))
    assert len({i['path'] for i in items})==len(items)
    return items


def reuse_comparison(items,out):
    previous=ROOT/'output/auditoria_min_silence_vctk-mic1_20260928_223636_544140'
    if not (previous/'protocol.json').is_file():
        return []
    protocol=json.loads((previous/'protocol.json').read_text())
    assert protocol['silero_version']=='6.2.3'
    assert all(protocol['fixed'][k]==v for k,v in PARAMS.items() if k!='min_silence_duration_ms')
    assert protocol['pre_emphasis'] is False and protocol['extra_rms_trim'] is False
    lookup={i['path']:i for i in items}
    rows=[]
    for file in sorted(previous.glob('amostras/*/audit.json')):
        audit=json.loads(file.read_text())
        path=audit['source']
        if path not in lookup:
            continue
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=audit['source_sha256']:
            continue
        info=sf.info(path)
        c=audit['conditions']['200']
        row=summarize(lookup[path],merge_segments(c['silero_segments'],round(c['duration_seconds']*16000)),
                      round(c['duration_seconds']*16000),info.samplerate,info.channels,info.frames,0,str(file))
        rows.append(row)
    json_write(out/'reused_cache.json',dict(source=str(previous),verified_files=len(rows),
            source_sha256_verified=True,protocol_matched=True))
    return rows


def validate_counting():
    assert [frames(n) for n in [0,511,512,767,768,1024]]==[0,0,1,1,2,3]
    assert frames(400)+frames(400)==0 and frames(800)==2
    assert frames(512)+frames(512)==2 and frames(1024)==3
    a=merge_segments([dict(start=100,end=700),dict(start=700,end=900)],1200)
    assert a==[[100,900]] and complement(a,1200)==[[0,100],[900,1200]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--inventory-only',action='store_true')
    args=p.parse_args()
    assert importlib.metadata.version('silero-vad')=='6.2.3'
    validate_counting()
    out=args.output.resolve()
    if not out.exists():
        out.mkdir(parents=True,exist_ok=False)
        items=inventory()
        json_write(out/'inventory.json',items)
        csv_write(out/'inventory.csv',items)
        json_write(out/'protocol.json',dict(silero_version='6.2.3',vad=PARAMS,window_samples=WINDOW,
            hop_samples=HOP,frame_rule='sum(max(0,1+floor((end-start-512)/256))) over separate continuous segments',
            center=False,padding=False,pre_emphasis=False,rms_trim=False,features=False,training=False,
            scope='all available source files, each microphone independently; paired report on intersection',
            resampling='integer: decimate n8 IIR zero_phase; otherwise resample_poly; unchanged from VAD comparison',
            brsd_speaker_mapping='(filename_number-1)//5+1; same mapping as src/sr/datasets/index.py',
            workers=args.workers,started_at=datetime.now(timezone.utc).isoformat()))
    else:
        items=json.loads((out/'inventory.json').read_text())
        assert json.loads((out/'protocol.json').read_text())['vad']==PARAMS
    totals=Counter(i['group'] for i in items)
    print('INVENTORY '+json.dumps(dict(totals)),flush=True)
    print('SPEAKERS '+json.dumps({g:len({i['speaker'] for i in items if i['group']==g}) for g in totals}),flush=True)
    if args.inventory_only:
        return
    completed={}
    if (out/'segments_counts.jsonl').exists():
        for line in (out/'segments_counts.jsonl').read_text().splitlines():
            r=json.loads(line)
            completed[r['path']]=r
    with (out/'segments_counts.jsonl').open('a') as f:
        if not completed:
            for row in reuse_comparison(items,out):
                f.write(json.dumps(row,separators=(',',':'))+'\n')
                completed[row['path']]=row
            f.flush()
        jobs=[i for i in items if i['path'] not in completed]
        if args.limit:
            # Benchmark covers both long BRSD and short VCTK tracks.
            jobs=[i for g in totals for i in [j for j in jobs if j['group']==g][:args.limit]]
        start=time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            for count,row in enumerate(pool.map(scan,jobs,chunksize=8),1):
                f.write(json.dumps(row,separators=(',',':'))+'\n')
                completed[row['path']]=row
                if count%100==0 or count==len(jobs):
                    f.flush()
                    progress=dict(status='running',completed=len(completed),total=len(items),
                        successful=sum('error' not in r for r in completed.values()),
                        errors=sum('error' in r for r in completed.values()),
                        elapsed_seconds=time.monotonic()-start,
                        rate_recordings_per_second=count/max(time.monotonic()-start,.001),
                        updated_at=datetime.now(timezone.utc).isoformat())
                    json_write(out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
    final=dict(status='complete' if len(completed)==len(items) else 'partial',completed=len(completed),
               total=len(items),errors=sum('error' in r for r in completed.values()),
               updated_at=datetime.now(timezone.utc).isoformat())
    json_write(out/'progress.json',final)
    print(json.dumps(final),flush=True)


if __name__=='__main__':
    main()
