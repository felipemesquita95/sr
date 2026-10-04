#!/usr/bin/env python3
"""Paired VCTK activity experiment, 80 continuous activity frames."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'output/disponibilidade_frames_vad200_20260928_2310'
CLASSES=('activity',)
MICS=('mic1','mic2')
NORMS=('zscore','cmn','cmvn','rasta')
ARCHS=('cnn','temporal_cnn')
MFCCS=(20,30,40)
K,HOP,WINDOW=80,256,512
NEEDED=WINDOW+(K-1)*HOP


def save(path,value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def intersect(first,second):
    result=[]
    i=j=0
    while i<len(first) and j<len(second):
        a,b=max(first[i][0],second[j][0]),min(first[i][1],second[j][1])
        if b-a>=NEEDED:
            result.append([a,b])
        if first[i][1]<second[j][1]:
            i+=1
        else:
            j+=1
    return result


def stages():
    return [dict(n_mfcc=n,normalization=norm,architecture=arch,train_class=condition,
                 train_mic=mic,fold=fold)
            for fold,n,norm,arch,condition,mic in itertools.product(range(1,6),MFCCS,NORMS,ARCHS,CLASSES,MICS)]


def make_plan(out):
    import numpy as np
    protocol=json.loads((AUDIT/'protocol.json').read_text())
    assert protocol['vad']==dict(sampling_rate=16000,threshold=.5,neg_threshold=.35,
        min_speech_duration_ms=250,min_silence_duration_ms=200,speech_pad_ms=30)
    pairs={}
    for line in (AUDIT/'segments_counts.jsonl').open():
        row=json.loads(line)
        if row['group'].startswith('VCTK '):
            pairs.setdefault((row['speaker'],row['utterance']),{})[row['group'].split()[-1]]=row
    available=defaultdict(list)
    for (speaker,utterance),mics in sorted(pairs.items()):
        if set(mics)!=set(MICS):
            continue
        assert mics['mic1']['samples_16k']==mics['mic2']['samples_16k']
        common={c:intersect(mics['mic1'][c+'_segments_samples'],mics['mic2'][c+'_segments_samples']) for c in CLASSES}
        if not all(common.values()):
            continue
        windows={}
        for c in CLASSES:
            a,b=max(common[c],key=lambda r:(r[1]-r[0],-r[0]))
            total_frames=1+(b-a-WINDOW)//HOP
            first=(total_frames-K)//2
            start=a+first*HOP
            assert a<=start and start+NEEDED<=b
            windows[c]=dict(segment_start=a,segment_end=b,window_start=start,
                            window_end=start+NEEDED,first_frame=first,frames=K)
        available[speaker].append(dict(speaker=speaker,utterance=utterance,
            paths={m:mics[m]['path'] for m in MICS},source_samples_16k=mics['mic1']['samples_16k'],windows=windows))
    assert len(available)==108
    per_speaker=min(map(len,available.values()))//5*5
    assert per_speaker==100
    rng=np.random.default_rng(42)
    items=[]
    speakers=sorted(available)
    for label,speaker in enumerate(speakers):
        rows=available[speaker]
        positions=sorted(rng.choice(len(rows),size=per_speaker,replace=False).tolist())
        assignments=rng.permutation(per_speaker)%5
        for rank,pos in enumerate(positions):
            items.append(dict(**rows[pos],label=label,fold_group=int(assignments[rank])))
    assert len({(i['speaker'],i['utterance']) for i in items})==len(items)
    assert set(Counter((i['speaker'],i['fold_group']) for i in items).values())=={20}
    digest=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=False)
    save(out/'cohort.json',dict(seed=42,speakers=speakers,speaker_count=108,
        recordings_per_speaker=per_speaker,paired_recordings=len(items),frames_per_recording=K,
        available_paired_recordings=sum(map(len,available.values())),eligible_per_speaker={s:len(available[s]) for s in speakers},
        sampling='uniform without replacement per speaker; balanced maximum multiple of five',
        windows='center of longest continuous common VAD class segment; identical sample times in mic1/mic2',
        manifest_sha256=digest,items=items))
    plan=dict(dataset='VCTK only',silero_version='6.2.3',vad=protocol['vad'],
        frame_samples=WINDOW,hop_samples=HOP,frames=K,mfcc_counts=list(MFCCS),
        representations='MFCC + delta + delta_delta',classes=list(CLASSES),mixed=False,
        normalizations=list(NORMS),architectures=list(ARCHS),
        channel_directions=[f'{a}->{b}' for a,b in itertools.product(MICS,repeat=2)],
        class_directions=[f'{a}->{b}' for a,b in itertools.product(CLASSES,repeat=2)],
        folds=5,split='60/20/20 per speaker; paired utterance stays in same fold in all classes and microphones',
        epochs=150,patience=15,batch_size=128,learning_rate=.001,seed=42,
        pre_emphasis_before_vad=False,pre_emphasis_mfcc=.97,rms_trim=False,
        feature_window='hamming',n_mels=128,center=False,
        log_mel=dict(ref=1.,amin=1e-10,top_db=80.),dct=dict(type=2,norm='ortho'),
        delta_width=9,delta_scope='selected 80-frame window only',
        rasta_scope='continuous common VAD segment before selected window, independently per segment',
        zscore_scope='train source class/microphone only; same statistics applied to every test target',
        cmn_cmvn_scope='per selected recording window, isolated from global zscore',
        models=240,test_evaluations=480,summary_combinations=96,
        source_counts=str(AUDIT/'segments_counts.jsonl'),comparison_note='New activity-only cohort; previous 40-frame results do not isolate duration',stages=stages())
    save(out/'protocol.json',plan)
    save(out/'status.json',dict(status='planned',models_completed=0,models_total=240,evaluations_total=480))
    with (out/'selection.csv').open('w',newline='') as f:
        columns=['speaker','utterance','label','fold_group','mic1','mic2','activity_start','activity_end']
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for i in items:
            writer.writerow(dict(speaker=i['speaker'],utterance=i['utterance'],label=i['label'],fold_group=i['fold_group'],
                **i['paths'],**{f'{c}_{edge}':i['windows'][c]['window_'+edge] for c in CLASSES for edge in ('start','end')}))
    (out/'README.md').write_text(f'# VCTK — 80 frames — somente atividade\n\n108 locutores × 100 gravações pareadas = {len(items)} pares. Seed 42. Cinco folds 60/20/20: 60/20/20 gravações por locutor.\n\n80 frames = 1.296 s por entrada, em segmento contínuo comum aos dois microfones. MFCC 20/30/40 + Δ + ΔΔ. Silero 6.2.3, silêncio mínimo 200 ms, demais parâmetros preservados.\n\nZ-score, CMN, CMVN, RASTA isolados; CNN e CNN temporal; mic1→mic1, mic1→mic2, mic2→mic1, mic2→mic2. 240 modelos, 480 avaliações, 96 combinações resumidas em cinco folds.\n\nCoorte nova: comparação com a grade anterior de 40 frames também muda as gravações e sua quantidade. Uma comparação isolada de duração exige reexecutar 40 frames nesta mesma seleção.\n\nResultados: resultados_folds.csv e resultados_resumo.csv, métricas em fração 0–1. Estado: status.json. Protocolo: protocol.json. Seleção e tempos exatos: selection.csv/cohort.json.\n')
    print(json.dumps({k:plan[k] for k in ('models','test_evaluations','summary_combinations')})+f' · {len(items)} pairs',flush=True)


def summarize(out):
    import numpy as np
    rows=[]
    for file in sorted((out/'models').glob('*/evaluations.json')):
        rows.extend(json.loads(file.read_text()))
    if not rows:
        return
    def csv_write(path,data):
        temporary=path.with_suffix('.tmp.csv')
        with temporary.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
        temporary.replace(path)
    csv_write(out/'resultados_folds.csv',rows)
    fields=['n_mfcc','normalization','architecture','train_class','test_class','train_mic','test_mic']
    grouped=defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in fields)].append(r)
    summaries=[]
    for key,group in sorted(grouped.items()):
        record=dict(zip(fields,key),folds_completed=len(group),folds_expected=5)
        for metric in ('accuracy','precision_macro','recall_macro','f1_macro','one_vs_rest_eer'):
            values=[r[metric] for r in group]
            record[metric+'_mean']=float(np.mean(values));record[metric+'_std']=float(np.std(values))
        summaries.append(record)
    csv_write(out/'resultados_resumo.csv',summaries)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'cohort.json').exists():
        raise RuntimeError('Prepare a reviewable plan first')
    status=dict(status='preparing_features',models_completed=0,models_total=240,updated_at=datetime.now(timezone.utc).isoformat())
    save(out/'status.json',status)
    env=dict(os.environ,KERAS_BACKEND='torch',KERAS_TORCH_DEVICE='cuda',OMP_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/sr-vctk80-grid-mpl')
    def call(args,**kwargs):
        resource_failures=0
        while True:
            try:
                return subprocess.run(args,env=env,check=True,**kwargs)
            except BlockingIOError as error:
                if error.errno != 11:
                    raise
                save(out/'status.json',dict(status='waiting_for_process_resources',
                    models_completed=status['models_completed'],models_total=240,
                    updated_at=datetime.now(timezone.utc).isoformat()))
                print('Process creation temporarily unavailable; retrying in 30 seconds',flush=True)
                time.sleep(30)
            except subprocess.CalledProcessError:
                stream=kwargs.get('stdout')
                if stream is None:
                    raise
                stream.flush()
                with Path(stream.name).open('rb') as reader:
                    reader.seek(max(0,Path(stream.name).stat().st_size-4096))
                    tail=reader.read().decode(errors='replace')
                if resource_failures>=3 or not any(message in tail for message in
                        ("can't start new thread",'Resource temporarily unavailable')):
                    raise
                resource_failures+=1
                save(out/'status.json',dict(status='retrying_resource_failure',
                    current_model=status.get('current_model'),models_completed=status['models_completed'],
                    models_total=240,retry=resource_failures,updated_at=datetime.now(timezone.utc).isoformat()))
                print(f'Temporary child resource failure; retry {resource_failures}/3 in 30 seconds',flush=True)
                time.sleep(30)
    call([sys.executable,str(ROOT/'experiments/vctk80_grid_features.py'),str(out)])
    lock=(ROOT/'tmp/vctk16_channel_suite.lock').open('a')
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            save(out/'status.json',dict(status='waiting_for_existing_gpu_experiment',models_completed=0,
                                       models_total=240,updated_at=datetime.now(timezone.utc).isoformat()))
            time.sleep(30)
    if not (out/'smoke_completed.json').exists():
        save(out/'status.json',dict(status='gpu_integration_checks',models_completed=0,models_total=240))
        for arch,norm,n in (('cnn','zscore',20),('temporal_cnn','rasta',40)):
            stage=dict(fold=1,n_mfcc=n,normalization=norm,architecture=arch,
                       train_class='activity',train_mic='mic1')
            call([sys.executable,str(ROOT/'experiments/vctk80_grid_train.py'),str(out),
                            '--stage',json.dumps(stage),'--smoke'])
        save(out/'smoke_completed.json',dict(status='passed',architectures=list(ARCHS)))
    for index,stage in enumerate(stages(),1):
        import shutil
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise RuntimeError('Less than 2 GiB free in the results volume; stopping before the next model')
        name='_'.join(str(stage[k]) for k in ('fold','n_mfcc','normalization','architecture','train_class','train_mic'))
        path=out/'models'/name
        if not (path/'completed.json').exists():
            status.update(status='training',current_model=name,models_completed=index-1,
                          updated_at=datetime.now(timezone.utc).isoformat())
            save(out/'status.json',status)
            path.mkdir(parents=True,exist_ok=True)
            with (path/'execution.log').open('a') as log:
                args=[sys.executable,str(ROOT/'experiments/vctk80_grid_train.py'),str(out),'--stage',json.dumps(stage)]
                call(args,stdout=log,stderr=subprocess.STDOUT)
        summarize(out)
        status.update(models_completed=index,updated_at=datetime.now(timezone.utc).isoformat())
        save(out/'status.json',status)
        print(f'{index}/240 models completed: {name}',flush=True)
    status.update(status='complete',current_model=None,models_completed=240)
    save(out/'status.json',status)
    summarize(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--launch',action='store_true')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--copy-plan',type=Path)
    args=p.parse_args()
    out=args.output.resolve()
    if args.copy_plan:
        import shutil
        source=args.copy_plan.resolve()
        out.mkdir(parents=True,exist_ok=False)
        for filename in ('cohort.json','protocol.json','selection.csv','README.md','status.json'):
            shutil.copy2(source/filename,out/filename)
        save(source/'results_location.json',dict(output=str(out)))
    if args.plan:
        make_plan(out)
    elif args.summarize:
        summarize(out)
    elif args.launch:
        log=out/'suite_execution.log'
        with log.open('a') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(out)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'execution.json',dict(pid=child.pid,log=str(log),started_at=datetime.now(timezone.utc).isoformat()))
        print('Suite PID:',child.pid,flush=True)
    else:
        try:
            run(out)
        except Exception as error:
            state=json.loads((out/'status.json').read_text())
            state.update(status='failed',error=repr(error),updated_at=datetime.now(timezone.utc).isoformat())
            save(out/'status.json',state)
            raise


if __name__=='__main__':
    main()
