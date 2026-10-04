#!/usr/bin/env python3
"""Extract only the new paired cohort, in fixed continuous VAD windows."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import importlib.metadata
import json
from math import gcd
import multiprocessing
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
import numpy as np
import soundfile as sf
from scipy.signal import decimate, resample_poly
from scipy.fft import dct
from vctk80_activity_grid import save, CLASSES, MICS, K


@lru_cache(maxsize=1)
def libraries():
    import librosa
    from build_vctk16_rasta_features import rasta_filter
    return librosa,rasta_filter


def extract(item):
    librosa,rasta_filter=libraries()
    result={}
    for mic in MICS:
        raw,rate=sf.read(item['paths'][mic],dtype='float64')
        mono=raw if raw.ndim==1 else raw.mean(axis=1)
        if rate==16000:
            reduced=mono.copy()
        elif rate%16000==0:
            reduced=decimate(mono,rate//16000,n=8,ftype='iir',zero_phase=True)
        else:
            factor=gcd(rate,16000)
            reduced=resample_poly(mono,16000//factor,rate//factor)
        assert len(reduced)==item['source_samples_16k'] and np.isfinite(reduced).all()
        for condition in CLASSES:
            window=item['windows'][condition]
            a,b=window['segment_start'],window['segment_end']
            first=window['first_frame']
            audio=np.asarray(reduced[a:b],dtype=np.float32)
            emphasized=np.empty_like(audio)
            emphasized[0]=audio[0]
            emphasized[1:]=audio[1:]-.97*audio[:-1]
            mel=librosa.feature.melspectrogram(y=emphasized,sr=16000,n_fft=512,hop_length=256,
                window='hamming',center=False,n_mels=128,power=2.)
            logmel=librosa.power_to_db(mel,ref=1.,amin=1e-10,top_db=80.)
            for name,values in [('baseline',logmel),('rasta',rasta_filter(logmel))]:
                static=dct(values,type=2,axis=0,norm='ortho')[:40,first:first+K].astype(np.float32)
                assert static.shape==(40,K)
                delta=librosa.feature.delta(static,width=9).astype(np.float32)
                delta2=librosa.feature.delta(static,width=9,order=2).astype(np.float32)
                tensor=np.stack([static,delta,delta2])
                assert tensor.shape==(3,40,K) and np.isfinite(tensor).all()
                result[(name,condition,mic)]=tensor
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    out=args.output.resolve()
    assert importlib.metadata.version('silero-vad')=='6.2.3'
    cohort=json.loads((out/'cohort.json').read_text())
    if args.smoke:
        result=extract(cohort['items'][0])
        print('Verified 4 finite activity feature tensors, two paired microphones, each (3,40,80).',flush=True)
        return
    feature_root=out/'features'
    feature_root.mkdir(exist_ok=True)
    if (feature_root/'completed.json').exists():
        assert json.loads((feature_root/'completed.json').read_text())['manifest_sha256']==cohort['manifest_sha256']
        return
    keys=[(r,c,m) for r in ('baseline','rasta') for c in CLASSES for m in MICS]
    checkpoint=feature_root/'progress.json'
    start=json.loads(checkpoint.read_text())['completed_pairs'] if checkpoint.exists() else 0
    arrays={}
    for key in keys:
        path=feature_root/('_'.join(key)+'.npy')
        arrays[key]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',
            dtype=np.float32,shape=(len(cohort['items']),3,40,K))
    began=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for index,result in enumerate(pool.map(extract,cohort['items'][start:],chunksize=4),start+1):
            for key in keys:
                arrays[key][index-1]=result[key]
            if index%100==0 or index==len(cohort['items']):
                for array in arrays.values():
                    array.flush()
                state=dict(completed_pairs=index,total_pairs=len(cohort['items']),
                    elapsed_seconds=time.monotonic()-began,manifest_sha256=cohort['manifest_sha256'])
                save(checkpoint,state)
                print(f'Features: {index}/{len(cohort["items"])} paired recordings',flush=True)
    save(feature_root/'completed.json',dict(status='complete',paired_recordings=len(cohort['items']),
        shape=[len(cohort['items']),3,40,K],manifest_sha256=cohort['manifest_sha256'],
        acoustic_frames_crossing_seams=0,delta_scope='selected continuous window only',
        numpy=np.__version__,librosa=libraries()[0].__version__))


if __name__=='__main__':
    main()
