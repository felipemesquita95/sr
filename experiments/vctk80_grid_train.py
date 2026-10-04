#!/usr/bin/env python3
"""One model, two held-out paired channel targets; train-only zscore."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('KERAS_BACKEND','torch')
os.environ.setdefault('KERAS_TORCH_DEVICE','cuda')
os.environ.setdefault('MPLCONFIGDIR','/tmp/sr-vctk80-grid-mpl')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils import set_random_seed
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from vctk80_activity_grid import ROOT, CLASSES, MICS, K, save
from vctk16_normalization import apply_cmvn, global_statistics

sys.path.insert(0,str(ROOT/'src'))
from sr.models import build_model


def load_array(out,normalization,condition,mic,n):
    representation='rasta' if normalization=='rasta' else 'baseline'
    original=np.load(out/'features'/f'{representation}_{condition}_{mic}.npy',mmap_mode='r')
    matrix=np.asarray(original[:,:,:n,:]).reshape(len(original),3*n,K).copy()
    if normalization=='cmn':
        matrix[:,:n]-=matrix[:,:n].mean(axis=2,keepdims=True)
    elif normalization=='cmvn':
        # Same affine convention as the prior normalization experiments.
        static=matrix[:,:n]
        mean=static.mean(axis=2,keepdims=True)
        std=np.maximum(static.std(axis=2,keepdims=True),1e-8)
        matrix[:,:n]=(static-mean)/std
        matrix[:,n:2*n]/=std
        matrix[:,2*n:3*n]/=std
    assert matrix.shape==(len(original),3*n,K) and np.isfinite(matrix).all()
    return matrix


class EpochState(Callback):
    def __init__(self,path,stage):
        super().__init__();self.path=path;self.stage=stage
    def on_epoch_end(self,epoch,logs=None):
        save(self.path,dict(stage=self.stage,epoch=epoch+1,
            metrics={k:float(v) for k,v in (logs or {}).items()},updated_at=datetime.now(timezone.utc).isoformat()))


def evaluate(model,x,y,folder):
    # Direct batched inference avoids the prediction adapter creating worker threads.
    with torch.no_grad():
        scores=np.concatenate([model(x[i:i+128],training=False).detach().cpu().numpy()
                               for i in range(0,len(x),128)])
    assert np.isfinite(scores).all()
    assert np.allclose(scores.sum(axis=1),1.,atol=1e-5)
    predicted=scores.argmax(axis=1)
    report=classification_report(y,predicted,labels=np.arange(scores.shape[1]),output_dict=True,zero_division=0)
    binary=(np.arange(scores.shape[1])[None,:]==y[:,None]).ravel()
    fpr,tpr,_=roc_curve(binary,scores.ravel(),drop_intermediate=False)
    fnr=1-tpr;cross=int(np.argmin(np.abs(fpr-fnr)))
    folder.mkdir(exist_ok=True)
    result=dict(n_recordings=len(y),accuracy=float(np.mean(predicted==y)),
        precision_macro=float(report['macro avg']['precision']),recall_macro=float(report['macro avg']['recall']),
        f1_macro=float(report['macro avg']['f1-score']),one_vs_rest_eer=float((fpr[cross]+fnr[cross])/2))
    save(folder/'metrics.json',result)
    save(folder/'classification_report.json',report)
    np.savez_compressed(folder/'predictions.npz',true=y,predicted=predicted,scores=scores)
    np.save(folder/'confusion.npy',confusion_matrix(y,predicted,labels=np.arange(scores.shape[1])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('--stage',required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    out=args.output.resolve();stage=json.loads(args.stage)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; do not fall back to CPU training')
    torch.set_num_threads(4)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    cohort=json.loads((out/'cohort.json').read_text())
    name='_'.join(str(stage[k]) for k in ('fold','n_mfcc','normalization','architecture','train_class','train_mic'))
    model_dir=out/('smoke' if args.smoke else 'models')/name
    model_dir.mkdir(parents=True,exist_ok=True)
    n=stage['n_mfcc'];fold=stage['fold']-1
    groups=np.array([i['fold_group'] for i in cohort['items']])
    labels=np.array([i['label'] for i in cohort['items']],dtype=np.int64)
    train=np.flatnonzero((groups!=fold)&(groups!=(fold+1)%5))
    validation=np.flatnonzero(groups==(fold+1)%5)
    test=np.flatnonzero(groups==fold)
    assert not (set(train)&set(test) or set(train)&set(validation) or set(test)&set(validation))
    assert len(train)==6480 and len(validation)==len(test)==2160
    if args.smoke:
        # Only use rows that have already been flushed by the feature builder.
        prepared=json.loads((out/'features/progress.json').read_text())['completed_pairs']
        train=train[train<prepared][:128]
        validation=validation[validation<prepared][:128]
        assert len(train)>=64 and len(validation)>=32
    norm=stage['normalization']
    array=load_array(out,norm,stage['train_class'],stage['train_mic'],n)
    mean,std=global_statistics(array[train],norm=='zscore')
    train_x=(array[train]-mean)/std
    val_x=(array[validation]-mean)/std
    np.savez(model_dir/'normalization.npz',mean=mean,std=std)
    save(model_dir/'input_protocol.json',dict(stage=stage,manifest_sha256=cohort['manifest_sha256'],
        train=train.tolist(),validation=validation.tolist(),test=test.tolist(),
        input_shape=[3*n,K],global_zscore=norm=='zscore',gpu=torch.cuda.get_device_name(0),torch=torch.__version__))
    set_random_seed(cohort['seed']+fold)
    model=build_model(stage['architecture'],input_shape=(3*n,K),num_classes=108,learning_rate=.001)
    checkpoint=model_dir/'best.keras'
    initial=0
    if checkpoint.exists() and (model_dir/'history.csv').exists() and not args.smoke:
        model=load_model(checkpoint)
        with (model_dir/'history.csv').open() as f:
            previous=list(csv.DictReader(f))
        # Resume optimizer state from the best stored epoch, not a later epoch.
        best=max(previous,key=lambda r:float(r['val_accuracy']))
        initial=int(best['epoch'])+1
    callbacks=[ModelCheckpoint(checkpoint,monitor='val_accuracy',mode='max',save_best_only=True),
        CSVLogger(model_dir/'history.csv',append=initial>0),
        EpochState(model_dir/'epoch_state.json',stage),
        ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=.2,patience=15,min_lr=1e-6),
        EarlyStopping(monitor='val_accuracy',mode='max',patience=15,restore_best_weights=True)]
    started=time.monotonic()
    if args.smoke:
        model.fit(train_x[:128],labels[train[:128]],validation_data=(val_x[:128],labels[validation[:128]]),
            epochs=1,batch_size=64,callbacks=callbacks,verbose=2)
        assert checkpoint.is_file()
        model=load_model(checkpoint)
        smoke_test=test[test<prepared][:128]
        assert len(smoke_test)>=32
        for target_class in CLASSES:
            for target_mic in MICS:
                target=load_array(out,norm,target_class,target_mic,n)
                evaluate(model,(target[smoke_test]-mean)/std,labels[smoke_test],
                         model_dir/f'{target_class}_{target_mic}')
                del target
        print('GPU fit, checkpoint reload and two paired channel evaluations passed.',flush=True)
        return
    model.fit(train_x,labels[train],validation_data=(val_x,labels[validation]),
        initial_epoch=initial,epochs=150,batch_size=128,callbacks=callbacks,verbose=2)
    model=load_model(checkpoint)
    evaluations=[]
    for target_class in CLASSES:
        for target_mic in MICS:
            target=load_array(out,norm,target_class,target_mic,n)
            target_x=(target[test]-mean)/std
            metrics=evaluate(model,target_x,labels[test],model_dir/f'{target_class}_{target_mic}')
            evaluations.append(dict(**stage,test_class=target_class,test_mic=target_mic,**metrics))
            del target,target_x
    save(model_dir/'evaluations.json',evaluations)
    save(model_dir/'completed.json',dict(stage=stage,manifest_sha256=cohort['manifest_sha256'],
        evaluations=2,elapsed_seconds=time.monotonic()-started,
        eer_note='Closed-set softmax one-versus-rest diagnostic, not an open-set embedding verification protocol'))


if __name__=='__main__':
    main()
