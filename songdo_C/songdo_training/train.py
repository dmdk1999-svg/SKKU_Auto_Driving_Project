import argparse
import json
from pathlib import Path
import numpy as np
from models import train
from collect import ROOT,TRAIN,VALID

def load(names,suffix=''):
    arrays=[np.load(ROOT/'data'/(s+suffix+'.npz')) for s in names]
    return np.concatenate([a['observation'] for a in arrays]),np.concatenate([a['action_raw'] for a in arrays])

def original(path,seed):
    d=np.load(path);x=d['observation'];y=d['action_raw'];ep=d['episode']
    ids=np.random.default_rng(seed).permutation(np.unique(ep))
    mask=np.isin(ep,ids[:max(1,round(len(ids)*.2))])
    return x[~mask],y[~mask],x[mask],y[mask]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['original','human','matched'])
    p.add_argument('--source',type=Path);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=50);a=p.parse_args()
    if a.kind=='original':x,y,vx,vy=original(a.source,a.seed)
    else:
        suffix='_human' if a.kind=='human' else ''
        x,y=load(TRAIN,suffix);vx,vy=load(VALID,suffix)
    output=ROOT/'models'/f'{a.kind}_{a.seed}.pt'
    output.parent.mkdir(exist_ok=True)
    output.with_suffix('.config.json').write_text(json.dumps({
        'kind':a.kind,'seed':a.seed,'epochs':a.epochs,'train_rows':len(x),'validation_rows':len(vx),
        'dimensions':x.shape[1],'batch_size':256,'learning_rate':.0003,
        'validation':VALID if a.kind!='original' else '20% episode split',
        'test_sessions_excluded':['PM4','PM5']},indent=2))
    train(x,y,vx,vy,output,a.seed,a.epochs)

