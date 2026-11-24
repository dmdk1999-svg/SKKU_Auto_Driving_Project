"""Dataset aggregation on learner states using counterfactual SUMO expert steps."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from collect import ROOT,TRAIN,flatten
from models import BC,train
from train import load
from environment import SongdoIntersectionEnv
from build import ROUTES

def query(env,snapshot):
    t=env.t;v=t.vehicle;id=env.ego
    speed=v.getSpeed(id);time=t.simulation.getTime()
    speed_mode=v.getSpeedMode(id);lane_mode=v.getLaneChangeMode(id);max_lat=v.getMaxSpeedLat(id)
    t.simulation.saveState(str(snapshot))
    v.setSpeed(id,-1);v.setSpeedMode(id,31);v.setLaneChangeMode(id,1621)
    v.setMaxSpeedLat(id,.8)
    t.simulationStep()
    if id in v.getIDList():
        a=(v.getSpeed(id)-speed)/.2
        answer=np.clip([a/(2 if a>=0 else 7),v.getLateralSpeed(id)/.8],-1,1)
    else:answer=np.zeros(2)
    t.simulation.loadState(str(snapshot))
    # SUMO loadState restores kinematics, but TraCI mode overrides survive it.
    v.setSpeedMode(id,speed_mode);v.setLaneChangeMode(id,lane_mode);v.setMaxSpeedLat(id,max_lat)
    assert abs(t.simulation.getTime()-time)<1e-6
    assert abs(v.getSpeed(id)-speed)<1e-6
    return answer.astype('f')

def collect(model,seed,roundno,episodes=24):
    rng=np.random.default_rng(1000+seed*100+roundno)
    xs=[];ys=[];results=[]
    with tempfile.TemporaryDirectory(prefix='songdo_dagger_') as folder:
        snapshot=Path(folder)/'branch.xml'
        for ep in range(episodes):
            session=TRAIN[ep%len(TRAIN)];env=SongdoIntersectionEnv(session)
            route=list(ROUTES)[ep%len(ROUTES)]
            try:
                obs,_=env.reset(seed=4000+seed*100+roundno*30+ep,options={'route':route})
                total=0
                for step in range(env.max_steps):
                    action=model.act(flatten(obs))
                    if step%5==0:
                        label=query(env,snapshot);xs.append(flatten(obs));ys.append(label)
                        # Mix expert control early to keep coverage beyond immediate failures.
                        if rng.random()<(.5 if roundno==0 else .2):
                            action=label
                    obs,r,done,trunc,info=env.step(action);total+=r
                    if done or trunc:break
                results.append({'session':session,'route':route,'outcome':info['outcome'],'steps':step+1})
                print('dagger',seed,roundno,ep,info['outcome'],step+1,flush=True)
            finally:env.close()
    return np.array(xs,dtype='f'),np.array(ys,dtype='f'),results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0)
    p.add_argument('--rounds',type=int,default=2);p.add_argument('--episodes',type=int,default=24)
    a=p.parse_args()
    x,y=load(TRAIN);vx,vy=load(('PM3',))
    source=ROOT/'models'/f'matched_{a.seed}.pt'
    base=BC.load(source)
    logs=[]
    for k in range(a.rounds):
        dx,dy,rows=collect(base,a.seed,k,a.episodes)
        np.savez_compressed(ROOT/'data'/f'dagger_{a.seed}_{k}.npz',observation=dx,action_raw=dy)
        # Learner-visited examples receive equal total weight to the original demonstrations.
        repeat=max(1,int(len(x)/max(1,len(dx))))
        x=np.concatenate([x,np.tile(dx,(repeat,1))]);y=np.concatenate([y,np.tile(dy,(repeat,1))])
        target=ROOT/'models'/f'dagger_{a.seed}_round{k}.pt'
        base=train(x,y,vx,vy,target,a.seed,epochs=15,initial=source)
        source=target
        logs.append({'round':k,'new_rows':len(dx),'effective_training_rows':len(x),'rollouts':rows})
    base.save(ROOT/'models'/f'dagger_{a.seed}.pt')
    (ROOT/'models'/f'dagger_{a.seed}.collection.json').write_text(json.dumps(logs,indent=2))

