"""Collect current-observation / next-action pairs from responsive SUMO expert traffic."""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import libsumo as t
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'songdo_intersection'))
from environment import SongdoIntersectionEnv
from run import command

TRAIN=('AM1','AM2','AM3','AM4','AM5','PM1','PM2')
VALID=('PM3',)
TEST=('PM4','PM5')

def flatten(obs):
    return np.concatenate([obs['ego'],obs['neighbors'].ravel(),obs['neighbor_mask']]).astype('f')

def collect(session):
    out=ROOT/'data';out.mkdir(exist_ok=True)
    env=SongdoIntersectionEnv(session)
    records={v['id']:v for v in env.vehicles}
    args=command(session,seed=37)+['--no-warnings','true']
    t.start(args)
    xs=[];ys=[];episodes=[]
    rng=np.random.default_rng(37);tick=0
    try:
        while t.simulation.getMinExpectedNumber():
            before={}
            if tick%5==0:
                ids=list(t.vehicle.getIDList())
                if len(ids)>16:ids=list(rng.choice(ids,16,replace=False))
                for vid in ids:
                    env.ego=vid;env.route=records[vid]['route']
                    before[vid]=(flatten(env.observe()),t.vehicle.getSpeed(vid))
            t.simulationStep();tick+=1
            alive=set(t.vehicle.getIDList())
            for vid,(obs,speed) in before.items():
                if vid not in alive:continue
                a=(t.vehicle.getSpeed(vid)-speed)/.2
                lat=t.vehicle.getLateralSpeed(vid)/.8
                xs.append(obs);ys.append([a/(2 if a>=0 else 7),lat])
                episodes.append(int(vid[1:]))
    finally:t.close()
    np.savez_compressed(out/(session+'.npz'),observation=np.array(xs,dtype='f'),
                        action_raw=np.clip(ys,-1,1).astype('f'),episode=episodes)
    print(session,len(xs),'SUMO responsive-expert pairs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);a=p.parse_args()
    collect(a.session)

