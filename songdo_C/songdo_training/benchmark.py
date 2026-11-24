"""Paired evaluation on fixed, held-out vehicles and simulator seeds."""
import argparse
import json
import math
import numpy as np
from models import BC
from collect import ROOT,flatten
from environment import SongdoIntersectionEnv
from build import ROUTES

def cases(session,count=4):
    e=SongdoIntersectionEnv(session);rng=np.random.default_rng(6101)
    chosen=[]
    for route in ROUTES:
        rows=[v for v in e.vehicles if v['route']==route]
        for k,index in enumerate(rng.choice(len(rows),count,replace=len(rows)<count)):
            chosen.append({'session':session,'route':route,'ego_id':rows[index]['id'],
                           'seed':22000+len(chosen),'case':session+'_'+str(len(chosen))})
    return chosen

def original_obs(env):
    t=env.t;v=t.vehicle;id=env.ego
    speed=v.getSpeed(id);edge=v.getRoadID(id);index=v.getLaneIndex(id);pos=v.getLanePosition(id)
    limit=16.67;obs=[speed/limit];connect=[];density=[]
    for offset in (2,1,0,-1,-2):
        front=rear=None;j=index+offset
        if offset==0:
            front=v.getLeader(id,50);rear=v.getFollower(id,50)
            front=(front[1],v.getSpeed(front[0])) if front and front[0] else None
            rear=(rear[1],v.getSpeed(rear[0])) if rear and rear[0] else None
        elif abs(offset)==1:
            modes=(2,0) if offset==1 else (3,1)
            neighbors=[v.getNeighbors(id,m) for m in modes]
            def nearest(values):
                if not values:return None
                vid,gap=min(values,key=lambda a:a[1]);return gap,v.getSpeed(vid)
            front,rear=[nearest(n) for n in neighbors]
        elif 0<=j<t.edge.getLaneNumber(edge):
            for other in t.lane.getLastStepVehicleIDs(f'{edge}_{j}'):
                if other==id:continue
                delta=v.getLanePosition(other)-pos
                if delta>=0:
                    gap=delta-v.getLength(other)
                    if front is None or gap<front[0]:front=(gap,v.getSpeed(other))
                else:
                    gap=-delta-v.getLength(id)
                    if rear is None or gap<rear[0]:rear=(gap,v.getSpeed(other))
        for car,is_front in [(front,True),(rear,False)]:
            if car is None or not 0<=car[0]<=50:
                obs.extend([1,1] if is_front else [-1,-1])
            else:obs.extend([car[0]/50*(1 if is_front else -1),(car[1]-speed)/limit])
        if edge.startswith(':'):
            connect.append(1);density.append(0);continue
        if not 0<=j<t.edge.getLaneNumber(edge):
            connect.append(-1);density.append(1);continue
        lane=f'{edge}_{j}';remain=t.lane.getLength(lane)-pos;links=t.lane.getLinks(lane)
        connect.append(1 if links or remain>50 else np.clip(remain/50,0,1))
        usable=50 if links else min(50,remain)
        n=sum(0<v.getLanePosition(o)-pos<=min(50,remain) for o in t.lane.getLastStepVehicleIDs(lane) if o!=id)
        if links and remain<50:
            n+=sum(v.getLanePosition(o)<=50-remain for o in t.lane.getLastStepVehicleIDs(links[0][0]))
        density.append(min(1,n*7.5/max(.01,usable)))
    return np.clip(obs+connect+density,-1,1).astype('f')

class Actor:
    def __init__(self,path,original=False):
        self.model=BC.load(path);self.original=original;self.target=None
    def act(self,env,obs):
        if not self.original:return self.model.act(flatten(obs))
        action=self.model.act(original_obs(env))
        accel=action[0]*5.4
        lon=accel/(2 if accel>=0 else 7)
        lane=env.t.vehicle.getLaneIndex(env.ego)
        edge=env.t.vehicle.getRoadID(env.ego)
        if abs(action[1])>=.5 and self.target is None:
            self.target=(edge,lane+int(np.sign(action[1])))
        lateral=0.
        if self.target:
            targetedge,targetlane=self.target
            if targetedge!=edge:self.target=None
            else:
                width=env.t.lane.getWidth(env.t.vehicle.getLaneID(env.ego))
                error=(targetlane-lane)*width-env.t.vehicle.getLateralLanePosition(env.ego)
                if abs(error)<.15:self.target=None
                else:lateral=np.clip(error/.8,-1,1)
        return np.clip([lon,lateral],-1,1)

def episode(actor,case):
    env=SongdoIntersectionEnv(case['session'])
    try:
        obs,_=env.reset(seed=case['seed'],options={'ego_id':case['ego_id']})
        start=env.t.simulation.getTime();total=0;wait=0;maxdecel=0;min_gap=1000
        while True:
            actor.target=getattr(actor,'target',None)
            action=actor.act(env,obs)
            leader=env.t.vehicle.getLeader(env.ego,100)
            if leader and leader[0]:min_gap=min(min_gap,leader[1])
            follower,_=env.t.vehicle.getFollower(env.ego,50)
            if follower:maxdecel=max(maxdecel,-env.t.vehicle.getAcceleration(follower))
            if env.t.vehicle.getSpeed(env.ego)<.2:wait+=.2
            obs,reward,done,truncated,info=env.step(action);total+=reward
            if done or truncated:
                return {**case,'outcome':info['outcome'],'return':total,'seconds':info['sim_time']-start,
                        'waiting_seconds':wait,'min_gap_m':min_gap,'max_follower_decel':maxdecel,
                        'red_stop':info['stopped_for_red']}
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--kind',default='matched')
    p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--out',required=True)
    a=p.parse_args()
    casefile=ROOT/('validation_cases.json' if a.split=='validation' else 'test_cases.json')
    if not casefile.exists():
        selected=cases('PM3',2) if a.split=='validation' else cases('PM4')+cases('PM5')
        casefile.write_text(json.dumps(selected,indent=2))
    rows=[]
    for case in json.loads(casefile.read_text()):
        actor=Actor(a.model,a.kind=='original')
        row=episode(actor,case);rows.append(row)
        print(a.kind,case['case'],row['outcome'],round(row['seconds'],1),flush=True)
    from pathlib import Path
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(rows,indent=2))

