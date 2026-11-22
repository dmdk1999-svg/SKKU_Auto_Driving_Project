"""Gymnasium environment with responsive traffic on the full Songdo C intersection."""
import json
import math
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import libsumo
from build import ROOT,ROUTES
from run import command

class SongdoIntersectionEnv(gym.Env):
    """One instance per process. All NPCs respond through SUMO IDM and SL2015."""
    metadata={'render_modes':[],'render_fps':5}

    def __init__(self,session='AM3',max_steps=1500,safe_control=False):
        super().__init__()
        self.session=session
        self.max_steps=max_steps
        self.safe_control=safe_control
        self.action_space=spaces.Box(-1,1,shape=(2,),dtype=np.float32)
        self.observation_space=spaces.Dict({
            'ego':spaces.Box(-1,1,shape=(14,),dtype=np.float32),
            'neighbors':spaces.Box(-1,1,shape=(8,8),dtype=np.float32),
            'neighbor_mask':spaces.Box(0,1,shape=(8,),dtype=np.float32)})
        self.vehicles=json.loads((ROOT/'sessions'/session/'vehicles.json').read_text())
        self.geo=json.loads((ROOT/'network/geometry.json').read_text())['sections']
        self.t=libsumo
        self.running=False
        self.steps=0

    def close(self):
        if self.running:
            self.t.close()
            self.running=False

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.close()
        options=options or {}
        candidates=[v for v in self.vehicles if
                    ('route' not in options or v['route']==options['route']) and
                    ('ego_id' not in options or v['id']==options['ego_id'])]
        if not candidates: raise ValueError('No vehicle matches the requested route/id')
        record=candidates[int(self.np_random.integers(len(candidates)))]
        self.ego=record['id'];self.route=record['route']
        # GUI uses run.py. Independent libsumo workers each own their simulator.
        args=command(self.session,seed=int(seed or 0))
        args+=['--save-state.rng','true','--no-warnings','true']
        self.t.start(args)
        self.running=True;self.steps=0;self.lat=0;self.travel=0
        while self.ego not in self.t.vehicle.getIDList():
            self.t.simulationStep()
            if self.t.simulation.getMinExpectedNumber()==0:
                self.close();raise RuntimeError('Selected expert vehicle was never inserted')
        self.t.vehicle.setColor(self.ego,(255,35,35,255))
        self.t.vehicle.setSpeedMode(self.ego,31 if self.safe_control else 96)
        self.t.vehicle.setLaneChangeMode(self.ego,512 if self.safe_control else 0)
        self.last_distance=self.t.vehicle.getDistance(self.ego)
        self.stopped_for_red=False
        return self.observe(),{'vehicle':self.ego,'route':self.route,'session':self.session,
                              'safe_control':self.safe_control}

    def observe(self):
        v=self.t.vehicle;id=self.ego
        xy=np.array(v.getPosition(id));speed=v.getSpeed(id)
        heading=math.radians(v.getAngle(id))
        forward=np.array([math.sin(heading),math.cos(heading)])
        left=np.array([-forward[1],forward[0]])
        lane=v.getLaneID(id);edge=v.getRoadID(id)
        width=self.t.lane.getWidth(lane);index=v.getLaneIndex(id)
        count=self.t.edge.getLaneNumber(edge)
        lateral=v.getLateralLanePosition(id)
        lights=v.getNextTLS(id)
        signal,dist=0.,100.
        if lights:
            signal={'r':-1,'R':-1,'y':.5,'Y':.5,'g':1,'G':1}.get(lights[0][3],0)
            dist=lights[0][2]
        route=ROUTES[self.route]
        remaining=v.getDrivingDistance(id,route[-1],self.t.lane.getLength(route[-1]+'_0'))
        a=np.array(self.geo[route[0]]['end'])-self.geo[route[0]]['start']
        b=np.array(self.geo[route[-1]]['end'])-self.geo[route[-1]]['start']
        angle=math.atan2(a[0]*b[1]-a[1]*b[0],a@b)/math.pi
        ego=np.array([speed/20,v.getAcceleration(id)/7,lateral/(width/2),
                      v.getLateralSpeed(id)/.8,max(0,remaining)/500,angle,signal,dist/100,
                      index/max(1,count-1),float(index<count-1),float(index>0),
                      float(edge.startswith(':')),v.getLength(id)/15,v.getWidth(id)/3],dtype=np.float32)
        neighbors=np.zeros((8,8),np.float32);mask=np.zeros(8,np.float32)
        near=[]
        for other in v.getIDList():
            if other==id:continue
            delta=np.array(v.getPosition(other))-xy
            distance=np.linalg.norm(delta)
            if distance<=60:near.append((distance,other,delta))
        for i,(_,other,delta) in enumerate(sorted(near,key=lambda r:r[0])[:8]):
            theta=math.radians(v.getAngle(other))
            velocity=np.array([math.sin(theta),math.cos(theta)])*v.getSpeed(other)-forward*speed
            difference=theta-heading
            neighbors[i]=[delta@forward/60,delta@left/60,velocity@forward/20,velocity@left/20,
                          math.sin(difference),math.cos(difference),v.getLength(other)/15,v.getWidth(other)/3]
            mask[i]=1
        return {'ego':np.clip(ego,-1,1),'neighbors':np.clip(neighbors,-1,1),'neighbor_mask':mask}

    def step(self,action):
        if not self.running:raise RuntimeError('Call reset before step')
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():raise ValueError('Two finite actions required')
        action=np.clip(action,-1,1)
        v=self.t.vehicle;id=self.ego
        edge=v.getRoadID(id);lights=v.getNextTLS(id)
        red=bool(lights and lights[0][3].lower()=='r')
        dist=lights[0][2] if lights else float('inf')
        speed=v.getSpeed(id)
        accel=action[0]*(2 if action[0]>=0 else 7)
        v.setSpeed(id,float(max(0,speed+.2*accel)))
        # Physical sublane commands use SUMO movement and collision geometry.
        # SUMO follows the selected route's curved internal lane through the intersection.
        if not edge.startswith(':'):
            self.lat=0 if abs(action[1])<1e-8 else self.lat+np.clip(.8*action[1]-self.lat,-.14,.14)
            v.setMaxSpeedLat(id,abs(self.lat))
            v.changeSublane(id,float(self.lat*.2))
        self.t.simulationStep();self.steps+=1
        hit=id in self.t.simulation.getCollidingVehiclesIDList()
        arrived=id in self.t.simulation.getArrivedIDList()
        teleported=id in self.t.simulation.getStartingTeleportIDList()
        alive=id in v.getIDList()
        crossed_red=alive and red and dist<max(5,speed*.2+2) and v.getRoadID(id)!=edge
        if alive:
            distance=v.getDistance(id)
            progress=max(0,distance-self.last_distance)
            self.last_distance=distance
            obs=self.observe()
            reward=progress*.05-.002
            self.stopped_for_red |= red and speed<.2 and dist<20
        else:
            obs={k:np.zeros(s.shape,dtype=s.dtype) for k,s in self.observation_space.spaces.items()}
            reward=0.
        outcome='running'
        if hit:outcome='collision';reward=-100
        elif crossed_red:outcome='red_light';reward=-50
        elif arrived:outcome='arrival';reward=20
        elif teleported or not alive:outcome='removed';reward=-100
        truncated=self.steps>=self.max_steps and outcome=='running'
        if truncated:outcome='timeout'
        return obs,float(reward),outcome in ('collision','red_light','arrival','removed'),truncated,{
            'outcome':outcome,'route':self.route,'vehicle':id,'stopped_for_red':self.stopped_for_red,
            'sim_time':self.t.simulation.getTime(),'requested_action':action.astype(np.float32),
            'applied_speed':v.getSpeed(id) if alive else 0}

if __name__=='__main__':
    from gymnasium.utils.env_checker import check_env
    env=SongdoIntersectionEnv(session='AM1')
    try:
        check_env(env,skip_render_check=True)
        print('Gymnasium API check passed')
    finally:env.close()

