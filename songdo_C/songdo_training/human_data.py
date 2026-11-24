"""Real Songdo driver BC samples. Signal labels use the reconstructed controller."""
import json
import math
import numpy as np
import pandas as pd
import sumolib
from collect import ROOT,flatten,TRAIN,VALID
from build import ROUTES
ENV=ROOT.parent/'songdo_intersection'

def lane_segments(net,route):
    edges=set(route)
    for a,b in zip(route,route[1:]):
        for connections in net.getEdge(a).getOutgoing().values():
            for c in connections:
                if c.getTo().getID()!=b:continue
                via=c.getViaLaneID()
                while via:
                    lane=net.getLane(via);edges.add(lane.getEdge().getID())
                    outs=lane.getOutgoing()
                    via=outs[0].getViaLaneID() if outs else ''
    start=[];delta=[];length=[];offset=[];lanes=[]
    for name in sorted(edges):
        for lane in net.getEdge(name).getLanes():
            shape=np.array(lane.getShape());diff=np.diff(shape,axis=0)
            lens=np.linalg.norm(diff,axis=1);off=np.r_[0,np.cumsum(lens)]
            for i in range(len(lens)):
                if lens[i]<.001:continue
                start.append(shape[i]);delta.append(diff[i]);length.append(lens[i]);offset.append(off[i]);lanes.append(lane)
    return np.array(start),np.array(delta),np.array(length),np.array(offset),lanes

def convert(session):
    net=sumolib.net.readNet(str(ENV/'network/C.net.xml'),withInternal=True)
    segments={k:lane_segments(net,v) for k,v in ROUTES.items()}
    manifest=json.loads((ENV/'sessions/manifest.json').read_text())[session]
    records={r['id']:r for r in json.loads((ENV/'sessions'/session/'vehicles.json').read_text())}
    links=net.getTLS('C').getConnections()
    link={}
    for a,b,index in links:link[(a.getEdge().getID(),b.getEdge().getID())]=index
    df=pd.read_csv(ENV/'sessions'/session/'expert.csv.gz')
    scenes={}
    for vid,g in df.groupby('id',sort=False):
        g=g.sort_values('tick')
        xy=g[['x','y']].to_numpy();speeds=g.speed.to_numpy();ticks=g.tick.to_numpy()
        route=g.route.iloc[0];path=ROUTES[route];r=records[vid]
        start,delta,lens,offset,lanes=segments[route]
        for j in range(3,len(g)-1):
            if ticks[j]%5:continue
            fraction=np.clip(((xy[j]-start)*delta).sum(1)/(lens*lens),0,1)
            points=start+fraction[:,None]*delta
            k=int(np.argmin(np.linalg.norm(points-xy[j],axis=1)))
            lane=lanes[k];edge=lane.getEdge();name=edge.getID()
            forward=delta[k]/lens[k];left=np.array([-forward[1],forward[0]])
            actual=(xy[j]-xy[j-3])/.6
            heading=actual/max(np.linalg.norm(actual),1e-6) if speeds[j]>1 else forward
            lateral=float((xy[j]-points[k])@left)
            position=offset[k]+fraction[k]*lens[k]+r['length']/2
            signal=0;distance=100
            if name==path[0]:
                phase_time=(ticks[j]*.2+manifest['signal_fit']['offset'])%manifest['signal_fit']['cycle']
                for duration,state in manifest['signal_phases']:
                    if phase_time<duration:break
                    phase_time-=duration
                signal={'r':-1,'y':.5,'G':1,'g':1}.get(state[link[(path[0],path[1])]],0)
                distance=max(0,lane.getLength()-position)
            angle_a=np.array(net.getEdge(path[0]).getShape()[-1])-net.getEdge(path[0]).getShape()[0]
            angle_b=np.array(net.getEdge(path[-1]).getShape()[-1])-net.getEdge(path[-1]).getShape()[0]
            turn=math.atan2(angle_a[0]*angle_b[1]-angle_a[1]*angle_b[0],angle_a@angle_b)/math.pi
            if name in path:
                at=path.index(name)
                remaining=lane.getLength()-position+sum(net.getEdge(e).getLength() for e in path[at+1:])
            else:remaining=sum(net.getEdge(e).getLength() for e in path[1:])
            ego=[speeds[j]/20,(speeds[j]-speeds[j-1])/.2/7,lateral/(lane.getWidth()/2),
                 actual@left/.8,max(0,remaining)/500,turn,signal,distance/100,
                 lane.getIndex()/max(1,edge.getLaneNumber()-1),
                 float(lane.getIndex()<edge.getLaneNumber()-1),float(lane.getIndex()>0),
                 float(name.startswith(':')),r['length']/15,r['width']/3]
            a=(speeds[j+1]-speeds[j])/.2
            lat=0 if name.startswith(':') else (xy[j+1]-xy[j])@left/.2/.8
            scenes.setdefault(int(ticks[j]),[]).append((vid,xy[j],heading,speeds[j],ego,[a/(2 if a>=0 else 7),lat],r))
    x=[];y=[];episode=[]
    for tick,scene in scenes.items():
        for vid,xy,heading,speed,ego,act,r in scene:
            left=np.array([-heading[1],heading[0]])
            near=sorted([(np.linalg.norm(s[1]-xy),s) for s in scene if s[0]!=vid],key=lambda z:z[0])[:8]
            neighbors=np.zeros((8,8),dtype='f');mask=np.zeros(8,dtype='f')
            for i,(dist,s) in enumerate(near):
                if dist>60:continue
                delta=s[1]-xy;velocity=s[2]*s[3]-heading*speed
                neighbors[i]=[delta@heading/60,delta@left/60,velocity@heading/20,velocity@left/20,
                              s[2][0]*heading[1]-s[2][1]*heading[0],s[2]@heading,s[6]['length']/15,s[6]['width']/3]
                mask[i]=1
            x.append(flatten({'ego':np.clip(ego,-1,1),'neighbors':np.clip(neighbors,-1,1),'neighbor_mask':mask}))
            y.append(np.clip(act,-1,1));episode.append(int(vid[1:]))
    out=ROOT/'data';out.mkdir(exist_ok=True)
    np.savez_compressed(out/(session+'_human.npz'),observation=np.array(x,dtype='f'),
                        action_raw=np.array(y,dtype='f'),episode=episode)
    print(session,len(x),'recorded human samples',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);a=p.parse_args();convert(a.session)

