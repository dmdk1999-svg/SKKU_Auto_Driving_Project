"""Reconstruct the complete observed Songdo C junction from supplied trajectories."""
from pathlib import Path
import argparse
import json
import zipfile
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import sumo
import sumolib

ROOT=Path(__file__).resolve().parent
INCOMING=('1_2','2_2','3_3')
OUTGOING=('1_1','2_1','3_1','3_2')
ROUTES={
    '1_2>2_1':['1_2','2_1'],
    '1_2>3_2':['1_2','3_1','3_2'],
    '2_2>1_1':['2_2','1_1'],
    '2_2>3_2':['2_2','3_1','3_2'],
    '3_3>1_1':['3_3','1_1'],
    '3_3>2_1':['3_3','2_1'],
}
COLORS={'1_2':(50,150,240),'2_2':(250,160,40),'3_3':(170,80,220)}

def exe(name):
    return str(Path(sumo.__file__).parent/'bin'/(name+'.exe'))

def write_xml(path,root):
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def read(z,name):
    d=pd.read_csv(z.open(name),dtype={'Road_Section':str})
    d['t']=pd.to_timedelta(d.Local_Time).dt.total_seconds()
    d['t']-=d.t.min()
    return d

def geometry(d,polygons):
    xy=d[['Local_X','Local_Y']].to_numpy()
    pix=d[['Ortho_X','Ortho_Y']].to_numpy()
    idx=np.arange(0,len(d),10)
    affine=np.linalg.lstsq(np.c_[pix[idx],np.ones(len(idx))],xy[idx],rcond=None)[0]
    origin=np.floor(xy.min(axis=0)/10)*10
    result={}
    for section,p in polygons.groupby('Section'):
        corners=p[['tlx','tly','blx','bly','brx','bry','trx','try']].to_numpy().reshape(-1,2)
        quads=(np.c_[corners,np.ones(len(corners))]@affine-origin).reshape(-1,4,2)
        first=quads[:,[0,3]].mean(axis=(0,1))
        last=quads[:,[1,2]].mean(axis=(0,1))
        g=d[d.Road_Section==section].groupby('Vehicle_ID')
        delta=(g[['Local_X','Local_Y']].last()-g[['Local_X','Local_Y']].first()).to_numpy()
        delta=delta[np.linalg.norm(delta,axis=1)>5]
        direction=np.median(delta,axis=0)
        if (last-first)@direction<0: first,last=last,first
        u=(last-first)/np.linalg.norm(last-first)
        normal=np.array([-u[1],u[0]])
        width=float(np.median(abs((quads[:,3]-quads[:,0])@normal)))
        result[section]={'start':first.tolist(),'end':last.tolist(),'width':width,
                         'lanes':len(p),'polygons':quads.tolist()}
    inner=np.array([result[s]['end' if s in INCOMING else 'start']
                    for s in ('1_1','1_2','2_1','2_2','3_1','3_3')])
    center=inner.mean(axis=0)
    return result,origin,center

def make_network(geo,center):
    folder=ROOT/'network'; folder.mkdir(parents=True,exist_ok=True)
    nodes=ET.Element('nodes'); edges=ET.Element('edges'); connections=ET.Element('connections')
    ET.SubElement(nodes,'node',id='C',x=str(center[0]),y=str(center[1]),type='traffic_light')
    join=(np.array(geo['3_1']['end'])+geo['3_2']['start'])/2
    ET.SubElement(nodes,'node',id='east_split',x=str(join[0]),y=str(join[1]),type='priority')
    for name,g in geo.items():
        a,b=np.array(g['start']),np.array(g['end'])
        if name in INCOMING:
            src,dst=name+'_boundary','C'
            pts=[a,b]
            node=src; pos=a
        elif name=='3_1':
            src,dst='C','east_split'; pts=[a,join]; node=None
        elif name=='3_2':
            src,dst='east_split','3_exit'; pts=[join,b];node=dst;pos=b
        else:
            src,dst='C',name+'_boundary';pts=[a,b];node=dst;pos=b
        if node:
            ET.SubElement(nodes,'node',id=node,x=str(pos[0]),y=str(pos[1]),type='priority')
        # Remove coincident shape points before invoking netconvert.
        pts=[p for i,p in enumerate(pts) if not i or np.linalg.norm(p-pts[i-1])>.1]
        ET.SubElement(edges,'edge',id=name,**{'from':src,'to':dst},numLanes=str(g['lanes']),
                      width=str(g['width']),speed='16.67',spreadType='center',
                      shape=' '.join(f'{p[0]:.4f},{p[1]:.4f}' for p in pts))
    # Dataset lane 1 is the leftmost lane. SUMO lane 0 is the rightmost.
    links=[('1_2','2_1',0,0),('1_2','2_1',1,1),('1_2','2_1',2,2),
           ('1_2','3_1',3,1),('2_2','1_1',0,0),('2_2','1_1',1,1),
           ('2_2','1_1',2,2),('2_2','3_1',0,0),
           ('3_3','1_1',0,0),('3_3','1_1',1,1),('3_3','2_1',2,2),
           ('3_1','3_2',0,0),('3_1','3_2',0,1),('3_1','3_2',1,2)]
    for a,b,i,j in links:
        ET.SubElement(connections,'connection',**{'from':a,'to':b,'fromLane':str(i),'toLane':str(j)})
    for name,root in [('nodes',nodes),('edges',edges),('connections',connections)]:
        write_xml(folder/(name+'.xml'),root)
    subprocess.run([exe('netconvert'),'-n',str(folder/'nodes.xml'),'-e',str(folder/'edges.xml'),
                    '-x',str(folder/'connections.xml'),'-o',str(folder/'C.net.xml'),
                    '--offset.disable-normalization','true','--junctions.corner-detail','12',
                    '--junctions.internal-link-detail','20','--tls.allred.time','2',
                    '--no-turnarounds','true'],check=True)
    return sumolib.net.readNet(str(folder/'C.net.xml'),withInternal=True)

def project(point,shape):
    pts=np.array(shape); delta=np.diff(pts,axis=0)
    lengths=np.linalg.norm(delta,axis=1)
    fraction=np.clip(((point-pts[:-1])*delta).sum(axis=1)/np.maximum(lengths**2,1e-6),0,1)
    closest=pts[:-1]+fraction[:,None]*delta
    i=np.argmin(np.linalg.norm(closest-point,axis=1))
    return float(np.r_[0,np.cumsum(lengths)][i]+fraction[i]*lengths[i])

def extract(d,origin,geo,net):
    entries=[]; trajectories=[]; crossings=[]; excluded={}
    for vid,g in d.groupby('Vehicle_ID',sort=False):
        g=g.sort_values('t').drop_duplicates('t')
        if len(g)<60: excluded['short']=excluded.get('short',0)+1;continue
        if int(g.Vehicle_Class.mode().iloc[0]) not in (0,1,2):
            excluded['class']=excluded.get('class',0)+1;continue
        # Require stable labels for at least 0.5 seconds; suppress isolated section glitches.
        counts=g.groupby('Road_Section').size()
        stable=set(counts[counts>=15].index)
        before=g[g.Road_Section.isin(set(INCOMING)&stable)]
        after=g[g.Road_Section.isin(set(OUTGOING)&stable)]
        if before.empty or after.empty or before.t.iloc[0]>=after.t.iloc[-1]:
            excluded['incomplete_route']=excluded.get('incomplete_route',0)+1;continue
        src=before.Road_Section.iloc[0]; dst=after.Road_Section.iloc[-1]
        if dst=='3_1': dst='3_2'
        key=src+'>'+dst
        if key not in ROUTES:
            excluded['unsupported_turn']=excluded.get('unsupported_turn',0)+1;continue
        start=before[before.Road_Section==src].iloc[0]
        stop=before[before.Road_Section==src].iloc[-1]
        g=g[g.t>=start.t]
        if np.diff(g.t).max()>.3:
            excluded['tracking_gap']=excluded.get('tracking_gap',0)+1;continue
        edge=net.getEdge(src)
        raw_lane=int(stop.Lane_Number) if pd.notna(stop.Lane_Number) else 1
        lane=int(np.clip(geo[src]['lanes']-raw_lane,0,geo[src]['lanes']-1))
        next_edge=net.getEdge(ROUTES[key][1])
        allowed=[i for i,l in enumerate(edge.getLanes()) if any(
                 c.getToLane().getEdge()==next_edge for c in l.getOutgoing())]
        lane=min(allowed,key=lambda i:abs(i-lane))
        point=start[['Local_X','Local_Y']].to_numpy(dtype=float)-origin
        departpos=project(point,edge.getLane(lane).getShape())
        length=float(np.clip(g.Vehicle_Length.median(),3,14))
        width=float(np.clip(g.Vehicle_Width.median(),1.5,2.5))
        speed=g.Vehicle_Speed.interpolate(limit_direction='both').to_numpy()/3.6
        if not np.isfinite(speed).all(): continue
        vmax=float(np.clip(np.quantile(speed,.95),8,20))
        entries.append({'id':'v'+str(vid),'route':key,'depart':float(start.t),
                        'departPos':float(np.clip(departpos,length+.1,edge.getLength()-1)),
                        'departLane':lane,'departSpeed':float(np.clip(speed[0],0,min(vmax,16.67))),
                        'length':length,'width':width,'desiredSpeed':vmax})
        crossings.append({'id':'v'+str(vid),'route':key,'t':float(stop.t)})
        ticks=np.arange(np.ceil(g.t.iloc[0]/.2),np.floor(g.t.iloc[-1]/.2)+1,dtype=int)
        time=ticks*.2
        x=np.interp(time,g.t,g.Local_X)-origin[0]
        y=np.interp(time,g.t,g.Local_Y)-origin[1]
        v=np.interp(time,g.t,speed)
        for j,k in enumerate(ticks):
            trajectories.append((k,'v'+str(vid),x[j],y[j],v[j],key))
    return entries,trajectories,crossings,excluded

def signals(net,crossings,duration):
    # Conflict-free protected approach phases. Green proportions use observed demand.
    # No controller logs exist in the supplied files, so this is a reconstructed plan.
    groups={s:sum(c['route'].startswith(s) for c in crossings) for s in INCOMING}
    def group(route):
        return 2 if route.startswith('3_3') else 1 if route=='1_2>3_2' else 0
    times=[np.array([c['t'] for c in crossings if group(c['route'])==k]) for k in range(3)]
    demand=np.array([len(t) for t in times],dtype=float)
    demand/=max(1,demand.sum())
    best=None
    for cycle in range(90,181,5):
        green=np.round(12+(cycle-15-36)*demand).astype(int)
        green[0]+=cycle-15-int(green.sum())
        starts=np.r_[0,np.cumsum(green+5)[:-1]]
        for offset in range(0,cycle,2):
            scores=[np.mean(((t+offset-starts[k])%cycle)<green[k]) for k,t in enumerate(times) if len(t)]
            score=float(np.mean(scores))
            if best is None or score>best[0]:
                best=(score,green,offset,cycle)
    score,green,offset,cycle=best
    links=net.getTLS('C').getConnections()
    count=1+max(row[2] for row in links)
    phases=[]
    for k in range(3):
        state=['r']*count
        for lane,out,index in links:
            src=lane.getEdge().getID();dst=out.getEdge().getID()
            route=src+'>'+('3_2' if dst=='3_1' else dst)
            if group(route)==k: state[index]='G'
        phases.extend([(int(green[k]),''.join(state)),(3,''.join('y' if x=='G' else 'r' for x in state)),(2,'r'*count)])
    return phases,groups,{'offset':int(offset),'cycle':int(cycle),'balanced_crossing_agreement':score}

def routes_file(folder,entries):
    root=ET.Element('routes')
    for key,route in ROUTES.items():
        ET.SubElement(root,'route',id=key,edges=' '.join(route))
    for row in entries:
        ident=row['id']
        ET.SubElement(root,'vType',id=ident+'_type',length=f"{row['length']:.3f}",
                      width=f"{row['width']:.3f}",carFollowModel='IDM',laneChangeModel='SL2015',
                      accel='1.8',decel='3.5',emergencyDecel='7',tau='1.3',minGap='1.2',
                      maxSpeed=f"{row['desiredSpeed']:.3f}",speedFactor='1',speedDev='0',
                      actionStepLength='0.6',maxSpeedLat='0.8',lcAccelLat='0.7',
                      lcStrategic='1',lcCooperative='1',lcSpeedGain='0.3')
    for row in sorted(entries,key=lambda r:r['depart']):
        color=','.join(map(str,COLORS[row['route'].split('>')[0]]))
        ET.SubElement(root,'vehicle',id=row['id'],type=row['id']+'_type',route=row['route'],
                      depart=f"{row['depart']:.3f}",departPos=f"{row['departPos']:.3f}",
                      departLane=str(row['departLane']),departSpeed="max",color=color)
    write_xml(folder/'traffic.rou.xml',root)

def build(archive,segments):
    (ROOT/'sessions').mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(segments) as z:
        polygons=pd.read_csv(z.open('segmentations/C.csv'))
    summary={}
    with zipfile.ZipFile(archive) as z:
        names=sorted(n for n in z.namelist() if n.endswith('.csv') and '_C_' in n)
        first=read(z,names[0])
        geo,origin,center=geometry(first,polygons)
        net=make_network(geo,center)
        (ROOT/'network/geometry.json').write_text(json.dumps({'sections':geo,'origin':origin.tolist(),
                                                          'center':center.tolist()},indent=2))
        for name in names:
            tag=Path(name).stem.split('_')[-1]
            folder=ROOT/'sessions'/tag;folder.mkdir(exist_ok=True)
            d=first if name==names[0] else read(z,name)
            entries,tracks,crossings,excluded=extract(d,origin,geo,net)
            routes_file(folder,entries)
            phases,counts,fit=signals(net,crossings,d.t.max())
            add=ET.Element('additional')
            tls=ET.SubElement(add,'tlLogic',id='C',type='static',programID='reconstructed',offset=str(fit['offset']))
            for seconds,state in phases: ET.SubElement(tls,'phase',duration=str(seconds),state=state)
            write_xml(folder/'signals.add.xml',add)
            config=ET.Element('configuration')
            inputs=ET.SubElement(config,'input')
            for k,v in [('net-file','../../network/C.net.xml'),('route-files','traffic.rou.xml'),
                        ('additional-files','signals.add.xml')]:
                ET.SubElement(inputs,k,value=v)
            timing=ET.SubElement(config,'time')
            ET.SubElement(timing,'step-length',value='0.2')
            processing=ET.SubElement(config,'processing')
            for k,v in [('lateral-resolution','0.2'),('time-to-teleport','-1'),
                        ('step-method.ballistic','true'),('collision.action','warn'),
                        ('collision.check-junctions','true'),('collision.mingap-factor','0')]:
                ET.SubElement(processing,k,value=v)
            write_xml(folder/'scenario.sumocfg',config)
            pd.DataFrame(tracks,columns=['tick','id','x','y','speed','route']).to_csv(folder/'expert.csv.gz',index=False)
            (folder/'vehicles.json').write_text(json.dumps(entries))
            (folder/'crossings.json').write_text(json.dumps(crossings))
            summary[tag]={'source_vehicles':int(d.Vehicle_ID.nunique()),'accepted_routes':len(entries),
                          'excluded':excluded,'duration':float(d.t.max()),'approach_counts':counts,
                          'route_counts':pd.Series([r['route'] for r in entries]).value_counts().to_dict(),
                          'signal_phases':phases,'signal_fit':fit}
            print(tag,summary[tag],flush=True)
    (ROOT/'sessions/manifest.json').write_text(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',required=True);parser.add_argument('--segments',required=True)
    args=parser.parse_args()
    build(args.archive,args.segments)

