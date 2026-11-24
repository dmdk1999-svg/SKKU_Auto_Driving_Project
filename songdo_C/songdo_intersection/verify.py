"""Integration checks for geometry, signals, responsive NPCs and the Gym API."""
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import libsumo as t
import sumolib
from build import ROOT,ROUTES
from run import command
from environment import SongdoIntersectionEnv

def geometry():
    net=sumolib.net.readNet(str(ROOT/'network/C.net.xml'),withInternal=True)
    max_join=0
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            for c in lane.getOutgoing():
                via=c.getViaLaneID()
                following=net.getLane(via) if via else c.getToLane()
                gap=np.linalg.norm(np.array(lane.getShape()[-1])-following.getShape()[0])
                max_join=max(max_join,float(gap))
    for route in ROUTES.values():
        for a,b in zip(route,route[1:]):
            assert net.getEdge(b) in net.getEdge(a).getOutgoing(),(a,b)
    assert max_join<.1,max_join
    # No conflicting connections receive protected green at the same time.
    tree=ET.parse(ROOT/'network/C.net.xml')
    junction=next(j for j in tree.findall('junction') if j.get('id')=='C')
    requests={int(r.get('index')):r.get('foes')[::-1] for r in junction.findall('request')}
    for path in (ROOT/'sessions').glob('*/signals.add.xml'):
        for phase in ET.parse(path).findall('.//phase'):
            green=[i for i,s in enumerate(phase.get('state')) if s=='G']
            for i in green:
                for j in green:
                    if i!=j:assert requests[i][j]=='0',(path,i,j)
    return {'max_lane_join_gap_m':max_join,'routes_connected':len(ROUTES)}

def traffic():
    args=command('AM3')
    args+=['--no-warnings','true']
    t.start(args)
    previous={};red_stops=set();green_departures=set();internal=set()
    intervention=None;min_follow_speed=100;max_jump_error=0;hits=0;teleports=0
    try:
        while t.simulation.getMinExpectedNumber()>0 and t.simulation.getTime()<1800:
            now=t.simulation.getTime()
            ids=t.vehicle.getIDList()
            if intervention is None and now>30:
                for vid in ids:
                    if t.vehicle.getRoadID(vid)!='2_2':continue
                    s=t.vehicle.getLanePosition(vid)
                    remaining=t.lane.getLength(t.vehicle.getLaneID(vid))-s
                    follower,gap=t.vehicle.getFollower(vid,40)
                    if follower and 12<gap<35 and remaining>65 and t.vehicle.getSpeed(follower)>5:
                        intervention={'leader':vid,'follower':follower,'t':now,
                                      'initial_follower_speed':t.vehicle.getSpeed(follower)}
                        t.vehicle.setStop(vid,'2_2',pos=s+30,laneIndex=t.vehicle.getLaneIndex(vid),duration=12)
                        break
            if intervention and now<intervention['t']+25 and intervention['follower'] in ids:
                min_follow_speed=min(min_follow_speed,t.vehicle.getSpeed(intervention['follower']))
            for vid in ids:
                xy=np.array(t.vehicle.getPosition(vid));speed=t.vehicle.getSpeed(vid)
                if vid in previous:
                    prev,old_speed=previous[vid]
                    distance=np.linalg.norm(xy-prev)
                    max_jump_error=max(max_jump_error,distance-max(speed,old_speed)*.2)
                previous[vid]=(xy,speed)
                if t.vehicle.getRoadID(vid).startswith(':C'): internal.add(vid)
                lights=t.vehicle.getNextTLS(vid)
                if lights and lights[0][3].lower()=='r' and lights[0][2]<25 and speed<.2:
                    red_stops.add(vid)
                if vid in red_stops and lights and lights[0][3].lower()=='g' and speed>2:
                    green_departures.add(vid)
            t.simulationStep()
            hits+=len(t.simulation.getCollidingVehiclesIDList())
            teleports+=len(t.simulation.getStartingTeleportIDList())
        remaining=t.simulation.getMinExpectedNumber()
    finally:t.close()
    assert intervention is not None,'No responsive follower case found'
    assert min_follow_speed<intervention['initial_follower_speed']-1
    assert len(green_departures)>5
    assert hits==0 and teleports==0 and remaining==0
    assert max_jump_error<.2,max_jump_error
    return {'red_light_stops':len(red_stops),'green_restarts':len(green_departures),
            'vehicles_using_internal_junction':len(internal),'max_step_displacement_excess_m':max_jump_error,
            'intervention':intervention,'intervention_follower_min_speed':min_follow_speed,
            'colliding_vehicle_events':hits,'teleports':teleports,'remaining':remaining}

def gym_check():
    from gymnasium.utils.env_checker import check_env
    e=SongdoIntersectionEnv('AM1')
    routes=[]
    try:
        check_env(e,skip_render_check=True)
        for k,route in enumerate(ROUTES):
            obs,info=e.reset(seed=100+k,options={'route':route})
            assert e.observation_space.contains(obs)
            for _ in range(10):
                obs,reward,done,truncated,info=e.step(np.array([-.2,0],dtype=np.float32))
                assert e.observation_space.contains(obs) and np.isfinite(reward)
                if done or truncated:break
            routes.append(route)
    finally:e.close()
    return {'gymnasium_checker':'passed','controlled_routes':routes}

if __name__=='__main__':
    result={'geometry':geometry(),'traffic':traffic(),'gym':gym_check()}
    (ROOT/'validation/integration.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

