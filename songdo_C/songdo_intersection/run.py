"""Play or audit responsive Songdo traffic; no recorded-position forcing."""
import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from build import ROOT,exe

def command(session,gui=False,seed=1,report=None):
    args=[exe('sumo-gui' if gui else 'sumo'),'-c',str(ROOT/'sessions'/session/'scenario.sumocfg'),
          '--step-length','.2','--lateral-resolution','.2','--seed',str(seed),
          '--time-to-teleport','-1','--collision.action','warn','--collision.check-junctions','true',
          '--collision.mingap-factor','0','--no-step-log','true','--duration-log.statistics',str(bool(report)).lower(),
          '--step-method.ballistic','true']
    if gui:
        args+=['--start','--delay','80','--gui-settings-file',str(ROOT/'network/view.xml')]
    else:
        args+=['--end',str(json.loads((ROOT/'sessions/manifest.json').read_text())[session]['duration']+1200)]
    if report:
        args+=['--tripinfo-output',str(report/'trips.xml'),'--statistic-output',str(report/'statistics.xml'),
               '--collision-output',str(report/'collisions.xml'),'--error-log',str(report/'warnings.log')]
    return args

def audit(session,seed=1):
    report=ROOT/'validation'/f'{session}_{seed}';report.mkdir(parents=True,exist_ok=True)
    p=subprocess.run(command(session,seed=seed,report=report),capture_output=True,text=True,
                     encoding='utf8',errors='replace',timeout=240)
    (report/'console.txt').write_text(p.stdout+p.stderr,encoding='utf8')
    if p.returncode: raise RuntimeError(p.stderr)
    manifest=json.loads((ROOT/'sessions/manifest.json').read_text())[session]
    trips=ET.parse(report/'trips.xml').findall('tripinfo')
    stat=ET.parse(report/'statistics.xml')
    collisions=ET.parse(report/'collisions.xml').findall('collision')
    waiting=[float(v.get('waitingTime')) for v in trips]
    vehicles={r['id']:r for r in json.loads((ROOT/'sessions'/session/'vehicles.json').read_text())}
    passed={}
    for v in trips:
        r=vehicles[v.get('id')]['route'];passed[r]=passed.get(r,0)+1
    result={'session':session,'seed':seed,'expected':manifest['accepted_routes'],'arrived':len(trips),
            'collisions':len(collisions),'teleports':stat.find('teleports').attrib,
            'route_arrivals':passed,'stopped_vehicles':sum(w>.2 for w in waiting),
            'mean_wait_s':sum(waiting)/max(1,len(waiting)),
            'max_depart_delay_s':max(float(v.get('departDelay')) for v in trips)}
    (report/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',default='AM3');p.add_argument('--seed',type=int,default=1)
    p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.audit: audit(a.session,a.seed)
    else: subprocess.run(command(a.session,True,a.seed),check=True)

