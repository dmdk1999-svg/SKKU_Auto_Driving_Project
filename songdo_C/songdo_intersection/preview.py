"""Capture the actual responsive SUMO GUI without altering vehicle positions."""
import traci
from build import ROOT
from run import command

def capture():
    args=command('AM3',gui=True)
    args[args.index('--delay')+1]='0'
    args+=['--quit-on-end']
    traci.start(args)
    try:
        traci.gui.setBoundary('View #0',-50,-70,155,150)
        while traci.simulation.getTime()<100:
            traci.simulationStep()
        traci.gui.screenshot('View #0',str(ROOT/'validation/sumo_intersection.png'),width=1200,height=1100)
        traci.simulationStep()
    finally:traci.close()
if __name__=='__main__':capture()

