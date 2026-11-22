from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sumolib
from build import ROOT

def draw():
    geo=json.loads((ROOT/'network/geometry.json').read_text())
    net=sumolib.net.readNet(str(ROOT/'network/C.net.xml'),withInternal=True)
    fig,ax=plt.subplots(figsize=(11,10))
    for name,g in geo['sections'].items():
        for poly in g['polygons']:
            p=np.array(poly+[poly[0]])
            ax.fill(p[:,0],p[:,1],color='#b9ccd6',alpha=.4)
        mid=(np.array(g['start'])+g['end'])/2
        ax.text(*mid,name,fontsize=12,color='#146987')
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            p=np.array(lane.getShape())
            internal=edge.getID().startswith(':')
            ax.plot(p[:,0],p[:,1],color='#d25f20' if internal else '#273f4b',
                    linewidth=1.3 if internal else 1,alpha=.8)
    ax.set_aspect('equal');ax.set_xlabel('Local east [m]');ax.set_ylabel('Local north [m]')
    ax.set_title('Songdo C - all seven road sections and internal junction lanes')
    ax.grid(alpha=.15)
    (ROOT/'validation').mkdir(exist_ok=True)
    fig.savefig(ROOT/'validation/network.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__': draw()

