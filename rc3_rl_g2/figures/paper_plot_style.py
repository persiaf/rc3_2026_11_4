"""Shared style, input provenance and exact data export for the G2 pilot."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
EVAL = ROOT/'outputs/final_evaluation'
COLORS = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#777777','#332288','#882255']
NAMES = {'ppo_final':'PPO 32k','bc_frozen':'Frozen BC','bc_continued_2048':'Continued BC',
         'original_fixed3':'Fixed3','original_detour':'Original Detour','stop':'Menu STOP',
         'detour_memory':'Detour + memory','ils':'ILS','random':'Random'}
ORDER = list(NAMES)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'], 'font.size':10,
    'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':8,'figure.dpi':120,'savefig.dpi':300,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'mathtext.fontset':'stix'})

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(name):
    data=read(EVAL/name/'results.json')
    if any(r['status']!='complete' for r in data):
        raise ValueError('Figure requires explicit review of failed rows, not silent filtering')
    return data

def group(data,method):
    selected=sorted([r for r in data if r['method']==method],key=lambda r:int(r['case_id']))
    if len(selected)!=24 or len({(r['case_id'],r['seed'],r['mode']) for r in selected})!=24:
        raise ValueError('Expected all 24 unique validation cases')
    return selected

def save(fig,name,data,sources):
    OUT.mkdir(exist_ok=True)
    for extension in ('pdf','svg','png'):
        fig.savefig(OUT/(name+'.'+extension),bbox_inches='tight',pad_inches=.08)
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader();writer.writerows(data)
    provenance={str(Path(p).relative_to(ROOT)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    (OUT/(name+'.sources.json')).write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    plt.close(fig)

def panel(ax,letter):
    ax.text(0,1.03,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
