"""Reproduce actual batch-loss and gradient curves; no invented data points."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from figure_style import setup, save, COLORS, STYLES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
SEEDS = (11622026,11622027,11622028)

def main():
    setup();fig, axes = plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    out=[]
    for seed,color,style in zip(SEEDS,COLORS,STYLES):
        rows=[json.loads(s) for s in (RUN/f'seed_{seed}/S1/updates.jsonl').read_text().splitlines()]
        x=np.arange(1,len(rows)+1);loss=np.array([r['loss'] for r in rows])
        smooth=np.array([loss[max(0,i-15):i+1].mean() for i in range(len(loss))])
        axes[0].plot(x,loss,color=color,alpha=.18,lw=.6)
        axes[0].plot(x,smooth,color=color,linestyle=style,lw=1.6,label=str(seed))
        axes[1].plot(x,[r['gradient_norm_before_clip'] for r in rows],color=color,linestyle=style,lw=.85,alpha=.85,label=str(seed))
        for i,r in enumerate(rows):
            out.append(dict(seed=seed,update=i+1,batch_CE=r['loss'],trailing16_CE=float(smooth[i]),
                gradient_norm_before_clip=r['gradient_norm_before_clip'],gradient_norm_after_clip=r['gradient_norm_after_clip'],
                cumulative_stage_wall_s=r['stage_elapsed_s']))
    axes[0].set_ylabel('Batch route-mean masked CE')
    axes[1].set_ylabel('Gradient L2 norm before clipping');axes[1].set_yscale('log')
    axes[1].axhline(1,color='#555555',linestyle=':',lw=1,label='Clip threshold = 1')
    for i,ax in enumerate(axes):
        ax.set_xlabel('Scientific S1 update');ax.set_xlim(1,256);ax.grid(axis='y',alpha=.2)
        ax.text(-.12,1.02,chr(97+i)+')',transform=ax.transAxes,fontweight='bold')
        ax.legend(loc='upper right',frameon=False)
    save(fig,HERE/'s1b_training_curves')
    with (HERE/'training_curves.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)

if __name__=='__main__':main()
