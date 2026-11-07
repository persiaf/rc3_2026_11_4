"""Paired fixed-endpoint teacher-prefix diagnostics, all 256 training routes."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from figure_style import setup, save, COLORS, STYLES

HERE=Path(__file__).resolve().parent

def main():
    setup();data=json.loads((HERE.parent/'TEACHER_STATE_DIAGNOSTICS.json').read_text())['groups']
    seeds=sorted({r['seed'] for r in data})
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for seed,color,style in zip(seeds,COLORS,STYLES):
        pair=[next(r for r in data if r['seed']==seed and r['endpoint']==ep) for ep in ['initial','SL256']]
        axes[0].plot([0,1],[r['route_macro_CE'] for r in pair],marker='o',color=color,linestyle=style,label=str(seed))
        axes[1].plot([0,1],[r['top1'] for r in pair],marker='o',color=color,linestyle=style,label=str(seed))
    axes[0].set_ylabel('All-teacher route-mean masked CE')
    axes[1].set_ylabel('Teacher action top-1 agreement')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylim(0,1)
    for i,ax in enumerate(axes):
        ax.set_xticks([0,1],['Untrained','SL256']);ax.set_xlim(-.15,1.15);ax.grid(axis='y',alpha=.2)
        ax.text(-.12,1.02,chr(97+i)+')',transform=ax.transAxes,fontweight='bold');ax.legend(frameon=False)
    save(fig,HERE/'s1b_teacher_agreement')

if __name__=='__main__':main()
