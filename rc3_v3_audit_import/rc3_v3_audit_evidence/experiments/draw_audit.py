"""Scientific audit figure from the independently reconstructed frozen metrics."""
import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
OUT=args.data_dir.resolve()


def rows(name):return list(csv.DictReader((OUT/name).open()))


def main():
    stats=rows('group_statistics.csv');comparisons=rows('paired_comparisons.csv');engineering=rows('engineering_pairs_recomputed.csv')
    lookup={(r['group'],r['config']):r for r in stats}
    colors={'fixed3':'#263e55','random':'#9b9e9f','detour':'#d78326','learned':'#236fbb'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.4),layout='constrained')
    ax=axes[0,0];scales=[12,36,64,100,144,196,256,400]
    for config,label,method in [('fixed3','Fixed3','fixed3'),('detour_calls_8','Detour, up to 8 calls','detour'),('learned_calls_8','Learned, up to 8 calls','learned')]:
        values=[float(lookup[f'tables_{n}',config]['planning_median_ms']) for n in scales]
        ax.plot(scales,values,'o-',color=colors[method],label=label,lw=1.7,ms=5)
    ax.set_yscale('log');ax.set_xlabel('Number of tables');ax.set_ylabel('Full planning median (ms, logarithmic)')
    ax.set_xticks(scales);ax.set_xticklabels(scales,rotation=45)
    ax.axvspan(230,415,color='#eaeef2',alpha=.8,zorder=-3)
    ax.text(315,3.6,'Stress only\nn = 3 / size',ha='center',fontsize=9)
    ax.set_xlim(0,415);ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',frameon=False,fontsize=9)
    ax.set_title('A  Per-size planning cost: 9 scenes / size through 196')
    ax=axes[0,1]
    anchor=lookup['primary_up_to_196','fixed3'];ax.scatter(float(anchor['planning_median_ms']),0,c=colors['fixed3'],s=65,zorder=4,label='Fixed3')
    for method,label in [('random','Random'),('detour','Detour'),('learned','Learned')]:
        selected=[lookup['primary_up_to_196',f'{method}_calls_{k}'] for k in (1,2,4,8)]
        xs=[float(r['planning_median_ms']) for r in selected];ys=[float(r['mean_improvement_pct']) for r in selected]
        ax.plot(xs,ys,'o-',color=colors[method],label=label,lw=1.7,ms=5)
        for x,y,k in zip(xs,ys,(1,2,4,8)):
            ax.annotate(str(k),(x,y),xytext=(3,5),textcoords='offset points',fontsize=8,color=colors[method])
    ax.set_xlabel('Full planning median across 54 main scenes (ms)');ax.set_ylabel('Mean cost reduction vs Fixed3 (%)')
    ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9);ax.set_title('B  Main-scene quality / time trade-off; labels = call cap')
    ax=axes[1,0]
    order=[('primary_up_to_196','Main, n = 54'),('in_distribution_100','100 tables, n = 9'),('smaller_transfer','12–64 tables, n = 27'),('larger_transfer','144–196 tables, n = 18'),('stress_over_196','256 / 400 tables, n = 6')]
    for i,(group,label) in enumerate(order):
        r=next(c for c in comparisons if c['group']==group and c['config']=='learned_calls_8' and c['reference']=='detour_calls_8')
        mean,lo,hi=(float(r[k]) for k in ('mean_reduction_pct','ci_low','ci_high'))
        ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color=colors['learned'],capsize=4,lw=1.6)
    ax.axvline(0,color='#444444',ls='--',lw=1);ax.set_yticks(range(len(order)));ax.set_yticklabels([v for _,v in order]);ax.invert_yaxis()
    ax.set_xlabel('Learned cost reduction vs Detour (%)\npositive = learned better; 20,000 scene-paired bootstrap 95% CI')
    ax.grid(axis='x',alpha=.15);ax.set_title('C  Eight-call learning advantage is not established')
    ax=axes[1,1]
    xs=np.array([float(r['old_planning_ms']) for r in engineering]);ys=np.array([float(r['new_planning_ms']) for r in engineering])
    ax.scatter(xs,ys,c=colors['fixed3'],s=35,alpha=.85)
    limit=max(xs.max(),ys.max())*1.08;ax.plot([0,limit],[0,limit],'--',color='#8a8a8a',lw=1,label='equal time')
    ax.set_xlim(0,limit);ax.set_ylim(0,limit);ax.set_xlabel('Old Fixed3 full planning (ms)');ax.set_ylabel('New Fixed3 full planning (ms)')
    ax.text(.05,.95,'18 / 18 faster, identical costs\nMedians: 141.56 → 77.36 ms\nMedian paired speedup: 1.802×',transform=ax.transAxes,va='top',fontsize=10,bbox={'boxstyle':'round,pad=.4','fc':'#f4f6f7','ec':'none'})
    ax.grid(alpha=.15);ax.set_title('D  Engineering gain: 18 validation scenes, no learning')
    fig.suptitle('V3 evidence audit — recorded results, independently recomputed',fontsize=16,weight='bold')
    fig.supxlabel('J = route length + 0.75 × service stops. Planning includes reference and final validation; preprocessing is separate.\nCIs omit training-seed uncertainty and multiplicity correction. Stress samples do not establish broad generalization.',fontsize=9)
    fig.savefig(OUT/'v3_evidence_audit.png',dpi=190)
    fig.savefig(OUT/'v3_evidence_audit.svg')


if __name__=='__main__':main()
