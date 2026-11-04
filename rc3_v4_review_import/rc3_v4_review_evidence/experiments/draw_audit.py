"""Draw scientific figure from independently reconstructed v4 results."""
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent);args=p.parse_args()
    root=args.data_dir
    def rows(name):return list(csv.DictReader((root/name).open()))
    stats=rows('group_statistics.csv');comp=rows('paired_comparisons.csv')
    s={v['config']:v for v in stats if v['group']=='main_100'}
    c={(v['config'],v['reference']):v for v in comp if v['group']=='main_100'}
    colors={'fixed3':'#34495e','detour8':'#df8927','v3_mlp8':'#168468','control8':'#979b9c','structured8':'#3c67b7'}
    names={'fixed3':'Fixed3','detour8':'Detour','v3_mlp8':'Frozen v3 MLP','control8':'Control','structured8':'Structured'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,axs=plt.subplots(2,2,figsize=(13.5,9.3),layout='constrained')
    ax=axs[0,0]
    for k in names:
        v=s[k];x=float(v['median_planning_ms']);y=float(v['mean_improvement_pct'])
        ax.scatter(x,y,s=75,color=colors[k],zorder=5)
        offsets={'fixed3':(5,8),'detour8':(-42,-20),'v3_mlp8':(-62,12),'control8':(-42,-22),'structured8':(-62,10)}
        ax.annotate(names[k],(x,y),xytext=offsets[k],textcoords='offset points',fontsize=10)
    ax.set(xlabel='Full planning median (ms)',ylabel='Mean cost reduction vs Fixed3 (%)',xlim=(70,470),ylim=(-.08,1.62))
    ax.set_title('A  Quality / time on 18 new 100-table scenes')
    ax.grid(alpha=.2)
    ax=axs[0,1]
    specs=[('structured8','detour8','Structured / Detour'),('structured8','control8','Structured / Control'),
           ('structured8','v3_mlp8','Structured / v3 MLP'),('v3_mlp8','detour8','v3 MLP / Detour')]
    for i,(a,b,label) in enumerate(specs):
        v=c[a,b];m=float(v['mean_reduction_pct']);lo=float(v['ci_low']);hi=float(v['ci_high'])
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=colors[a],lw=1.8,ms=6)
    ax.axvline(0,color='#555',ls='--',lw=1)
    ax.set_yticks(range(len(specs)));ax.set_yticklabels([v[2] for v in specs]);ax.invert_yaxis()
    ax.set_xlabel('Paired cost reduction (%) with 95% scene CI')
    ax.set_title('B  Distinguish learned benefit from input benefit')
    ax.grid(axis='x',alpha=.2)
    ax=axs[1,0]
    seeds=[17,29,43];x=np.arange(3);width=.32
    for offset,variant,color in [(-width/2,'control',colors['control8']),(width/2,'structured',colors['structured8'])]:
        vals=[float(s[f'{variant}_seed{seed}_calls8']['mean_improvement_pct']) for seed in seeds]
        ax.bar(x+offset,vals,width,color=color,label=variant.title())
        for xx,val in zip(x+offset,vals):ax.text(xx,val+.025,f'{val:.3f}',ha='center',fontsize=9)
    ax.set_xticks(x);ax.set_xticklabels(seeds);ax.set_xlabel('Training seed; checkpoints selected on old validation set')
    ax.set_ylabel('Mean cost reduction vs Fixed3 (%)');ax.set_ylim(0,1.75)
    ax.set_title('C  Positive direction across all three seeds')
    ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.2)
    ax.text(.99,.04,'Same 18 scenes; one run per seed.\nThese are not 54 independent scenes.',transform=ax.transAxes,ha='right',fontsize=9)
    ax=axs[1,1]
    labels=['Detour','Control','Structured'];configs=['detour_wall400','control_wall400','structured_wall400']
    vals=[float(s[k]['mean_improvement_pct']) for k in configs]
    ax.bar(range(3),vals,color=[colors['detour8'],colors['control8'],colors['structured8']],width=.55)
    for i,k in enumerate(configs):
        v=s[k]
        ax.text(i,vals[i]+.03,f"{vals[i]:.3f}%\n{float(v['median_planning_ms']):.1f} ms\n{float(v['mean_dp_calls']):.1f} repairs",ha='center',fontsize=9)
    ax.set_xticks(range(3));ax.set_xticklabels(labels);ax.set_ylim(0,2.05)
    ax.set_ylabel('Mean cost reduction vs Fixed3 (%)')
    ax.set_title('D  Similar actual time: Detour remains competitive')
    ax.grid(axis='y',alpha=.2)
    ax.text(.5,.03,'400 ms soft cap; 54 / 54 overruns per method.\nFrozen v3 MLP was not evaluated at this cap.',transform=ax.transAxes,ha='center',fontsize=9)
    fig.suptitle('V4 audit — useful quality signal, deployment gate not met',fontweight='bold',fontsize=17)
    fig.supxlabel('Recorded Windows timings; common preprocessing is separate. J = route length + 0.75 × service stops.\n20,000 scene-paired bootstrap samples; CIs omit training-seed uncertainty and multiplicity correction.',fontsize=9)
    fig.savefig(root/'v4_evidence_audit.png',dpi=170)
    fig.savefig(root/'v4_evidence_audit.svg')
    plt.close(fig)

if __name__=='__main__':main()
