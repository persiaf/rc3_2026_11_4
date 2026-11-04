"""Reproduce three bounded offline diagnostic curves from saved raw rows."""
import json
from paper_plot_style import *

def series(run,key,positive=False):
    x=[];y=[]
    for path in sorted((OUT/run).glob('diagnostic_*.json')):
        rows=json.loads(path.read_text(encoding='utf-8'))['rows']
        rows=[r for r in rows if r['is_root'] and (not positive or r['known_positive'])]
        x.append(int(path.stem.split('_')[-1]));y.append(100*sum(r[key] for r in rows)/len(rows))
    return x,y

fig,axes=plt.subplots(1,3,figsize=(11.2,3.2),layout='constrained')
axes[0].plot(*series('toy16','evidence_greedy_near_best_hit'),color='#0072B2',linestyle='-',label='Near-best (16)')
axes[0].plot(*series('toy16','evidence_known_positive_continue',True),color='#D55E00',linestyle='--',label='Positive continue (8)')
axes[0].set_ylabel('Queried-action success (%)');axes[0].set_xlabel('Toy optimizer updates')
for run,label,color,ls in [('sampling_original','Original sampling','#0072B2','-'),('sampling_root_half','Root-half sampling','#D55E00','--')]:
    axes[1].plot(*series(run,'full_known_positive_continue',True),label=label,color=color,linestyle=ls)
axes[1].set_ylabel('Full-action known positive (%)');axes[1].set_xlabel('Sampling-control updates')
for run,label,color,ls in [('gain_control','Continued CE','#0072B2','-'),('gain_gain','CE + gain regression','#D55E00','--')]:
    axes[2].plot(*series(run,'full_known_positive_continue',True),label=label,color=color,linestyle=ls)
axes[2].set_ylabel('Full-action known positive (%)');axes[2].set_xlabel('Additional matched updates')
for ax in axes:
    ax.set_ylim(-2,105);ax.legend(loc='lower right',frameon=False);ax.grid(axis='y',alpha=.15)
axes[1].legend(loc='upper left',frameon=False)
save(fig,'training_curves')
