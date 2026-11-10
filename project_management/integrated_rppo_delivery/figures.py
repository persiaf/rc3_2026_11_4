"""Scientific plots from closed CSV/JSON only; no model or planning execution."""
import csv
from rc3_integrated_rppo.common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':130,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    def table(name):
        with (HOME/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    rows=table('TRAINING_CURVES.csv');kl=table('POLICY_KL.csv');fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained')
    for seed in sorted({r['seed'] for r in rows if r['variant']=='main'}):
        rr=[r for r in rows if r['seed']==seed and r['variant']=='main'];x=[int(r['round']) for r in rr]
        y=np.array([float(r['H_J']) for r in rr]);axes[0].plot(x,y,alpha=.25,lw=.7)
        smooth=[float(y[max(0,i-15):i+1].mean()) for i in range(len(y))];axes[0].plot(x,smooth,label=seed)
        axes[2].plot(x,[float(r['elapsed_s']) for r in rr],label=seed,lw=1.)
    axes[0].set(ylabel='Training J after H',title='Varying training schedule: diagnostic, not a generalization comparison');axes[0].legend()
    kk=[r for r in kl if r.get('stage')=='main' and r.get('accepted')=='True']
    if kk:
        axes[1].plot([float(r['old_mean']) for r in kk],label='old -> new mean',lw=.8)
        axes[1].plot([float(r['old_p95']) for r in kk],label='old -> new p95',lw=.8)
        axes[1].plot([float(r['base_mean']) for r in kk],label='base -> current mean',lw=.8)
    for v in (.01,.05,.10):axes[1].axhline(v,color='gray',ls='--',lw=.6)
    axes[1].set(ylabel='Exact sampled-state KL',xlabel='Accepted epoch index');axes[1].legend()
    axes[2].set(xlabel='Rollout round',ylabel='Actual round time (s)')
    fig.savefig(HOME/'TRAINING_DIAGNOSTICS.png');plt.close(fig)
    result=read(HOME/'RESULT_SUMMARY.json');effects=result['effects'];refs=list(effects);y=[100*effects[c]['mean'] for c in refs]
    error=np.array([[100*(effects[c]['mean']-effects[c]['CI95'][0]) for c in refs],[100*(effects[c]['CI95'][1]-effects[c]['mean']) for c in refs]])
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.bar(refs,y,color=['#2c7fb8' if x>0 else '#d95f0e' for x in y]);ax.errorbar(refs,y,yerr=error,fmt='none',ecolor='black',capsize=4)
    ax.axhline(0,color='black',lw=.8);ax.axhline(.5,color='gray',ls='--',lw=.8);ax.set(ylabel='RL cost improvement (%)',title=result['scope']+'; physical-parent 95% interval')
    fig.savefig(HOME/'QUALITY_CONTRASTS.png');plt.close(fig)


if __name__=='__main__':main()
