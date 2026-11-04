"""All 24 paired cases. Positive values favor the final PPO checkpoint."""
from paper_plot_style import *

def main():
    raw=rows('final_fixed_work');ppo=group(raw,'ppo_final');data=[]
    methods=['bc_frozen','bc_continued_2048','detour_memory','stop']
    fig,axes=plt.subplots(2,4,figsize=(7.6,4.6),sharey='row',sharex=True,layout='constrained')
    modes=['regular','mixed','blocked'];markers=['o','s','^']
    for j,method in enumerate(methods):
        comparator=group(raw,method)
        for k,(a,b) in enumerate(zip(ppo,comparator)):
            if (a['case_id'],a['seed'])!=(b['case_id'],b['seed']):raise ValueError('Pair mismatch')
            m=modes.index(a['mode']);ret=a['total_reward']-b['total_reward'];cost=100*(b['J']-a['J'])/b['J']
            for i,value in enumerate((ret,cost)):
                axes[i,j].scatter(k+1,value,s=14,marker=markers[m],facecolor='none',edgecolor=COLORS[m],linewidth=.9,
                                  label=a['mode'] if j==0 and i==0 and k<3 else None)
            data.append(dict(comparator=method,case_id=a['case_id'],seed=a['seed'],mode=a['mode'],
                             ppo_return_minus_comparator=ret,ppo_cost_reduction_percent=cost))
        for i in range(2):
            axes[i,j].axhline(0,color='#777777',linewidth=.7);axes[i,j].set_xticks([1,12,24]);panel(axes[i,j],chr(97+i*4+j))
        axes[0,j].text(.5,1.04,NAMES[method],transform=axes[0,j].transAxes,ha='center',fontsize=10)
        axes[1,j].set_xlabel('Validation case')
    axes[0,0].set_ylabel('PPO return advantage');axes[1,0].set_ylabel('PPO cost reduction (%)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    save(fig,'f2_paired',data,[EVAL/'final_fixed_work/results.json'])
if __name__=='__main__':main()
