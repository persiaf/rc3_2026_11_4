"""Fresh complete planning time; frozen originals have no synthetic G2 work."""
from paper_plot_style import *

def main():
    datasets=['final_fixed_work','final_soft_time_0400ms'];data=[]
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.8),layout='constrained',sharey=True)
    markers=['o','s','D','^','v','P','X','<','>']
    for i,name in enumerate(datasets):
        raw=rows(name);ax=axes[i]
        for j,method in enumerate(ORDER):
            gg=group(raw,method);timing=1000*np.median([r['complete_s'] for r in gg]);gain=np.mean([r['quality_gain_percent'] for r in gg])
            ax.scatter(timing,gain,s=60,facecolors='none',edgecolors=COLORS[j],marker=markers[j],label=NAMES[method],linewidth=1.2)
            for r in gg:data.append(dict(dataset=name,method=method,case_id=r['case_id'],seed=r['seed'],mode=r['mode'],
                complete_ms=1000*r['complete_s'],quality_gain_percent=r['quality_gain_percent'],time_overrun_ms=1000*r['time_overrun_s']))
        if i==1:ax.axvline(400,color='#888888',linestyle=':',linewidth=.8,label='400 ms soft budget')
        ax.set_xlabel('Median complete planning time (ms)');panel(ax,chr(97+i));ax.set_ylim(bottom=-.03)
    axes[0].set_ylabel('Mean cost reduction from Fixed3 (%)')
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=5,frameon=False)
    save(fig,'f3_quality_time',data,[EVAL/d/'results.json' for d in datasets])
if __name__=='__main__':main()
