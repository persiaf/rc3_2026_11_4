import statistics
from .paper_plot_style import plt,COLORS,save
from .sources import table,report
def main():
    rows=[r for r in table('PARENT_PAIRED_RESULTS.csv') if r['tier']=='development' and r['endpoint']=='4096' and r['method']=='D5-PG' and r['metric']=='J']
    parents={r['physical_key']:r['group'] for r in rows};keys=sorted(parents,key=lambda k:(parents[k].startswith('independent'),k))
    labels=[];counts={True:0,False:0}
    for k in keys:
        dev=parents[k].startswith('independent');counts[dev]+=1;labels.append(('DEV' if dev else 'SEL')+str(counts[dev]).zfill(2))
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.6),sharey=True,layout='constrained')
    for i,(ax,control,target) in enumerate(zip(axes,['FrozenSL','D5-SL-U'],[1.,.5])):
        lookup={r['physical_key']:float(r['relative_gain'])*100 for r in rows if r['control']==control};values=[lookup[k] for k in keys]
        ax.scatter(values,range(len(keys)),color=COLORS[i],marker=['o','s'][i]);ax.axvline(0,color='.45',linewidth=.8)
        ax.axvline(statistics.mean(values),color=COLORS[i],linestyle='--',label='Parent mean');ax.axvline(target,color=COLORS[2],linestyle=':',label='P1 requirement')
        ax.set_xlabel(f'H-J gain vs {control} (%)');ax.set_yticks(range(len(keys)),labels);ax.legend(frameon=False,loc='lower left')
    axes[0].invert_yaxis()
    save(fig,'parent_paired_HJ')
if __name__=='__main__':main()
