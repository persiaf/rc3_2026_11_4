from .paper_plot_style import plt,COLORS,save
from .sources import report
def main():
    points=[0,1024,2048,4096];base=report('baseline_FrozenSL')['summary']['overall']['parent_equal_mean']
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    for ax,metric,label in zip(axes,['raw_J','J'],['Raw Actor cost J (lower is better)','After-H cost J (lower is better)']):
        for i,method in enumerate(['D5-PG','D5-SL-U']):
            values=[base[metric]]+[report(f'2026111101_{method}_{p}')['summary']['overall']['parent_equal_mean'][metric] for p in points[1:]]
            ax.plot(points,values,color=COLORS[i],marker=['o','s'][i],linestyle=['-','--'][i],label=method)
        ax.axhline(base[metric],color=COLORS[4],linestyle=':',label='FrozenSL')
        ax.set_xlabel('Optimizer updates');ax.set_ylabel(label);ax.set_xticks(points);ax.tick_params(axis='x',labelrotation=25);ax.legend(frameon=False)
    save(fig,'fixed_endpoint_quality')
if __name__=='__main__':main()
