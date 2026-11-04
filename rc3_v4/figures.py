"""Figures from frozen results; predetermined first blocked scenes, no best-case selection."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rc3_baseline.baseline import plot_scene


def main():
    root=Path('rc3_v4');folder=root/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    rows=list(csv.DictReader((root/'training/history.csv').open(encoding='utf-8-sig')))
    summary=json.loads((root/'results/summary.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for variant,color in [('control','#5276a7'),('structured','#bd6d28')]:
        selected=[r for r in rows if r['variant']==variant];epochs=sorted({int(r['epoch']) for r in selected})
        values=np.array([[float(r['val_gain_pct']) for r in selected if int(r['epoch'])==e] for e in epochs])
        axes[0].plot(epochs,values.mean(1),color=color,label=variant)
        axes[0].fill_between(epochs,values.mean(1)-values.std(1),values.mean(1)+values.std(1),color=color,alpha=.16)
    axes[0].set(xlabel='Training epoch',ylabel='Validation reduction from Fixed3 (%)',title='Mean +/- SD over 3 training seeds')
    axes[0].legend();axes[0].grid(alpha=.2)
    colors=['#606060','#32856b','#a47eb5','#5276a7','#bd6d28']
    for config,label,color in zip(('fixed3','detour8','v3_mlp8','control8','structured8'),
                                  ('Fixed3','Detour','v3 MLP (optimized)','Control','Structured'),colors):
        r=next(r for r in summary['summaries'] if r['group']=='main_100' and r['config']==config)
        x,y=r['median_planning_ms'],r['mean_improvement_pct']
        axes[1].scatter(x,y,color=color,s=65,label=label)
    axes[1].set(xlabel='Median full planning time (ms)',ylabel='Mean reduction from Fixed3 (%)',title='New 100-table test: 18 scenes, up to 8 calls')
    axes[1].legend(fontsize=8,loc='best');axes[1].grid(alpha=.2)
    fig.savefig(folder/'training_and_test.png',dpi=180);plt.close(fig)
    records=[]
    for tables in (100,400):
        for path in sorted((root/'results/plans').glob('scene_plan_*.json')):
            result=json.loads(path.read_text(encoding='utf-8'))
            if result['metadata']['tables']==tables and result['metadata']['mode']=='blocked':break
        fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
        for ax,config,label in zip(axs,('fixed3','detour8','structured8'),('Fixed3','Detour (8 calls)','Structured (8 calls)')):
            key=config+'_rep0';plan=result['plans'][key]
            plot_scene(ax,result,f'{label}\nJ={plan["objective_length_plus_stops_m"]:.2f}',method=key)
        fig.suptitle(f'{tables} tables | first predeclared blocked case | seed {result["metadata"]["seed"]}')
        handles=[Line2D([],[],color='#1565c0',label='Base path'),
                 Line2D([],[],color='#7542b8',marker='^',linestyle='',label='Service stop'),
                 Line2D([],[],color='#12866e',marker='o',linestyle='',label='Left-arm task'),
                 Line2D([],[],color='#b8512c',marker='o',linestyle='',label='Right-arm task'),
                 Patch(facecolor='#e4e9ef',edgecolor='#8fa4bb',label='Table'),
                 Line2D([],[],color='#d59b20',marker='*',linestyle='',label='Start / return')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=6,frameon=False)
        fig.savefig(folder/f'paths_{tables}.png',dpi=160,bbox_inches='tight');plt.close(fig)
        records.append(dict(tables=tables,path=str(path),seed=result['metadata']['seed'],selection='first blocked case in frozen metadata order'))
    (folder/'selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(folder)


if __name__=='__main__':main()
