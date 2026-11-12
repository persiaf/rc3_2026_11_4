import json,numpy as np
from .paper_plot_style import plt,COLORS,save,HERE
from .sources import report,record
def main():
    pg=report('2026111101_D5-PG_4096');baseline={r['case_id']:r for r in report('baseline_FrozenSL')['rows']};bins={'improved':[],'tied':[],'degraded':[]};eps=1e-8
    for r in sorted(pg['rows'],key=lambda x:x['case_id']):
        delta=r['metrics']['J']-baseline[r['case_id']]['metrics']['J'];kind='improved' if delta<-eps else 'degraded' if delta>eps else 'tied';bins[kind].append(r)
    selected={k:(v[0] if v else None) for k,v in bins.items()};fig,axes=plt.subplots(2,3,figsize=(7.1,4.5),layout='constrained')
    for col,(kind,row) in enumerate(selected.items()):
        if row is None:
            for ax in axes[:,col]:ax.set_axis_off();ax.text(.5,.5,'NOT_AVAILABLE',ha='center',va='center',transform=ax.transAxes,fontsize=9)
            continue
        a=record(row);b=record(baseline[row['case_id']])
        for r,(key,mode) in enumerate([('raw_plan','raw'),('plan','H')]):
            ax=axes[r,col]
            for data,label,color,style in [(b,'FrozenSL',COLORS[0],'-'),(a,'D5-PG',COLORS[1],'--')]:
                xy=np.asarray(data[key]['route_polyline']);ax.plot(xy[:,0],xy[:,1],color=color,linestyle=style,linewidth=.9,label=label)
            xy=np.asarray(a[key]['route_polyline']);ax.scatter(xy[0,0],xy[0,1],marker='o',s=14,c='black');ax.scatter(xy[-1,0],xy[-1,1],marker='s',s=12,c='black')
            ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('x (m)');ax.set_ylabel(('Raw' if r==0 else 'After H')+' y (m)');ax.tick_params(labelsize=7)
    axes[0,0].legend(frameon=False,fontsize=7)
    save(fig,'saved_route_examples')
    data=dict(rule='Lexicographically first case per H-J difference sign; absolute tie tolerance1e-8; all66 cases retained in statistics',
        sign_counts={k:len(v) for k,v in bins.items()},selected={k:None if r is None else dict(case_id=r['case_id'],physical_key=r['physical_key'],PG_metrics=r['metrics'],FrozenSL_metrics=baseline[r['case_id']]['metrics']) for k,r in selected.items()},
        new_planning_or_model_calls=0,scope='Saved navigation polylines only; not certified arm joint trajectories')
    with (HERE/'ROUTE_EXAMPLE_SELECTION.json').open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
