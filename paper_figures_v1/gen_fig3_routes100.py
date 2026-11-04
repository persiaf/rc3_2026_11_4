"""Five frozen Fixed8 route panels; no planning, inference or timing runs."""
from shared import *
from matplotlib.patches import Rectangle


DISPLAY_ID='first_main100_fixed_call_case'
METHODS=(('fixed3_rep0','Fixed3'),('detour8_rep0','Detour8'),
         ('v3_mlp8_rep0','旧 MLP8'),('control8_rep0','Control8'),
         ('structured8_rep0','Structured8'))


def frozen_case():
    item=display(DISPLAY_ID);data=source('main100');scene=resolve(item['scene'])
    assert data['metadata']['seed']==item['seed']==1565649625
    assert len(scene['tables'])==100 and len(scene['tasks'])==158 and len(scene['candidates'])==800
    rows=[]
    for label,name in METHODS:
        config=data['runs'][label]['config']
        assert label.endswith('_rep0') and config['kind']=='calls' and config['seconds'] is None
        assert config['horizon']==(0 if label=='fixed3_rep0' else 8)
        plan=data['plans'][label]
        rows.append({'label':label,'name':name,'plan':plan,'summary':cost_summary(plan),
                     'config':config,'plan_pointer':'/plans/'+label})
    # The central 20% on each dimension is fixed from the saved navigation
    # envelope, independently of any method's route or objective.
    x0,x1=min(scene['x_lines']),max(scene['x_lines'])
    y0,y1=min(scene['y_lines']),max(scene['y_lines'])
    cx,cy=(x0+x1)/2,(y0+y1)/2
    crop=(cx-.1*(x1-x0),cx+.1*(x1-x0),cy-.1*(y1-y0),cy+.1*(y1-y0))
    return item,scene,rows,scene_bounds(scene,pad=.6),crop


def panel_layout(scene,rows,bounds,crop,zoom=False):
    fig,axes=plt.subplots(3,2,figsize=(7.1,8.4))
    fig.subplots_adjust(left=.082,right=.985,bottom=.052,top=.975,wspace=.27,hspace=.39)
    for number,(ax,row) in enumerate(zip(axes.flat,rows)):
        plan=row['plan']
        draw_scene(ax,scene,bounds=bounds)
        draw_route(ax,plan,lw=1.05 if zoom else .8)
        draw_tasks(ax,scene,plan,size=25 if zoom else 8)
        draw_stops(ax,scene,plan,size=28 if zoom else 9,headings=zoom)
        if zoom:
            ax.set_xticks([9,10,11,12,13]);ax.set_yticks([8,9,10,11])
        else:
            ax.set_xticks([0,5,10,15,20]);ax.set_yticks([0,5,10,15])
            ax.add_patch(Rectangle((crop[0],crop[2]),crop[1]-crop[0],crop[3]-crop[2],
                                   fill=False,edgecolor='#5E3C99',lw=.85,ls=(0,(4,2)),zorder=10))
        # Panel identifiers lie outside the plotting rectangle; no figure title.
        ax.text(0,1.045,f"({chr(97+number)}) {row['name']}",transform=ax.transAxes,
                ha='left',va='bottom',fontsize=10)
        ax.tick_params(labelsize=8)
    legend_ax=axes.flat[-1];legend_ax.axis('off')
    handles=legend_handles()
    if zoom:
        handles=[h for h in handles if h.get_label()!='起终点']
        handles.append(Line2D([],[],color='#222222',lw=1,marker='>',markersize=4,label='停靠朝向'))
        note='同一中心区域\n起终点位于图外\n任务符号与全局图一致'
    else:
        handles.append(Rectangle((0,0),1,1,fill=False,edgecolor='#5E3C99',ls=(0,(4,2)),label='共同放大区域'))
        note=f"{len(scene['tables'])} 张桌 / {len(scene['tasks'])} 个任务\n{len(scene['candidates'])} 个候选配置\n统一 rep0；同坐标范围"
    legend_ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.03,1.02),
                     frameon=False,fontsize=9,handlelength=2.15,labelspacing=.60)
    legend_ax.text(.06,.14 if not zoom else .22,note,transform=legend_ax.transAxes,
                   ha='left',va='top',fontsize=9,linespacing=1.55)
    return fig


def main():
    item,scene,rows,bounds,crop=frozen_case()
    fig=panel_layout(scene,rows,bounds,crop)
    outputs=save(fig,'fig3_routes100');plt.close(fig)
    result={'display_id':DISPLAY_ID,'seed':scene['seed'],'tables':len(scene['tables']),
            'tasks':len(scene['tasks']),'candidates':len(scene['candidates']),
            'selection_rule':item['selection_rule'],'source':INDEX['sources']['main100'],
            'common_axes_limits':list(bounds),'central_crop':list(crop),
            'crop_rule':'Central 20% in x and y of the saved x_lines/y_lines envelope; selected without inspecting method performance.',
            'all_panels_share_style_and_axes':True,'new_planning_or_inference':False,'new_timing':False,
            'methods':[{'label':r['label'],'display_label':r['name'],'plan_pointer':r['plan_pointer'],
                        'config':r['config'],**r['summary']} for r in rows],
            'outputs':outputs,'scope':'A fixed illustrative case, not an independent comparison population; route overlaps are drawn on top of one another.'}
    write_json('fig3_metadata.json',result)


if __name__=='__main__':main()
