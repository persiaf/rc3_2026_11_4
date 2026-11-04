"""Plot the frozen 3x3 input/output illustration; no planner is invoked."""
from shared import *


def main():
    entry=display('integration_3x3_default_fixed3')
    scene=resolve(entry['scene']);plan=resolve(entry['plans'][0]['plan'])
    summary=cost_summary(plan)
    fig=plt.figure(figsize=(7.1,5.3))
    grid=fig.add_gridspec(1,2,width_ratios=[2.55,1.0],left=.07,right=.99,bottom=.22,top=.97,wspace=.12)
    ax=fig.add_subplot(grid[0,0]);side=fig.add_subplot(grid[0,1]);side.axis('off')
    draw_scene(ax,scene,candidates=True)
    draw_route(ax,plan,lw=1.35,arrows=True)
    draw_tasks(ax,scene,plan,size=30)
    draw_stops(ax,scene,plan,size=85,numbered=True,headings=True)
    assigned={a['task_id']:a['arm'] for a in plan['assignments']}
    # Annotation positions only; task positions/IDs/arms all come from the file.
    labels=[]
    for t in scene['tasks']:
        table=scene['tables'][t['table_id']]
        dx=-5 if t['x']<table['cx'] else 5
        dy=-9 if t['y']<table['cy'] else 6
        dx,dy={7:(8,10),8:(-10,2),9:(8,-11),10:(-8,-10),11:(8,-8)}.get(t['id'],(dx,dy))
        arm=assigned[t['id']]
        art=ax.annotate(f"{t['id']}{arm}",(t['x'],t['y']),xytext=(dx,dy),textcoords='offset points',
                       ha='right' if dx<0 else 'left',va='top' if dy<0 else 'bottom',
                       color=COLORS[arm],fontsize=8,zorder=10,
                       arrowprops={'arrowstyle':'-','color':'#999999','lw':.45,'shrinkA':1.5,'shrinkB':3.5})
        labels.append(art)
    rows=[]
    for s in plan['service_stops']:
        task_text=lambda arm: ','.join(str(t['task_id']) for t in s['tasks'] if t['arm']==arm) or '—'
        rows.append([str(s['sequence']),str(s['candidate_id']),task_text('L'),task_text('R')])
    table=side.table(cellText=rows,colLabels=['次序','候选','L任务','R任务'],
                     colWidths=[.18,.19,.33,.30],cellLoc='center',bbox=[0,.36,1,.44])
    table.auto_set_font_size(False);table.set_fontsize(8)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#AAAAAA');cell.set_linewidth(.35)
        if row==0:cell.set_facecolor('#F0F0F0')
    side.text(0,.91,'圆圈编号表示服务次序',fontsize=9,transform=side.transAxes)
    side.text(0,.85,'任务标注：ID + 用臂',fontsize=9,transform=side.transAxes)
    side.text(0,.29,f"{len(scene['tables'])} 张桌 · {len(scene['tasks'])} 个任务\n"
              f"L = {summary['L_m']:.3f} m\nK = {summary['K']}\n"
              f"J = {summary['J_equivalent_m']:.3f} 等效 m",fontsize=9,linespacing=1.65,
              va='top',transform=side.transAxes)
    handles=legend_handles(include_candidates=True,include_obstacles=bool(scene.get('obstacles')))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,frameon=False,
               handlelength=1.5,columnspacing=1.4,labelspacing=.65,fontsize=8.5)
    outputs=save(fig,'fig1_scene');plt.close(fig)
    write_json('fig1_metadata.json',dict(source=entry['scene'],plan_source=entry['plans'][0]['plan'],
        seed=scene['seed'],tables=len(scene['tables']),tasks=len(scene['tasks']),candidates=len(scene['candidates']),
        **summary,outputs=outputs,scope='Frozen proxy scene/format illustration, not a new timing sample.',
        task_ids_are_zero_based=True,stop_sequence_is_one_based=True,all_tasks_assigned=len(assigned)==len(scene['tasks'])))


if __name__=='__main__':main()
