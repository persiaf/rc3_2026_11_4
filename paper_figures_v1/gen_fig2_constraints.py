"""Render the frozen synthetic common-constraint counterexample; no planning."""
from shared import *


def arms(value):
    return ''.join(a for a,flag in zip(('L','R'),value) if flag) or '—'


def main():
    spec=display('external_common_constraint_counterexample')
    data=source('external');scene=resolve(spec['scene'])
    plan=resolve(spec['plans'][0]['plan']);cost=cost_summary(plan)
    ids=plan['visit_order_candidate_ids'];task_ids=[t['id'] for t in scene['tasks']]
    F=np.asarray(resolve(spec['planning_F']),bool)
    A=np.asarray(resolve(spec['audit_A']),bool);effective=F&A
    matrices=[F,A,effective]
    labels=[r'$F$',r'$A$',r'$F\wedge A$']
    nav_to_candidates=lambda route:[n-1 for n in route[1:-1]]
    old_ids=nav_to_candidates(data['f_only_canonical_route'])
    new_ids=nav_to_candidates(data['common_canonical_route'])
    lost=[i for i in task_ids if not effective[i,old_ids].any()]
    assert new_ids==ids and all(effective[i,new_ids].any() for i in task_ids)
    fig=plt.figure(figsize=(7.3,4.05))
    grid=fig.add_axes([.045,.37,.435,.55]);grid.set_axis_off()
    grid.set_xlim(-.65,6.9);grid.set_ylim(-.12,4.8)
    grid.text(-.58,4.64,'(a)',fontsize=10,va='top')
    n=len(task_ids);cell_w=.76;cell_h=.61
    starts=[.15,2.50,4.85]
    matrix_records=[]
    for block,(values,title,xstart) in enumerate(zip(matrices,labels,starts)):
        grid.text(xstart+.79,4.28,title,ha='center',fontsize=12)
        for col,candidate in enumerate(ids):
            grid.text(xstart+col*.91+.38,3.79,rf'$c_{{{candidate}}}$',ha='center',fontsize=9)
        records=[]
        for row,tid in enumerate(task_ids):
            y=3.10-row*.71
            if block==0:grid.text(-.21,y+.30,rf'$t_{{{tid}}}$',ha='right',va='center',fontsize=10)
            cells=[]
            for col,candidate in enumerate(ids):
                val=values[tid,candidate];symbol=arms(val);filled=bool(val.any())
                rect=patches.Rectangle((xstart+col*.91,y),cell_w,cell_h,
                    facecolor='#DCE7F0' if filled else 'white',edgecolor='#666666',lw=.6)
                grid.add_patch(rect)
                grid.text(xstart+col*.91+.38,y+.305,symbol,ha='center',va='center',fontsize=10)
                cells.append(dict(candidate_id=candidate,arms=symbol))
            records.append(dict(task_id=tid,cells=cells))
        matrix_records.append(dict(matrix=('F','A','F_and_A')[block],rows=records))
    # The orange outline identifies the common-coverage loss, not a collision.
    for block in (1,2):
        xstart=starts[block]
        row=task_ids.index(lost[0]);y=3.10-row*.71
        grid.add_patch(patches.Rectangle((xstart-.025,y-.025),cell_w+.05,cell_h+.05,
                      fill=False,edgecolor=COLORS['R'],lw=1.3))
    grid.text(.15,-.01,'L：左臂允许；—：该表中无可用臂',fontsize=8.5,va='bottom')

    note=fig.add_axes([.045,.13,.435,.21]);note.set_axis_off()
    old_text=', '.join(str(i) for i in old_ids);new_text=', '.join(str(i) for i in new_ids)
    task_text=', '.join(f't{i}' for i in lost)
    note.text(.01,.91,f'仅按 F 删站：候选 [{old_text}]',fontsize=9.5,va='top')
    note.text(.01,.57,task_text+' 失去共同覆盖，方案被拒绝。',fontsize=9.5,va='top',color='#9D4200')
    note.text(.01,.20,f'按 F∧A 保留：候选 [{new_text}]',fontsize=9.5,va='top')

    ax=fig.add_axes([.555,.40,.425,.49])
    draw_scene(ax,scene,bounds=(-.35,4.75,-.30,2.22))
    draw_route(ax,plan,lw=1.4,arrows=False)
    draw_stops(ax,scene,plan,size=95,numbered=True,headings=True)
    # Only stored task coordinates and assignments are shown. No arm trajectories.
    draw_tasks(ax,scene,plan,size=34,labels=False)
    for task in scene['tasks']:
        delta={0:(-16,-11),1:(4,5),2:(-17,5),3:(4,-12)}[task['id']]
        ax.annotate(rf'$t_{{{task["id"]}}}$',(task['x'],task['y']),xytext=delta,
                    textcoords='offset points',fontsize=9,zorder=10)
    ax.text(-.08,1.12,'(b)',transform=ax.transAxes,fontsize=10,va='top')
    ax.set_xticks([0,1,2,3,4]);ax.set_yticks([0,1,2])
    service=fig.add_axes([.545,.13,.445,.19]);service.set_axis_off()
    service.text(0,.97,'停靠序号    候选 / 高度        服务任务（左臂）',fontsize=8.5,va='top')
    for row,stop in enumerate(plan['service_stops']):
        tasks=', '.join(str(a['task_id']) for a in stop['tasks'])
        service.text(.035,.61-row*.30,
            f"{stop['sequence']}             {stop['candidate_id']} / {stop['pose']['h']:.1f} m              {tasks}",
            fontsize=8.5,va='top')
    legend=legend_handles()
    fig.legend(handles=legend[:5],loc='lower center',bbox_to_anchor=(.53,.005),ncol=5,
               frameon=False,fontsize=8.5,handlelength=1.35,columnspacing=1.2)
    files=save(fig,'fig2_constraints');plt.close(fig)
    write_json('fig2_metadata.json',dict(figure='fig2_constraints',files=files,
        source=INDEX['sources']['external'],source_pointers=dict(scene=spec['scene']['json_pointer'],
        planning_F=spec['planning_F']['json_pointer'],audit_A=spec['audit_A']['json_pointer'],
        plan=spec['plans'][0]['plan']['json_pointer']),seed=scene['seed'],
        task_ids=task_ids,candidate_ids=ids,matrices=matrix_records,
        f_only_selected_candidates=old_ids,common_selected_candidates=new_ids,
        tasks_losing_common_coverage=lost,cost=cost,
        shown_polyline=plan['route_polyline'],
        notes=['Only the stored successful full base path is plotted.',
               'Synthetic service tables: cross-table assignments are not physical reachability evidence.',
               'Numbered circles are stop sequence; c identifiers are zero-based candidates.',
               'No planning, inference, training or timing measurement.']))
    (HERE/'fig2_caption.txt').write_text(
        '图2 共同约束贯穿路线规范化的必要性（合成约束案例，seed='+str(scene['seed'])+'）。'
        '（a）仅显示已选候选c0、c23对应的F、A与F∧A；L表示该表允许左臂，横线表示无可用臂。'
        '仅按F判断时，候选c0已覆盖全部任务，c23会被删除；但任务t0在c0不满足A，因而失去共同覆盖。'
        '采用F∧A后保留c0与c23。（b）保存的成功方案完整底盘折线；圆圈中的1、2为停靠顺序，'
        '黑箭头表示停靠朝向，下方给出候选、高度和服务任务。失败方案未绘制为已认证路径。'
        f'L={cost["L_m"]:.1f} m，K={cost["K"]}，J=L+0.75K={cost["J_equivalent_m"]:.1f}等效m；'
        'J不是实测任务总时间。可达表及跨桌任务分配为刻意构造的合成约束，不表示真实机械臂工作空间、臂轨迹或物理可执行性。',encoding='utf-8')


if __name__=='__main__':main()
