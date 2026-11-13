"""R08-B saved-output arithmetic only; no new model or planning calls."""
import csv,time,json
from collections import defaultdict
import numpy as np
from .common import ROOT,HOME,read,put,unarchive,sha

def table(name,rows):
    with (HOME/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def stats(values):
    return dict(mean=float(np.mean(values)),p50=float(np.median(values)),p95=float(np.quantile(values,.95)),max=float(max(values))) if values else None
def assignments(plan):return {a['task_id']:(a['candidate_id'],a['arm']) for a in plan['assignments']}
def costs(plan):
    L=sum(plan['route_leg_costs_m']);K=len(plan['visit_order_candidate_ids']);return dict(J=L+.75*K,L=L,K=K)
def gain(a,b):return (a-b)/max(a,1e-7)

def main():
    tick=time.perf_counter();protocol=read(HOME/'PROTOCOL.json');supervised=read(HOME/'controls/SUPERVISOR_RECEIPT.json')
    if supervised['status']!='COMPLETE' or read(HOME/'STATUS.json')['status']!='SCIENCE_COMPLETE':raise RuntimeError('Incomplete B cannot receive a science verdict')
    tasks=read(HOME/'TASKS.json')['records'];rows=[];parents=[];curves=[];requests=[];menus=[];audit=0
    for task in tasks:
        d=read(HOME/'tasks'/(task['id']+'.json'));menu=read(HOME/'menus'/(task['id']+'.json'));base=unarchive(d['anchor']['record']);bm=base['metrics']
        br=base['raw_plan']['visit_order_candidate_ids'];bh=base['plan']['visit_order_candidate_ids'];ar=assignments(base['raw_plan']);ah=assignments(base['plan']);local=[]
        requests.append(dict(request_id=base['request_id'],purpose='anchor',task_id=task['id'],size=task['size'],full_request_s=d['anchor']['full_request_s'],H_s=base['H_s']))
        for rr in d['results']:
            out=unarchive(rr['record']);a=rr['action'];m=out['metrics'];raw=out['raw_plan']['visit_order_candidate_ids'];post=out['plan']['visit_order_candidate_ids']
            if out['problem_identity']!=base['problem_identity'] or out['physical_key']!=task['physical_key'] or out['H_identity']!=base['H_identity'] or out['anchor']!=d['anchor']:raise ValueError('Exact matched edit source')
            if set(a['drop'])&set(a['add']) or not a['legal'] or not max(len(a['drop']),len(a['add']))<=2:raise ValueError('Canonical menu action invalid')
            for label,plan,prefix in [('raw',out['raw_plan'],'raw_'),('H',out['plan'],'')]:
                c=costs(plan)
                if not plan['validation']['valid_model_plan'] or plan['unserved_task_ids'] or any(abs(c[k]-m[prefix+k])>1e-7 for k in ('J','L','K')):raise ValueError('Saved complete route/cost/validation mismatch')
                audit+=4
            rassign=assignments(out['raw_plan']);hassign=assignments(out['plan'])
            if set(rassign)!=set(ar) or set(hassign)!=set(ah):raise ValueError('Task completion pairing')
            raw_drop=set(br)-set(raw);raw_add=set(raw)-set(br);H_drop=set(bh)-set(post);H_add=set(post)-set(bh)
            r=dict(task_id=task['id'],physical_key=task['physical_key'],size=task['size'],menu_index=a['menu_index'],block_kind=a['block_kind'],
                explicit_drop_count=len(a['drop']),explicit_add_count=len(a['add']),net_joint=a['net_joint'],
                raw_realized_drop_count=len(raw_drop),raw_realized_add_count=len(raw_add),raw_realized_joint=max(len(raw_drop),len(raw_add))>=2,
                H_drop_vs_anchor=len(H_drop),H_add_vs_anchor=len(H_add),H_actual_set_joint_vs_anchor=max(len(H_drop),len(H_add))>=2,
                requested_adds_retained_raw=len(set(a['add'])&set(raw)),requested_adds_retained_H=len(set(a['add'])&set(post)),
                requested_drops_absent_H=len(set(a['drop'])-set(post)),
                raw_config_assignment_change=sum(ar[i][0]!=rassign[i][0] for i in ar)/max(len(ar),1),H_config_assignment_change=sum(ah[i][0]!=hassign[i][0] for i in ah)/max(len(ah),1),
                H_arm_assignment_change=sum(ah[i][1]!=hassign[i][1] for i in ah)/max(len(ah),1),
                raw_identical_to_anchor=br==raw,H_identical_route=bh==post,H_identical_set=set(bh)==set(post),
                H_cost_identical=abs(m['J']-bm['J'])<=1e-7,H_erased_raw_cost_difference=abs(m['raw_J']-bm['raw_J'])>1e-7 and abs(m['J']-bm['J'])<=1e-7,
                relative_gain=gain(bm['J'],m['J']),material_improvement=gain(bm['J'],m['J'])>=.01,
                **{'anchor_'+k:bm[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},**{'edit_'+k:m[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},
                **{'delta_'+k:m[k]-bm[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},
                full_request_s=rr['full_request_s'],lower_s=out['lower']['lower_s'],H_s=out['H_s'],zero_gain_cleanup_count=len(out['lower']['cleanup_removed_global']),fallback_used=out['fallback_used'])
            rows.append(r);local.append(r);requests.append(dict(request_id=out['request_id'],purpose='edit',task_id=task['id'],size=task['size'],full_request_s=rr['full_request_s'],H_s=out['H_s']))
        if len(local)!=len(menu['actions']) or len(local)>16:raise ValueError('Executed menu completeness')
        menus.append(menu['menu_build_s']);best=max([0.,*[r['relative_gain'] for r in local]])
        parents.append(dict(task_id=task['id'],physical_key=task['physical_key'],size=task['size'],actions=len(local),missing=16-len(local),
            best_oracle_gain=best,material_any=best>=.01,net_joint_actions=sum(r['net_joint'] for r in local),
            material_edits=sum(r['material_improvement'] for r in local),material_net_joint_edits=sum(r['material_improvement'] and r['net_joint'] for r in local),
            anchor_request_s=d['anchor']['full_request_s'],menu_build_s=menu['menu_build_s'],all_edit_s=sum(r['full_request_s'] for r in local),
            best_menu_index=next((r['menu_index'] for r in local if r['relative_gain']==best),None)))
        for k in (1,2,4,8,16):
            prefix=local[:k];best=max([0.,*[r['relative_gain'] for r in prefix]])
            curves.append(dict(task_id=task['id'],physical_key=task['physical_key'],size=task['size'],prefix_budget=k,actual_actions=len(prefix),best_oracle_gain=best,material_any=best>=.01,
                edit_cumulative_s=sum(r['full_request_s'] for r in prefix),anchor_and_menu_and_prefix_s=d['anchor']['full_request_s']+menu['menu_build_s']+sum(r['full_request_s'] for r in prefix)))
    if len(parents)!=48 or len({p['physical_key'] for p in parents})!=48:raise ValueError('48 fixed parent denominator')
    rsrc=read(HOME/'RESOURCE_ACCOUNTING.json')['counts']
    if len(requests)!=rsrc['requests_completed'] or rsrc['requests_failed'] or rsrc['actor_optimizer_attempts'] or rsrc['critic_optimizer_attempts']:raise ValueError('Actual full request ledger')
    full=[p for p in parents if p['size']=='10x10'];local=[p for p in parents if p['size']!='10x10'];material=[r for r in rows if r['material_improvement']]
    mean_req=float(np.mean([r['full_request_s'] for r in requests]));p95edit=float(np.quantile([r['full_request_s'] for r in rows],.95));menu_unit=max(float(np.mean(menus)),float(np.quantile(menus,.95)))
    # D:3072 training episodes*4 menus +4 edited methods*148 evaluation cases*4 menus.
    D_menu_calls=3*128*8*4+4*74*2*4
    setup=max(0,supervised['elapsed_s']-sum(r['full_request_s'] for r in requests)-sum(menus))
    forecast=max(mean_req,p95edit)*22428+D_menu_calls*menu_unit+1920+3000+setup
    checks=dict(all_plans_valid=True,full_parent_count=sum(p['material_any'] for p in full)>=8,local_parent_count=sum(p['material_any'] for p in local)>=4,
        overall_fraction=sum(p['material_any'] for p in parents)/48>=.30,full_fraction=sum(p['material_any'] for p in full)/32>=.25,
        local_fraction=sum(p['material_any'] for p in local)/16>=.25,oracle_mean=float(np.mean([p['best_oracle_gain'] for p in parents]))>=.015,
        material_net_joint_fraction=bool(material) and sum(r['net_joint'] for r in material)/len(material)>=.5,
        throughput_fits=forecast<=21600)
    quality=all(v for k,v in checks.items() if k!='throughput_fits');passed=quality and checks['throughput_fits']
    status='OPPORTUNITY_PASS' if passed else ('TECHNICAL_THROUGHPUT_NOT_ADMITTED' if quality else 'OPPORTUNITY_INSUFFICIENT')
    groups={}
    for size in ('ALL','10x10','local','4x3','6x6','8x8'):
        pp=[p for p in parents if size=='ALL' or (size=='local' and p['size']!='10x10') or p['size']==size]
        rr=[r for r in rows if size=='ALL' or (size=='local' and r['size']!='10x10') or r['size']==size]
        groups[size]=dict(tasks=len(pp),actions=len(rr),material_tasks=sum(p['material_any'] for p in pp),material_task_fraction=sum(p['material_any'] for p in pp)/len(pp),
            oracle_parent_mean=float(np.mean([p['best_oracle_gain'] for p in pp])),net_joint_actions=sum(r['net_joint'] for r in rr),material_edits=sum(r['material_improvement'] for r in rr),
            material_joint_edits=sum(r['net_joint'] and r['material_improvement'] for r in rr),H_identical_routes=sum(r['H_identical_route'] for r in rr),H_cost_identical=sum(r['H_cost_identical'] for r in rr),
            H_erased_raw_differences=sum(r['H_erased_raw_cost_difference'] for r in rr),edit_times=stats([r['full_request_s'] for r in rr]),
            improved=sum(r['delta_J']<-1e-7 for r in rr),tied=sum(abs(r['delta_J'])<=1e-7 for r in rr),degraded=sum(r['delta_J']>1e-7 for r in rr))
    curve_summary=[]
    for size in ('ALL','10x10','local','4x3','6x6','8x8'):
        for k in (1,2,4,8,16):
            cc=[r for r in curves if r['prefix_budget']==k and (size=='ALL' or (size=='local' and r['size']!='10x10') or r['size']==size)]
            curve_summary.append(dict(size=size,prefix_budget=k,tasks=len(cc),actual_actions=sum(r['actual_actions'] for r in cc),material_fraction=sum(r['material_any'] for r in cc)/len(cc),
                oracle_parent_mean=float(np.mean([r['best_oracle_gain'] for r in cc])),mean_total_s=float(np.mean([r['anchor_and_menu_and_prefix_s'] for r in cc])),mean_edit_s=float(np.mean([r['edit_cumulative_s'] for r in cc]))))
    result=dict(status=status,passed=passed,opportunity_quality_passed=quality,checks=checks,groups=groups,actual_requests=len(requests),actual_edits=len(rows),fixed_parents=48,
        material_net_joint_fraction=sum(r['net_joint'] for r in material)/len(material) if material else None,
        forecast=dict(D_requests=22428,D_menu_calls=D_menu_calls,mean_complete_request_s=mean_req,p95_edit_s=p95edit,conservative_menu_s=menu_unit,
            source_model_misc_setup_s=setup,gradient_checkpoint_reserve_s=1920,other_reserve_s=3000,estimated_D_wall_s=forecast,D_cap_s=21600,
            scope='B one-edit throughput projection, not measured multi-step training or deployment speed'),
        oracle_not_learned_policy=True,C_D_E='NOT_RUN_NOT_AUTHORIZED',analysis_wall_s=time.perf_counter()-tick,audit_checks=audit)
    table('ACTION_RESULTS.csv',rows);table('TASK_PARENT_RESULTS.csv',parents);table('PREFIX_OPPORTUNITY.csv',curves);table('PREFIX_SUMMARY.csv',curve_summary);table('REQUEST_TIMES.csv',requests)
    put(HOME/'RESULTS.json',result);put(HOME/'NUMERICAL_AUDIT.json',dict(status='PASS',checks=audit,complete_requests=len(requests),edits=len(rows),mask_and_validation_evidence='Actual raw/H independent certificates; existing A coverage precheck and real solver assertions',new_planning_calls_in_analysis=0))
    print(dict(status=status,requests=len(requests),edits=len(rows),groups=groups,checks=checks,forecast=result['forecast']),flush=True)

if __name__=='__main__':main()
