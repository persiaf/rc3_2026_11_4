"""Recompute saved-plan validity and feedback without any planner or proposal."""
from __future__ import annotations
from collections import Counter
import csv
import gzip
import math
import numpy as np
from .common import ROOT,WORKSPACE,METHODS,read,write,sha,jsonable,teacher_index,feedback_label,schedule

def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-7):raise ValueError(f'Recorded values disagree: {a} vs {b}')

def load_evidence(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        import json
        return json.load(f)

def summarize(rows,feedback):
    summary={}
    for method in METHODS:
        own=[r for r in rows if r['method']==method]
        gains=[r['quality_gain_percent'] for r in own]
        item=dict(n=len(own),mean_J=float(np.mean([r['J'] for r in own])),mean_L=float(np.mean([r['L'] for r in own])),
            mean_K=float(np.mean([r['K'] for r in own])),mean_gain_percent=float(np.mean(gains)),
            median_complete_ms=1000*float(np.median([r['complete_s'] for r in own])),
            median_complete_plus_model_load_ms=1000*float(np.median([r['complete_plus_model_load_s'] for r in own])),
            mean_proposals=float(np.mean([r['proposal_calls'] for r in own])),
            total_proposals=sum(r['proposal_calls'] for r in own),
            quality_improved_cases=sum(g>1e-8 for g in gains),fallbacks=sum(r['fallback'] for r in own),
            stop_at_root=sum(r['stop_at_root'] for r in own),stop_reasons=dict(Counter(r['stop_reason'] for r in own)),
            overruns=sum(r['time_overrun_s']>0 for r in own),
            models_loaded_once=method in ('continued_ce','ce_gain'))
        if own[0]['shared_environment']:
            delta=[r['return_vs_stop'] for r in own]
            item.update(mean_return=float(np.mean([r['return_'] for r in own])),mean_return_vs_stop=float(np.mean(delta)),
                return_vs_stop_wins=sum(v>1e-8 for v in delta),return_vs_stop_ties=sum(abs(v)<=1e-8 for v in delta),
                return_vs_stop_losses=sum(v< -1e-8 for v in delta),mean_work=float(np.mean([r['work_used'] for r in own])))
        else:item.update(mean_return=None,mean_return_vs_stop=None,return_vs_stop_wins=None,return_vs_stop_ties=None,return_vs_stop_losses=None,mean_work=None)
        summary[method]=item
    model_feedback={}
    for method in ('continued_ce','ce_gain'):
        own=[r for r in feedback if r['method']==method and r['action_kind']!='stop' and r['proposal_calls']>0]
        by_label={}
        for label in ('action_queried','action_unqueried','state_unseen'):
            selected=[r for r in own if r['teacher_status']==label]
            by_label[label]=dict(n=len(selected),immediate_best_improvements=sum(r['immediate_best_quality_percent']>1e-8 for r in selected),
                positive_immediate_rewards=sum(r['reward']>1e-8 for r in selected),
                positive_actual_policy_suffixes=sum(r['realized_policy_suffix_return']>1e-8 for r in selected),
                positive_vs_same_state_analytic_stop=sum(r['policy_return_vs_analytic_stop']>1e-8 for r in selected),
                mean_immediate_reward=float(np.mean([r['reward'] for r in selected])) if selected else None,
                mean_actual_policy_suffix=float(np.mean([r['realized_policy_suffix_return'] for r in selected])) if selected else None)
        model_feedback[method]=by_label
    return dict(methods=summary,model_feedback=model_feedback,
        interpretation='Twelve old development scenes, one frozen greedy trajectory each. Suffix outcomes are not independent state samples or optimal Q labels; times have one repetition per method/scene.')

def main():
    from rc3_baseline.validator import validate
    from rc3_rl_g2.teacher_dataset import load_teacher
    p=read(ROOT/'outputs/protocol.json');directory=ROOT/'outputs/evaluation';rows=read(directory/'results.json')
    if sha(ROOT/'outputs/protocol.json')!=(ROOT/'outputs/protocol.sha256').read_text().strip():raise ValueError('Protocol changed')
    if sha(WORKSPACE/'rc3_rl_g2/outputs/teacher_extra/teacher.pkl')!=p['extra_teacher_sha256']:raise ValueError('Teacher changed')
    samples=load_teacher(WORKSPACE/'rc3_rl_g2/outputs/teacher_extra/teacher.pkl')['samples']
    import json
    attempts=[json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    expected=[dict(slot=i,case_id=c['case_id'],method=m) for i,(c,m) in enumerate(schedule(p['cases']))]
    if attempts!=expected or len(rows)!=72:raise ValueError('Attempt schedule/cap mismatch')
    feedback=[];checks=[];case_map={c['case_id']:c for c in p['cases']}
    by_case={i:{r['method']:r for r in rows if r['case_id']==i} for i in case_map}
    for row in rows:
        if row['status']!='complete' or not row['valid']:raise ValueError('A planning failure must be reported before aggregate success: '+str(row))
        case=case_map[row['case_id']];case_rows=by_case[case['case_id']]
        scene=read(WORKSPACE/case['scene_path']);index=teacher_index(samples,case)
        destination=directory/f"case_{case['case_id']:02d}_{row['method']}"
        evidence=load_evidence(destination/'evidence.json.gz');plan=read(destination/'plan.json')
        if plan!=evidence['plan']:raise ValueError('Standalone and evidence plans disagree')
        check=validate(scene,plan,evidence['sparse_feasibility'])
        if not check['valid_model_plan']:raise ValueError('Independent saved-plan verification failed: '+str(check))
        close(row['J'],row['L']+.75*row['K']);close(row['J'],plan['objective_length_plus_stops_m']);close(row['L'],plan['route_length_m'])
        if row['K']!=len(plan['service_stops']):raise ValueError('Stop count mismatch')
        close(row['anchor_J'],case_rows['fixed3']['J'])
        close(row['quality_gain_percent'],100*(case_rows['fixed3']['J']-row['J'])/case_rows['fixed3']['J'])
        if row['complete_s']<0 or row['J']>row['anchor_J']+1e-7:raise ValueError('Invalid time or worsened anchor')
        action_rows=evidence['actions'];events=evidence['events']
        if row['shared_environment']:
            close(row['return_'],sum(v['reward'] for v in action_rows)+sum(v['reward'] for v in events))
            paid=sum(v['info']['work_units'] for v in action_rows)+sum(e['menu_work_units'] for e in events)
            if paid!=row['work_used']:raise ValueError('Work ledger mismatch')
            close(row['return_'],row['quality_gain_percent']-p['config']['lambda_work']*row['work_used']/p['config']['work_scale'])
            close(row['return_vs_stop'],row['return_']-case_rows['stop']['return_'])
            if sum(v['proposal_calls'] for v in action_rows)!=row['proposal_calls']:raise ValueError('Proposal counter mismatch')
            for action in action_rows:
                obs=action['environment_observation'];a=action['action_id'];info=action['info']
                if not obs['mask'][a] or action['action']!=obs['actions'][a]:raise ValueError('Selected action/legality mismatch')
                label=feedback_label(index,obs,a)
                if any(action[k]!=v for k,v in label.items()):raise ValueError('Teacher feedback category or null values changed')
                close(action['reward'],info['quality_reward']-info['work_penalty']+info['shaping_reward'])
                close(action['realized_policy_suffix_return'],row['return_']-action['prefix_return'])
                close(action['realized_after_action_return'],row['return_']-action['prefix_return']-action['reward'])
                feedback.append(dict(case_id=case['case_id'],method=row['method'],mode=case['mode'],step=action['step'],
                    action_id=a,action_kind=action['action']['kind'],teacher_status=label['teacher_status'],teacher_depth=label['teacher_depth'],
                    known_teacher_Q=label['known_Q'],known_teacher_delta=label['known_delta_vs_stop'],
                    proposal_calls=action['proposal_calls'],accepted=info['accepted'],route_changed=info['route_changed'],best_improved=info['best_improved'],
                    immediate_best_quality_percent=action['immediate_best_quality_percent'],reward=action['reward'],
                    realized_policy_suffix_return=action['realized_policy_suffix_return'],realized_after_action_return=action['realized_after_action_return'],
                    prefix_return=action['prefix_return'],after_work=action['after_work'],before_best_J=action['before_best_J'],after_best_J=action['after_best_J'],
                    hypothetical_stop_total_return=action['hypothetical_stop_total_return'],final_episode_return=row['return_'],
                    analytic_stop_remaining_return=action['hypothetical_stop_total_return']-action['prefix_return'],
                    policy_return_vs_analytic_stop=row['return_']-action['hypothetical_stop_total_return'],
                    analytic_stop_scope='Zero-potential accounting at same visited state; assumes prefix final validation no rollback; no counterfactual environment call',
                    episode_return_vs_stop=row['return_vs_stop'],suffix_scope='Observed frozen-policy trajectory, not optimal or teacher Q'))
            for raw in evidence.get('policy_observations',[]):
                if not raw['executed']:continue
                allowed=np.asarray(raw['observation']['mask'],bool);values=np.asarray([v if v is not None else -np.inf for v in raw['scores']['logits']])
                if int(values.argmax())!=raw['action_id'] or not allowed[raw['action_id']]:raise ValueError('Full-menu greedy decision mismatch')
                probabilities=np.asarray(raw['scores']['probabilities'],dtype=np.float64)
                tolerance=4*len(probabilities)*np.finfo(np.float32).eps
                if (not np.isfinite(probabilities).all() or (probabilities<0).any()
                        or (probabilities[~allowed]!=0).any() or abs(probabilities.sum()-1)>tolerance):
                    raise ValueError('Invalid FP32 probability normalization or legality')
            if row['method'] in ('continued_ce','ce_gain') and action_rows:
                run='gain_control' if row['method']=='continued_ce' else 'gain_gain'
                old=read(WORKSPACE/f'rc3_rl_g21/outputs/{run}/extra_existing_teacher_diagnostic.json')
                previous=next(v for v in old['rows'] if v['is_root'] and int(v['scene_id'].split(':',1)[0])==case['case_id'])
                if action_rows[0]['action_id']!=previous['full_greedy_action']:raise ValueError('Frozen root choice differs from G21')
        elif row['return_'] is not None or row['work_used'] is not None:raise ValueError('Original algorithm has no shared-work reward')
        checks.append(dict(case_id=case['case_id'],method=row['method'],valid=True,plan_sha256=sha(destination/'plan.json'),
            evidence_sha256=sha(destination/'evidence.json.gz'),independent_validation=check))
    summary=summarize(rows,feedback)
    # Retrospective stopping accounting is a diagnostic, not an evaluated method.
    late=[]
    for row in rows:
        if not row['shared_environment']:continue
        own=[v for v in feedback if v['case_id']==row['case_id'] and v['method']==row['method']]
        after_improvement=[v for v in own if v['step']>0 and v['hypothetical_stop_total_return']>row['return_']+1e-8
                           and v['before_best_J']<row['anchor_J']-1e-8]
        if row['best_improvements']>0 and after_improvement:
            late.append(dict(case_id=row['case_id'],method=row['method'],actual_return=row['return_'],
                retrospectively_better_visited_stop_return=max(v['hypothetical_stop_total_return'] for v in after_improvement),
                scope='Accounting on already visited states, assumes prefix final validation no rollback; not deployed or optimal'))
    summary['retrospective_late_stop_diagnostics']=late
    summary['counts']=dict(plans=72,scenes=12,shared_episodes=48,shared_actions=sum(r['steps'] for r in rows if r['shared_environment']),
        shared_proposal_calls=sum(r['proposal_calls'] for r in rows if r['shared_environment']),
        original_detour_attempts=sum(r['proposal_calls'] for r in rows if r['method']=='detour'),
        optimizer_updates=0,ppo_actions=0,extra_teacher_branches=0,unique_saved_plans_independently_validated=72)
    write(ROOT/'outputs/summary.json',summary);write(ROOT/'outputs/action_feedback.json',feedback)
    with (ROOT/'outputs/action_feedback.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(feedback[0]));writer.writeheader();writer.writerows(feedback)
    write(ROOT/'outputs/plan_audit.json',dict(all_passed=True,plans=72,checks=checks,counts=summary['counts'],
        scope='Saved plan validation and ledger/teacher/greedy-logit recomputation, no planner or environment step'))
    print(json.dumps(dict(all_passed=True,counts=summary['counts'],methods=summary['methods']),ensure_ascii=False))

if __name__=='__main__':main()
