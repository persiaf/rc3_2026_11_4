"""Reproduce all G0 log-audit JSON/CSV outputs without running any planner.

Usage:
    python audit_records.py --source /path/to/rc3_rl_research --output /path/to/audit
Requires only Python and NumPy. The source directory is read-only to this script.
"""
from pathlib import Path
import argparse,gzip,json,math,collections,csv
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True,type=Path,help='Unpacked rc3_rl_research directory containing outputs/g0')
parser.add_argument('--output',required=True,type=Path,help='Separate directory for derived audit JSON/CSV')
args=parser.parse_args()
ROOT=args.source.resolve()
OUT=args.output.resolve()
if not (ROOT/'outputs/g0/protocol.json').is_file():
    parser.error('--source must contain outputs/g0/protocol.json')
if OUT==ROOT or ROOT in OUT.parents:
    parser.error('--output must be outside the source tree to preserve original results')
OUT.mkdir(parents=True,exist_ok=True)
TOL=1e-8

def read_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]

def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),p25=float(np.percentile(a,25)),p75=float(np.percentile(a,75)),p95=float(np.percentile(a,95))) if len(a) else {}

def compact(r):
    return dict(path=r['path'],returned_cost=r['finalization']['returned_cost'],total_reward=r['total_reward'],work_used=r['work_used'])

case_stats=[];counts=collections.Counter();proposal_counts=collections.Counter();timings=collections.defaultdict(list)
kind_changes=collections.defaultdict(collections.Counter);step_times=collections.defaultdict(list)
menu_roots=[];menu_all=[];query_gain=[];witness_details=[];errors=[]
teacher_counts=collections.Counter();teacher_rows=[];multistep_rows=[]
sensitivity={lam:dict(nonmonotone_cases=0,stop_cases=0,advantages=[]) for lam in (0.,.001,.003,.01,.03)}

for d in sorted((ROOT/'outputs/g0').glob('case_*')):
    old=json.loads((d/'summary.json').read_text())
    rows=read_gz(d/'queries.jsonl.gz');menus=read_gz(d/'menus.jsonl.gz')
    bypath={tuple(r['path']):r for r in rows};roots=[r for r in rows if r['depth']==1];leaves=[r for r in rows if r['leaf']]
    anchor=old['anchor_cost'];one=min(r['best_cost'] for r in roots)
    stop=next(r for r in roots if r['action']['kind']=='stop')
    best=min(leaves,key=lambda r:(r['finalization']['returned_cost'],r['path']))
    bestR=max(leaves,key=lambda r:(r['total_reward'],-r['work_used'],[-x for x in r['path']]))
    perroot=[]
    for r in roots:
        ll=[v for v in leaves if v['path'][0]==r['action_id']]
        b=min(ll,key=lambda v:v['finalization']['returned_cost'])
        br=max(ll,key=lambda v:v['total_reward'])
        perroot.append(dict(row=r,leaves=ll,best=b,bestR=br))
    def first_worse(r):
        root=bypath[(r['path'][0],)]
        return root['info']['accepted'] and root['current_cost']>anchor+TOL
    def whole_monotone(r):return all(bypath[tuple(r['path'][:i])]['current_change']<=TOL for i in range(1,len(r['path'])+1))
    mono_leaves=[r for r in leaves if whole_monotone(r)]
    bestmono=min(mono_leaves,key=lambda r:(r['finalization']['returned_cost'],r['path']))
    bestmonoR=max(mono_leaves,key=lambda r:r['total_reward'])
    worse_leaves=[r for r in leaves if first_worse(r)]
    strict=[r for r in worse_leaves if r['finalization']['returned_cost']<one-TOL]
    positive_strict=[r for r in strict if r['total_reward']>stop['total_reward']+TOL]
    nonworse_first_leaves=[r for r in leaves if bypath[(r['path'][0],)]['current_cost']<=anchor+TOL]
    mono_first_cost=min(r['finalization']['returned_cost'] for r in nonworse_first_leaves)
    flags=dict(
        worse_first_beats_one_step=bool(strict),
        worse_first_then_record=any(r['finalization']['returned_cost']<anchor-TOL for r in worse_leaves),
        nonimproving_first_then_record=any(r['row']['info']['accepted'] and r['row']['current_cost']>=anchor-TOL and r['row']['action']['kind']!='stop' and r['best']['finalization']['returned_cost']<anchor-TOL for r in perroot),
        ranking_reversal=any(a['row']['best_cost']<b['row']['best_cost']-TOL and a['best']['finalization']['returned_cost']>b['best']['finalization']['returned_cost']+TOL for a in perroot for b in perroot),
        worse_first_beats_all_monotone_first=any(r['finalization']['returned_cost']<mono_first_cost-TOL for r in worse_leaves),
        strict_witness_positive_return=bool(positive_strict),
        any_leaf_beats_stop_return=bestR['total_reward']>stop['total_reward']+TOL,
        any_leaf_beats_whole_monotone_quality=best['finalization']['returned_cost']<bestmono['finalization']['returned_cost']-TOL,
        any_leaf_beats_whole_monotone_return=bestR['total_reward']>bestmonoR['total_reward']+TOL,
        best_return_first_worse=first_worse(bestR),
        best_return_has_worsening=not whole_monotone(bestR),
        best_return_first_stop=bestR['action']['kind']=='stop' and bestR['depth']==1,
    )
    for k,v in flags.items():counts[k]+=int(v)
    for k in ('worse_first_beats_one_step','worse_first_then_record','nonimproving_first_then_record','ranking_reversal','worse_first_beats_all_monotone_first'):
        if flags[k]!=old[k]:errors.append([d.name,k,flags[k],old[k]])
    first_bestR=bypath[(bestR['path'][0],)]
    # Mechanism-display trajectories were not selected to maximize return.
    selected=bypath[tuple(old['selected_path'])]
    root_returns={r['row']['action_id']:r['bestR']['total_reward'] for r in perroot}
    ordered_root_returns=sorted(root_returns.values(),reverse=True)
    def proposal_count(r):
        return sum(bypath[tuple(r['path'][:i])]['action']['kind']!='stop' for i in range(1,len(r['path'])+1))
    best_proposals=proposal_count(bestR)
    teacher_counts[f'best_return_proposal_count_{best_proposals}']+=1
    teacher_counts['selected_strict_witness']+=old['selected_is_strict_witness']
    teacher_counts['selected_return_below_oracle']+=selected['total_reward']<bestR['total_reward']-TOL
    teacher_counts['selected_return_below_STOP']+=selected['total_reward']<stop['total_reward']-TOL
    teacher_counts['root_action0_unchanged']+=bypath[(0,)]['current']==bypath[(0,)]['before_current']
    teacher_counts['root_action0_is_return_optimal']+=root_returns[0]>=ordered_root_returns[0]-TOL
    teacher_counts['best_return_initial_same_as_immediate_quality_opt']+=bestR['path'][0] in [r['action_id'] for r in roots if abs(r['best_cost']-one)<TOL]
    teacher_rows.append(dict(case_id=old['case']['case_id'],selected_path=old['selected_path'],selected_return=selected['total_reward'],
        oracle_return=bestR['total_reward'],stop_return=stop['total_reward'],root_return_margin=ordered_root_returns[0]-ordered_root_returns[1],
        optimal_proposal_count=best_proposals,oracle_first_action=bestR['path'][0]))
    # Preserve log order for ties, as in the original supplemental computation.
    best_return_logorder=max(leaves,key=lambda r:r['total_reward'])
    one_proposal=max((r for r in leaves if proposal_count(r)<=1),key=lambda r:r['total_reward'])
    multistep_rows.append(dict(case_id=old['case']['case_id'],best_return=best_return_logorder['total_reward'],
        best_one_proposal_return=one_proposal['total_reward'],advantage=best_return_logorder['total_reward']-one_proposal['total_reward'],
        best_path=best_return_logorder['path'],one_path=one_proposal['path'],
        quality_gain_pct=100*(one_proposal['finalization']['returned_cost']-best_return_logorder['finalization']['returned_cost'])/one_proposal['finalization']['returned_cost']))
    # Counterfactual coefficient sensitivity uses saved costs/work only, not reruns.
    for lam,aggregate in sensitivity.items():
        def adjusted_return(r):
            return 100*(anchor-r['finalization']['returned_cost'])/anchor-lam*r['work_used']/10000
        rb=max(leaves,key=adjusted_return);rm=max(mono_leaves,key=adjusted_return)
        advantage=adjusted_return(rb)-adjusted_return(rm)
        aggregate['nonmonotone_cases']+=advantage>TOL
        aggregate['stop_cases']+=adjusted_return(rb)<=adjusted_return(stop)+TOL
        aggregate['advantages'].append(advantage)
    counts['reward_optimal_first_kind_'+first_bestR['action']['kind']]+=1
    counts['scenes']+=1;counts['queries']+=len(rows);counts['menus']+=len(menus);counts['leaves']+=len(leaves)
    trace=[]
    for i in range(1,len(bestR['path'])+1):
        t=bypath[tuple(bestR['path'][:i])]
        trace.append(dict(action=t['action'],current=t['current_cost'],best=t['best_cost'],return_so_far=t['total_reward']))
    gains=dict(
        best_depth3_gain_vs_anchor_pct=100*(anchor-best['finalization']['returned_cost'])/anchor,
        best_depth3_gain_vs_one_step_pct=100*(one-best['finalization']['returned_cost'])/one,
        best_depth3_gain_vs_whole_monotone_pct=100*(bestmono['finalization']['returned_cost']-best['finalization']['returned_cost'])/bestmono['finalization']['returned_cost'],
        best_return_gain_vs_anchor_pct=100*(anchor-bestR['finalization']['returned_cost'])/anchor,
        best_return_advantage_vs_stop=bestR['total_reward']-stop['total_reward'],
        best_return_advantage_vs_whole_monotone=bestR['total_reward']-bestmonoR['total_reward'],
    )
    if strict:
        strictbest=min(strict,key=lambda r:r['finalization']['returned_cost'])
        gains['strict_best_gain_vs_one_step_pct']=100*(one-strictbest['finalization']['returned_cost'])/one
        gains['strict_best_reward_advantage_vs_stop']=max(r['total_reward'] for r in strict)-stop['total_reward']
    same_state_actions=collections.Counter()
    nochange=collections.Counter()
    for r in rows:
        kind=r['action']['kind'];counts['action_'+kind]+=1
        prop=r['info']['proposal']
        if prop is not None:
            same_state_actions[(tuple(r['before_current']),r['action']['key'])]+=1
            proposal_counts['queries']+=1
            proposal_counts['accepted']+=int(r['info']['accepted'])
            proposal_counts['changed_route']+=int(r['current']!=r['before_current'])
            proposal_counts['best_improved']+=int(r['info']['best_improved'])
            proposal_counts['current_worse']+=int(r['current_change']>TOL)
            proposal_counts['current_equal']+=int(abs(r['current_change'])<=TOL)
            proposal_counts['current_better']+=int(r['current_change']<-TOL)
            proposal_counts['globally_pruned_retained']+=int(bool(prop['diagnostics'].get('outside_removed_nodes')))
            ch='worse' if r['current_change']>TOL else ('better' if r['current_change']<-TOL else 'equal')
            kind_changes[kind][ch]+=1
            kind_changes[kind]['route_unchanged']+=int(r['current']==r['before_current'])
            step_times[kind].append(r['step_elapsed_s'])
            # Same route+same action earlier on this path: deterministic duplicate proposal.
            prevkeys=[(tuple(bypath[tuple(r['path'][:i])]['before_current']),bypath[tuple(r['path'][:i])]['action']['key']) for i in range(1,len(r['path']))]
            proposal_counts['repeated_same_route_action_within_path']+=int((tuple(r['before_current']),r['action']['key']) in prevkeys)
        else:step_times[kind].append(r['step_elapsed_s'])
    proposal_counts['unique_route_actions']+=len(same_state_actions)
    proposal_counts['duplicate_route_actions_across_tree']+=sum(n-1 for n in same_state_actions.values())
    for m in menus:
        ob=m['observation'];fn=ob['feature_names'];acts=ob['actions'];valid=[i for i,v in enumerate(ob['mask']) if v and acts[i]['kind']!='stop']
        selected=m['selected_action_ids'];allrow={'valid_nonstop':len(valid),'queried_nonstop':sum(acts[i]['kind']!='stop' for i in selected),'pending_menu_work':ob['pending_menu_work']}
        for name in ('retained_overlap_fraction','pool_extra_covered_tasks'):
            vals=[ob['features'][i][fn.index(name)] for i in valid]
            allrow[name+'_positive']=sum(v>0 for v in vals);allrow[name+'_count']=len(vals)
        menu_all.append(allrow)
        if not m['path']:menu_roots.append(allrow)
    replay=json.loads((d/'selected_replay.json').read_text())['state']
    for k,v in replay['timings'].items():timings[k].append(v)
    timings['selected_replay_elapsed_s'].append(old['verification_replay']['elapsed_s'])
    timings['preparation_s'].append(old['preparation_s'])
    timings['tree_elapsed_s'].append(old['tree_elapsed_s'])
    case_stats.append(dict(case_id=old['case']['case_id'],mode=old['case']['mode'],anchor_cost=anchor,one_step_cost=one,
        stop_return=stop['total_reward'],best_quality=compact(best),best_whole_monotone_quality=compact(bestmono),best_return=compact(bestR),best_whole_monotone_return=compact(bestmonoR),
        best_return_trace=trace,flags=flags,gains=gains,root_menu_work=menus[0]['observation']['pending_menu_work'],root_action_count=len(menus[0]['observation']['actions']),
        reward_optimal_first_kind=first_bestR['action']['kind'],reward_optimal_first_current_change=first_bestR['current_change']))

gain_names=sorted(set().union(*(c['gains'].keys() for c in case_stats)))
results=dict(scope='Offline reanalysis of existing G0 tree logs only. No new planning, training or timing. Finite first-seven action menus, max depth3; whole-monotone means current cost never increases by >1e-8 along a logged path.',
    official_metric_mismatches=errors,counts=dict(counts),proposal_counts=dict(proposal_counts),kind_changes={k:dict(v) for k,v in kind_changes.items()},
    gains={k:stats([c['gains'][k] for c in case_stats if k in c['gains']]) for k in gain_names},
    selected_replay_recorded_timings={k:stats(v) for k,v in timings.items()},queried_step_recorded_times={k:stats(v) for k,v in step_times.items()},
    menus=dict(total=len(menu_all),root_pending_work=stats([m['pending_menu_work'] for m in menu_roots]),
        valid_nonstop=stats([m['valid_nonstop'] for m in menu_all]),queried_nonstop=stats([m['queried_nonstop'] for m in menu_all]),
        total_action_entries=sum(m['valid_nonstop'] for m in menu_all),
        retained_overlap_positive_entries=sum(m['retained_overlap_fraction_positive'] for m in menu_all),
        pool_extra_covered_tasks_positive_entries=sum(m['pool_extra_covered_tasks_positive'] for m in menu_all)),
    cases=case_stats)
(OUT/'audit_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'teacher_audit.json').write_text(json.dumps(dict(counts=teacher_counts,rows=teacher_rows),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'multistep_return_audit.json').write_text(json.dumps(multistep_rows,ensure_ascii=False,indent=2),encoding='utf-8')
sensitivity_rows=[dict(lambda_work=lam,nonmonotone_strictly_better_return_cases=v['nonmonotone_cases'],
    STOP_optimal_cases=v['stop_cases'],mean_return_advantage_vs_monotone=float(np.mean(v['advantages']))) for lam,v in sensitivity.items()]
(OUT/'reward_sensitivity.json').write_text(json.dumps(sensitivity_rows,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'cases.csv').open('w',newline='',encoding='utf-8') as f:
    names=['case_id','mode','anchor_cost','one_step_cost','stop_return','reward_optimal_first_kind','reward_optimal_first_current_change']+list(case_stats[0]['flags'])+gain_names
    w=csv.DictWriter(f,fieldnames=names);w.writeheader()
    for c in case_stats:w.writerow({**{k:c[k] for k in names if k in c},**c['flags'],**c['gains']})
print(json.dumps(dict(output=str(OUT),official_metric_mismatches=errors,counts=dict(counts),teacher_counts=dict(teacher_counts),
    multistep_vs_one_proposal=dict(positive_scenes=sum(r['advantage']>TOL for r in multistep_rows),
        mean_return_advantage=float(np.mean([r['advantage'] for r in multistep_rows]))),
    written_files=['audit_results.json','cases.csv','teacher_audit.json','multistep_return_audit.json','reward_sensitivity.json']),ensure_ascii=False,indent=2))
