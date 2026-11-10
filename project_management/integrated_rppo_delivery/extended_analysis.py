"""Raw/H, size/endpoint and actual optimizer identities from closed results."""
import csv
import json
import io
from collections import defaultdict
from rc3_integrated_rppo.common import *
from rc3_integrated_rppo.statistics import paired_effects
from rc3_integrated_rppo.report import derived,write_new


def csv_out(name,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
    derived(HOME/name,f.getvalue().encode('utf-8-sig'))


def main():
    summary=read(HOME/'RESULT_SUMMARY.json');events,tail=read_chain(OUT/'ledger/events.jsonl');choices=summary['choices'];seeds=tuple(int(s) for s in choices)
    kind={'final_confirmation':'quality','selection_development':'selection','promotion':'promotion'}[summary['scope']]
    starts={e['key']:e['query'] for e in events if e['event']=='request_start'};rows=[];diagnostics=defaultdict(list)
    for e in events:
        if e['event']!='request_done':continue
        q=starts[e['key']]
        if q['kind']==kind:
            r=object_read(e['object'])
            if kind=='selection' and r['study_method'] in ('RL','residual_SL'):
                if r['update']!=choices[str(r['seed'])]['chosen'][r['study_method']]['round']:continue
            rows.append(r)
        if q['kind'] in ('train','ablation'):
            r=object_read(e['object']);total=0.;G=[]
            for reward in reversed(r['rewards']):total+=reward/r['normalization'];G.append(total)
            G=np.asarray(list(reversed(G)));adv=G-np.asarray(r['V_old'])
            diagnostics[(q['seed'],q['stage'],q['round'])].extend(adv.tolist())
    table=[]
    groups=['all']+sorted({r['spec']['role'] for r in rows})
    for group in groups:
        for size in ('all',(4,3),(6,6),(8,8)):
            for endpoint in ('all','closed_depot','open_first_last_connected'):
                selected=[r for r in rows if (group=='all' or r['spec']['role']==group) and
                    (size=='all' or (r['spec']['width'],r['spec']['height'])==size) and (endpoint=='all' or r['spec']['endpoints']==endpoint)]
                if not selected:continue
                for metric in ('raw_J','J'):
                    changed=[dict(r,metrics=dict(J=r['metrics'][metric])) for r in selected]
                    for ref in ('SL256','residual_SL','Fixed3','Detour16'):
                        e=paired_effects(changed,ref,seeds=seeds)
                        table.append(dict(scope=summary['scope'],group=group,size=size,endpoint=endpoint,metric=metric,reference=ref,
                            gain=e['mean'],CI95=e['CI95'],parent_count=len(e['parents']),seed_means=e['seed_means'],
                            note='Development screening, not independent confirmation' if kind=='selection' else 'Predeclared physical-parent aggregation'))
    csv_out('RAW_AND_H_CONTRASTS.csv',table)
    lookup={(r['case_id'],r['seed'],r['study_method']):r for r in rows};attribution=[]
    for r in rows:
        if r['study_method']!='RL':continue
        for role in ('SL256','residual_SL'):
            c=lookup[(r['case_id'],r['seed'],role)];m=r['metrics'];b=c['metrics']
            raw_gain=b['raw_J']-m['raw_J'];final_gain=b['J']-m['J'];extra_H=(m['raw_J']-m['J'])-(b['raw_J']-b['J'])
            require(abs(final_gain-raw_gain-extra_H)<1e-7,'Raw/H contribution identity')
            attribution.append(dict(scope=summary['scope'],case=r['case_id'],parent=r['physical_key'],seed=r['seed'],reference=role,
                raw_route_changed=r['raw_route']!=c['raw_route'],final_route_changed=r['route']!=c['route'],raw_cost_gain=raw_gain,
                final_cost_gain=final_gain,RL_H_reduction=m['raw_J']-m['J'],reference_H_reduction=b['raw_J']-b['J'],
                difference_in_H_reductions=extra_H,note='Identity only; absolute H benefit is not a neural learning gain'))
    csv_out('ROUTE_CHANGE_AND_H_ATTRIBUTION.csv',attribution)
    csv_out('ADVANTAGE_DIAGNOSTICS.csv',[dict(seed=k[0],variant=k[1],round=k[2],states=len(a),mean=float(np.mean(a)),
        std=float(np.std(a)),RMS=float(np.sqrt(np.mean(np.square(a)))),negative=sum(v<0 for v in a),positive=sum(v>0 for v in a),
        exactly_zero=sum(v==0 for v in a),abs_le_1e_8=sum(abs(v)<=1e-8 for v in a)) for k,a in diagnostics.items()])
    endpoints=[];closed={(e['seed'],e['variant'],e['round']):e for e in events if e['event']=='round_complete'}
    index=read(HOME/'ACTOR_INDEX.json')
    for actor in index['actors']:
        role=actor['checkpoint_role'];r=actor['update'];stage=actor.get('training_source',{}).get('stage')
        e=closed.get((actor['seed'],stage,r))
        endpoints.append(dict(actor=actor['path'],actor_sha=actor['actor_sha'],tensor_sha=actor['tensor_sha'],role=role,run_id=EXP,
            metadata_update_unit='rollout round for new residual models; original supervised step256 forSL256',metadata_update=r,
            actual_accepted_Actor_Adam_steps=(256 if role=='SL256' else None) if e is None else (e['counts'].get('SL_optimizer_steps',0) if role=='residual_SL' else e['PPO']['actor_cursor']),
            actual_critic_Adam_steps=0 if role in ('SL256','residual_SL') else (None if e is None else e['PPO']['critic_cursor']),
            actual_matched_SL_Adam_steps=None if e is None else e['counts'].get('SL_optimizer_steps',0)))
    write_new(HOME/'MODEL_ENDPOINT_IDENTITY_MAP.json',dict(actors=endpoints,old_role_files_changed=False,
        meaning='Metadata update denotes rollout round, not4x Adam epoch count; exact Adam counters derive from immutable round-completion events',ledger_tail=tail))
    if kind=='quality':
        cells=[]
        for group in ('final_seen','final_unseen'):
            for size in ((4,3),(6,6),(8,8)):
                sub=[r for r in rows if r['spec']['role']==group and (r['spec']['width'],r['spec']['height'])==size]
                effects={ref:paired_effects(sub,ref,seeds=seeds) for ref in ('SL256','residual_SL')}
                cells.append(dict(group=group,size=size,effects=effects,
                    supports_increment=all(e['mean']>=.005 and e['CI95'][0]>0 and all(g>0 for g in e['seed_means'].values()) for e in effects.values())))
        write_new(HOME/'QUALITY_SIX_CELLS.json',cells)
    else:write_new(HOME/'QUALITY_SIX_CELLS.json',dict(status='NOT_EVALUATED_FINAL_STAGE_NOT_STARTED',reason='No substitution of development cases for final216'))
    print(json.dumps(dict(scope=summary['scope'],raw_H_contrasts=len(table),advantage_rounds=len(diagnostics),actor_endpoints=len(endpoints))))


if __name__=='__main__':main()
