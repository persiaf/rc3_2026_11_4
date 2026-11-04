"""Independent NumPy/stdlib recomputation of saved G2.1 supervised evidence.

No model loading, inference, optimizer step, or environment query is performed.
Chosen actions are taken from the diagnostic records; their known quality is
recomputed from hash-verified frozen teacher labels. Saved logits are absent,
so this audit does not independently reproduce argmax or probability values.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_data import load_split, recomputed_weights
from .gates import evaluate


RUNS = ('toy16', 'sampling_original', 'sampling_root_half', 'gain_control', 'gain_gain')
METRICS = ('evidence_greedy_near_best_hit', 'full_legal_greedy_observed_near_best_hit',
    'full_greedy_unknown', 'full_greedy_stop', 'stop_probability', 'unknown_probability_mass',
    'evidence_known_positive_continue', 'full_known_positive_continue',
    'evidence_greedy_Q_gap', 'observed_full_greedy_Q_gap')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected, name):
    if isinstance(expected, (float, np.floating)):
        ok=actual is not None and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-10, atol=1e-12)
    else:
        ok=actual == expected
    if not ok:
        raise ValueError('Mismatch in '+name+': '+repr(actual)+' != '+repr(expected))


def root_half(samples):
    scenes=defaultdict(list)
    for i,s in enumerate(samples):scenes[s['scene_id']].append(i)
    weights=np.zeros(len(samples))
    for indices in scenes.values():
        roots=[i for i in indices if not samples[i]['path']]
        descendants=[i for i in indices if samples[i]['path']]
        if len(roots)!=1 or not descendants:raise ValueError('Invalid scene composition')
        weights[roots[0]]=.5/len(scenes)
        routes=defaultdict(list)
        for i in descendants:routes[tuple(samples[i]['current_route'])].append(i)
        for route in routes.values():
            weights[route]=.5/(len(scenes)*len(routes)*len(route))
    return weights


def recompute_diagnostic(path, samples):
    saved=read(path)
    same(saved['new_environment_queries'],0,'diagnostic environment queries')
    same(len(saved['rows']),len(samples),'diagnostic row count')
    rows=[]
    for index,(row,sample) in enumerate(zip(saved['rows'],samples)):
        legal=np.asarray(sample['legality_mask'],bool)
        evidence=np.asarray(sample['evidence_mask'],bool)
        q=np.asarray(sample['q_values'],float);stop=int(sample['stop_id'])
        ties=evidence & (q >= q[evidence].max()-sample['tie_tolerance'])
        search=evidence.copy();search[stop]=False
        delta=float(q[search].max()-q[stop])
        expected=dict(sample_index=index,scene_id=sample['scene_id'],is_root=not sample['path'],
            teacher_depth=sample['teacher_depth'],delta=delta,known_positive=delta>.001,
            clear_positive=delta>=.1,clear_stop=delta<=-.02,
            legal_actions=int(legal.sum()),evidence_actions=int(evidence.sum()),
            unknown_legal_actions=int((legal & ~evidence).sum()),near_best_actions=int(ties.sum()),
            stop_teacher_near_best=bool(ties[stop]))
        for prefix in ('full','evidence'):
            action=row[prefix+'_greedy_action']
            if type(action) is not int or not 0<=action<len(legal) or not legal[action]:
                raise ValueError('Illegal saved chosen action')
            known=bool(evidence[action])
            if prefix=='evidence' and not known:raise ValueError('Evidence argmax chose unknown action')
            expected[prefix+'_greedy_stop']=action==stop
            expected[prefix+'_known_positive_continue']=bool(known and action!=stop and q[action]-q[stop]>.001)
            expected[prefix+'_selected_gain']=float(q[action]-q[stop]) if known else None
            if prefix=='full':
                expected.update(full_greedy_unknown=not known,full_greedy_quality_known=known,
                    full_legal_greedy_observed_near_best_hit=bool(ties[action]),
                    observed_full_greedy_Q_gap=float(q[evidence].max()-q[action]) if known else None)
            else:
                expected.update(evidence_greedy_near_best_hit=bool(ties[action]),
                    evidence_greedy_Q_gap=float(q[evidence].max()-q[action]))
        for key,value in expected.items():same(row[key],value,str(path.name)+' row%d '%index+key)
        for key in ('stop_probability','unknown_probability_mass','observed_near_best_probability_mass','evidence_probability_mass'):
            value=row[key]
            if not np.isfinite(value) or not 0<=value<=1+1e-12:raise ValueError('Invalid saved probability')
        same(row['evidence_probability_mass']+row['unknown_probability_mass'],1.,'probability partition')
        rows.append(dict(row,**expected))
    groups={}
    for name,predicate in [('all',lambda r:True),('roots',lambda r:r['is_root']),
        ('successors',lambda r:not r['is_root']),('positive_roots',lambda r:r['is_root'] and r['known_positive']),
        ('clear_stop_roots',lambda r:r['is_root'] and r['clear_stop'])]:
        own=[row for row in rows if predicate(row)]
        if not own:continue
        metrics={}
        for key in METRICS:
            values=[r[key] for r in own if r[key] is not None]
            metrics[key]=dict(mean=float(np.mean(values)) if values else None,n=len(values),
                count=sum(bool(v) for v in values) if values and isinstance(values[0],bool) else None)
        groups[name]=dict(n=len(own),metrics=metrics)
    same(saved['groups'],groups,'saved diagnostic aggregates')
    return dict(groups=groups,rows=rows,new_environment_queries=0)


def exposure_audit(request, receipt, updates, samples, seed):
    stage=request['stage'];variant=request['variant'];settings=request['settings']
    weights=None if stage=='toy' else recomputed_weights(samples) if stage=='sampling' and variant=='original' else root_half(samples)
    if weights is not None:
        weights/=weights.sum()
        if not np.allclose(request['sampling_weights'],weights,rtol=0,atol=1e-15):
            raise ValueError('Registered sampling weights differ')
    else:same(request['sampling_weights'],None,'toy full batch')
    rng=np.random.default_rng(seed+91);roots=np.asarray([not s['path'] for s in samples])
    cumulative_root=cumulative_total=0
    for number,row in enumerate(updates,1):
        same(row['update'],number,'update sequence')
        indices=np.arange(len(samples)) if weights is None else rng.choice(len(samples),settings['batch_size'],p=weights)
        cumulative_root+=int(roots[indices].sum());cumulative_total+=len(indices)
        same(row['root_exposures'],cumulative_root,'replayed root exposures')
        same(row['total_exposures'],cumulative_total,'replayed total exposures')
        for key in ('loss','ce','gain','grad_norm'):
            if not np.isfinite(row[key]):raise ValueError('Nonfinite update metric')
        # The sum was formed in float32 before individual values were serialized.
        if not np.isclose(row['loss'],row['ce']+row['gain'],rtol=1e-6,atol=1e-7):
            raise ValueError('Loss decomposition differs beyond float32 rounding')
        if not(stage=='gain' and variant=='gain'):same(row['gain'],0.,'control gain loss')
    same(receipt['root_exposures'],cumulative_root,'receipt root exposure')
    same(receipt['total_exposures'],cumulative_total,'receipt total exposure')
    return dict(root_exposures=cumulative_root,total_exposures=cumulative_total,
        actual_root_fraction=cumulative_root/cumulative_total,
        registered_root_mass=1. if weights is None else float(weights[roots].sum()),
        exact_seeded_sampler_replay=True)


def audit_run(folder, protocol, protocol_sha, samples, selection, teacher_sha, workspace):
    request=read(folder/'request.json');receipt=read(folder/'receipt.json')
    stage=request['stage'];count=protocol[stage]['updates']
    for value in (request,receipt):
        same(value['protocol_sha256'],protocol_sha,'protocol identity')
        same(value['new_environment_queries'],0,'supervised environment queries')
    same(request['settings'],protocol[stage],'prescribed settings')
    same(request['initial_teacher_sha256'],teacher_sha,'initial teacher source')
    same(receipt['request_sha256'],sha(folder/'request.json'),'request identity')
    same(receipt['checkpoint_sha256'],sha(folder/'model.pt'),'checkpoint identity')
    same(receipt['completed'],True,'run completed')
    same(receipt['updates'],count,'registered final update')
    same(receipt['stage'],stage,'stage agreement');same(receipt['variant'],request['variant'],'variant agreement')
    same(request['trainer_sha256'],sha(folder.parent.parent/'train_supervised.py'),'trainer identity')
    initial_path=workspace/'rc3_rl_g2/outputs/training/bc_initial.pt' if stage!='gain' else folder.parent/'sampling_root_half/model.pt'
    same(request['initial_sha256'],sha(initial_path),'initial checkpoint identity')
    if stage!='gain':same(request['initial_sha256'],protocol['initial_bc_sha256'],'frozen BC initialization')
    own=[samples['initial'][i] for i in selection['sample_indices']] if stage=='toy' else samples['initial']
    updates=[json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines() if line]
    same(len(updates),count,'update count')
    initial=recompute_diagnostic(folder/'diagnostic_0000.json',own)
    final_path=folder/('diagnostic_%04d.json'%count)
    final=recompute_diagnostic(final_path,own)
    extra=recompute_diagnostic(folder/'extra_existing_teacher_diagnostic.json',samples['extra'])
    same(receipt['final_diagnostic'],final['groups'],'receipt final aggregates')
    exposure=exposure_audit(request,receipt,updates,own,protocol['seed'])
    gate=None
    if stage=='toy':
        rows=final['rows'];pos=[r for r in rows if r['clear_positive']];neg=[r for r in rows if r['clear_stop']]
        composition=(len(rows)==16 and len(pos)==8 and len(neg)==8
            and all(r['is_root'] and r['teacher_depth']==2 for r in rows)
            and len({r['scene_id'] for r in rows})==16
            and all(bool(r['clear_positive']) != bool(r['clear_stop']) for r in rows))
        near=sum(r['evidence_greedy_near_best_hit'] for r in rows)
        positive=sum(r['evidence_known_positive_continue'] for r in pos)
        gate=dict(passed=composition and near>=15 and positive>=7,composition_valid=composition,
                  evidence_near_best=near,positive_continue=positive,n=16,positive_n=8)
        for key,value in gate.items():same(receipt['gate'][key],value,'toy gate '+key)
        for chosen,sample in zip(selection['samples'],own):
            same(chosen['scene_id'],sample['scene_id'],'fixed toy scene')
            if chosen['selection_label']=='stop':
                if not(sample['ties'][sample['stop_id']] and sum(sample['ties'])==1):raise ValueError('Toy negative lacks unique STOP')
    else:
        gate=evaluate(final,extra,protocol['opportunity_gate'])
        saved_gate=folder/'opportunity_gate.json'
        if saved_gate.exists():
            saved=read(saved_gate)
            for key in ('passed','checks','metrics','scope'):same(saved[key],gate[key],'opportunity gate '+key)
    return dict(run=folder.name,all_passed=True,stage=stage,variant=request['variant'],
        updates=len(updates),initial_sha256=request['initial_sha256'],settings=request['settings'],
        checkpoint_sha256=receipt['checkpoint_sha256'],request_sha256=sha(folder/'request.json'),
        receipt_sha256=sha(folder/'receipt.json'),updates_sha256=sha(folder/'updates.jsonl'),
        final_diagnostic_sha256=sha(final_path),extra_diagnostic_sha256=sha(folder/'extra_existing_teacher_diagnostic.json'),
        exposure=exposure,initial_groups=initial['groups'],final_groups=final['groups'],extra_groups=extra['groups'],gate=gate)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-all',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent;out=root/'outputs';workspace=root.parent
    protocol=read(out/'protocol.json');protocol_sha=sha(out/'protocol.json')
    same(protocol_sha,(out/'protocol.sha256').read_text().strip(),'protocol sidecar')
    selection=read(out/'selection.json');same(sha(out/'selection.json'),protocol['selection_sha256'],'selection identity')
    samples={};sources={}
    for split in ('initial','extra'):
        samples[split],_,sources[split]=load_split(workspace,split)
    completed={};pending=[]
    for name in RUNS:
        folder=out/name
        if not(folder/'receipt.json').exists():pending.append(name);continue
        completed[name]=audit_run(folder,protocol,protocol_sha,samples,selection,sources['initial']['teacher_sha256'],workspace)
    pairs={}
    for name,left,right in [('sampling','sampling_original','sampling_root_half'),('gain','gain_control','gain_gain')]:
        if left not in completed or right not in completed:continue
        a,b=completed[left],completed[right]
        checks=dict(same_initial=a['initial_sha256']==b['initial_sha256'],same_settings=a['settings']==b['settings'],
                    same_updates=a['updates']==b['updates'],same_initial_diagnostic=a['initial_groups']==b['initial_groups'])
        if name=='gain':checks['same_actual_root_exposure']=a['exposure']==b['exposure']
        if not all(checks.values()):raise ValueError('Paired comparison mismatch: '+name)
        pairs[name]=checks
    if args.require_all and pending:raise ValueError('Required runs still pending: '+repr(pending))
    report=dict(schema='rc3-g21-independent-supervised-result-audit-v1',all_passed=True,
        complete_five_run_scope=not pending,completed_runs=list(completed),pending_runs=pending,
        script_sha256=sha(__file__),protocol_sha256=protocol_sha,teacher_sources=sources,
        comparisons=pairs,runs=completed,new_environment_queries=0,torch_used=False,
        limitations=['Recorded chosen actions audited against frozen finite-depth teacher labels; no inference reproduced.',
            'Probabilities are range/partition checked and reaggregated, not regenerated without saved logits.',
            'Checkpoint SHA and lineage receipts checked; tensors not deserialized.',
            'Existing teacher states are development data, not unseen generalization or planner-quality evidence.'])
    destination=out/'results_audit.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(destination),all_passed=True,completed=list(completed),pending=pending,
                         paired_checks=pairs),ensure_ascii=False))


if __name__=='__main__':main()
