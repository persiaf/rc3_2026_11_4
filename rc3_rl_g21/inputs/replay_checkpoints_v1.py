"""Zero-update CPU replay of final G2.1 checkpoints on saved teacher states.

This script never creates an optimizer, scene, search environment or proposal.
State evaluations count repeated model/state pairs; forward calls count batches
of at most 16 states, not environment actions or independent scenes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from .train_supervised import ROOT, G2, checked_samples, configure, diagnose, read, write
from rc3_rl_g2.checkpoints import load_model, sha256

RUNS=('toy16','sampling_original','sampling_root_half','gain_control','gain_gain')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def compare(saved,replayed,absolute_tolerance=1e-12,relative_tolerance=1e-10):
    """Compare every saved action, probability and aggregate, retaining deltas."""
    result=dict(matches=True,exactly_equal=True,numeric_values=0,max_absolute_difference=0.,
        max_relative_difference=0.,max_difference_path=None,mismatch_count=0,mismatches=[])
    def mismatch(path,reason,a,b):
        result['matches']=False;result['mismatch_count']+=1
        if len(result['mismatches'])<20:
            result['mismatches'].append(dict(path=path,reason=reason,saved=a,replayed=b))
    def visit(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            if a.keys()!=b.keys():mismatch(path,'different keys',list(a),list(b))
            for key in a.keys()&b.keys():visit(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):mismatch(path,'different lengths',len(a),len(b))
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif type(a) in (int,float) and type(b) in (int,float):
            if not math.isfinite(a) or not math.isfinite(b):
                mismatch(path,'nonfinite numeric value',a,b);return
            delta=abs(a-b);relative=delta/max(abs(a),abs(b),1e-30)
            result['numeric_values']+=1
            if delta:
                result['exactly_equal']=False
                if delta>result['max_absolute_difference']:
                    result['max_absolute_difference']=delta;result['max_difference_path']=path
                result['max_relative_difference']=max(result['max_relative_difference'],relative)
            # IDs/counts must agree exactly; tolerance is for floating metrics.
            if (type(a) is int and type(b) is int and a!=b) or not math.isclose(a,b,rel_tol=relative_tolerance,abs_tol=absolute_tolerance):
                mismatch(path,'numeric mismatch',a,b)
        elif type(a)!=type(b) or a!=b:
            result['exactly_equal']=False;mismatch(path,'value/type mismatch',a,b)
    visit(saved,replayed,'')
    result['exactly_equal']=result['exactly_equal'] and result['matches']
    result.update(absolute_tolerance=absolute_tolerance,relative_tolerance=relative_tolerance,
        saved_canonical_sha256=canonical_sha(saved),replayed_canonical_sha256=canonical_sha(replayed))
    return result


def parameters_sha(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        array=value.detach().cpu().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def counted_diagnose(model,samples):
    count=dict(state_evaluations=0,forward_calls=0,batch_sizes=[])
    def count_forward(module,args):
        size=len(args[0]);count['state_evaluations']+=size;count['forward_calls']+=1;count['batch_sizes'].append(size)
    hook=model.register_forward_pre_hook(count_forward)
    started=time.perf_counter()
    try:diagnostic=diagnose(model,samples)
    finally:hook.remove()
    count['elapsed_s']=time.perf_counter()-started
    if count['state_evaluations']!=len(samples) or count['forward_calls']!=(len(samples)+15)//16:
        raise AssertionError('observed forward counter differs from diagnose batching')
    return diagnostic,count


def main():
    output=ROOT/'outputs';destination=output/'checkpoint_replay.json'
    baseline_paths=[output/'frozen_bc_diagnostic.json',output/'frozen_bc_extra_diagnostic.json']
    if destination.exists() or any(path.exists() for path in baseline_paths):
        raise FileExistsError('Preserve prior replay/baseline evidence; do not overwrite')
    protocol=read(output/'protocol.json');protocol_sha=sha256(output/'protocol.json')
    if protocol_sha!=(output/'protocol.sha256').read_text().strip():raise ValueError('Protocol changed')
    if sha256(output/'selection.json')!=protocol['selection_sha256']:raise ValueError('Toy selection changed')
    configure(protocol['seed'])
    import torch
    if torch.get_num_threads()!=1:raise AssertionError('CPU execution is not single-threaded')
    initial=checked_samples();extra=checked_samples('extra');selection=read(output/'selection.json')
    selected=[initial[index] for index in selection['sample_indices']]
    runs=[];started=time.perf_counter()
    for name in RUNS:
        folder=output/name;receipt=read(folder/'receipt.json');checkpoint=folder/'model.pt'
        actual_sha=sha256(checkpoint)
        if actual_sha!=receipt['checkpoint_sha256']:raise ValueError('Checkpoint SHA changed before loading: '+name)
        if (receipt['completed'] is not True or receipt['protocol_sha256']!=protocol_sha
                or receipt['updates']!=protocol[receipt['stage']]['updates']
                or receipt['request_sha256']!=sha256(folder/'request.json')):
            raise ValueError('Final receipt identity changed: '+name)
        model,payload=load_model(checkpoint,'cpu')
        if payload.get('protocol_sha256')!=protocol_sha or payload.get('updates')!=receipt['updates']:
            raise ValueError('Final checkpoint metadata differs: '+name)
        model.eval();before=parameters_sha(model)
        primary,primary_count=counted_diagnose(model,selected if name=='toy16' else initial)
        held,extra_count=counted_diagnose(model,extra)
        final_path=folder/f"diagnostic_{receipt['updates']:04d}.json"
        primary_compare=compare(read(final_path),primary)
        extra_path=folder/'extra_existing_teacher_diagnostic.json'
        extra_compare=compare(read(extra_path),held)
        receipt_compare=compare(receipt['final_diagnostic'],primary['groups'])
        after=parameters_sha(model)
        unchanged=before==after and sha256(checkpoint)==actual_sha
        if not unchanged:raise AssertionError('Read-only replay changed checkpoint/parameters')
        item=dict(run=name,checkpoint_sha256=actual_sha,checkpoint_sha_checked_before_load=True,
            final_diagnostic_sha256=sha256(final_path),extra_diagnostic_sha256=sha256(extra_path),
            primary=primary_count,extra=extra_count,primary_comparison=primary_compare,
            extra_comparison=extra_compare,receipt_groups_comparison=receipt_compare,
            parameters_unchanged=True,parameter_sha256=before,
            matches=primary_compare['matches'] and extra_compare['matches'] and receipt_compare['matches'])
        runs.append(item)
        print(json.dumps(dict(run=name,matches=item['matches'],state_evaluations=primary_count['state_evaluations']+extra_count['state_evaluations'],
            forward_calls=primary_count['forward_calls']+extra_count['forward_calls'])),flush=True)
    baseline_checkpoint=G2/'outputs/training/bc_initial.pt'
    if sha256(baseline_checkpoint)!=protocol['initial_bc_sha256']:raise ValueError('Frozen initial BC SHA changed')
    baseline,_=load_model(baseline_checkpoint,'cpu');baseline.eval();baseline_before=parameters_sha(baseline)
    primary,primary_count=counted_diagnose(baseline,initial)
    held,extra_count=counted_diagnose(baseline,extra)
    if parameters_sha(baseline)!=baseline_before:raise AssertionError('Frozen BC parameter change')
    write(baseline_paths[0],primary);write(baseline_paths[1],held)
    baseline_record=dict(checkpoint_sha256=protocol['initial_bc_sha256'],checkpoint_sha_checked_before_load=True,
        primary=primary_count,extra=extra_count,primary_output_sha256=sha256(baseline_paths[0]),
        extra_output_sha256=sha256(baseline_paths[1]),parameters_unchanged=True,
        scope='New zero-update inference on existing teacher states; not rollout or new teacher labels')
    pairs=[item[split] for item in runs for split in ('primary','extra')]
    baseline_pairs=[primary_count,extra_count]
    result=dict(schema='rc3-g21-zero-update-checkpoint-replay-v1',protocol_sha256=protocol_sha,
        script_sha256=sha256(__file__),trainer_sha256=sha256(ROOT/'train_supervised.py'),
        cpu_threads=torch.get_num_threads(),cpu_interop_threads=torch.get_num_interop_threads(),device='cpu',
        all_matches=all(item['matches'] for item in runs),runs=runs,frozen_initial_bc=baseline_record,
        checkpoint_replay_state_evaluations=sum(item['state_evaluations'] for item in pairs),
        checkpoint_replay_forward_calls=sum(item['forward_calls'] for item in pairs),
        all_state_evaluations=sum(item['state_evaluations'] for item in pairs+baseline_pairs),
        all_forward_calls=sum(item['forward_calls'] for item in pairs+baseline_pairs),
        distinct_teacher_states=len(initial)+len(extra),distinct_teacher_scenes=len({('initial',s['scene_id']) for s in initial}|{('extra',s['scene_id']) for s in extra}),
        counts_scope='Repeated checkpoint/state evaluations; one batched forward handles up to 16 states. None are environment actions or independent new scenes.',
        optimizer_updates=0,new_environment_queries=0,new_teacher_queries=0,elapsed_s=time.perf_counter()-started)
    write(destination,result)
    print(json.dumps({key:result[key] for key in ('all_matches','checkpoint_replay_state_evaluations','checkpoint_replay_forward_calls','all_state_evaluations','all_forward_calls','optimizer_updates','new_environment_queries')}),flush=True)
    if not result['all_matches']:raise SystemExit('Replay differences retained for review')


if __name__=='__main__':main()
