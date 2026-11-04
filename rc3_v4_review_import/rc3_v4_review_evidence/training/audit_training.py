"""Read-only v4 training/data/export audit; no Torch/PT load or retraining.

Numeric pickle caches are opcode-inspected and opened with a restricted NumPy
whitelist. The source root is never modified. Optional cache diagnostics use
only the original validation set and cannot change the frozen test decision.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from safe_numeric import load


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def same(a,b):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,type(a)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return bool(a==b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cache-summary',type=Path)
    args=parser.parse_args();root=args.source_root.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True);sys.dont_write_bytecode=True
    sys.path.insert(0,str(root))
    from rc3_v4.model import NumpyPolicy
    started=time.perf_counter();r=root/'rc3_v4'
    manifest=json.loads((r/'data/manifest.json').read_text())
    summary=json.loads((r/'training/summary.json').read_text())
    v4,op4=load(r/'data/data.pkl')
    v3,op3=load(root/'rc3_v3/data_v3_20261102/data.pkl')
    rows=v4['states'];old_rows=v3['states']
    fields_unchanged={k:same(v3[k],v4[k]) for k in v3 if k!='states'}
    unchanged_rows=[same({k:v for k,v in row.items() if k!='graphs'},old) for row,old in zip(rows,old_rows)]
    cases=json.loads((r/'data/test_cases.json').read_text())
    train_seeds={int(s['seed']) for s in v4['scenes']['train']}
    val_seeds={int(s['seed']) for s in v4['scenes']['validation']}
    test_seeds={int(s['seed']) for s in cases}
    old_test_path=root/'rc3_v3/data_v3_20261102/test_cases.json'
    old_test=json.loads(old_test_path.read_text()) if old_test_path.exists() else None
    old_test_seeds={int(s['seed']) for s in old_test} if old_test is not None else None
    positive_states=sum(bool((row['gains'][row['mask']]>1e-7).any()) for row in rows)
    gain=np.concatenate([row['gains'][row['mask']] for row in rows])
    flat=np.concatenate([row['features'][row['mask']] for row in rows])
    mean,std=flat.mean(0),flat.std(0).clip(.05)
    pos=gain>1e-7
    graph_counts=[];shape_ok=True;finite_ok=True;max_c=0;max_t=0
    for row in rows:
        graph_counts.append(len(row['graphs']))
        shape_ok &= len(row['graphs'])==len(row['features'])==len(row['mask'])==len(row['gains'])
        for g in row['graphs']:
            c,t=len(g['cand']),len(g['task']);max_c=max(c,max_c);max_t=max(t,max_t)
            shape_ok &= g['cand'].shape==(c,10) and g['task'].shape==(t,4) and g['reach'].shape==(t,c,2) and g['dist'].shape==(c,c)
            finite_ok &= all(np.isfinite(v).all() for v in g.values())
    hashes={
        'v3_data':dict(expected=manifest['source_sha256'],actual=sha(root/'rc3_v3/data_v3_20261102/data.pkl')),
        'v4_data':dict(expected=manifest['data_sha256'],actual=sha(r/'data/data.pkl')),
        'test_cases':dict(expected=manifest['test_sha256'],actual=sha(r/'data/test_cases.json'))}
    for p,digest in summary['source_sha256'].items():
        name=p.replace('\\','/').rsplit('/',1)[-1]
        hashes[name]=dict(expected=digest,actual=sha(r/name))
    with (r/'training/history.csv').open(encoding='utf-8-sig',newline='') as f:history=list(csv.DictReader(f))
    runs=[]
    for run in summary['runs']:
        variant,seed=run['variant'],run['seed']
        entries=[row for row in history if row['variant']==variant and int(row['seed'])==seed]
        chosen=max(entries,key=lambda row:(float(row['val_gain_pct']),-float(row['val_calls'])))
        path=r/'training'/f'{variant}_seed{seed}.npz';model=NumpyPolicy.load(path)
        count=sum(v.size for k,v in model.w.items() if k not in ('mean','std'))
        runs.append(dict(variant=variant,seed=seed,selected_epoch=run['epoch'],
            epochs_recorded=[int(row['epoch']) for row in entries],
            history_selection_correct=int(chosen['epoch'])==run['epoch'],
            metadata_matches=model.metadata['seed']==seed and model.metadata['epoch']==run['epoch'] and same(model.metadata['validation'],run['validation']),
            parameter_count=count,parameter_count_matches=count==run['parameter_count'],
            bytes=path.stat().st_size,sha256=sha(path),
            normalization_exact=np.array_equal(model.w['mean'],mean) and np.array_equal(model.w['std'],std),
            phase=model.metadata['phase'],stop_rule=model.stop_rule,stop_threshold=model.stop_threshold))
    selected={}
    for variant,item in summary['selected'].items():
        model=NumpyPolicy.load(r/'training'/f'{variant}.npz')
        best=max([run for run in summary['runs'] if run['variant']==variant],key=lambda row:(row['validation']['mean_gain_pct'],-row['validation']['mean_calls']))
        selected[variant]=dict(seed=model.metadata['seed'],epoch=model.metadata['epoch'],
            val_gain_pct=model.metadata['validation']['mean_gain_pct'],
            chosen_by_validation=same(item,best),
            exported_npz_identical_to_selected_seed=sha(r/'training'/f'{variant}.npz')==sha(r/'training'/f'{variant}_seed{item["seed"]}.npz'))
    expected_updates=len(summary['runs'])*summary['config']['epochs']*int(np.ceil(len(rows)/summary['config']['batch_size']))
    updates_history_correct=all(int(row['updates'])==i//9*3840+int(row['epoch'])*48 for i,row in enumerate(history))
    result=dict(scope='Read-only training evidence audit. No PT load, no Torch, no training, no new-test tuning.',
        pickle_safety=dict(opcode_scan_first=True,numeric_numpy_whitelist=True,v3_opcodes=op3,v4_opcodes=op4),
        counts=dict(train_scenes=len(train_seeds),validation_scenes=len(val_seeds),new_test_scenes=len(test_seeds),
            train_states=len(rows),training_scenes_represented=len({row['scene_id'] for row in rows}),
            states_per_scene=dict(collections.Counter(collections.Counter(row['scene_id'] for row in rows).values())),
            positive_states=positive_states,zero_states=len(rows)-positive_states,
            legal_actions=len(gain),positive_actions=int(pos.sum()),zero_actions=int((~pos).sum()),
            teacher_counts=dict(collections.Counter(row['teacher'] for row in rows)),
            step_counts=dict(collections.Counter(int(row['step']) for row in rows)),
            train_sizes=dict(collections.Counter(str((s['nx'],s['ny'])) for s in v4['scenes']['train'])),
            test_sizes=dict(collections.Counter(str((s['nx'],s['ny'])) for s in cases))),
        data_reuse=dict(all_old_top_level_fields_unchanged=all(fields_unchanged.values()),field_checks=fields_unchanged,
            rows_equal_except_added_graphs=len(rows)==len(old_rows) and all(unchanged_rows),
            graph_shapes_and_lengths_valid=bool(shape_ok),all_graph_values_finite=bool(finite_ok),max_candidates=max_c,max_tasks=max_t),
        split_checks=dict(train_validation_disjoint=not train_seeds&val_seeds,test_train_disjoint=not test_seeds&train_seeds,
            test_validation_disjoint=not test_seeds&val_seeds,new_old_test_disjoint=not test_seeds&old_test_seeds if old_test_seeds is not None else None,
            old_test_metadata_available=old_test is not None,
            historical_workspace_exclusion='Uploaded scan records 2863 seeds from 3111 source files; full original historical workspace is not independently available, so this audit does not attest all history.'),
        hashes={k:dict(**v,match=v['expected']==v['actual']) for k,v in hashes.items()},
        training=dict(runs=len(runs),epochs_per_run=summary['config']['epochs'],batch_size=summary['config']['batch_size'],
            updates_expected=expected_updates,updates_recorded=summary['updates'],history_rows=len(history),
            all_history_update_counts_correct=updates_history_correct,runs_details=runs,selected=selected,
            gain_scale_derived=float(max(np.quantile(gain[pos],.9),.05)),
            positive_weight_derived=float(min((~pos).sum()/pos.sum(),8.)),
            actually_retrained=False),
        source_files_modified=False)
    if args.cache_summary:
        diagnostic=json.loads(args.cache_summary.read_text());errs=[]
        for i,reference in enumerate(v4['reference_costs']['validation']):
            found=[row for row in diagnostic['records'] if row['scene']==i]
            if not found:continue
            expected=summary['selected']['structured']['validation']['gains_pct'][i]
            errs.extend(abs(100*(reference-row['objective'])/reference-expected) for row in found)
        result['validation_cache_crosscheck']=dict(scenes=diagnostic['scenes'],plans=len(diagnostic['records']),
            source='Existing first N validation scenes; complete objective checked against selected training history.',
            max_gain_percentage_point_difference=max(errs) if errs else None,
            all_head_arrays_bitwise_equal=diagnostic['all_bitwise_equal'],all_final_plans_equal=diagnostic['all_final_plan_objects_equal'])
    result['elapsed_s']=time.perf_counter()-started
    (out/'training_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('pickle_safety','hashes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
