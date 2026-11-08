"""Independent evidence-only audit. Does not import an evaluation runner/provider."""
from pathlib import Path
import json
import hashlib
import time
from collections import Counter
import numpy as np
from rc3_s3_evaluator.common import read, digest, identity, write_new
from rc3_s3_evaluator.journal import grid
from rc3_s3_recovery_20261108.storage import read_chain

START = time.perf_counter()
ROOT = Path.cwd()
HOME = ROOT / 'rc3_s3_recovery_20261108'
TRACE = HOME / 'paper/review-traces/experiment-result-to-claim/20261108_run03'
RUN = ROOT / 'rc3_s3_evaluator/runs/rc3_v3_global_local_20261106_01'
CONT = RUN / 'continuations/persistence_r1'
SEED = CONT / '11622026'
SOURCE = '767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea'
OLDFAIL = '68e89307acb21cdff0501aa2c6531990909421d266140a88943876f8fe4b0ea9'
NEWFAIL = '88c202d54990cee9fe80d3b8a01fba6c7275a98069a822d3c25e9b701d7eff68'

def check(x, message):
    if not x:
        raise AssertionError(message)
    check.n += 1
    if time.perf_counter() - START > 90:
        raise TimeoutError('Read-only audit90s cap')
check.n = 0

check(digest(HOME/'SOURCE_BINDING_02.json') == SOURCE, 'Exact source02 binding')
source = read(HOME/'SOURCE_BINDING_02.json')
source_checks = {p: digest(ROOT/p) == sha for p,sha in source['source_sha256'].items()}
check(all(source_checks.values()), 'All bound source files unchanged')
index = read(ROOT/'rc3_s3_results_20261108/ARTIFACT_INDEX.json')
old_checks = {r['path']: digest(ROOT/r['path']) == r['sha256'] and (ROOT/r['path']).stat().st_size == r['bytes'] for r in index['artifacts']}
check(len(old_checks) == 180 and all(old_checks.values()), 'Original180 exact bytes')
old, old_tail = read_chain(RUN/'11622026/journal.jsonl')
new, new_tail = read_chain(SEED/'journal.jsonl')
head = read(SEED/'HEAD.json')
check(head['sequence'] == len(new) and head['sha256'] == new_tail, 'New head exactly closes chain')
check(not (SEED/'HEAD.pending').exists(), 'No new pending')
check((RUN/'11622026/HEAD.pending').exists(), 'Original pending retained')

def events_by(chain, name, idfield='id'):
    out = {}
    for e in chain:
        if e['event'] == name:
            rid = e['request']['id'] if idfield == 'request' else e[idfield]
            check(rid not in out, 'No duplicate '+name)
            out[rid] = e
    return out

os_ = events_by(old,'request_start','request')
ns = events_by(new,'request_start','request')
oc = events_by(old,'request_complete')
nc = events_by(new,'request_complete')
oe = events_by(old,'request_error')
ne = events_by(new,'request_error')
check((len(os_),len(oc),len(oe)) == (163,162,1), 'Original paid history')
check((len(ns),len(nc),len(ne)) == (1896,1895,1), 'New paid history')
check(not set(os_) & set(ns), 'No original terminal IDs replayed')
check(set(ns) == set(nc) | set(ne) and not set(nc) & set(ne), 'Each new paid ID terminal')
check(set(oe) == {OLDFAIL} and set(ne) == {NEWFAIL}, 'Exact failed slots retained')

registry = read(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')
check(digest(ROOT/'rc3_global_local_execution_v3/REGISTRY.json') == source['registry_sha256'], 'Registry unchanged')
cases = [{'id':f"S3_{r['case']:03d}"} for r in registry['S3']]
expected_list = [q for s in (11622026,11622027,11622028) for q in grid(cases,s)]
expected = {q['id']:q for q in expected_list}
starts = {**os_, **ns}
for rid,e in starts.items():
    check(expected[rid] == e['request'], 'Exact declared debit')
expected_new = [q['id'] for q in grid(cases,11622026) if q['id'] not in os_]
check(list(ns) == expected_new[:len(ns)], 'Unexecuted requests resumed in exact original order')
check(list(ns)[-1] == NEWFAIL, 'Failing request stopped original order')
check(not (CONT/'11622027/journal.jsonl').exists() and not (CONT/'11622028/journal.jsonl').exists(), 'Other seeds not started')

rows=[]
artifact_checks=[]
for folder, completed in ((RUN/'11622026',oc),(SEED,nc)):
    check(len(list((folder/'results').glob('*.json'))) == len(completed), 'No orphan result artifacts')
    for rid,e in completed.items():
        p=folder/'results'/(rid+'.json')
        check(digest(p) == e['artifact_sha256'], 'Committed result exact hash')
        r=read(p)
        check(r['request'] == expected[rid] and r['valid'] is True, 'Stored valid result and exact request')
        check(all(np.isfinite(r['metrics'][k]) for k in ('J','L','K','raw_J','raw_L','raw_K')), 'Finite stored costs')
        r['_complete_time_s']=e['complete_time_s']; rows.append(r)
        artifact_checks.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=e['artifact_sha256']))

stable={}; raw={}; counts=Counter()
for r in rows:
    q=r['request']; k=(r['case_id'],q['seed'],q['method'])
    value=(r['raw_route_sha256'],r['route_sha256'],r['metrics'])
    check(k not in stable or stable[k] == value,'Same frozen route/cost across repeats and revisions')
    stable[k]=value
    if q['method'] in ('RL_raw','RL_plus_H'):
        k=(r['case_id'],q['seed'],q['kind'],q['repeat'])
        check(k not in raw or raw[k] == r['raw_route_sha256'],'RL pair same raw route')
        raw[k]=r['raw_route_sha256']
    counts[q['kind']]+=1
check(dict(counts) == {'quality':294,'cold':881,'hot':882}, 'Independent coverage tally')

model_checks=[]
model_index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')
for seed, models in model_index['models'].items():
    for role,m in models.items():
        for pk,hk in [('path','checkpoint_sha256'),('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')]:
            check(digest(ROOT/m[pk]) == m[hk], 'Frozen model source bytes')
            model_checks.append(m[pk])
check(len(model_checks)==36,'12 model triplets checked without deserialization')

finals=[]
opens={e['token']:e for e in new if e['event']=='session_open'}
closes={e['token']:e for e in new if e['event']=='session_close'}
check(set(opens)==set(closes) and len(opens)==2,'Both parent sessions closed')
for token in opens:
    d=SEED/'sessions'/token
    final=read(d/'FINAL.json'); seal=read(d/'FINAL_SEAL.json')
    check(digest(d/'FINAL.json')==seal['final_sha256'],'Independent FINAL seal')
    check(final['token']==token and final['status']=='FAILED' and final['journal_close_error'] is None and final['exit_receipt_error'] is None,'Failures preserved and closure healthy')
    check(final['elapsed_s']>=closes[token]['elapsed_s'],'Final includes all heavier closure work')
    finals.append(dict(token=token,elapsed_s=final['elapsed_s'],final_sha256=digest(d/'FINAL.json')))
paid=592+sum(f['elapsed_s'] for f in finals)
check(abs(paid-2896.9277158999903)<1e-8,'No wall refund/reset')
retries=[read(p) for p in (SEED/'io_receipts').glob('ANCHOR_RETRY_*.json')]
chain_hash_counts=Counter(e['sha256'] for e in new)
check(len(retries)==6 and all(chain_hash_counts[r['payload']['same_event_sha256']]==1 for r in retries),'Six real same-anchor retries, no duplicate event')

folder=CONT/'inputs/S3_042'
receipt=read(folder/'RECEIPT.json'); scene=read(folder/'scene.json')
check(digest(folder/'scene.json')==receipt['scene_sha256'] and digest(folder/'arrays.npz')==receipt['arrays_sha256'],'Failed input saved exact bytes')
check(identity(scene)==receipt['content']['scene'],'Failed scene content identity')
with np.load(folder/'arrays.npz',allow_pickle=False) as a:
    check(set(a.files)=={'F','A'}, 'Only F/A saved, no navigation D')
    F=a['F'];A=a['A']
def ah(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
check(ah(F)==receipt['content']['F'] and ah(A)==receipt['content']['A'],'Failed saved F/A content hashes')
cover=(F&A).any((1,2)); uncovered=np.flatnonzero(~cover).tolist()
check(F.shape==(233,800,2) and A.shape==F.shape and cover.all(),'All233 globally have same-arm F&A coverage')
faults=[read(p) for p in (SEED/'faults').glob('REQUEST_FAILURE_*.json')]
check(len(faults)==1 and faults[0]['payload']['request']['id']==NEWFAIL,'Exact new input failure receipt')
failure=faults[0]['payload']
check(failure['journal_healthy'] is True and failure['counts']['navigation_solves']==1 and 'inputs.py\", line 188' in failure['traceback'],'Input/endpoint preparation failure after navigation')
check(not any(k in failure['counts'] for k in ('model_loads','neural_greedy_constructions','H_calls','native_calls','optimizer_updates')),'No recorded policy/native/H operation for failed request')

allcomplete=set(oc)|set(nc)
missing=set(expected)-allcomplete
unexecuted=set(expected)-set(starts)
quality_cells=[]; timing_cells=[]
for group in ('seen_layout_new_tasks','unseen_layout_new_tasks'):
    for size in ((4,3),(6,6),(8,8)):
        ids={f"S3_{c['case']:03d}" for c in registry['S3'] if c['group']==group and tuple(c['size'])==size}
        for kind in ('quality','cold','hot'):
            subset=[q for q in expected_list if q['case_id'] in ids and q['kind']==kind]
            cell=dict(group=group,size=list(size),kind=kind,expected=len(subset),observed=sum(q['id'] in allcomplete for q in subset),missing=sum(q['id'] in missing for q in subset))
            cell['status']='INCOMPLETE' if cell['missing'] else 'COMPLETE'
            (quality_cells if kind=='quality' else timing_cells).append(cell)
check(all(c['status']=='INCOMPLETE' for c in quality_cells+timing_cells),'No original scientific grid cell evaluable')
check(len(missing)==13819 and len(unexecuted)==13817,'Exact distinct missing and unexecuted totals')
main=read(HOME/'results/RESOURCE_ACCOUNTING.json')
check(main['total_debits']==len(starts) and main['complete_requests']==len(rows) and main['failed_requests']==2 and main['not_executed_requests']==len(unexecuted),'Main report independent count agreement')
check(main['original_exact_first_stage_wall_s'] is None,'Exact original wall remains unknown')

elapsed=time.perf_counter()-START
out=dict(status='PASS',scope='READ_ONLY_INDEPENDENT_STOPPED_RUN_AUDIT',source_binding_sha256=SOURCE,source_files_checked=len(source_checks),original_files_checked=len(old_checks),model_source_files_checked=len(model_checks),checkpoint_deserializations=0,old=dict(events=len(old),debits=len(os_),success=len(oc),failures=len(oe)),continuation=dict(events=len(new),debits=len(ns),success=len(nc),failures=len(ne),head_closed=True,new_pending=False,tail_sha256=new_tail),total=dict(debits=len(starts),success=len(rows),failed=2,unexecuted=len(unexecuted),missing_results=len(missing)),coverage=dict(counts),quality_cells=quality_cells,timing_cells=timing_cells,sessions=finals,original_exact_wall_s=None,conservative_carry_s=592,charged_cumulative_wall_s=paid,remaining_wall_s=14400-paid,real_anchor_retry_receipts=len(retries),all_result_files_hash_verified=len(artifact_checks),frozen_route_cost_keys=len(stable),matched_RL_raw_pair_keys=len(raw),failed_input=dict(case_id='S3_042',request=failure['request'],F_shape=list(F.shape),A_shape=list(A.shape),global_same_arm_FA_covered=int(cover.sum()),globally_uncovered_task_ids=uncovered,navigation_D_persisted=False,navigation_recomputed=False,exact_endpoint_unreachable_task_ids='NOT_IDENTIFIABLE_FROM_SAVED_F_A_ONLY',local_window_infeasible='NOT_ESTABLISHED',failure_counts=failure['counts']),reviewer_actual_scientific_calls=dict(target_generation=0,navigation=0,model_load=0,model_inference=0,native=0,H=0,optimizer=0,validation_rerun=0),checks=check.n,wall_s=elapsed,main_ledger_agreement=True)
write_new(TRACE/'READONLY_AUDIT.json',out)
print(json.dumps({k:out[k] for k in ['status','checks','wall_s','source_files_checked','original_files_checked','model_source_files_checked','old','continuation','total','coverage','charged_cumulative_wall_s','remaining_wall_s','real_anchor_retry_receipts','failed_input']},ensure_ascii=False))
