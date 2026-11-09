"""Saved-data and delivery closure only; no new experiments."""
import ast
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];HOME=ROOT/'rc3_gl_v5_20261109'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,m):
    if not x:raise ValueError(m)
def write(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    begin=time.perf_counter();b=read(HOME/'SOURCE_BINDING.json')
    for n,h in b['source_sha256'].items():need(sha(ROOT/n)==h,'Frozen source changed '+n)
    for n,h in b['artifacts'].items():need(sha(HOME/n)==h,'Registration changed '+n)
    for p in (ROOT/'project_management/gl_v5_delivery').glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    required=['EXPERIMENT_REPORT.md','RESULT_TO_CLAIM.md','INDEPENDENT_REVIEW.md','NEXT_STEP_DECISION.md',
        'RESOURCE_ACCOUNTING.json','WEIGHT_LINEAGE.json','GIT_VERSION_AUDIT.json','TRAINING_AND_ROUTE_RESULTS.csv',
        'TRAINING_CURVES.csv','SAMPLE_REFERENCE_COMPARISON.csv','GRADIENTS_BY_INSTANCE.csv','CLAIM_REVIEW_VERDICT.json']
    for n in required:need((HOME/n).is_file(),'Missing delivery '+n)
    links=0
    for p in HOME.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if re.match(r'\w+://',target) or target.startswith('#'):continue
            need((p.parent/target.split('#')[0]).exists(),'Broken relative link '+str(p)+' '+target);links+=1
    with (HOME/'TRAINING_AND_ROUTE_RESULTS.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    counts=Counter(x['row_type'] for x in rows);need(sorted(counts.values())==[128,432],'Wrong row counts')
    r=read(HOME/'RESOURCE_ACCOUNTING.json');need(r['total_requests']==2480 and r['optimizer_updates']==128,'Budget ledger wrong')
    need(r['phase']['status']=='COMPLETE' and r['phase']['wall_s']<=7200,'Phase not complete within budget')
    a=read(HOME/'INDEPENDENT_AUDIT.json');need(a['status']=='PASS','Audit failed')
    inclusive=Counter();exclusive=Counter();native=Counter()
    for p in (HOME/'runs/SCIENCE/DEV432/results').glob('*.json'):
        x=read(p);inclusive.update(x['internal_search']['inclusive_s']);exclusive.update(x['internal_search']['exclusive_s']);native.update(x['internal_search']['counts'])
    prep=Counter()
    for event in r['preparation_events']:
        if event['event']=='new_DEV_input_frozen':prep.update(event['counts'])
    write(HOME/'RESOURCE_ACCOUNTING_DETAIL.json',dict(
        scientific_phase=r['phase'],top_requests_including_all_cache_hits=2480,
        sampled_H_calls=1536,reference_H_calls=486,reference_paid_cache_hits=26,evaluation_common_H_calls=288,
        actual_common_H_total=2310,evaluation_native_requests=144,
        evaluation_search_counts=dict(native),evaluation_internal_inclusive_s=dict(inclusive),evaluation_internal_exclusive_s=dict(exclusive),
        preparation_operation_counts=dict(prep),frozen_development_views=72,
        fresh_reference_greedy_forwards=512,baseline_and_G_share_dynamic_cache=False,
        note='Derived from original saved operation and plan records; no new calls. Internal nested inclusive clocks not summed with outer complete wall.'))
    errors=[p.name for p in (HOME/'runs').rglob('*') if p.is_file() and any(x in p.name.lower() for x in ('pending','fault','emergency','retry'))]
    need(not errors,'Execution abnormality receipt present')
    write(HOME/'DELIVERY_VERIFICATION.json',dict(status='PASS',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        frozen_source_files=len(b['source_sha256']),relative_links_checked=links,csv_rows=dict(counts),
        all_required_deliveries=required,execution_fault_receipts=errors,wall_s=time.perf_counter()-begin,
        new_scientific_requests=0,new_model_inferences=0,new_optimizer_steps=0))
    write(HOME/'MAINTENANCE_RESOURCE_ACCOUNTING.json',dict(scope='Post-science saved-data analysis/review/publication; separate from unchanged scientific budget',
        analysis_wall_s=read(HOME/'results/ANALYSIS_CLOSURE.json')['wall_s'],numerical_audit_wall_s=a['wall_s'],
        closeout_wall_s=time.perf_counter()-begin,original_phase=r['phase'],read_only_analysis_failure_01=read(HOME/'READ_ONLY_ANALYSIS_FAILURE_01.json'),read_only_audit_failure_01=read(HOME/'READ_ONLY_AUDIT_FAILURE_01.json'),new_scientific_requests=0,
        new_model_inferences=0,new_optimizer_steps=0,new_navigation_native_H=0,
        note='Measured suboperations only; interactive code/report/reviewer/Git transport time is maintenance, not a fabricated exact optimizer/planning wall or refunded budget.'))
    print(dict(status='PASS',links=links,rows=dict(counts)))


if __name__=='__main__':main()
