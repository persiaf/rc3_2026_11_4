"""Freeze the saved-input diagnosis before its single allowed navigation call."""
import time
T0=time.perf_counter()
from rc3_s3_evaluator.common import ROOT,read,digest,identity,write_new,label,require,now
from rc3_fast.planner import map_key
from rc3_global_local_impl_review.runtime_guards import current_navigation_sources
from .scan import HOME,INPUT


def main():
    search=read(HOME/'NAVIGATION_SEARCH.json');require(not search['authenticated_matches'],'Reuse authenticated navigation rather than recomputing')
    records=[]
    for case in range(3):
        folder=ROOT/'rc3_global_local/parents';s=folder/f'{case:02}_scene.json';r=folder/f'{case:02}_receipt.json'
        receipt=read(r);require(digest(s)==receipt['scene_sha256'],'S0 saved geometry differs')
        key=map_key(read(s));records.append(dict(scene=label(s),receipt=label(r),receipt_sha256=digest(r),
            physical_map_key=key,same_physical_map=key==search['physical_map_key']))
    require(not any(r['same_physical_map'] for r in records),'Existing S0 navigation map found; examine/reuse first')
    write_new(HOME/'NAVIGATION_SEARCH_SUPPLEMENT.json',dict(S0_records=records,
        reason='Explicit S0 parent artifact path supplement to99 map-key-named navigation receipts',
        authenticated_same_map_matches=0,navigation_calls=0))
    oldsource=read(ROOT/'rc3_s3_recovery_20261108/SOURCE_BINDING_02.json')
    require(digest(ROOT/'rc3_s3_recovery_20261108/SOURCE_BINDING_02.json')=='767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea','Old source binding changed')
    pinned={}
    for name,sha in oldsource['source_sha256'].items():
        p=ROOT/name;require(digest(p)==sha,'Old source changed');pinned[name]=dict(sha256=sha,bytes=p.stat().st_size)
    for base in (ROOT/'rc3_s3_evaluator/runs/rc3_v3_global_local_20261106_01',ROOT/'rc3_s3_recovery_20261108'):
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:pinned[label(p)]=dict(sha256=digest(p),bytes=p.stat().st_size)
    index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')
    for models in index['models'].values():
        for model in models.values():
            for pk,hk in [('path','checkpoint_sha256'),('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')]:
                p=ROOT/model[pk];require(digest(p)==model[hk],'Frozen model source changed');pinned[label(p)]=dict(sha256=model[hk],bytes=p.stat().st_size)
    write_new(HOME/'FROZEN_EVIDENCE_INDEX.json',dict(baseline='c4219abe6838a9d5416cb311c09a0565ef21ac8b',files=dict(sorted(pinned.items()))))
    sources=dict(oldsource['source_sha256'])
    for p in HOME.glob('*.py'):sources[label(p)]=digest(p)
    reg=dict(schema='rc3-s3-042-one-call-navigation-diagnostic-v1',utc=now(),
        authorization_source='Actual user task 2026-11-09: absent existing same-map/same-source artifact, at most1 frozen navigation invocation and300s; no other scientific compute',
        experiment_id='rc3_v3_global_local_20261106_01',input_files=search['input_files'],
        physical_map_key=search['physical_map_key'],navigation_sources=current_navigation_sources(),source_sha256=sources,
        navigation_call_cap=1,wall_cap_s=300,new_task_generation=0,model=0,native=0,H=0,teacher=0,optimizer=0,
        new_FA_generation=0,original60='SEALED',old_D_sha256=None,
        result_identity_rule='Diagnostic recomputation only; cannot claim original D byte recovery without original D digest',
        scientific_S3_grid_requests=0,diagnostic_account_separate=True,formal_S3_inherited_debits=2059,
        formal_S3_inherited_success=2057,formal_S3_inherited_failures=2,formal_S3_inherited_wall_s=2896.9277158999903,
        formal_remaining_requests=[3233,5292,5292],formal_remaining_wall_s=[11503.07228410001,14400,14400],
        receipt_rule='No automatic rerun of a diagnostic failure or any old scientific failure',
        command='.\\rc3_learning\\.venv\\Scripts\\python.exe -B -X utf8 -m rc3_s3_connectivity_20261109.diagnose',
        frozen_evidence_index_sha256=digest(HOME/'FROZEN_EVIDENCE_INDEX.json'))
    write_new(HOME/'DIAGNOSTIC_REGISTRATION.json',reg)
    write_new(HOME/'PREPARATION_RESOURCE.json',dict(registration_and_frozen_hash_wall_s=time.perf_counter()-T0,
        S0_scene_receipts_read=3,source_files_checked=len(oldsource['source_sha256']),frozen_files_pinned=len(pinned),
        navigation_calls=0,model_materializations=0,task_generation=0,
        preliminary_inspection_note='Earlier shell source/Git inspection is maintenance; two fetch connection failures preserved in final Git audit. No scientific calls preceded this registration.'))
    print(dict(registration_sha256=digest(HOME/'DIAGNOSTIC_REGISTRATION.json'),frozen_files=len(pinned),navigation_call_cap=1,wall_cap_s=300))


if __name__=='__main__':main()
