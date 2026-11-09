"""Future read-only amended-grid statistics, never substitute failed result slots."""
import time
from rc3_s3_evaluator.common import SEEDS,read,digest,identity,require,write_new
from rc3_s3_recovery_20261108.authority import admit as old_admit
from rc3_s3_recovery_20261108.analysis import analyze,saved_rows
from rc3_s3_recovery_20261108.storage import read_chain
from .recovery import RUN,FAILED_OLD,FAILED_SCOPE,admit
from .scope import CONTRACT


def analyze_saved():
    start=time.perf_counter();permit,cases,baseline,binding=admit()
    _,legacy,_=old_admit();rows=saved_rows(legacy)
    keep=('request','case_id','group','size','endpoint','physical_key','valid','metrics','raw_route_sha256','route_sha256','persistence_revision','input_contract')
    for seed in SEEDS:
        folder=RUN/str(seed)
        seal=read(folder/'SCHEDULE_SEAL.json');require(seal['schedule_sha256']==digest(folder/'SCHEDULE_COMPLETE.json'),'Unknown scheduling completion')
        events,_=read_chain(folder/'journal.jsonl')
        for e in events:
            if e['event']!='request_complete':continue
            p=folder/'results'/(e['id']+'.json');require(digest(p)==e['artifact_sha256'],'Scoped result changed')
            r=read(p);thin={k:r[k] for k in keep if k in r};thin['complete_time_s']=e['complete_time_s'];rows.append(thin)
    output=RUN/'SCIENTIFIC_ANALYSIS_SCOPE_R1.json';rid=identity(rows)
    if output.exists():
        old=read(output);require(old['binding']==binding and old['input_result_identity']==rid,'Existing amended analysis changed');return old
    result=analyze(rows,cases)
    require(set(result['missing_ids'])=={FAILED_OLD,FAILED_SCOPE},'Unregistered missing slots after scheduling')
    result.update(scope='V3_PLUS_TARGET_SCOPE_R1_NOT_UNAMENDED_V3_COMPLETE_GRID',input_contract=CONTRACT,binding=binding,
        input_result_identity=rid,original_failed_slots_not_refilled=True,
        gate_interpretation='Original numeric and parent/seed aggregation rules under explicit amended input admission; missing cells retain null. No complete original-grid or overall AND PASS.',
        version_comparability='Original accepted views unchanged by algebra and regression; source-r1 admits outside-target failures. Timings retain persistence/input-contract tags; do not assert equal pre/post overhead.',
        original_protocol_rewritten=False,postprocessing_wall_s=time.perf_counter()-start,new_scientific_calls=0)
    write_new(output,result);return result
