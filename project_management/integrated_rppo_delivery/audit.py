"""One closeout checksum audit; no task generation/inference/navigation/H."""
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path
from rc3_integrated_rppo.common import ROOT,HOME,BASE,EXP,OUT,read,write_new,digest,label,read_chain,object_read,source_path,now


def main():
    start=time.perf_counter();prior=json.loads(subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'],cwd=ROOT,encoding='utf8'))
    allowed={'.gitattributes','.gitignore','README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json'}
    user={'第二章与第三章修订稿_20261029.docx'};bad=[];checked=0
    def one(item):
        name,e=item
        if name in allowed or name in user:return name,'DECLARED_METADATA_OR_USER_FILE'
        p=ROOT/name
        return name,('PASS' if p.is_file() and digest(p)==e['sha256'] else 'CHANGED_OR_MISSING')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name,status in pool.map(one,prior['files'].items()):
            checked+=status=='PASS'
            if status=='CHANGED_OR_MISSING':bad.append(name)
            if checked and checked%5000==0:print('old_hashes',checked,flush=True)
    events,tail=read_chain(OUT/'ledger/events.jsonl');head=read(OUT/'ledger/HEAD.json')
    if head['sequence']!=len(events) or head['sha256']!=tail:raise ValueError('Ledger not closed')
    starts={e['key']:e for e in events if e['event']=='request_start'};done={e['key']:e for e in events if e['event']=='request_done'};failed={e['key']:e for e in events if e['event']=='request_failed'}
    if len(starts)!=sum(e['event']=='request_start' for e in events):raise ValueError('Duplicate predebit')
    valid=0;roles=[];H=0;old_sources={digest(HOME/'SOURCE_BINDING.json')}
    for p in HOME.glob('SOURCE_BINDING_R*.json'):old_sources.add(digest(p))
    for key,e in done.items():
        r=object_read(e['object']);q=starts[key]['query']
        if starts[key]['source_binding_sha256'] not in old_sources or r['request']!=q:raise ValueError('Request provenance changed')
        plan=r.get('plan',r.get('result',{}).get('plan'))
        if not plan['validation']['valid_model_plan'] or not plan['validation']['all_tasks_served']:raise ValueError('Invalid saved route')
        H+=r.get('counts',r.get('counters',{})).get('full_H_calls',0);valid+=1
        m=r.get('model_identity')
        if 'metrics' in r and m is not None:
            expect='RL' if r['study_method'] in ('RL','RL_raw') else r['study_method']
            if r['checkpoint_role']!=expect or r['actor_sha']!=m['actor_sha'] or r['tensor_sha']!=m['tensor_sha'] or m['run_id']!=EXP:roles.append(key)
    status='PASS' if not bad and not roles and not(set(starts)-set(done)-set(failed)) else 'REVISE'
    record=dict(status=status,experiment=EXP,baseline=BASE,source_binding_sha256=digest(source_path()),
        old_manifest_files=len(prior['files']),old_files_byte_checked=checked,unexpected_old_changes=bad,
        allowed_root_maintenance=sorted(allowed),unrelated_user_file_excluded=sorted(user),
        new_request_predebits=len(starts),new_successes=len(done),new_failures=len(failed),valid_route_certificates=valid,role_mismatches=roles,
        new_H=H,unclosed_requests=sorted(set(starts)-set(done)-set(failed)),ledger_tail=tail,
        source_and_model_identity_distinct=True,old_missing_slots_reexecuted=0,original60_semantic_access=0,
        scope='Cryptographic byte and existing certificate audit; no new planner/model/H calls. Not an independent physical robot certification.',wall_s=time.perf_counter()-start,utc=now())
    write_new(HOME/'CLOSEOUT_INTEGRITY_AUDIT.json',record);print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
