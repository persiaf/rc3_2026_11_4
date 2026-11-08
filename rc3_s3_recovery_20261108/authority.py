"""Exact legacy import and one user-scoped repair/resume revision."""
from pathlib import Path
from rc3_s3_evaluator.common import (ROOT,HERE,EXPERIMENT,SEEDS,PROTOCOL,REGISTRY,H_ID,TRAIN_SOURCE,
    require,read,digest,project_file,identity,write_new,label)
from rc3_s3_evaluator.provenance import admit_formal,verify_training_anchors,environment
from .storage import read_chain,regular

HOME=Path(__file__).resolve().parent
RUN=HERE/'runs'/EXPERIMENT
CONT=RUN/'continuations'/'persistence_r1'
OLD_SOURCE='b348fb7261c8e6b8c5b2391649c8bf8d19dfca31f9b26f473e62eb4b82232492'
OLD_INDEX='65088e7e4ff0457d98e053c0413e19a995b6b2983348fc6099a512215ce2d6f5'
FAILED_ID='68e89307acb21cdff0501aa2c6531990909421d266140a88943876f8fe4b0ea9'


def validate_activation(root=RUN,expected_binding=None):
    root=Path(root);parent=root/'continuations';cont=parent/'persistence_r1';marker=root/'PERSISTENCE_RECOVERY_ACTIVATION.json'
    if parent.exists():
        require(all(p.name=='persistence_r1' and p.is_dir() and not p.is_symlink() for p in parent.iterdir()),'Unknown/renamed continuation; cannot create a fresh budget')
    if marker.exists():
        regular(marker);m=read(marker)
        require(m['schema']=='rc3-s3-recovery-activation-v1' and m['experiment_id']==EXPERIMENT and
            m['canonical_continuation']=='continuations/persistence_r1' and cont.is_dir(),'Activation marker without original continuation')
        a=read(cont/'AUTHORITY.json');require(identity(a)==m['authority_identity'],'Activation authority changed')
        if expected_binding is not None:require(a['binding']==expected_binding,'Activation bound to another source/recovery')
    else:require(not cont.exists(),'Continuation without immutable outside activation marker')


def old_permit():
    require(digest(HERE/'S3_SOURCE_BINDING.json')==OLD_SOURCE,'Approved legacy evaluator changed')
    return admit_formal(ROOT/'rc3_s3_results_20261108/authorizations/S3_AUTHORIZATION.json',
        ROOT/'rc3_s3_results_20261108/authorizations/EVALUATION_SOURCE_APPROVAL.json')


def inspect_legacy(root=RUN,index_path=None,*,enforce_exact=True):
    root=Path(root);index_path=index_path or ROOT/'rc3_s3_results_20261108/ARTIFACT_INDEX.json'
    if enforce_exact:require(digest(index_path)==OLD_INDEX,'Wrong legacy evidence index')
    index=read(index_path)['artifacts'];paths={}
    for e in index:
        p=project_file(e['path']) if enforce_exact else root/e['path']
        regular(p);require(p.is_relative_to(root.resolve()) and digest(p)==e['sha256'] and p.stat().st_size==e['bytes'],'Legacy bytes/source differ')
        paths[str(p.resolve())]=e
    validate_activation(root)
    actual={str(p.resolve()) for p in root.rglob('*') if p.is_file() and
        p.relative_to(root).parts[:2]!=('continuations','persistence_r1') and p!=root/'PERSISTENCE_RECOVERY_ACTIVATION.json'}
    require(actual==set(paths),'Unexpected/orphan legacy file or missing evidence')
    if enforce_exact:require(len(index)==180,'Wrong legacy file set')
    folder=root/str(SEEDS[0]);events,tail=read_chain(folder/'journal.jsonl')
    head,pending=read(folder/'HEAD.json'),read(folder/'HEAD.pending')
    for a in (head,pending):
        require(type(a['sequence']) is int and 0<a['sequence']<=len(events) and events[a['sequence']-1]['sha256']==a['sha256'],'Unproved legacy anchor')
    starts={};results={};errors={};open_token=None
    for e in events:
        if e['event']=='session_open':require(open_token is None,'Overlapping legacy session');open_token=e['token']
        elif e['event']=='session_close':require(open_token==e['token'],'Bad legacy closure');open_token=None
        elif e['event']=='request_start':
            q=e['request'];require(q['id'] not in starts,'Duplicate legacy debit');starts[q['id']]=q
        elif e['event']=='request_complete':
            rid=e['id'];require(rid in starts and rid not in results and rid not in errors,'Wrong legacy completion')
            p=folder/'results'/(rid+'.json');require(digest(p)==e['artifact_sha256'],'Legacy committed result differs')
            row=read(p);require(row['request']==starts[rid] and row['valid'] and row['plan']['validation']['valid_model_plan'],'Bad saved legacy result')
            results[rid]=dict(request=starts[rid],artifact_path=label(p) if enforce_exact else str(p),
                artifact_sha256=e['artifact_sha256'],complete_time_s=e['complete_time_s'])
        elif e['event']=='request_error':
            rid=e['id'];require(rid in starts and rid not in results and rid not in errors,'Wrong legacy failure');errors[rid]=e
    require({p.stem for p in (folder/'results').glob('*.json')}==set(results),'Orphan legacy result')
    require(set(starts)==set(results)|set(errors),'Undisposed unknown legacy debit')
    if enforce_exact:
        require(len(events)==3961 and head['sequence']==3959 and pending['sequence']==3960,'Unknown legacy dirty-head pattern')
        require(len(starts)==163 and len(results)==162 and set(errors)=={FAILED_ID} and open_token is not None,'Recovery scope is not the approved incident')
        q=starts[FAILED_ID]
        require(q['case_id']=='S3_003' and q['method']=='Fixed3_endpoint_adapter' and q['kind']=='cold' and q['repeat']==1 and q['seed']==SEEDS[0],'Failed slot substituted')
    return dict(schema='rc3-s3-legacy-import-v1',experiment_id=EXPERIMENT,index_sha256=digest(index_path),
        journal_sha256=digest(folder/'journal.jsonl'),journal_events=len(events),tail_sha256=tail,
        head=head,pending=pending,head_sha256=digest(folder/'HEAD.json'),pending_sha256=digest(folder/'HEAD.pending'),
        request_starts=starts,results=results,errors=errors,old_exact_wall_s=None,
        wall_carry_forward_s=592,first_seed_debits=len(starts),first_seed_success=len(results),first_seed_failure=len(errors),
        historical_open_session=open_token,disposition='Known historical infrastructure failure remains missing; only unexecuted IDs may run')


def baseline_for(legacy,seed):
    if seed==SEEDS[0]:
        return dict(legacy_import_sha256=identity(legacy),debits=163,success=162,infra_failure=1,
            terminal_ids=sorted(legacy['request_starts']),failure_ids=[FAILED_ID],wall_carry_s=592,old_exact_wall_s=None)
    require(seed in SEEDS,'Wrong seed')
    return dict(legacy_import_sha256=identity(legacy),debits=0,success=0,infra_failure=0,terminal_ids=[],failure_ids=[],wall_carry_s=0,old_exact_wall_s=0)


def admit():
    p=HOME/'RECOVERY_REVISION.json';d=read(p)
    require(d['status']=='ACCEPTED_UNDER_USER_SCOPED_REPAIR_AND_RESUME_AUTHORIZATION','Recovery not accepted')
    require(d['experiment_id']==EXPERIMENT and d['protocol_sha256']==PROTOCOL and d['registry_sha256']==REGISTRY and
        d['H_identity']==H_ID and d['training_source_sha256']==TRAIN_SOURCE and d['old_evaluation_source_sha256']==OLD_SOURCE,'Scientific identity changed')
    require(d['policy_seeds']==list(SEEDS) and d['wall_carry_forward_s']==[592,0,0] and
        d['remaining_requests']==[5129,5292,5292] and d['remaining_wall_s']==[13808,14400,14400] and
        d['failed_request_id']==FAILED_ID and d['refill_failed_slot'] is False,'Changed recovery budget/disposition')
    require(digest(project_file(d['user_scope_message']))==d['user_scope_message_sha256'],'Recovery instruction changed')
    source=read(project_file(d['source_binding_path']))
    require(digest(project_file(d['source_binding_path']))==d['source_binding_sha256'] and source['environment']==environment(),'Recovery source/runtime differs')
    for name,sha in source['source_sha256'].items():require(digest(project_file(name))==sha,'Changed runtime: '+name)
    for name,sha in d['engineering_evidence_sha256'].items():require(digest(project_file(name))==sha,'Engineering evidence changed')
    require(read(project_file(d['engineering_result_path']))['status']=='PASS','Repair not verified')
    permit=old_permit();verify_training_anchors();legacy=inspect_legacy()
    require(identity(legacy)==d['legacy_import_identity'],'Legacy import differs from accepted decision')
    binding=dict(experiment_id=EXPERIMENT,old_stage_authorization_sha256=permit.authority_sha256,
        old_scientific_binding=permit.stable(),recovery_revision_sha256=digest(p),
        recovery_source_binding_sha256=d['source_binding_sha256'],legacy_import_identity=identity(legacy))
    validate_activation(RUN,binding)
    return permit,legacy,binding


def initialize(binding,legacy):
    expected=dict(schema='rc3-s3-continuation-authority-v1',binding=binding)
    path=CONT/'AUTHORITY.json'
    validate_activation(RUN,binding)
    if path.exists():require(read(path)==expected,'New directory/authority cannot reset continuation resources')
    else:
        require(not CONT.exists(),'Unbound continuation already exists')
        write_new(RUN/'PERSISTENCE_RECOVERY_ACTIVATION.json',dict(schema='rc3-s3-recovery-activation-v1',experiment_id=EXPERIMENT,
            canonical_continuation='continuations/persistence_r1',authority_identity=identity(expected)))
        write_new(path,expected);write_new(CONT/'BASELINE_IMPORT.json',legacy)
    require(read(CONT/'BASELINE_IMPORT.json')==legacy,'Baseline import changed')
