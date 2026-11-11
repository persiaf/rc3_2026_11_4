"""One metadata-only lock; charged before reads, no model/label inspection."""
import time
ENTRY=time.perf_counter()
from datetime import datetime,timezone
import hashlib
import json
import subprocess
from .ledger import Budget,HOME,ROOT,digest,write_new,canonical,CAPS
from . import data

BASE='ec4f5fafe788dc27c592db765cc87c3c855ca891'
PROPOSAL='rc3_r04_review_20261111/R04_ABLATION_PROTOCOL.json'

def main():
    if (HOME/'M1_EXECUTION_PROTOCOL.json').exists(): raise RuntimeError('M1 registration immutable; no reselection')
    budget=Budget('metadata',ENTRY)
    try:
        registry=data.metadata_registry(budget.charge)
        schedule=data.batch_schedule(registry)
        write_new(HOME/'M1_DATA_AND_SOURCE_IDENTITY.json',registry)
        write_new(HOME/'BATCH_SCHEDULE.json',schedule)
        proposal=json.loads((ROOT/PROPOSAL).read_text(encoding='utf-8'))
        protocol=dict(schema='rc3-r04-m1-execution-v1',base_commit=BASE,
            parent_proposal=dict(path=PROPOSAL,sha256=digest(ROOT/PROPOSAL)),
            seed=11622026,scope='M1 offline C0/C1 only; M2/M3 and Actor optimizer not authorized',
            original_M1=proposal['M1_proposed_offline_gate'],
            execution_status='AUTHORIZED_THIS_TASK',caps=CAPS,
            selection='64 fit records: fixed33 parent roundrobin, withinparent SHA256(seed|R04-M1-trajectory|request_id). 16 reserve parents selected by exact R04-validation hash from46outsideprevious50; eachreserve x4sizes one smallest trajectory hash, missingcells not replaced.',
            input_recovery='Full268 tensors were not saved; restore exact saved prefixes with accurate frozen SL and immutable existing inputs; no new action/task/navigation/H. Persist reconstructed features for reproducibility.',
            clock=budget.data['scientific_clock'],
            midpoint_rule='At128 C1 value MSE / fit-only conditional constant >1.25 in BOTH state_pooled and parent_equal stops BOTH arms. C0 error alone does not terminate C1.',
            per_size_rule='Each of10x10,4x3,6x6,8x8 >=3 reserve physicalparents; C1value/conditional<=1 in BOTH aggregations',
            component_rule='Each normalizedL,K,H C1/C0<=1.10 in BOTH aggregations',
            initialization='Same default fresh network tensors, seed11622026; no target-informed bias. Physical prediction differs due only to output units.',
            normalization='fixed initial taskcount n>0; no empirical normalization fitted on either fold',
            fit_reduction='state_SUM_per_trajectory_then_episode_MEAN',
            bootstrap=dict(draws=20000,generator='PCG64',seed=11622026,quantile='linear',unit='complete physical_parent',paired=True),
            engineering='Zero-step unit tests and exact actual endpoint Critic/Adam/RNG/cursor/accounting save/load; no duplicate optimizer attempts, no Actor transaction/G2 claim',
            source_identity_excludes_mutable_downstream_authorization=True)
        write_new(HOME/'M1_EXECUTION_PROTOCOL.json',protocol)
        old=json.loads((ROOT/'rc3_conservative_refinement/integration_20261110/SOURCE_BINDING_R03.json').read_text(encoding='utf-8'))
        sources=dict(old['source_sha256'])
        for folder in ('rc3_critic_diagnostics_20261111','rc3_r04_m1_20261111'):
            for path in sorted((ROOT/folder).glob('*.py')): sources[path.relative_to(ROOT).as_posix()]=digest(path)
        artifacts=dict(registry['artifact_sha256'])
        artifacts.update(old['artifacts'])
        artifacts[PROPOSAL]=digest(ROOT/PROPOSAL)
        artifacts['rc3_r04_review_20261111/R04_ACCEPTANCE_CRITERIA.md']=digest(ROOT/'rc3_r04_review_20261111/R04_ACCEPTANCE_CRITERIA.md')
        sources_identity=dict(schema='rc3-r04-m1-source-v1',base_commit=BASE,
            source_sha256=sources,artifact_sha256=artifacts,H_identity=old['H_identity'],
            protocol_sha256=digest(HOME/'M1_EXECUTION_PROTOCOL.json'),
            data_sha256=digest(HOME/'M1_DATA_AND_SOURCE_IDENTITY.json'),schedule_sha256=digest(HOME/'BATCH_SCHEDULE.json'),
            environment=dict(python='3.10.20',torch='2.12.1+cu126',numpy='2.2.6',device='cpu',threads=1,deterministic_algorithms=True))
        write_new(HOME/'SOURCE_BINDING.json',sources_identity)
        authority='用户本轮明确授权：固化M1独立执行版本，必要C0/C1接口，各最多256步离线拟合，预登记留出校准；总512Critic尝试/128保存轨迹/1800秒，Actor及新场景导航教师nativeH均0。'
        write_new(HOME/'PROTOCOL_APPROVAL.json',dict(schema='rc3-explicit-user-protocol-approval-v1',
            approved=True,authority=authority,protocol_sha256=digest(HOME/'M1_EXECUTION_PROTOCOL.json'),
            source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),old_proposal_unchanged=True,
            scope_interpretation='Task authorizes M1 execution; no request to reapprove already explicit scope. Metadata selection and source binding instantiate that approved scope.'))
        write_new(HOME/'STAGE_AUTHORIZATION.json',dict(schema='rc3-r04-m1-stage-authorization-v1',authorized=True,
            authority=authority,stage='M1',protocol_approval_sha256=digest(HOME/'PROTOCOL_APPROVAL.json'),
            protocol_sha256=digest(HOME/'M1_EXECUTION_PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
            caps=CAPS,implementation_GO_required=True,R03_256=False,M2=False,M3=False))
        budget.emit('metadata_lock',protocol_sha256=digest(HOME/'M1_EXECUTION_PROTOCOL.json'),
            data_sha256=digest(HOME/'M1_DATA_AND_SOURCE_IDENTITY.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
            trajectory_count=len(registry['trajectories']),metadata_status=registry['status'])
        budget.close('METADATA_REGISTERED')
        print(json.dumps(dict(status=registry['status'],trajectories=len(registry['trajectories']),coverage=registry['coverage'],
                             protocol_sha256=digest(HOME/'M1_EXECUTION_PROTOCOL.json'),wall_s=budget.data['wall_s']),ensure_ascii=False),flush=True)
    except BaseException as exc:
        budget.close('FAILED_METADATA',exc);raise

if __name__=='__main__':main()
