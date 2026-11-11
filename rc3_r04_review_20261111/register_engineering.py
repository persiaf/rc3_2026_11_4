"""Identity and resource registration before any R04 numerical engineering."""
from datetime import datetime,timezone
import json
from pathlib import Path
from .run_engineering import digest,write_new
from .evidence_audit import AUDIT_INPUTS

HOME=Path(__file__).resolve().parent
ROOT=HOME.parent


def main():
    if (HOME/'ENGINEERING_REGISTRATION.json').exists():raise RuntimeError('No duplicate registration')
    old=json.loads((ROOT/'rc3_conservative_refinement/integration_20261110/SOURCE_BINDING_R03.json').read_text(encoding='utf-8'))
    sources=dict(old['source_sha256'])
    for package in ('rc3_critic_diagnostics_20261111','rc3_r04_review_20261111'):
        for path in (ROOT/package).glob('*.py'):sources[path.relative_to(ROOT).as_posix()]=digest(path)
    artifacts=dict(old['artifacts'])
    for name in AUDIT_INPUTS:artifacts[name]=digest(ROOT/name)
    artifacts['rc3_r04_review_20261111/R04_ABLATION_PROTOCOL.json']=digest(HOME/'R04_ABLATION_PROTOCOL.json')
    plan=dict(schema='rc3-r04-zero-optimizer-engineering-registration-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),base_commit='853eb6b016df3aee08c233894ab6c24beb3765d8',
        branch='codex/rc3-r04-value-relative-review-20261111',
        authorized_scope='Explicit R04 user task: design, independent unit tests and offline interfaces only; no scientific training',
        source_sha256=sources,artifact_sha256=artifacts,
        environment=dict(python='3.10.20',torch='2.12.1+cu126',numpy='2.2.6',device='cpu'),
        expected_unit_tests=37,unit_fixtures='artificial tensors / synthetic fit-only statistics',
        proposed_scientific_protocol_sha256=digest(HOME/'R04_ABLATION_PROTOCOL.json'),
        proposed_scientific_protocol_approved=False,
        offline_saved_state_rule='first8 groups in predecessor candidate CSV byte order; synthetic zero206 features, no actual encoder/real weight load/rollout',
        independent_saved_csv_audit='one full inherited arithmetic audit plus independently streamed window/phase sums; writes only this new stage',
        tolerances=dict(zero_SL_logits='bitwise exact on legal entries; identical mask',
            mask_invalid='NaN/Inf excluded before mean; all-invalid raises',
            residual_bound=.25,component_loss_adapter_atol=1e-6,component_loss_adapter_rtol=1e-6,
            predecessor_numeric_audit='retains prior ATOL1e-9/RTOL1e-7; component composition2e-5'),
        caps=dict(wall_s=900,module_forward_calls=2048,backward_calls=64,autograd_grad_calls=64,
            unit_test_cases=37,offline_interface_suites=1,saved_csv_audits=1,
            source_reads=10000,source_hashes=5000,synthetic_deserializations=16,
            Actor_optimizer_updates=0,Critic_optimizer_updates=0,real_weight_loads=0,
            scientific_training_runs=0,new_scenes=0,navigation_requests=0,teacher_requests=0,
            native_requests=0,H_requests=0,original60_reads=0,S3_reads=0),
        failure_policy='one-shot; debit before call; failures retained; no unregistered retry or cap reset',
        H_identity=old['H_identity'],old_active_pointers_unchanged=True,
        proposed_R04_scientific_protocol_is_not_execution_authorization=True)
    write_new('ENGINEERING_REGISTRATION.json',plan)
    print(json.dumps(dict(sha256=digest(HOME/'ENGINEERING_REGISTRATION.json'),sources=len(sources),artifacts=len(artifacts),optimizer_cap=0)))


if __name__=='__main__':main()
