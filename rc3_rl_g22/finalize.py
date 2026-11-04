"""Finalize recorded feedback and frozen-input checks; never runs a plan."""
from .common import ROOT,WORKSPACE,read,write,sha
from .register import frozen_snapshot
from .evaluate import authorize

def main():
    p=authorize();out=ROOT/'outputs';before=read(out/'freeze_before.json');after=frozen_snapshot()
    if before!=after:raise ValueError('Old source, models, data or gates changed')
    after.update(before_after_unchanged=True,all_passed=True,before_snapshot_sha256=sha(out/'freeze_before.json'))
    write(out/'freeze_after.json',after)
    receipt=read(out/'evaluation/receipt.json');summary=read(out/'summary.json')
    audit=read(out/'plan_audit.json');replay=read(out/'inference_replay.json')
    if receipt['attempted_plans']!=72 or receipt['complete_plans']!=72 or not audit['all_passed'] or not replay['all_passed']:
        raise ValueError('Incomplete evaluation or verification')
    if receipt['optimizer_updates']!=0 or receipt['ppo_actions']!=0:raise ValueError('Unexpected training')
    rows=read(out/'evaluation/results.json')
    index=[]
    for row in sorted(rows,key=lambda r:(r['case_id'],r['method'])):
        directory=out/f"evaluation/case_{row['case_id']:02d}_{row['method']}"
        index.append(dict(case_id=row['case_id'],method=row['method'],mode=row['mode'],J=row['J'],L=row['L'],K=row['K'],
            plan_path=(directory/'plan.json').relative_to(ROOT).as_posix(),plan_sha256=sha(directory/'plan.json'),
            evidence_path=(directory/'evidence.json.gz').relative_to(ROOT).as_posix(),evidence_sha256=sha(directory/'evidence.json.gz')))
    write(out/'plan_index.json',index)
    decision=dict(schema='rc3-g22-frozen-feedback-decision-v1',client_date='2026-11-04',evaluation_completed=True,
        planned_and_attempted=72,valid_returned_plans=72,old_development_scenes=12,training_updates=0,ppo_actions=0,
        original_gate_preserved=True,models_unchanged=True,default_deployment='Frozen Fixed3/Detour',
        next_action='Archive frozen-policy feedback; separately preregister feedback coverage and joint ranking/stopping calibration with matched continued-CE control',
        auto_expand=False,adopt_new_model=False,scope='Some unqueried actions help; no overall superiority to strong rules and no DRL increment established',
        protocol_sha256=sha(out/'protocol.json'),counts=summary['counts'])
    write(out/'decision.json',decision)
    log=(ROOT/'tests_initial.log').read_text(encoding='utf-8-sig')
    if 'Ran 16 tests' not in log or not log.rstrip().endswith('OK'):raise ValueError('New pure tests missing')
    checks=dict(all_passed=True,new_pure_tests=16,saved_plans_validated=72,frozen_model_inputs_replayed=52,
        exact_model_outputs=True,frozen_g2_local_files=before['g2_and_legacy']['summary']['local_matched_count'],
        frozen_g21_payloads=len(before['g21_payloads']),old_gates_unchanged=True,training_updates=0,ppo_actions=0,
        counts=summary['counts'],receipts={name:sha(out/name) for name in ('protocol.json','decision.json','plan_audit.json','inference_replay.json','freeze_before.json','freeze_after.json')})
    write(out/'final_checks.json',checks)
    print({k:v for k,v in checks.items() if k not in ('receipts','counts')})

if __name__=='__main__':main()
