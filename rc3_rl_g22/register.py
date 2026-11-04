"""Freeze a new evaluation-only contract without touching the G2.1 gates."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
from .common import ROOT,WORKSPACE,METHODS,read,write,sha,check_contract

def frozen_snapshot():
    from rc3_rl_g21.audit_freeze import inventory
    old=inventory(WORKSPACE)
    if not old['all_passed']:raise ValueError('Frozen G2/legacy audit failed')
    manifest=read(WORKSPACE/'rc3_rl_g21/MANIFEST.sha256.json')
    rows={name:dict(expected=identity['sha256'],actual=sha(WORKSPACE/name)) for name,identity in manifest['files'].items()}
    if not all(r['actual']==r['expected'] for r in rows.values()):raise ValueError('G21 frozen payload changed')
    return dict(g2_and_legacy=old,g21_payloads=rows,g21_manifest_sha256=sha(WORKSPACE/'rc3_rl_g21/MANIFEST.sha256.json'),
        g21_archive_sha256=sha(WORKSPACE/'rc3_rl_g21_delivery.zip'),
        g21_gate_files_exist={name:(WORKSPACE/'rc3_rl_g21/outputs'/name).exists() for name in ('opportunity_gate.json','ppo_authorization.json')})

def main():
    output=ROOT/'outputs';output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise FileExistsError('Keep preregistration immutable')
    request=Path('C:/Users/Administrator/.codex/attachments/d5107243-ee0b-43de-8369-72008eeaff05/已粘贴的文本.txt')
    (ROOT/'inputs').mkdir(exist_ok=True);(ROOT/'inputs/user_request.txt').write_bytes(request.read_bytes())
    old=read(WORKSPACE/'rc3_rl_g2/outputs/protocol.json')
    selected=[]
    for mode in ('regular','mixed','blocked'):
        selected.extend(sorted((c for c in old['splits']['teacher_extra'] if c['mode']==mode),key=lambda c:c['case_id'])[:4])
    selected.sort(key=lambda c:c['case_id'])
    for c in selected:
        source=WORKSPACE/f"rc3_rl_g2/outputs/teacher_extra/case_{c['case_id']:04d}/input.json"
        original=read(source);scene=original['source']['scene']
        if len(scene['tables'])!=100 or scene['seed']!=c['seed']:raise ValueError('Scene identity mismatch')
        destination=ROOT/f"inputs/scene_{c['case_id']:02d}.json";write(destination,scene)
        c['scene_path']=destination.relative_to(WORKSPACE).as_posix();c['scene_sha256']=sha(destination)
        c['source_input_sha256']=sha(source)
    models={}
    for name,run in [('continued_ce','gain_control'),('ce_gain','gain_gain')]:
        path=WORKSPACE/f'rc3_rl_g21/outputs/{run}/model.pt';receipt=read(path.parent/'receipt.json')
        if sha(path)!=receipt['checkpoint_sha256']:raise ValueError('Frozen model identity mismatch')
        models[name]=dict(path=path.relative_to(WORKSPACE).as_posix(),sha256=sha(path),parent_run=run)
    protocol=dict(schema='rc3-g22-frozen-evaluation-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        client_date='2026-11-04',evaluation_authorized=True,training_allowed=False,ppo_allowed=False,
        max_plans=72,methods=list(METHODS),cases=selected,models=models,config=old['config'],
        case_selection='First four original numeric teacher_extra case IDs per mode, sorted by case ID; no performance filtering',
        scene_scope='12 previously used development scenes, not a new independent test set',
        early_stop=dict(max_proposals=4,patience=1,progress_tolerance=1e-12),
        soft_complete_limit_s=5.0,soft_limit_scope='No new search starts after five seconds; current prepare/proposal and mandatory final validation may overrun',
        original_detour_horizon=16,method_order='Six-method rotation by selected case index; sequential timing, CPU one thread',
        timing_scope='JSON scene read, fresh geometry/navigation, one certified Fixed3 anchor, menus, policy, repairs and final validation; model resident; evidence serialization excluded',
        prepared_input_cache=False,model_load_scope='Load once before the 72 measured plans; record per-model load time separately and provide add-load sensitivity',
        old_gate_scope='No G21 passed flag/threshold/authorization changes; separate evaluation-only authority from latest user request',
        feedback_scope='Observed immediate effect and this frozen-policy suffix, not optimal Q or original finite-tree labels',
        supplemental_link_status='Supplement not local; linked library returned cache miss. This new contract is derived from the complete pasted body; unseen proposal limits not claimed.',
        source_request_sha256=sha(request),g2_protocol_sha256=sha(WORKSPACE/'rc3_rl_g2/outputs/protocol.json'),
        extra_teacher_sha256=sha(WORKSPACE/'rc3_rl_g2/outputs/teacher_extra/teacher.pkl'),
        success_decision='Report actual paired quality, net return and complete time. No automatic model adoption or training, regardless of result.',
        failure_scope='Count each attempted scene-method once, preserve failures, never retry a plan beyond the 72 slots')
    protocol['implementation_sha256']={name:sha(ROOT/name) for name in ('common.py','register.py','evaluate.py')}
    check_contract(protocol)
    snapshot=frozen_snapshot();write(output/'freeze_before.json',snapshot)
    write(output/'protocol.json',protocol);(output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n',encoding='ascii')
    print(dict(cases=[c['case_id'] for c in selected],plans=72,protocol_sha256=sha(output/'protocol.json')))

if __name__=='__main__':
    from pathlib import Path
    main()
