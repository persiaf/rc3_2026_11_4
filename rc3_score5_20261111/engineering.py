import copy
import torch
from rc3_conservative_refinement.runtime import tensor_digest
from rc3_conservative_refinement.trainer import checkpoint_equal
from rc3_global_local.environment import Environment
from .actor import load_actor,export,load_export,ALLOW,StaleCacheError
from .trainer import Trainer,rng
from .runtime import request,teacher_episode,replay_logits
from .common import HOME,put,read

def run(ledger,trace,inputs,identity):
    cases=read(HOME/'SCHEDULE.json')['engineering'];models=ledger.path/'models';models.mkdir()
    tests=[]
    def passed(name,**kw):tests.append(dict(check=name,status='PASS',**kw));ledger.event('check_pass',check=name)
    pg=load_actor('Score5-PG');sl=load_actor('Score5-SL')
    assert tensor_digest(pg.state_dict())==tensor_digest(sl.state_dict())==pg.base_identity['tensor_sha256']
    assert set(n for n,p in pg.named_parameters() if p.requires_grad)==set(ALLOW)
    assert sum(p.numel() for p in pg.parameters())==65088 and sum(p.numel() for p in pg.trainable_parameters())==25152
    frozen={n:p.detach().clone() for n,p in pg.named_parameters() if n not in ALLOW}
    passed('exact_SL256_initialization_five_tensor25152_frozen39936')
    # Initial full+three local complete greedy routes and logits against the
    # untouched original frozen Actor, no synthetic32-state or teacherTop1 gate.
    for i,spec in enumerate(cases):
        a=request(ledger,inputs,spec,f'eng:init:{i}','Score5-PG',actor=pg,greedy=True)
        b=request(ledger,inputs,spec,f'eng:SL:{i}','FrozenSL',greedy=True)
        assert a['record']['raw_route']==b['record']['raw_route'] and a['record']['route']==b['record']['route'] and a['record']['metrics']==b['record']['metrics']
        plain=__import__('rc3_conservative_refinement.runtime',fromlist=['load_frozen_sl']).load_frozen_sl()
        p=a['episode']['problem'];ca=pg.encode(p);cb=plain.encode(p)
        for state,z in zip(a['episode']['states'],a['episode']['old_logits']):
            with torch.no_grad():za=pg.score(state,ca);zb=plain.score(state,cb)
            assert torch.equal(za,zb) and torch.equal(za,z)
    passed('initial_original_logits_masks_raw_H_routes_equal_four_sizes')
    entry=export(models/'initial.engineering.actor.pt',pg,'engineering')
    original=request(ledger,inputs,cases[0],'eng:initial_export','Score5-PG',entry=entry,greedy=True)
    assert original['record']['actor_tensor_sha256']==pg.base_identity['tensor_sha256']
    for role,usage in [('Score5-SL','engineering'),('Score5-PG','science')]:
        try:load_export(entry,role,usage)
        except ValueError:pass
        else:raise AssertionError('Role/stage mismatch accepted')
    passed('initial_export_exact_and_engineering_role_stage_isolation')
    t=Trainer(pg,ledger,identity);control=Trainer(sl,ledger,identity)
    bad=torch.optim.Adam(pg.parameters(),lr=3e-5)
    try:pg.validate_optimizer(bad)
    except ValueError:pass
    else:raise AssertionError('All-parameter optimizer accepted')
    try:pg.validate_optimizer(torch.optim.Adam(sl.trainable_parameters(),lr=3e-5))
    except ValueError:pass
    else:raise AssertionError('Foreign optimizer accepted')
    passed('optimizer_all_parameter_and_foreign_instance_rejected')
    sample=request(ledger,inputs,cases[0],'eng:PG_online','Score5-PG',actor=pg,seed=1162202601)['episode']
    # Cache built under no_grad must support new independent backwards when
    # grad is enabled. No graph/key is reused across backward calls.
    with torch.no_grad():cache=pg.encode(sample['problem'])
    for pos,s in enumerate(sample['states'][:2]):
        z=pg.score(s,cache);assert z.requires_grad and not cache['history_hidden'].requires_grad
        (-torch.log_softmax(z,-1)[sample['actions'][pos]]).backward();ledger.counts['backward_calls']+=1
        pg.zero_grad(set_to_none=True)
    passed('no_grad_static_cache_reenters_grad_key_projection_prefix_no_shared_graph')
    stale=pg.encode(sample['problem']);before=tensor_digest(pg.state_dict());r=t.update([sample])
    assert r['accepted'] and r['after']!=before
    assert all(r['gradient_norms'][n] is not None and r['gradient_norms'][n]>0 for n in ALLOW)
    assert all(torch.equal(dict(pg.named_parameters())[n],v) for n,v in frozen.items())
    try:pg.score(sample['states'][0],stale)
    except StaleCacheError:pass
    else:raise AssertionError('Pre-update cache accepted')
    passed('real_PG_all_five_gradients_frozen_exact_stale_cache_reject',gradient_norms={n:r['gradient_norms'][n] for n in ALLOW})
    teacher=teacher_episode(ledger,inputs,sl,0,'eng:teacher:0');r=control.update([teacher]);assert r['accepted']
    assert all(r['gradient_norms'][n] is not None and r['gradient_norms'][n]>0 for n in ALLOW)
    passed('real_existing_label_SL_update_same_five_tensors_no_new_teacher')
    updated=export(models/'updated.engineering.actor.pt',pg,'engineering')
    a=request(ledger,inputs,cases[3],'eng:updated_live','Score5-PG',actor=pg,greedy=True)
    b=request(ledger,inputs,cases[3],'eng:updated_export','Score5-PG',entry=updated,greedy=True)
    assert a['record']['raw_route']==b['record']['raw_route'] and a['record']['route']==b['record']['route'] and a['record']['metrics']==b['record']['metrics']
    loaded=load_export(updated,'Score5-PG','engineering')
    for sa in a['episode']['states']:
        with torch.no_grad():assert torch.equal(pg.score(sa,pg.encode(sa.problem)),loaded.score(sa,loaded.encode(sa.problem)))
    passed('updated_export_frozen_eval_logits_masks_raw_H_equivalence')
    reject=request(ledger,inputs,cases[1],'eng:reject_sample','Score5-PG',actor=pg,seed=1162202602)['episode']
    before=copy.deepcopy(pg.state_dict());opt=copy.deepcopy(t.optimizer.state_dict());stale=pg.encode(reject['problem'])
    r=t.update([reject],force_reject=True);assert not r['accepted'] and r['rollback_exact']
    assert checkpoint_equal(pg.state_dict(),before) and checkpoint_equal(t.optimizer.state_dict(),opt)
    try:pg.score(reject['states'][0],stale)
    except StaleCacheError:pass
    else:raise AssertionError('Rollback incorrectly revalidated stale cache')
    passed('real_Adam_candidate_injected_reject_exact_weights_Adam_and_cache_invalidation')
    checkpoint=t.save(models/'closed.engineering.training.pt');saved=t.restore(checkpoint)
    restored=Trainer(load_actor('Score5-PG'),ledger,identity);restored.restore(checkpoint)
    assert checkpoint_equal(rng(),saved['rng']) and restored.cursor==t.cursor and restored.used==t.used
    assert t.actor.revision==restored.actor.revision
    nxt=request(ledger,inputs,cases[2],'eng:resume_next','Score5-PG',actor=pg,seed=1162202603)['episode']
    t.update([nxt]);restored.update([nxt])
    assert checkpoint_equal(t.actor.state_dict(),restored.actor.state_dict()) and checkpoint_equal(t.optimizer.state_dict(),restored.optimizer.state_dict())
    assert t.cursor==restored.cursor and t.actor.revision==restored.actor.revision
    passed('checkpoint_Model_Adam_RNG_cursor_role_resource_and_next_update_exact')
    assert ledger.counts['requests']==14 and ledger.counts['actor_optimizer_attempts']==5
    put(ledger.path/'CHECKS.json',dict(status='PASS',tests=tests,full_requests=14,Actor_attempts=5,Critic_attempts=0,
        engineering_models_forbidden_as_formal_initialization=True,no_teacher_Top1_or_fixed32_fit_gate=True,
        no_old_A1_residual_amplitude_or_permanent_SL_anchor=True))
