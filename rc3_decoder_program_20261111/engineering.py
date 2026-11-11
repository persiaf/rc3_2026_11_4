import copy
import torch
from rc3_conservative_refinement.runtime import tensor_digest,load_frozen_sl
from rc3_conservative_refinement.trainer import checkpoint_equal
from .actor import load_actor,export,load_export,ALLOW,StaleCacheError
from .trainer import Trainer,rng
from .runtime import request,teacher_episode
from .common import HOME,read,put

def run(ledger,inputs,identity):
    registration=read(HOME/'PROGRAM_PROTOCOL.json')['P0'];cases=registration['ordinary_training_specs'];models=ledger.path/'models';models.mkdir()
    tests=[]
    def passed(name,**kw):tests.append(dict(check=name,status='PASS',**kw));ledger.event('check_pass',check=name)
    pg=load_actor('D5-PG');sl=load_actor('D5-SL');full=load_actor('Full-PG')
    assert len({p.data_ptr() for a in (pg,sl,full) for p in a.parameters()})==sum(len(list(a.parameters())) for a in (pg,sl,full))
    assert tensor_digest(pg.state_dict())==tensor_digest(sl.state_dict())==tensor_digest(full.state_dict())==pg.base_identity['tensor_sha256']
    assert sum(p.numel() for p in pg.trainable_parameters())==25152
    frozen={n:p.clone() for n,p in pg.state_dict().items() if n not in ALLOW}
    passed('exact_SL256_and_independent_storage_and_five_tensor25152_of65088')
    references={};initial=[]
    for i,spec in enumerate(cases):
        a=request(ledger,inputs,spec,f'p0:init:{i}','D5-PG',actor=pg,greedy=True)
        b=request(ledger,inputs,spec,f'p0:reference:{i}','FrozenSL',greedy=True)
        assert a['record']['raw_route']==b['record']['raw_route'] and a['record']['route']==b['record']['route'] and a['record']['metrics']==b['record']['metrics']
        references[i]=b['record']['metrics']['J'];initial.append(a)
        plain=load_frozen_sl()
        with torch.no_grad():
            ca=pg.encode(a['episode']['problem']);cb=plain.encode(a['episode']['problem']);cc=sl.encode(a['episode']['problem'])
            for s,z in zip(a['episode']['states'],a['episode']['old_logits']):
                assert torch.equal(pg.score(s,ca),z) and torch.equal(plain.score(s,cb),z) and torch.equal(sl.score(s,cc),z)
        assert a['record']['complete_return']==-a['record']['metrics']['J']
    passed('initial_logits_masks_raw_H_routes_equal_in_full_and_three_local_sizes_complete_return_minus_J')
    tp=Trainer(pg,ledger,identity);ts=Trainer(sl,ledger,identity);tf=Trainer(full,ledger,identity)
    for bad in (torch.optim.Adam(pg.parameters()),torch.optim.Adam(sl.trainable_parameters())):
        try:pg.validate_optimizer(bad)
        except ValueError:pass
        else:raise AssertionError('Incorrect optimizer accepted')
    full.validate_optimizer(tf.optimizer)
    passed('exact_D5_whitelist_and_Full_all_parameters_foreign_optimizers_rejected')
    e=request(ledger,inputs,cases[0],'p0:pg_sample','D5-PG',actor=pg,seed=2026111101)['episode'];e['reference_J_H']=references[0]
    with torch.no_grad():cache=pg.encode(e['problem'])
    for index,s in enumerate(e['states'][:2]):
        z=pg.score(s,cache);(-torch.log_softmax(z,-1)[e['actions'][index]]).backward();ledger.counts['backward_calls']+=1
        assert dict(pg.named_parameters())['key.weight'].grad is not None and not cache['history_hidden'].requires_grad;pg.zero_grad(set_to_none=True)
    passed('no_grad_collection_cache_reenters_grad_key_rebuilt_frozen_prefix_detached')
    stale=pg.encode(e['problem']);update=tp.update([e])
    assert all(update['gradient_norms'][name] is not None for name in ALLOW)
    assert all(torch.equal(pg.state_dict()[n],v) for n,v in frozen.items())
    try:pg.score(e['states'][0],stale)
    except StaleCacheError:pass
    else:raise AssertionError('Stale cache accepted')
    passed('real_complete_return_PG_update_finite_gradients_and_cache_version',gradient_norms={n:update['gradient_norms'][n] for n in ALLOW},parameter_changed=update['before']!=update['after'],loss_sources=update['loss_sources'])
    teacher_ids=registration['teacher_indices'][:2]
    sles=[teacher_episode(ledger,inputs,sl,index,'p0:teacher:'+str(index)) for index in teacher_ids]
    upd=ts.update(sles)
    assert upd['before']!=upd['after'] and all(upd['gradient_norms'][n]>0 for n in ALLOW)
    assert all(torch.equal(sl.state_dict()[n],v) for n,v in frozen.items())
    passed('real_existing_label_SL_all_five_nonzero_gradients_frozen_weights_bitwise',gradient_norms={n:upd['gradient_norms'][n] for n in ALLOW})
    fe=request(ledger,inputs,cases[1],'p0:full_sample','Full-PG',actor=full,seed=2026111101)['episode'];fe['reference_J_H']=references[1]
    full_upd=tf.update([fe]);assert all(v is not None for v in full_upd['gradient_norms'].values())
    passed('FullActor_same_PG_recipe_all_original_parameters_graph_connected',trainable_scalars=65088,gradient_norms=full_upd['gradient_norms'])
    entry=export(models/'D5-PG.engineering.actor.pt',pg)
    live=request(ledger,inputs,cases[3],'p0:live_export_check','D5-PG',actor=pg,greedy=True)
    deployed=request(ledger,inputs,cases[3],'p0:deployed_check','D5-PG',entry=entry,greedy=True)
    assert live['record']['raw_route']==deployed['record']['raw_route'] and live['record']['route']==deployed['record']['route'] and live['record']['metrics']==deployed['record']['metrics']
    loaded=load_export(entry);assert not any(p.requires_grad for p in loaded.parameters())
    with torch.no_grad():
        c1=pg.encode(live['episode']['problem']);c2=loaded.encode(live['episode']['problem'])
        for s in live['episode']['states']:assert torch.equal(pg.score(s,c1),loaded.score(s,c2))
    try:load_export(entry,engineering=False)
    except ValueError:pass
    else:raise AssertionError('Engineering model admitted as science')
    passed('Actor_only_frozen_export_logits_masks_full_raw_H_routes_equal_usage_isolated')
    saved=tp.save(models/'D5-PG.engineering.training.pt');snapshot=tp.restore(saved)
    clone=Trainer(load_actor('D5-PG'),ledger,identity);clone.restore(saved)
    assert checkpoint_equal(rng(),snapshot['rng']) and clone.cursor==tp.cursor and clone.used==tp.used
    nxt=request(ledger,inputs,cases[2],'p0:resume_next','D5-PG',actor=pg,seed=2026111102)['episode'];nxt['reference_J_H']=references[2]
    tp.update([nxt]);clone.update([nxt])
    assert checkpoint_equal(pg.state_dict(),clone.actor.state_dict()) and checkpoint_equal(tp.optimizer.state_dict(),clone.optimizer.state_dict())
    assert clone.cursor==tp.cursor and clone.used==tp.used and clone.actor.revision==pg.revision
    passed('resume_model_Adam_RNG_cursor_ids_resource_and_next_update_exact')
    assert ledger.counts['actor_optimizer_attempts']==5 and ledger.counts['requests']==14
    put(ledger.path/'CHECKS.json',dict(status='PASS',tests=tests,new_complete_requests=13,carried_failed_requests=1,
        new_actor_attempts=5,critic_updates=0,engineering_models_forbidden_as_scientific_initialization=True,
        no_teacher_Top1_or_fixed32_gate=True,science_quality_claim=False))
    return tests
