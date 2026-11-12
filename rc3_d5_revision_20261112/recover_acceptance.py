"""Resume the real interrupted acceptance; do not replay its completed requests."""
import time,gzip,json,math
from .common import ROOT,HOME,read,put,sha,state,unarchive
from .science import fast_budget,reference_allocation,closeout
from .data import Pool,replay_teacher
from .training import Trainer
from .runtime import new_actor,sample_generator,policy_request,recover_episode,restore_generator,generator_text,References
from rc3_conservative_refinement.runtime import tensor_digest
from rc3_conservative_refinement.trainer import checkpoint_equal
from rc3_decoder_program_20261111.actor import ALLOW

def run(budget):
    fast_budget(budget)
    if read(HOME/'REFERENCE_MANIFEST.json')['status']!='PASS':raise PermissionError('Reference preparation must finish before quota-preserving continuation')
    pool=Pool(budget);seed=2026111101;schedule=read(HOME/'schedules'/f'{seed}.json')['batches'];refs=References(pool,budget)
    actor=new_actor('D5-PG',budget);t=Trainer(actor,budget,seed,sample_generator(seed),True)
    checkpoint=read(HOME/'acceptance/PG/LATEST.json');t.restore_current(checkpoint)
    clone=Trainer(new_actor('D5-PG',budget),budget,seed,sample_generator(seed),True);clone.restore_current(checkpoint)
    if t.cursor!=1:raise ValueError('The preserved acceptance checkpoint must be cursor1')
    pointers={}
    with (HOME/'acceptance/records/REQUEST_POINTERS.jsonl').open(encoding='utf8') as f:
        for line in f:
            row=json.loads(line);pointers[row['request_id']]=row['record']
    with gzip.open(HOME/'acceptance/updates/UPDATES.jsonl.gz','rt',encoding='utf8') as f:prior_updates=[json.loads(x) for x in f if x.strip()]
    pg1=next(u for u in prior_updates if u['role']=='D5-PG' and u['cursor']==1)
    sl1=next(u for u in prior_updates if u['role']=='D5-SL' and u['cursor']==1)
    if tensor_digest(t.actor.state_dict())!=pg1['after']:raise ValueError('Checkpoint not the actual completed PG candidate')
    aa=[];bb=[];reused=0;new_requests_before=budget.counts['requests'];actors_before=budget.counts['actor_optimizer_attempts']
    for slot,tid in enumerate(schedule[1]):
        pair=[]
        for which,trainer in [('original',t),('restored',clone)]:
            rid=f'acceptance:resume:{which}:{slot}'
            if rid in pointers:
                record=unarchive(pointers[rid]);episode=recover_episode(pool,record,'D5-PG')
                if episode['actor_tensor_sha']!=tensor_digest(trainer.actor.state_dict()):raise ValueError('Recovered trace is not current own policy')
                restore_generator(trainer.sampling,record['sampler_after']);out=dict(record=record,episode=episode);reused+=1
            else:out=policy_request(budget,pool,tid,rid,trainer.actor,purpose='engineering',folder='acceptance/records',generator=trainer.sampling,seed=seed)
            out['episode']['reference_J_H']=refs.get(tid);pair.append(out)
        for k in ('actions','mask_bits_b64','raw_route','route','metrics','sample_log_prob'):
            if pair[0]['record'][k]!=pair[1]['record'][k]:raise ValueError('Actually restored next full8 differs: '+k)
        aa.append(pair[0]['episode']);bb.append(pair[1]['episode'])
    if generator_text(t.sampling)!=generator_text(clone.sampling):raise ValueError('Resumed actual sampling states differ')
    t.update(aa);clone.update(bb)
    if tensor_digest(t.actor.state_dict())!=tensor_digest(clone.actor.state_dict()) or not checkpoint_equal(t.optimizer.state_dict(),clone.optimizer.state_dict()):raise ValueError('Next full8 Adam differs after actual resampling')
    # One fresh next scheduled complete8 is necessary because the failing job
    # did not persist its in-memory complete-batch timing receipt.
    ids=schedule[2];reference_values={tid:refs.get(tid) for tid in ids};input_s=pool.prepare_s
    tick=time.perf_counter();wall=time.time();episodes=[]
    for slot,tid in enumerate(ids):
        out=policy_request(budget,pool,tid,f'acceptance:measured_full8:{slot}',t.actor,purpose='engineering',folder='acceptance/records',generator=t.sampling,seed=seed)
        ep=out['episode'];ep['reference_J_H']=reference_values[tid];episodes.append(ep)
    result=t.update(episodes);t.save_current(HOME/'acceptance/MEASURED_PG',True);batch_s=max(time.perf_counter()-tick,time.time()-wall)
    if t.actor.frozen_digest()!=t.actor.initial_frozen_digest:raise ValueError('Measured full8 changed frozen bytes')
    estimate=4096*batch_s+reference_allocation()+352*input_s/max(len(pool.cache),1)
    passed=estimate<=28800
    put(HOME/'TARGETED_ACCEPTANCE.json',dict(status='PASS' if passed else 'BUDGET_THROUGHPUT_BLOCKED',
        source_identity=t.identity,actual_old_PG_and_SL_gradients_reused=True,old_PG_gradient_norms=pg1['gradient_norms'],SL_gradient_norms=sl1['gradient_norms'],
        current_PG_gradient_norms=result['gradient_norms'],frozen_bitwise=True,restored_own_sampling_actions_masks_raw_H_probabilities_exact=True,
        restored_next_full8_Adam_and_model_exact=True,reused_completed_resume_requests=reused,
        actual_new_requests_this_continuation=budget.counts['requests']-new_requests_before,actual_Actor_attempts_this_continuation=budget.counts['actor_optimizer_attempts']-actors_before,
        actual_PG_full8_batch_including_update_checkpoint_s=batch_s,estimated_4096_with_reference_and_input_s=estimate,
        engineering_total_requests=budget.counts['engineering_requests'],engineering_total_Actor_attempts=budget.counts['engineering_optimizer_attempts'],
        no_old_attempt_or_time_refund=True,model_initialization_for_science='Fresh legalSL256 only; all acceptance models prohibited',
        complete_batch_ids=ids,validation_purpose='Production interface and true batch throughput; not a route-benefit experiment'))
    if passed:state(phase='P1',status='TARGETED_ACCEPTANCE_AND_REFERENCE_PASS',next_action='Start fixed first-seed4096 D5-PG and4096 same-task SL-U')
    else:closeout('TECHNICAL_BUDGET_INCOMPLETE','Measured full8 plus exact reference/input cost does not fit8h; no formal training')
    print(dict(phase='ACCEPTANCE_RESUME',status='PASS' if passed else 'BLOCKED',batch_s=batch_s,estimated4096_s=estimate,new_requests=budget.counts['requests']-new_requests_before),flush=True)
