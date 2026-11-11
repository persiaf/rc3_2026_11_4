"""Once-only identity/configuration registration; no model or fit execution."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import time
from .data import build_selection
from .ledger import digest,write_new

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent


def main():
    if (HOME/'PREREGISTRATION.json').exists():
        raise RuntimeError('Registration is immutable; no configuration reselection')
    tick=time.perf_counter();selection=build_selection()
    write_new(HOME/'TRAJECTORY_SELECTION.json',selection)
    write_new(HOME/'CALIBRATION_SPLIT.json',dict(selection['split'],
        runner_split_mapping={'fit':'fit','holdout':'validation'},
        scope='Parents held out from this offline calibration; all selected trajectories are from historical R03 training, not unseen supervised/DRL parents'))
    rng=random.Random(11622026)
    groups={p:[] for p in selection['split']['fit_physical_parents']}
    for row in selection['trajectories']:
        if row['split']=='fit': groups[row['parent_id']].append(row['trajectory_id'])
    parents=sorted(groups);rng.shuffle(parents)
    for trajectories in groups.values(): rng.shuffle(trajectories)
    positions={p:0 for p in parents};batches=[];batch_size=min(8,len(parents))
    for step in range(256):
        batch=[]
        for j in range(batch_size):
            p=parents[(step*batch_size+j)%len(parents)]
            batch.append(groups[p][positions[p]%len(groups[p])]);positions[p]+=1
        batches.append(batch)
    write_new(HOME/'BATCH_SCHEDULE.json',dict(seed=11622026,steps=256,batch_size=batch_size,
        rule='Fixed shuffled physical-parent circular schedule, one rotating saved trajectory per selected parent; identical list for C0/C1/C2',batches=batches))
    old=json.loads((ROOT/'rc3_conservative_refinement/integration_20261110/SOURCE_BINDING_R03.json').read_text(encoding='utf-8'))
    source=dict(old['source_sha256'])
    for path in sorted(HOME.glob('*.py')): source[path.relative_to(ROOT).as_posix()]=digest(path)
    assert (HOME/'independent_recheck.py').is_file(), 'Independent checker must be bound before computation'
    artifacts=dict(old['artifacts'])
    for item in selection['source_metadata']: artifacts[item['path']]=item['sha256']
    for row in selection['trajectories']:
        artifacts[row['request_path']]=row['request_sha256']
        for key in ('before_checkpoint','after_checkpoint'):
            item=row[key];artifacts[item['path']]=item['sha256']
    # Bind every persisted saved-input dependency chosen by the fixed metadata.
    index=json.loads((ROOT/'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json').read_text(encoding='utf-8'))
    refs=json.loads((ROOT/'rc3_integrated_rppo/DATA_REFERENCES.json').read_text(encoding='utf-8'))
    for row in selection['trajectories']:
        spec=row['spec'];key=f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        entry=index.get(key,refs['bank'].get(key))
        if entry is None: raise ValueError('No persisted input for selected metadata')
        for item in entry['files'].values(): artifacts[item['path']]=item['sha256']
    metadata=['TRAJECTORY_SELECTION.json','CALIBRATION_SPLIT.json','BATCH_SCHEDULE.json']
    plan=dict(schema='rc3-offline-critic-rank-preregistration-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        base_commit='83b0df9ceb6b43c5e04de883db1376558393af4d',
        branch='codex/rc3-critic-rank-diagnostics-20261111',seed=11622026,
        user_authorized='2026-11-11 explicit bounded offline diagnostic task; max128 trajectories,3x256 Critic steps,3600s; no joint PPO',
        scientific_identity_excludes_mutable_git_publication_sha=True,
        old_R03_protocol_sha256='a20eca451e7094bd16390ae5b973c85240a42337e4ef6ae42bd243b9a8cfeb0c',
        old_R03_status='STOPPED_REGISTERED_RULE_AT128;256_NOT_STARTED',
        H_identity=old['H_identity'],source_sha256=source,artifact_sha256=artifacts,
        registration_files_sha256={name:digest(HOME/name) for name in metadata},
        selection_rule=selection['selection_rule'],trajectories=128,
        environment=dict(device='cpu',threads=1,deterministic_algorithms=True,python='3.10.20',torch='2.12.1+cu126',numpy='2.2.6'),
        actor=dict(amplitude=.25,maximum_pairwise_residual_difference=.5,
            structural_inability='strict base top1-top2 gap > .5; single legal actions reported separately; .5 may at most tie under closed amplitude bounds',
            variants=['ON_before','ON_after','OFF_before','OFF_after','ON_after_OFF','OFF_after_ON'],
            ratio_denominator='Exact before-checkpoint reconstructed full logits; saved actual selected logit atol1e-6 and probability atol2e-6 independently checked',
            pre_update_and_post_update_ratios_separate=True),
        critic=dict(configurations=['C0','C1','C2'],input_dim=268,hidden=[128,128],
            initializer='Identical raw net tensors from default RemainingCostCritic initialization, manual_seed11622026; no fitted bias initialization. C0 raw units and C1/C2 normalized units yield different initial physical predictions; both initial and final results mandatory.',
            C0='Original raw L/K softplus,signed H; .5*sum(((pred-target)/initial_n)^2)',
            C1='Direct normalized L/n,K/n softplus,H/n signed; .5*sum(component errors^2)',
            C2='C1 + .5*(composed value - complete RTG)^2; value loss coefficient1',
            composed_value='-(normalized_L + .75*normalized_K - normalized_H)',
            state_reduction='state mean within each trajectory, then equal mean across parent-balanced batch trajectories; differs from old PPO state SUM and never replaces old stop metric',
            optimizer=dict(name='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,gradient_clip=1),
            steps_per_configuration=256,total_optimizer_attempts_cap=768,
            endpoints=[0,256],intermediate_validation_or_model_selection=False,
            references='Fitting fold only, parent equal / trajectory equal / state mean. Global and window x observable progress bin floor(4*(1-v[256])); unknown bins fall back to fitting global.',
            aggregations=['state_pooled','parent_equal'],new_features=0),
        caps=dict(wall_s=3600,actor_optimizer_attempts=0,critic_optimizer_attempts=768,
            critic_backwards=768,critic_forwards=8192,actor_head_forwards=1024,
            saved_view_loads=128,base_encoder_forwards=128,base_state_forwards=100000,
            feature_states=100000,source_reads=20000,source_hashes=20000,model_loads=130,
            numerical_rechecks=1,new_scenes=0,navigation_requests=0,teacher_requests=0,
            native_requests=0,H_requests=0,original60_reads=0,S3_reads=0),
        failure_policy='Debit before forward/backward/optimizer. One-shot journal refuses restart. Failures and paid work retained; any code repair requires explicit same-budget receipt, never reset/rerun as new study.',
        preflight_metadata_only_wall_s=time.perf_counter()-tick,
        preflight_scope='Source inspection, hashes and schedule identities only; no new numerical model/state/label diagnostic, forward or optimizer')
    write_new(HOME/'PREREGISTRATION.json',plan)
    print(json.dumps(dict(registration_sha256=digest(HOME/'PREREGISTRATION.json'),trajectories=128,
        fit_parents=len(parents),holdout_parents=len(selection['split']['holdout_physical_parents']),
        sources=len(source),artifacts=len(artifacts),batch_size=batch_size)))


if __name__=='__main__': main()
