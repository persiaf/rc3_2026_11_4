"""Register G2 splits and fixed training decisions before measuring validation."""
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
import zipfile
import numpy as np
from rc3_v3.data import historical_seeds
from rc3_rl_research.types import SearchConfig

ROOT=Path(__file__).resolve().parent
def main():
    output=ROOT/'outputs';output.mkdir(exist_ok=True)
    path=output/'protocol.json'
    if path.exists():raise ValueError('refusing to change a registered protocol')
    excluded,scan=historical_seeds(ROOT.parent,exclude=ROOT)
    rng=np.random.default_rng(2026110302);used=set(excluded);splits={}
    for split,count in [('teacher_initial',64),('teacher_extra',32),('train',384),
                        ('validation',24),('rule_dev',12),('pipeline',6)]:
        cases=[]
        for i in range(count):
            while True:
                seed=int(rng.integers(100000,2**31-1))
                if seed not in used:break
            used.add(seed)
            cases.append(dict(case_id=i,split=split,nx=10,ny=10,seed=seed,
                              mode=('regular','mixed','blocked')[i%3]))
        splits[split]=cases
    protocol=dict(schema='rc3-g2-protocol-v1',date='2026-11-03',training_seed=20261103,
        split_generation_seed=2026110302,splits=splits,excluded_seed_count=len(excluded),
        seed_scan_unreadable=scan['unreadable'],config=asdict(SearchConfig()),
        teacher=dict(depth=2,branches_including_stop=8,tie_tolerance=.001,
                     label='max observed remaining return including final event; actor only',
                     unknown_action_policy='exclude from BC normalization; legal for PPO'),
        model=dict(kind='graph',hidden=64,message_rounds=2,dropout=0.,batchnorm=False),
        bc=dict(initial_updates=400,continued_updates=2048,batch_size=32,learning_rate=1e-4,
                continued_checkpoints=[512,1024,2048],weighting='equal scenes, equal current routes, duplicate states shared'),
        ppo=dict(actions=32768,workers=8,rollout_steps=32,epochs=4,minibatch=64,
                 gamma=1.,gae_lambda=.95,learning_rate=1e-4,clip=.2,grad_clip=.5,
                 target_kl=.02,entropy_coefficient=.01,value_coefficient=.5,
                 validation_actions=[8192,16384,32768],advantage_normalization='per rollout',
                 final_checkpoint='32768 actions regardless of development scores',
                 no_automatic_extension=True),
        pipeline=dict(mlp_actions=256,graph_actions=768,formal_actions_exclude_pipeline=True),
        baselines=['original_fixed3','original_detour','environment_stop','detour_memory','ils','random'],
        rule_selection=dict(dev_split='rule_dev',criterion='highest mean return; then lowest mean work',
                            frozen_before_validation=True),
        evaluation=dict(work_budget=2000000,horizon=16,soft_time_limit_s=.4,
                        learned_decoding='greedy',random_repetitions=1,
                        checkpoints_use_work_budget=True,
                        final_methods_share_work_and_full_time_protocol=True,
                        primary='paired return vs frozen BC, continued BC and shared rules',
                        secondary=['J','quality gain','actual work','full scene-to-plan time','STOP rate','fallback','soft budget overruns'],
                        scope='24 validation scenes repeatedly examined; not final confirmation or multiseed generalization'),
        continuation_gate=dict(autostart=False,conditions=[
            'PPO validation return improves over both frozen and continued BC at two checkpoints including final',
            'positive paired development return vs strongest shared rule with wins distributed across modes',
            'no feasibility or accounting failures; full planning time remains practically plausible']),
        restrictions=['no changes to frozen G01/v4/v5/v5.1','no physical IK','no dynamic task arrivals','one training seed only'])
    path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'protocol.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  protocol.json\n')
    (output/'historical_seed_scan.json').write_text(json.dumps(scan,ensure_ascii=False,indent=2),encoding='utf-8')
    frozen={}
    for archive in ('rc3_v4_delivery.zip','rc3_v5_delivery.zip','rc3_v5_1_maintenance.zip',
                    'rc3_rl_research/outputs/rc3_rl_research_g01.zip'):
        p=ROOT.parent/archive
        frozen[archive]=hashlib.sha256(p.read_bytes()).hexdigest()
    g01=json.loads((ROOT.parent/'rc3_rl_research/PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    bad=[r['path'] for r in g01['files'] if hashlib.sha256((ROOT.parent/'rc3_rl_research'/r['path']).read_bytes()).hexdigest()!=r['sha256']]
    if bad:raise ValueError('frozen G01 already differs: '+str(bad))
    (output/'freeze_before.json').write_text(json.dumps(dict(archive_sha256=frozen,g01_payloads=len(g01['files']),g01_matches=True),indent=2))
    with zipfile.ZipFile(ROOT.parent/'G01_audit_G2_tasks.zip') as z:
        m=json.loads(z.read('G01_audit_G2_tasks/MANIFEST.json'))
        checked=[hashlib.sha256(z.read('G01_audit_G2_tasks/'+r['path'])).hexdigest()==r['sha256'] for r in m['files']]
        if not all(checked):raise ValueError('input review package hash mismatch')
    (output/'input_receipt.json').write_text(json.dumps(dict(files_checked=len(checked),all_passed=True,
        archive_sha256=hashlib.sha256((ROOT.parent/'G01_audit_G2_tasks.zip').read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),splits={k:len(v) for k,v in splits.items()},excluded=len(excluded))))

if __name__=='__main__':main()
