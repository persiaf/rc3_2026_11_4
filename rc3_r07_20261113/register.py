import ast,hashlib,json,time,datetime
from .common import ROOT,HOME,D5,BASE,RUN,H_ID,read,put,sha

def main():
    if (HOME/'PROTOCOL.json').exists():raise FileExistsError('No reset')
    full=sorted(read(D5/'FULL_TASK_REGISTRATION.json')['records'],key=lambda r:(r['physical_key'],r['id']))[:32]
    used={r['physical_key'] for r in full};local=[];old=sorted(read(D5/'ORIGINAL_LOCAL_TEACHERS.json')['records'],key=lambda r:(r['physical_key'],r['id']))
    for i in range(16):
        size=((4,3),(6,6),(8,8))[i%3]
        choices=[r for r in old if r['physical_key'] not in used and (r['spec']['width'],r['spec']['height'])==size]
        if not choices:raise ValueError('Insufficient registered unique local parents; do not silently substitute')
        r=choices[0];local.append(r);used.add(r['physical_key'])
    tasks=[dict(r,size=f"{r['spec']['width']}x{r['spec']['height']}") for r in full+local]
    if len(used)!=48 or any(r['spec']['role']!='train' for r in tasks):raise ValueError('Training48 distinct physical parents only')
    put(HOME/'TASKS.json',dict(selection='First32 full by physical_key/id; local16 rotate4x3/6x6/8x8 selecting first eligible unused physical_key/id; disjoint parents; metadata only, no costs',records=tasks,physical_parents=48))
    origin=datetime.datetime(2026,11,13,3,23,tzinfo=datetime.timezone.utc).timestamp()
    p=dict(schema='R07-counterfactual-opportunity-v1',run_id=RUN,base_commit=BASE,origin_unix_s=origin,deadline_unix_s=origin+1800,
        origin_scope='Conservative before first task read; all implementation,input restore,rollout,H,validation,statistics,review,Git and waiting',
        max_requests=320,normal_primary_max=288,repeat_plan='8 fixed size/stage states x2 branches=16 extra requests; total<=304, remaining16 only same-input technical recovery',
        Actor_updates=0,Critic_updates=0,new_teacher_scene_navigation=0,
        state_selection='Saved exact frozenSL route; progress=(N_initial-N_remaining)/N_initial; bins[0,1/3),[1/3,2/3),[2/3,1); nearest1/6,1/2,5/6 among>=2 legal; tie earliest step; missing remains missing',
        ranking='Exact frozenSL top1/top2 legal logits, stable local candidate index ties; no score-gap filtering',
        intervention='Only forced action differs; suffix policy is identical frozenSL greedy; suffix actions may differ because resulting states differ',
        H_identity=H_ID,mask_cost_endpoints='Unchanged; J=L+.75K; full raw routes each H once, same2 joint passes, no branch-dependent search',
        actor='Exact SL256 tensor1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e; frozen eval, never D5/D6 endpoints',
        delta='alternative_H_J-baseline_H_J; negative improves',tie_absolute=1e-7,material_relative=.005,
        repeat_selection='Per size10x10/4x3/6x6/8x8: first physical parent early and next distinct parent late; exact same states/candidates, no cost selection; unavailable repeats stay missing',
        opportunity_gate=dict(overall_material_improvement_fraction_min=.20,full_and_local_each_min=.15,
            full_parents_with_material_improvement_min=8,local_parents_min=4,parent_equal_mean_oracle_per_state_gain_min=.005,
            scope='Oracle means average max(0,-delta/Jbase) within fixed states then parents, not additive gains along one route, not trained policy performance'),
        statistics='Report signed/absolute cost, improved/tied/degraded; cluster by actual physical parent. Stratified physical-parent bootstrap20000,seed2026111307, descriptive95% intervals; no independent-state p-values',
        teacher_gap='Existing same-task strong teacher only after state selection; no new teacher/H; exact problem identity and route cost',
        decision='Only opportunity verification. If gate fails, recommend closing current constructive learning direction and redesigning overall planning decisions, not another reward baseline. No automatic PG/Q/multiseed.',
        errors='Stop on real provenance/input/constraint error; no case/state/candidate replacement',delivery_reserve_s=300)
    put(HOME/'PROTOCOL.json',p)
    b=read(ROOT/'rc3_d6_20261113/SOURCE_BINDING.json')
    for f in HOME.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'));b['source_sha256'][f.relative_to(ROOT).as_posix()]=sha(f)
    art={str((HOME/n).relative_to(ROOT)).replace('\\','/'):sha(HOME/n) for n in ('PROTOCOL.json','TASKS.json')}
    for n in ('FULL_TASK_REGISTRATION.json','ORIGINAL_LOCAL_TEACHERS.json','FULL_TEACHER_MANIFEST.json','REFERENCE_MANIFEST.json'):
        art[(D5/n).relative_to(ROOT).as_posix()]=sha(D5/n)
    b['artifact_sha256'].update(art)
    stable=dict(protocol_sha256=sha(HOME/'PROTOCOL.json'),tasks_sha256=sha(HOME/'TASKS.json'),sources={k:v for k,v in b['source_sha256'].items() if k.startswith(HOME.name+'/')},base=BASE)
    ident=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest();put(HOME/'SCIENTIFIC_IDENTITY.json',dict(identity=ident,**stable))
    b.update(schema='R07-execution-source-v1',run_id=RUN,base_commit=BASE,scientific_identity=ident,R07_artifacts=art)
    put(HOME/'SOURCE_BINDING.json',b);put(HOME/'AUTHORIZATION.json',dict(authorized=True,authority='Current explicit R07 user task',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json')))
    put(HOME/'RESOURCE_ACCOUNTING.json',dict(counts=dict(requests=0,requests_completed=0,requests_failed=0,actor_optimizer_attempts=0,critic_optimizer_attempts=0,new_teacher_requests=0,new_scenes=0,navigation_solves=0,original60_reads=0,oldS3_reads=0,confirmation_requests=0,final_requests=0),wall_s=time.time()-origin,prior_D5_D6_ledgers_preserved=True))
    put(HOME/'STATUS.json',dict(run_id=RUN,status='REGISTERED_NOT_RUN',closed=False));print(dict(identity=ident,registered_wall_s=time.time()-origin))
if __name__=='__main__':main()
