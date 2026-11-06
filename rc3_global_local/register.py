"""S0 execution registration and separately held S1/S2/S3 study protocol."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from .problem import signature

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
BASE='6f595cfa89f74d7f7a0e39faaadf5b8655d83f33'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry():
    S0=[dict(case=i,map_seed=9700000000+i,task_seed=9800000000+i,mode=mode,A_condition=i)
        for i,mode in enumerate(('regular','mixed','blocked'))]
    train=[dict(parent=i,map_seed=10000000000+i,mode=('regular','mixed','blocked')[i%3],
        task_seeds=[11000000000+i*8+j for j in range(8)]) for i in range(96)]
    test=[]
    for group,count in [('seen_layout_new_tasks',6),('unseen_layout_new_tasks',12)]:
        for i in range(count):
            ms=10000000000+i if group.startswith('seen') else 12000000000+i
            for size in ((4,3),(6,6),(8,8)):
                for position in ('corner','center'):
                    test.append(dict(case=len(test),group=group,parent=i,map_seed=ms,
                        task_seed=13000000000+len(test),mode=('regular','mixed','blocked')[i%3],
                        size=list(size),position=position,endpoints='closed_depot' if position=='corner' else 'open_first_last_connected',
                        A_condition=len(test)%3,
                        note='Seen means training layout, not unseen-environment generalization' if group.startswith('seen') else 'Parent layout absent from training'))
    development=[]
    for i in range(3):
        for size in ((4,3),(6,6),(8,8)):
            for position in ('corner','center'):
                development.append(dict(case=len(development),map_seed=14000000000+i,
                    task_seed=15000000000+len(development),mode=('regular','mixed','blocked')[i],size=list(size),position=position,A_condition=len(development)%3))
    teachers=[]
    for i in range(256):
        parent=i%96;task_draw=(i//96)%8;width,height=((4,3),(6,6),(8,8))[i%3]
        teachers.append(dict(case=i,parent=parent,map_seed=train[parent]['map_seed'],task_seed=train[parent]['task_seeds'][task_draw],
            width=width,height=height,x0=(i*3)%(11-width),y0=(i*5)%(11-height),
            endpoints='closed_depot' if i%2==0 else 'open_first_last_connected',A_condition=(parent+task_draw)%3))
    return dict(S0=S0,S1_teachers=teachers,S2_train=train,S2_development=development,S3=test,policy_seeds=[11622026,11622027,11622028],S0_network_seed=11612026,
        sampling_seed=11632026,original60_protected=[7700000000,7700000059])


def audit_seeds(seeds):
    targets={r[k] for r in seeds['S0'] for k in ('map_seed','task_seed')}
    targets.update(r['map_seed'] for r in seeds['S2_train'])
    targets.update(s for r in seeds['S2_train'] for s in r['task_seeds'])
    targets.update(r[k] for r in seeds['S3'] for k in ('map_seed','task_seed'))
    targets.update(r[k] for r in seeds['S2_development'] for k in ('map_seed','task_seed'))
    known=read(ROOT/'rc3_nav_value/r1/seed_audit.json')['explicit_historical_banks']
    for study in ('rc3_nav_value/r1/seeds.json','rc3_teacher_distill/seeds.json'):
        data=read(ROOT/study)
        for split,rows in data.items():
            for k in ('map_seed','task_seed'):
                vals=[r[k] for r in rows];known.append(dict(name=study+':'+split+':'+k,start=min(vals),end=max(vals)))
    overlap=[dict(seed=s,bank=b) for s in targets for b in known if b['start']<=s<=b['end']]
    pattern=re.compile(rb'(?<![0-9A-Za-z_.])(?:9[7-9][0-9]{8}|1[0-5][0-9]{9})(?![0-9A-Za-z_.])')
    names=[n.decode('utf-8') for n in subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE],cwd=ROOT).split(b'\0') if n]
    inventory=[];hits=[];excluded=[]
    for name in names:
        p=ROOT/name;gz=name.endswith('.jsonl.gz')
        if 'future_test' in name.lower() or 'retained60' in name.lower():excluded.append(name);continue
        if not gz and p.suffix.lower() not in ('.json','.jsonl','.md','.py','.csv','.tsv','.yaml','.yml','.txt','.m','.log','.toml'):continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        total=0;tail=b''
        with (gzip.open(p,'rb') if gz else p.open('rb')) as f:
            for b in iter(lambda:f.read(1024*1024),b''):
                for m in pattern.finditer(tail+b):
                    if int(m.group()) in targets:hits.append(dict(path=name,seed=int(m.group())))
                total+=len(b);tail=b[-128:]
        inventory.append(dict(path=name,sha256=h.hexdigest(),scanned_bytes=total))
    result=dict(base=BASE,verdict='PASS' if not hits and not overlap else 'FAIL',unique_scene_seed_numbers=len(targets),
        collisions=hits,historical_bank_overlaps=overlap,inventory=inventory,protected_scene_exclusions=excluded,
        intended_overlap='S3 seen-layout map IDs intentionally reuse six S2 layouts; all S3 task seeds and all unseen S3 layouts are disjoint.',
        limits='Tracked text/gzip manifests plus declared banks; no binary archive internals or unrecorded external runs; protected scene contents not read')
    write(OUT/'seed_audit.json',result)
    if result['verdict']!='PASS':raise ValueError('Proposed scene seed collision')


def register():
    if (OUT/'S0_protocol.json').exists():raise FileExistsError('S0 registration already exists')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    manifest=json.loads(subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'],cwd=ROOT))
    old=['rc3_baseline/baseline.py','rc3_baseline/validator.py','rc3_fast/solver.py','rc3_fast/scenes.py',
        'rc3_constructive/baselines.py','rc3_constructive/problem.py','rc3_v5/external.py','rc3_nav_value/r1/oracle.py']
    for name in old:assert digest(ROOT/name)==manifest['files'][name]['sha256'],name
    core=read(OUT/'unit_receipt.json');policy=read(OUT/'policy_unit_receipt.json')
    for r in (core,policy):assert r['failures']==r['errors']==0
    for name,h in core['tested_source_sha256'].items():assert digest(OUT/name)==h,name
    assert digest(OUT/'test_environment.py')==core['test_sha256']
    for name,h in policy['source_sha256'].items():assert digest(OUT/name)==h,name
    review=read(OUT/'prereg_review.json')
    assert review['verdict']=='PASS_FOR_S0' and review['remaining_blockers']==0
    for name,h in review['source_sha256'].items():assert digest(OUT/name)==h,name
    import platform,sys,importlib.metadata
    environment=dict(python=sys.version,OS=platform.platform(),packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','threadpoolctl')},
        torch_threads=2,threadpool_limit=1,deterministic_algorithms=True,device='cpu',capture='Before real S0 scenes and probes')
    write(OUT/'execution_environment.json',environment)
    seeds=registry();write(OUT/'seeds.json',seeds);audit_seeds(seeds)
    hsrc={n:digest(ROOT/n) for n in ('rc3_constructive/baselines.py','rc3_fast/solver.py')}
    p=dict(schema='rc3-global-local-S0-v1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        client_date='2026-11-06',base=BASE,branch='codex/global-local-s0',pushed=False,
        fixed_research_goal='DRL trained on large complex10x10 parent scenes, frozen policy for variable4x3/6x6/8x8 local tasks, complete constraint-validated execution paths; supervision only auxiliary init/control',
        current_scope='S0 environment consistency, untrained policies, finite gradient probes, freeze future protocol; NO S1/S2 optimization or S3 performance execution',
        source_sha256={n:digest(ROOT/n) for n in sorted(set(old)|{q.relative_to(ROOT).as_posix() for q in OUT.glob('*.py')})},
        seeds_sha256=digest(OUT/'seeds.json'),seed_audit_sha256=digest(OUT/'seed_audit.json'),
        unit_sha256=digest(OUT/'unit_receipt.json'),policy_unit_sha256=digest(OUT/'policy_unit_receipt.json'),
        prereg_review_sha256=digest(OUT/'PREREG_REVIEW.md'),prereg_review_json_sha256=digest(OUT/'prereg_review.json'),
        environment_sha256=digest(OUT/'execution_environment.json'),H_source_sha256=hsrc,H_sha256=signature(hsrc),
        semantics=dict(window='table-index task subset, NOT physically cropped navigation map; all original obstacles and corridors retained; paths may leave task window',
            candidates='all same-arm(F&A) positive-coverage candidates with entry and exit connectivity, including candidates outside selected tables',
            state='global context, local effective F/C, remaining tasks, current node, explicit exit and complete ordered prefix',
            action='one positive-gain legal service pose; no early STOP; at most task_count actions',
            endpoints='local symbol0 outgoing uses entry→candidate, incoming uses candidate→exit; service-service distances symmetric. Empty task uses direct entry→exit cost.',
            reward='step -Dmove-.75; last adds -Dexit + Jraw-JH; sum reward=-JH. Empty task terminal reward=-Dentry,exit.',
            H='original byte-identical normalize/two_opt60 twice + joint_search2 + normalize, applied to explicit endpoint cost adapter; open path uses new independent validator',
            physical='proxy or externally supplied F/A table consistency; no invented IK/collision/joint-trajectory evidence'),
        cases=dict(parent_scenes=3,modes=['regular','mixed','blocked'],sizes=[[4,3],[6,6],[8,8]],positions=['corner','center'],
            crops=18,corner_endpoints=[0,0],center_endpoints='first and last depot-connected original candidate nodes',
            A_conditions='all allowed; deterministic same-arm restrictions retaining one valid pair per task; synthetic allow table only'),
        policies=['uniform random legal','gain/(navigation+.75) rule','untrained global/local/prefix stochastic policy greedy endpoint'],
        neural=dict(hidden=64,global_tokens='100 table14-feature tokens: geometry/task density/obstacle count/true neighboring navigation',
            local='task5/candidate7 effective bipartite encoder; true current/exit navigation; full-prefix incremental GRU',
            heuristic_logit_prior=False,global_context_identity='scene+task+F+A hash; never cache by static map alone',
            gradient_probe='three legal forced first stops per8x8 center view, same instance mean return baseline; forced step contributes no policy logprob; one accumulated backward, zero optimizer steps'),
        limits=dict(parent_generations=3,navigation_preparations=3,crops=18,ordinary_episodes=54,gradient_episodes=9,
            full_terminal_attempts=80,expected_real_episodes=63,wall_s=1800,optimizer_updates=0,
            termination='cooperative checks between work and before successful completion; overrun/error => incomplete, preserve attempts; no silent replacement seed'),
        gate='All view slices equal independent parent calculations, preserve parent inputs, all produced plans validate, rewards telescope, exact action replay, masked policy distribution and finite structural/empirical gradients, unchanged neural parameters, no optimizer/heldout60 calls',
        future_training='separately registered S2_S3_protocol.json held NOT_STARTED; source binding for future trainer/baseline conformance mandatory before any formal optimizer',
        original60='sealed, no content reads/generation/queries')
    write(OUT/'S0_protocol.json',p)
    future=dict(schema='rc3-global-local-S2-S3-design-v1',date='2026-11-06',status='FROZEN_DESIGN_NOT_STARTED',
        fixed_research_goal=p['fixed_research_goal'],seeds_sha256=p['seeds_sha256'],S0_protocol_sha256=digest(OUT/'S0_protocol.json'),
        mandatory_pre_optimizer_binding='A separate execution task must hash the completed training runner and endpoint-adapted Fixed3/Detour baselines, verify conformance and S0 pass, then write execution receipt. No optimizer is authorized by S0 implementation task.',
        S1=dict(role='bounded auxiliary supervised initialization, NOT final method',teacher_routes_max=256,supervised_updates=256,
            instances='Exactly seeds.json S1_teachers256 specs from S2 training bank; endpoint-adaptedDetour16 followed by sameH, no selection using S3',
            teacher_terminal_evaluations_max=512,
            architecture='same GlobalLocalPolicy as S2 and all controls',teacher='new endpoint-adapted Fixed3/Detour16 routes with same F&A/global nav/H; never import R1/T1 g0 action labels',
            save='untrained checkpoint, frozen supervised checkpoint, every teacher instance and resource'),
        S2=dict(core='genuine policy-gradient optimization on large complex parent scenes',parent_maps=96,task_draws_per_parent=8,
            episodes='50% full10x10 tasks,50% local windows balanced4x3/6x6/8x8, all embedded in original10x10 geometry/navigation',
            task_and_constraints='Sample declared parent/task bank only, A_condition=(parent_index+task_draw_index)%3 using frozen make_allow; unchanged proxy F and base footprint. Whole-parent task density/obstacles/F&A remain visible.',
            windows_and_endpoints='Full episodes closed depot; local size cyclic4x3/6x6/8x8, x0/y0 uniformly drawn among valid table-grid origins from sampling RNG. Alternate closed depot and open first/last depot-connected global candidates by local episode index.',
            schedule='even update indexes full; odd local sizes cyclic; four parent/task draws from fixed sampling seed; same instance3 trajectories: one fully policy-sampled including first action plus2 forced distinct legal first stops',
            starts='Primary trajectory samples first action with logprob; auxiliary forced starts are first/last legal original IDs and contribute no forced-first logprob. All complete same target tasks; no TSP symmetry claim.',
            loss='For each trajectory use detached leave-one-out mean return of the other2 as baseline. Negative advantage*logprob: primary weight1, each forced auxiliary weight.25, divide total by1.5. S0 forced-only probe checks plumbing, not this future optimizer.',
            objective='REINFORCE on negative complete JH with exact masks and explicit entry/exit; no teacher action-regression substitute',
            optimizer='Adam1e-4,clip1,512 fixed updates,batch4instances,3 optimizer seeds11622026..28; save fixed end, no S3 selection',
            controls='frozen supervised start, untrained same architecture, continued-SL512updates on same frozen S1 dataset; all same F&A and postprocessing',
            development='18 declared local cases on3 separate parent layouts, checked at updates256 and512 for feasibility/resources only; no best-checkpoint selection and no S3 reads',
            budgets_per_seed=dict(rl_terminal_rewards=6144,development_and_other=2048,total_terminal=8192,optimizer_updates=512,wall_s=7200),
            missing_reward='Failures charged; do not silently discard/reward-zero infeasible routes; stop incomplete. No new seed/epoch rescue.',
            boundary='Training and controls remain supervised+RL distinction; final contribution requires positive independent RL increment, not merely smaller degradation.'),
        S3=dict(status='NOT_STARTED',checkpoint='frozen fixed512-step endpoints, no local retraining',
            inference='One deterministic greedy constructive trajectory per neural method, no multi-start/teacher/search in action selection. Raw and H results refer to the same raw construction; H benefit and added time separate.',
            allow_mask='seeds.json S3 explicit A_condition=case%3 using same frozen make_allow formula',
            cases=108,seen_layout_parent_maps=6,unseen_layout_parent_maps=12,children_per_parent=6,
            groups='Seen-layout/new-task results and unseen-layout/new-task results separate; seen is not unseen-environment generalization.',
            sizes=[[4,3],[6,6],[8,8]],endpoints='corner closed-depot; center open connected endpoints; separate strata',
            methods=['Fixed3_endpoint_adapter','Detour16_endpoint_adapter','untrained','frozen_supervised','continued_SL','RL_raw','RL_plus_H'],
            timing='3 repeats, cold global-navigation and hot known-parent navigation; include parsing/prep/subset/all features/global encode/local decisions/H/export/independent validation; model load and parent caches explicit. All methods receive equal input/cache scope.',
            cache='Hot cache primes navigation only; global/local neural embeddings rebuilt per request and bound to task+F+A+model. Report any later cache optimization separately, never reuse task-conditioned hG by map key alone.',
            primary_metrics=['task completion','proxy/table constraint feasibility','J','L','K','complete median/p95 time'],
            independent_RL_gate='ALL valid; vs frozen SL and continued-SL mean paired gain>=.5% with parent-cluster bootstrap95 lower>0, each size reported; three optimizer seeds direction consistent; no use of action Top1 as final success',
            deployment_gate='vs Detour16 mean relative degradation one-sided95upper<=.5%; complete median time ratio<=.7 and p95<=1 in both cache modes; not dominated byFixed3; report seen/unseen and each size separately',
            bootstrap='20000 parent-cluster resamples seed11632026; independent parents6 seen/12 unseen, NOT108 independent environments',
            budget_per_seed=dict(complete_requests=5292,wall_s=14400),budget_all3seeds=dict(complete_requests=15876),
            no_blending='Report raw/H gains separately, no teacher search at inference, no returned-home/exit connection cost omission'),
        fail='Record full negative/incomplete results, no automatic optimizer/seed/network-depth extension',
        protected60='unchanged sealed',physical_joint_trajectory='not yet attested; requires real kinematic/collision/joint-path evidence for full physical execution claims')
    write(OUT/'S2_S3_protocol.json',future)
    write(OUT/'registration_receipt.json',dict(S0_sha256=digest(OUT/'S0_protocol.json'),future_sha256=digest(OUT/'S2_S3_protocol.json'),
        new_real_scenes=0,new_navigation=0,new_terminal_H=0,optimizer_updates=0,
        synthetic_pre_registration_tests='Core and network structure probes separately logged in two unit receipts; backward without parameter update is not formal training'))
    print(json.dumps(dict(registered=True,S0_sha256=digest(OUT/'S0_protocol.json'),S2_S3_sha256=digest(OUT/'S2_S3_protocol.json'))))


def verify():
    p=read(OUT/'S0_protocol.json');r=read(OUT/'registration_receipt.json')
    assert digest(OUT/'S0_protocol.json')==r['S0_sha256']
    assert digest(OUT/'S2_S3_protocol.json')==r['future_sha256']
    for name,h in p['source_sha256'].items():assert digest(ROOT/name)==h,name
    for filename,key in [('seeds.json','seeds_sha256'),('seed_audit.json','seed_audit_sha256'),('unit_receipt.json','unit_sha256'),
                         ('policy_unit_receipt.json','policy_unit_sha256'),('PREREG_REVIEW.md','prereg_review_sha256'),
                         ('prereg_review.json','prereg_review_json_sha256'),('execution_environment.json','environment_sha256')]:
        assert digest(OUT/filename)==p[key],filename
    return p


if __name__=='__main__':register()
