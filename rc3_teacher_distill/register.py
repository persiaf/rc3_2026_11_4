"""Preregister all T seeds, source hashes, budgets and gates before data/training."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
BASE='5d5fe25754ffa032f97937a3fb54f15901c2fb7d'
FROZEN='df8cf28b0ece2271bd86d5eceee8ea5ca71a65d0'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_frozen_sources():
    frozen_manifest=json.loads(subprocess.check_output(['git','show',FROZEN+':project_management/GIT_MANIFEST.json'],cwd=ROOT))
    old=read(ROOT/'rc3_nav_value/r1/protocol.json')
    names=set(old['source_sha256'])|{'rc3_nav_value/r1/oracle.py','rc3_nav_value/r1/protocol.json'}
    for n in sorted(names):
        assert digest(ROOT/n)==frozen_manifest['files'][n]['sha256'],('Frozen source changed',n)


def seed_audit(seeds):
    started=time.perf_counter()
    targets={r[k] for split in seeds.values() for r in split for k in ('map_seed','task_seed')}
    names=[n.decode() for n in subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE],cwd=ROOT).split(b'\0') if n]
    # Historical declared banks include protected60; read declarations, never scenes.
    old=read(ROOT/'rc3_nav_value/r1/seed_audit.json')
    banks=old['explicit_historical_banks']
    for split,rows in read(ROOT/'rc3_nav_value/r1/seeds.json').items():
        for k in ('map_seed','task_seed'):
            v=[r[k] for r in rows];banks.append(dict(name='R1_'+split+'_'+k,start=min(v),end=max(v)))
    overlaps=[dict(seed=t,bank=b) for t in sorted(targets) for b in banks if b['start']<=t<=b['end']]
    pattern=re.compile(rb'(?<![0-9A-Za-z_.])(?:9[1-6][0-9]{8})(?![0-9A-Za-z_.])')
    inventory=[];hits=[];excluded=[]
    suffixes={'.json','.jsonl','.yaml','.yml','.csv','.tsv','.py','.md','.txt','.log','.m','.toml'}
    for name in names:
        p=ROOT/name
        if 'future_test' in name.lower() or 'retained60' in name.lower() or re.search(r'(^|/)test[_/].*scene',name):
            excluded.append(name);continue
        gz=name.endswith('.jsonl.gz')
        if not gz and p.suffix.lower() not in suffixes:continue
        sha=hashlib.sha256();logical=0;tail=b''
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
        with (gzip.open(p,'rb') if gz else p.open('rb')) as f:
            for block in iter(lambda:f.read(1024*1024),b''):
                for m in pattern.finditer(tail+block):
                    if int(m.group()) in targets:hits.append(dict(path=name,seed=int(m.group())))
                logical+=len(block);tail=block[-128:]
        inventory.append(dict(path=name,sha256=sha.hexdigest(),scanned_bytes=logical))
    write(OUT/'seed_audit.json',dict(base=BASE,verdict='PASS' if not hits and not overlaps else 'FAIL',
        scenes=len(targets)//2,distinct_scene_seeds=len(targets),literal_hits=hits,range_overlaps=overlaps,
        inventory=inventory,protected_scene_exclusions=excluded,elapsed_s=time.perf_counter()-started,
        limitations='Tracked text/gzip records and declared historical/protected banks only; no binary archive internals or unrecorded external executions; no protected scene contents read'))
    if hits or overlaps:raise ValueError('New scene seed collision')


def register():
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already registered; no replacement')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=BASE:
        raise ValueError('Unexpected maintenance base')
    verify_frozen_sources()
    t0=read(OUT/'t0_diagnostics.json')
    assert (OUT/'PREREG_REVIEW.md').is_file(),'Independent preregistration review required'
    assert read(OUT/'unit_receipt.json')['tests_passed']==19
    assert all(t0['reference_versions']['frozen_sources_match'].values())
    assert t0['checks']['r1_actual_selected_action_versioned_terminal_join']==1024
    seeds={split:[dict(case=i,map_seed=ms+i,task_seed=ts+i,mode=('regular','mixed','blocked')[i%3],
             partition=('fit' if i<80 else 'calibration') if split=='train' else split) for i in range(n)]
        for split,ms,ts,n in [('train',9100000000,9200000000,96),('development',9300000000,9400000000,24),('T2',9500000000,9600000000,36)]}
    write(OUT/'seeds.json',seeds);seed_audit(seeds)
    old=read(ROOT/'rc3_nav_value/r1/protocol.json')
    sources=set(old['source_sha256'])|{'rc3_nav_value/r1/oracle.py'}
    sources.update(p.relative_to(ROOT).as_posix() for p in OUT.glob('*.py'))
    p=dict(schema='rc3-teacher-distill-T0-T2-v1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        maintenance_base=BASE,frozen_research_commit=FROZEN,branch='codex/teacher-distill-t0-t2',
        user_scope='T0/T1 and conditional T2 only; no T3/RL/API/PPO, no push, no old evidence writes',
        source_sha256={n:digest(ROOT/n) for n in sorted(sources)},
        seeds_sha256=digest(OUT/'seeds.json'),seed_audit_sha256=digest(OUT/'seed_audit.json'),
        T0_sha256=digest(OUT/'t0_diagnostics.json'),H_source_sha256=old['H_source_sha256'],H_sha256=old['H_sha256'],
        prereg_review_sha256=digest(OUT/'PREREG_REVIEW.md'),unit_receipt_sha256=digest(OUT/'unit_receipt.json'),
        old128_sha256=old['old128_sha256'],reference='gB-v1: per-scene native Fixed3 original node order, first legal positive-gain B action; never warm g0 labels',
        cost='J=L+.75K, full ordered prefix+forced action+Complete_gB followed by frozen complete H; old reachability/coverage/export/scalar checks unchanged',
        menu=dict(K=8,reference_first=True,fill_targets='gain to2; gain/(Dcurrent+.75) to4; scarcity=sum1/legalDegree to6; spatial farthest-minimum to8; numeric tie original ID',
            legal='remaining positive coverage AND finite Dcurrent AND finite Dhome',labels_not_used=True,rule='same-menu gain/(Dcurrent+.75) maximizer, original ID ties'),
        data=dict(train_maps=96,fit_maps=80,calibration_maps=16,development_maps=24,T2_maps=36,
            prefix_thresholds=[0,.25,.5,.75],prefix_distribution='actual gB trajectory, first state crossing each coverage threshold; duplicate prefixes skipped without replacements',
            split='all different map/task seeds; each map one task draw, balanced modes for dev24/T2_36; T2 not generated or read until T1 passes',
            legacy_labels='T0 analysis only; NO legacy labels enter T1',unknown='never zero/negative-filled; every offered T1 menu action fully queried'),
        models=dict(controls=['same-menu deterministic rule','linear rank/value','MLP38-64-64-1'],
            prefix_representation='exact full-prefix state kernel; model receives38 engineered features including prefix edge/order-weighted summaries, NOT a lossless sequence encoder',
            stochastic_seeds=[11512026,11512027,11512028],primary='mean of3 MLP endpoints',
            ensemble='linear and MLP each3 seeds, same fit states/features/standardization/minibatch schedule, 1024 updates per seed, batch32states, Adam1e-3, clip1, no dropout/checkpoint selection',
            target='advantage percent=100*(Yref-Y)/Yref; predicted reference score anchored exactly0',
            loss='Huber beta.5 + .25 * true-gap-weighted pairwise softplus temperature.25; ties1e-7 ignored',
            calibration='no fitting: scene maximum absolute advantage error across all calibration menus/actions; order ceil((16+1)*.9)=16, descriptive simultaneous intervals, not a safety theorem',
            choice='largest ensemble mean advantage; original node ties. No extra risk coefficient or dev tuning.'),
        T1_gate=dict(primary='MLP ensemble must meet ALL; linear recorded separately, cannot substitute for failed primary',
            aggregation='equal-weight independent scenes; equal-weight distinct available prefix states within scene; same unit for means and bootstrap',
            tied_top1_min_percent=60,top1_absolute_J_tolerance=1e-7,near_optimal_tolerance_percent=.1,
            near_optimal_min_percent=75,descriptive_near_optimal_tolerance_percent=.5,mean_regret_max_percent=.5,
            rule_regret_reduction='scene paired20000 bootstrap95 lower strictly positive',
            selected_optimism_over1pp_max_percent=10,scene_simultaneous_interval_coverage_min_percent=80,
            all_feasible_finite=True,unknown_menu_actions_allowed=0,
            fail='freeze all endpoints/labels/negative results; T2 NOT RUN, no extra epochs/seed/menu variants'),
        T2=dict(conditional_on_T1=True,methods=['gBH','menu_rule','linear','neural','fixed3','detour16','old128'],
            repetitions=3,nav_cache_modes=['cold','hot'],model_load='outside request, recorded separately; no neural feature cache reuse',
            full_request='include C/F/D prep + generating native Fixed3 B for each method needing B + menu/features/inference + construction + H + export/scalar validation; native baselines retain native pipeline',
            timing='randomized deterministic order seed11522026; deployment compute sum excludes fsync/logging IO, request wall and residual also recorded. Hot cache priming separately counted. No B cache shared across requests.',
            paired_teacher='new-scene deterministic menu-K8 rollout,4 coverage checkpoints, frozen gB completions and H for each action, teacher offline only. NOT included in student decisions.',
            mechanism_gate='neural vs gBH mean paired gain>=.5% AND bootstrap95 lower>0; vs same-menu rule bootstrap95 lower>0; all36 valid',
            compression_gate='new paired teacher mean positive; ratio(sum(gBH-neural)/sum(gBH-teacher))>=1/3, and paired mechanism gate; never divide by old R0 mean',
            deployment_gate='Detour degradation one-sided95upper<=.5%; complete cold/hot median ratio<=.7 and p95ratio<=1; not dominated in meanJ/median/p95 byFixed3',
            failure='retain full negative results, no new training or T3'),
        budgets=dict(T1=dict(train=3072,development=768,selected_action_audit_reserved=256,reference_and_other=512,total=4608),
            T2=dict(routes=3072,offline_paired_teacher=1368,total=4440),combined_top_level=9048,
            actual_optimizer_updates_max=6144,wall_T1_s=3600,wall_T2_s=3600,
            deadline='T1 starts before first new scene; T2 separate deadline only if gate passes. Cooperative checks between calls/steps and before successful completion, not OS preemption of a running solver. Overrun remains recorded but inconclusive_stop. Failures charged; no replacements.'),
        bootstrap=dict(resamples=20000,seed=11522026,unit='independent scene',CI='percentile95 exploratory screen, no multi-seed population-confirmation claim'),
        counters='terminal attempts/completions/failures/native/fullH/internal search/export/scalar, exact cache hits, label uses, scenes/preparations/navhits/hotpriming, gB raw completions/decisions, menu/features, train/inference batches and rows, optimizer updates, elapsed wall separately',
        original60='SEALED; no content reads/generation/queries; protected seed range7700000000..7700000059',
        T3='NOT AUTHORIZED, no automatic start')
    write(OUT/'protocol.json',p)
    write(OUT/'registration_receipt.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        registered_utc=p['registered_utc'],optimizer_updates_before_registration=0,new_label_queries_before_registration=0,
        new_scene_generation_before_registration=0,source_files=len(sources),T0_only_before_registration=True))
    print(json.dumps(dict(registered=True,protocol_sha256=digest(OUT/'protocol.json'),source_files=len(sources))))


def verify():
    verify_frozen_sources()
    p=read(OUT/'protocol.json')
    receipt=read(OUT/'registration_receipt.json')
    assert digest(OUT/'protocol.json')==receipt['protocol_sha256']
    for n,h in p['source_sha256'].items():assert digest(ROOT/n)==h,n
    for n,k in [('seeds.json','seeds_sha256'),('seed_audit.json','seed_audit_sha256'),('t0_diagnostics.json','T0_sha256'),
                ('PREREG_REVIEW.md','prereg_review_sha256'),('unit_receipt.json','unit_receipt_sha256')]:
        assert digest(OUT/n)==p[k],n
    assert digest(ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt')==p['old128_sha256']
    return p


if __name__=='__main__':register()
