"""True-parent matched effects; seeds/windows are never independent layouts."""
from .common import *


def parent_effects(rows,reference,method='RL',*,by_seed=False):
    buckets={}
    for x in rows:
        if x['study_method'] not in (reference,method):continue
        seed=x.get('seed')
        key=(x['physical_key'],seed if by_seed else None)
        buckets.setdefault(key,{}).setdefault(x['study_method'],[]).append(x['metrics']['J'])
    return [dict(parent=k[0],seed=k[1],gain=(float(np.mean(v[reference]))-float(np.mean(v[method])))/float(np.mean(v[reference])))
        for k,v in sorted(buckets.items(),key=str) if method in v and reference in v]


def paired_effects(rows,reference,method='RL',seeds=SEEDS):
    # Match own-seed model comparisons, deterministic native references broadcast.
    lookup={}
    for r in rows:lookup[(r['case_id'],r.get('seed'),r['study_method'])]=r
    grouped={};modes={}
    for r in rows:
        if r['study_method']!=method or r.get('seed') not in seeds:continue
        key=(r['case_id'],r['seed'],reference);c=lookup.get(key,lookup.get((r['case_id'],None,reference)))
        require(c is not None,'Missing paired reference')
        p=r['physical_key'];modes[p]=r['spec']['mode']
        grouped.setdefault(p,{}).setdefault(r['seed'],[[],[]]);grouped[p][r['seed']][0].append(c['metrics']['J']);grouped[p][r['seed']][1].append(r['metrics']['J'])
    require(grouped,'No complete parents for contrast')
    effects=[];seed_effects={s:[] for s in seeds}
    for p,v in sorted(grouped.items()):
        require(set(v)==set(seeds),'Incomplete seed pairing')
        g=[]
        for s,(c,m) in v.items():
            a=(float(np.mean(c))-float(np.mean(m)))/float(np.mean(c));g.append(a);seed_effects[s].append(a)
        effects.append(dict(parent=p,mode=modes[p],gain=float(np.mean(g))))
    rng=np.random.Generator(np.random.PCG64(11652026));strata=[]
    for mode in ('mixed','blocked'):
        a=np.array([e['gain'] for e in effects if e['mode']==mode],float)
        if len(a):strata.append(a[rng.integers(0,len(a),size=(20000,len(a)))])
    boots=np.concatenate(strata,axis=1).mean(1)
    return dict(mean=float(np.mean([e['gain'] for e in effects])),CI95=np.quantile(boots,[.025,.975]).tolist(),
        one_sided95_lower=float(np.quantile(boots,.05)),parents=effects,seed_means={str(s):float(np.mean(v)) for s,v in seed_effects.items()},
        bootstrap=20000,unit='physical map_key; six views averaged within parent; then matched seeds averaged')


def select_candidates(info,rows):
    seed=info['seed'];options=[]
    base=[r for r in rows if r['study_method']=='SL256' and r['seed']==seed]
    from .eval import gains
    for role in ('RL','residual_SL'):
        candidates=[]
        for point in read(protocol_path())['training']['selection_candidates']:
            if str(point) not in info['candidates']:continue
            entry=info['candidates'][str(point)][role]
            rr=[r for r in rows if r['seed']==seed and r['study_method']==role and r['update']==point]
            require(len(rr)==48,'Candidate selection grid incomplete')
            if role=='residual_SL':rr=[dict(r,study_method='RL') for r in rr]
            candidates.append((gains(base+rr,'SL256',seed),-point,entry))
        require(candidates,'No fixed selection candidates')
        best=max(candidates,key=lambda a:(a[0],a[1]));options.append((role,dict(entry=best[2],gain_vs_SL=best[0],round=-best[1])))
    chosen=dict(options);rl=chosen['RL']['entry'];sl=chosen['residual_SL']['entry']
    rr=[r for r in rows if r['seed']==seed and ((r['study_method']=='RL' and r['update']==rl['update']) or (r['study_method']=='residual_SL' and r['update']==sl['update']))]
    control=gains(rr,'residual_SL',seed)
    return dict(seed=seed,chosen=chosen,gain_vs_matched_SL=control,
        P2_pass=chosen['RL']['gain_vs_SL']>=.005 and control>=.005,threshold=.005,
        tie='Earlier fixed candidate; same candidates for RL and matched residual-SL')
