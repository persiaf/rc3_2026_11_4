"""Independent G2 ledger review. Default mode is stdlib-only, not a rerun.

--checkpoints imports Torch only after collection has stopped. --replay performs
six explicitly chosen fresh executions and one duplicate checkpoint restoration.
Neither operation adds training samples or finalizes the eight saved workers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, message, atol=1e-7):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(a, b, rel_tol=1e-8, abs_tol=atol), f'{message}: {a} != {b}')


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def lines(path, allow_incomplete=False):
    raw = Path(path).read_text(encoding='utf-8').splitlines()
    result = []
    for index, line in enumerate(raw):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if not allow_incomplete or index != len(raw)-1:
                raise
    return result


def audit_protocol():
    p = read(OUT/'protocol.json')
    digest = sha(OUT/'protocol.json')
    require((OUT/'protocol.sha256').read_text().strip().split()[0] == digest,
            'preregistered protocol checksum changed')
    expected = dict(teacher_initial=64, teacher_extra=32, train=384,
                    validation=24, rule_dev=12, pipeline=6)
    require({k: len(v) for k, v in p['splits'].items()} == expected, 'split counts')
    seen = set()
    for split, cases in p['splits'].items():
        for i, case in enumerate(cases):
            require(case['split'] == split and case['case_id'] == i, 'case index/split')
            require(case['nx'] == case['ny'] == 10, 'unexpected scene scale')
            require(case['seed'] not in seen, 'seed crosses splits or is duplicated')
            seen.add(case['seed'])
    require(Counter(c['mode'] for c in p['splits']['validation']) ==
            dict(regular=8, mixed=8, blocked=8), 'validation strata')
    manifest = read(OUT/'formal_source_manifest.json')['source_sha256']
    for name, digest0 in manifest.items():
        require(sha(ROOT/name) == digest0, 'formal production source changed: '+name)
    return p, dict(protocol_sha256=digest, split_counts=expected,
                   distinct_seeds=len(seen), production_files_verified=len(manifest))


def audit_teacher_metadata(p):
    result = {}
    for name in ('teacher_initial', 'teacher_extra'):
        directory = OUT/name
        spec, summary, prior = [read(directory/f) for f in
            ('collection_spec.json', 'summary.json', 'audit_receipt.json')]
        require(spec['cases'] == p['splits'][name], 'teacher cases changed')
        require(spec['config'] == p['config'], 'teacher config changed')
        for source_name, digest in spec['source_sha256'].items():
            require(sha(ROOT/source_name) == digest, 'teacher source changed: '+source_name)
        require(spec['depth'] == 2 and spec['branch_limit'] == 8 and
                spec['tie_tolerance'] == .001, 'teacher search contract')
        require(prior['all_passed'] and summary['completed_cases'] == len(spec['cases'])
                and not summary['failed_cases'], 'teacher was not fully audited')
        require(sha(directory/'teacher.pkl') == summary['teacher_sha256'] ==
                prior['teacher_sha256'], 'teacher pickle content changed')
        physical = Counter()
        cert = Counter()
        for i, case in enumerate(spec['cases']):
            case_dir = directory/f'case_{i:04d}'
            record = read(case_dir/'summary.json')
            require(record['case'] == case and record['status'] == 'complete', 'teacher case')
            with gzip.open(case_dir/'queries.jsonl.gz','rt',encoding='utf-8') as handle:
                queries=[json.loads(line) for line in handle]
            with gzip.open(case_dir/'menus.jsonl.gz','rt',encoding='utf-8') as handle:
                menus={tuple(m['path']):m for m in map(json.loads,handle)}
            # Expanded menu records do not include depth-cut leaf menus. Recover
            # their charge from the actual external event (including correction),
            # rather than omit the physically built final menu from oracle costs.
            computed=Counter(menu_builds=1,menu_work_units=menus[()]['observation']['pending_menu_work'])
            for query in queries:
                info=query['info']
                computed['selected_edges']+=1
                computed['proposal_calls']+=int(info['proposal'] is not None)
                computed['proposal_work_units']+=info['proposal_work_units']
                computed['certification_calls']+=int(info['proposal_valid'])
                if query['leaf']:
                    computed['certification_calls']+=1+int(query['finalization']['fallback_used'])
                if info['finalization'] is None:
                    computed['menu_builds']+=1
                    if query['leaf']:
                        units=(query['finalization']['quality_correction']-query['terminal_event_reward'])*spec['config']['work_scale']/spec['config']['lambda_work']
                        close(units,round(units),'leaf physical menu charge')
                        computed['menu_work_units']+=round(units)
                    else:
                        computed['menu_work_units']+=menus[tuple(query['path'])]['observation']['pending_menu_work']
            require(dict(computed)==record['physical_oracle_work'],'raw query/menu physical oracle work')
            physical.update(record['physical_oracle_work'])
            for key in ('hits', 'misses', 'bypasses', 'actual_certifications'):
                cert[key] += record['cache_after'][key]-record['cache_before'][key]
        require(dict(physical) == summary['physical_oracle_work'], 'teacher physical work sum')
        require(cert['hits']+cert['actual_certifications'] == physical['certification_calls'],
                'teacher certificate queries omitted')
        require(cert['actual_certifications'] == cert['misses']+cert['bypasses'], 'teacher cache')
        result[name] = dict(cases=len(spec['cases']), samples=summary['samples'],
            queries=summary['query_count'], teacher_sha256=summary['teacher_sha256'],
            physical_oracle_work=dict(physical), certificate_cache=dict(cert),
            unknown_legal_actions=prior['unknown_legal_actions'],
            raw_label_audit='Existing all-query audit receipt re-bound to the identical teacher SHA; full pickle/raw replay requires --checkpoints.')
    debug = read(OUT/'teacher_debug_v1_bound_rejection'/'interruption_receipt.json')
    result['excluded_debug_v1'] = debug
    result['scope'] = 'Teacher depth-2 observed evidence is actor-only, not PPO rollout or horizon-16 critic target; interrupted v1 costs separate, unflushed tail unknown.'
    return result


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def teacher_root_diagnostics():
    """Pure JSON post hoc diagnosis; all costs were paid during collection."""
    records = []
    for name in ('teacher_initial','teacher_extra'):
        spec = read(OUT/name/'collection_spec.json')
        for index, case in enumerate(spec['cases']):
            with gzip.open(OUT/name/f'case_{index:04d}'/'queries.jsonl.gz','rt',encoding='utf-8') as handle:
                queries = [json.loads(line) for line in handle]
            roots = [q for q in queries if len(q['path']) == 1]
            stop = [q for q in roots if q['action']['kind'] == 'stop']
            require(len(stop) == 1, 'root STOP missing')
            best = max(q['remaining_return'] for q in roots)
            stop_q = stop[0]['remaining_return']
            gain = best-stop_q
            records.append(dict(split=name, case=case, best_observed_remaining_return=best,
                root_STOP_return=stop_q, gain_over_STOP=gain,
                STOP_near_best=gain <= spec['tie_tolerance']+1e-12,
                best_observed_path=next(q['best_descendant_path'] for q in roots if q['remaining_return']==best),
                observed_nonstop_actions=len(roots)-1))
    groups = {}
    for split in ('all','teacher_initial','teacher_extra'):
        for mode in ('all','regular','mixed','blocked'):
            subset = [r for r in records if (split=='all' or r['split']==split)
                      and (mode=='all' or r['case']['mode']==mode)]
            values = [r['gain_over_STOP'] for r in subset]
            groups[split+'/'+mode] = dict(cases=len(subset),STOP_near_best=sum(r['STOP_near_best'] for r in subset),
                gain_over_STOP=dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values),
                    q25=quantile(values,.25),median=quantile(values,.5),q75=quantile(values,.75)),
                best_observed_return_mean=statistics.mean(r['best_observed_remaining_return'] for r in subset),
                root_STOP_return_mean=statistics.mean(r['root_STOP_return'] for r in subset))
    return dict(groups=groups,records=records,
        scope='Post hoc offline diagnosis of all 96 registered teacher roots; highest return among queried depth-2 suffixes only. No new queries, no PPO or horizon-16 oracle claims.')


def audit_bc(p):
    result = {}
    for kind, updates, samples, scenes in (('initial',400,512,64), ('continued',2048,768,96)):
        record = read(OUT/'training'/f'bc_{kind}_training.json')
        require(record['updates'] == updates and len(record['rows']) == updates, 'BC count')
        require(record['samples'] == samples and record['unique_scenes'] == scenes, 'BC dataset count')
        require(record['actor_only'] and not record['critic_supervised'], 'BC critic contaminated')
        require(finite(record) and [r['update'] for r in record['rows']] == list(range(1,updates+1)),
                'BC updates or finiteness')
        result[kind] = {k:v for k,v in record.items() if k != 'rows'}
    pipeline = read(OUT/'pipeline'/'summary.json')
    require(pipeline['all_passed'], 'pipeline failed')
    require({r['kind']:r['actual_actions'] for r in pipeline['reports']} ==
            dict(mlp=256,graph=768), 'pipeline/formal action confusion')
    result['pipeline_receipt'] = pipeline
    prior_pipeline=read(OUT/'pipeline_before_resume_guards'/'summary.json')
    result['prior_debug_pipeline_receipt']=prior_pipeline
    result['all_preparatory_pipeline_actual_actions']=sum(r['actual_actions'] for x in (pipeline,prior_pipeline) for r in x['reports'])
    result['scope'] = 'BC supervised updates and pipeline interactions are not included in formal PPO actions.'
    return result


def audit_rule_selection(p):
    directory=OUT/'rule_development'
    spec=read(directory/'selection_protocol.json')
    selected=read(directory/'selected_ils.json')
    require(spec['cases']==p['splits']['rule_dev'],'rule tuning used wrong cases')
    require(selected['frozen'] and selected['validation_used'] is False,'rule validation leakage')
    require(selected['selection_protocol_sha256']==sha(directory/'selection_protocol.json'),'rule protocol changed')
    scores=[]
    for score in selected['scores']:
        target=directory/f"grid_{score['grid_index']}"
        rows=read(target/'results.json')
        require(len(rows)==12 and {r['seed'] for r in rows}=={c['seed'] for c in p['splits']['rule_dev']},'rule rows')
        require(all(r['status']=='complete' and r['valid'] for r in rows),'rule development failure')
        close(score['mean_return'],statistics.mean(r['total_reward'] for r in rows),'rule mean return')
        close(score['mean_work'],statistics.mean(r['work_used'] for r in rows),'rule mean work')
        scores.append(score)
    winner=min(scores,key=lambda s:(-s['mean_return'],s['mean_work'],s['grid_index']))
    require(winner['settings']==selected['selected'],'rule selection criterion')
    return dict(selected=selected['selected'],cases=12,query_runs=48,scores=scores,
        selection_sha256=sha(directory/'selected_ils.json'),validation_used=False,
        scope='All four registered ILS development-grid means independently recomputed; separate from 24 validation cases.')


def audit_iterations(p, allow_incomplete, rows=None):
    rows = lines(OUT/'training'/'ppo_iterations.jsonl', allow_incomplete) if rows is None else rows
    batch = p['ppo']['workers']*p['ppo']['rollout_steps']
    cumulative = 0
    for i, row in enumerate(rows, 1):
        require(finite(row), 'nonfinite PPO iteration')
        require(row['iteration'] == i and row['actual_actions'] == i*batch, 'action counting')
        require(0 < row['updates'] <= 16, 'optimizer updates outside epoch/minibatch cap')
        cumulative += row['updates']
        require(row['optimizer_updates'] == cumulative, 'cumulative optimizer count')
        require(row['advantages_finite'] and row['gradients_finite'], 'PPO nonfinite')
        require(0 <= row['old_new_logp_error'] <= 3e-5, 'old/new log-prob mismatch')
        for key in ('stop_fraction','changed_fraction'):
            require(0 <= row[key] <= 1, 'invalid action fraction')
            close(row[key]*batch, round(row[key]*batch), 'fraction action count')
        require(0 <= row['clip_fraction'] <= 1 and row['approx_kl'] >= -1e-7, 'PPO clipping/KL')
        for key in ('collect_s','inference_s','worker_s','elapsed_s','validation_s','optimizer_s'):
            require(row[key] >= 0, 'negative training time')
    require(rows and rows[-1]['actual_actions'] <= p['ppo']['actions'], 'formal action cap')
    if not allow_incomplete:
        require(rows[-1]['actual_actions'] == p['ppo']['actions'], 'PPO not complete')
    return rows, dict(iterations=len(rows), actual_actions=rows[-1]['actual_actions'],
        optimizer_updates=cumulative, early_kl_iterations=sum(r['early_kl_stop'] for r in rows),
        sampled_STOP_actions=round(sum(r['stop_fraction']*batch for r in rows)),
        sampled_changed_actions=round(sum(r['changed_fraction']*batch for r in rows)),
        completed_episode_count_from_rollouts=sum(r['completed_episodes'] for r in rows),
        max_old_new_logp_error=max(r['old_new_logp_error'] for r in rows),
        optimizer_s=sum(r['optimizer_s'] for r in rows), collect_s=sum(r['collect_s'] for r in rows),
        inference_s=sum(r['inference_s'] for r in rows), worker_wait_s=sum(r['worker_s'] for r in rows),
        last=rows[-1], scope='Retained fresh on-policy interactions; collector boundaries do not terminate '
            'or finalize episodes. A separately recorded killed/uncommitted tail, if present, is not included.')


def audit_interruption(p, iterations, episodes):
    """Bind retained-prefix accounting without inventing the killed tail."""
    path=OUT/'interruption_receipt.json'
    if not path.exists():
        return dict(interrupted=False,retained_actions=iterations[-1]['actual_actions'],
                    physical_actions_interval=[iterations[-1]['actual_actions']]*2,
                    resumed_first_episodes=[])
    receipt=read(path)
    require(receipt['schema']=='rc3-g2-interruption-v1','unknown interruption record')
    boundary=receipt['committed_actual_actions'];batch=p['ppo']['workers']*p['ppo']['rollout_steps']
    require(type(boundary) is int and boundary>0 and boundary%batch==0,'invalid interruption boundary')
    require(boundary<=iterations[-1]['actual_actions'],'audit precedes retained checkpoint')
    row=iterations[boundary//batch-1]
    require(row['iteration']==receipt['iteration'] and row['optimizer_updates']==receipt['optimizer_updates'],
            'interruption checkpoint counters differ from retained log prefix')
    interval=receipt['possible_unlogged_physical_actions_interval']
    require(interval==[0,batch], 'unexpected unlogged-action bound')
    checkpoint=Path(receipt['checkpoint'].replace('\\','/'))
    if not checkpoint.is_absolute():checkpoint=ROOT.parent/checkpoint
    checkpoint=checkpoint.resolve()
    require(checkpoint.is_relative_to(OUT.resolve()),'interruption artifact escapes output directory')
    require(sha(checkpoint)==receipt['checkpoint_sha256'],'retained interruption checkpoint changed')
    prefix=[e for e in episodes if e['training_actions']<=boundary]
    require(len(prefix)==receipt['completed_episode_rows'],'interruption episode prefix count')
    owners={c['seed']:i%8 for i,c in enumerate(p['splits']['train'])}
    counts=Counter(owners[e['case']['seed']] for e in prefix)
    first=[dict(worker=i,episode=counts[i]+1) for i in range(8)]
    prefix_sha={}
    for name,key in [('ppo_iterations.jsonl','actual_actions'),('ppo_episodes.jsonl','training_actions')]:
        kept=[line for line in (OUT/'training'/name).read_bytes().splitlines(keepends=True)
              if line.strip() and json.loads(line)[key]<=boundary]
        prefix_sha[name]=dict(rows=len(kept),sha256=hashlib.sha256(b''.join(kept)).hexdigest())
    retained=iterations[-1]['actual_actions']
    return dict(interrupted=True,receipt_sha256=sha(path),checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=receipt['checkpoint_sha256'],retained_boundary=boundary,
        retained_actions=retained,unknown_discarded_actions_interval=interval,
        unknown_discarded_optimizer_updates_interval=[0,p['ppo']['epochs']*((batch+p['ppo']['minibatch']-1)//p['ppo']['minibatch'])],
        physical_actions_interval=[retained+interval[0],retained+interval[1]],
        resumed_first_episodes=first,retained_prefix_hashes=prefix_sha,
        unknown_tail_work_and_time=True,
        scope='One killed uncommitted rollout may have executed 0..256 additional physical actions. '
              'Up to one rollout optimizer cycle (0..16 attempted updates) may also be lost before log flush. '
              'It contributes neither retained actor samples nor retained optimizer updates. '
              'Final 32768 is the retained on-policy count, not an exact total physical-call claim. '
              'Trainer elapsed time excludes interruption idle time and unlogged tail time. '
              'Resume creates and validates adapters outside episode cache deltas; these costs are not silently counted as zero.')


def audit_episodes(p, iterations, allow_incomplete, all_rows=None, interruption=None):
    all_rows = lines(OUT/'training'/'ppo_episodes.jsonl', allow_incomplete) if all_rows is None else all_rows
    # The iteration summary is flushed BEFORE its episode rows. A live reader
    # uses the common complete prefix of both snapshots selected by main().
    rows = [r for r in all_rows if r['training_actions'] <= iterations[-1]['actual_actions']]
    registered = {c['seed']:(i%8,c) for i,c in enumerate(p['splits']['train'])}
    owned = defaultdict(list)
    counters, cert, preparation = Counter(), Counter(), Counter()
    interrupted_keys={(r['worker'],r['episode']) for r in (interruption or {}).get('resumed_first_episodes',[])}
    resume_cache_boundaries=[]
    for row in rows:
        require(finite(row) and row['validation'] is True, 'episode validity/finiteness')
        worker, expected = registered[row['case']['seed']]
        require(row['case'] == expected, 'unregistered training episode')
        owned[worker].append(row['episode'])
        close(row['quality_reward'], 100*(row['anchor_J']-row['J'])/max(row['anchor_J'],1e-8), 'quality')
        close(row['work_penalty'], p['config']['lambda_work']*row['work']/p['config']['work_scale'], 'work penalty')
        close(row['return_value'], row['quality_reward']-row['work_penalty'], 'complete return')
        require(row['J'] <= row['anchor_J']+1e-7, 'returned plan exceeds anchor')
        close(row['work'], row['counters']['menu_work']+row['counters']['proposal_work'], 'work ledger')
        require(not row['counters']['work_count_incomplete'], 'unknown training work')
        require(0 < row['steps'] <= p['config']['horizon'] and
                0 <= row['counters']['proposal_calls'] <= row['steps'], 'episode steps')
        require(row['overrun'] == max(0,row['work']-p['config']['work_budget']), 'work overrun')
        delta = {k:row['cache_after'][k]-row['cache_before'][k] for k in
                 ('hits','misses','bypasses','actual_certifications')}
        require(all(v>=0 for v in delta.values()), 'negative certificate cache delta')
        certificate_gap=row['counters']['certification_calls']-delta['hits']-delta['actual_certifications']
        if (worker,row['episode']) in interrupted_keys:
            require(certificate_gap>=0,'resume logical/physical certificate boundary is negative')
            resume_cache_boundaries.append(dict(worker=worker,episode=row['episode'],
                preserved_logical_certifications=certificate_gap,post_load_physical_delta=delta,
                checkpoint_prefix_verified=False))
        else:
            require(certificate_gap==0, 'episode actual certificate calls')
        require(delta['actual_certifications'] == delta['misses']+delta['bypasses'], 'certificate cache closure')
        cert.update(delta)
        counters.update({k:v for k,v in row['counters'].items() if not isinstance(v,bool)})
        stats = row['prepared_stats']
        preparation['memory' if stats.get('memory_hit') else 'disk' if stats.get('cache_hit') else 'cold'] += 1
    for worker, sequence in owned.items():
        require(sequence == list(range(1,len(sequence)+1)), f'worker {worker} episode sequence')
    require(len(rows) == sum(r['completed_episodes'] for r in iterations), 'episode vs rollout count')
    result = dict(completed_episodes=len(rows), completed_steps=sum(r['steps'] for r in rows),
        charged_work=sum(r['work'] for r in rows), counters=dict(counters), certificate_cache_delta=dict(cert),
        prepared_input_categories=dict(preparation), episodes_by_worker={str(k):len(v) for k,v in owned.items()},
        fallback_count=sum(r['fallback'] for r in rows), overrun_count=sum(r['overrun']>0 for r in rows),
        termination_reasons=dict(Counter(r['stop_reason'] for r in rows)),
        mean_return=statistics.mean(r['return_value'] for r in rows),
        unique_training_scenes=len({r['case']['seed'] for r in rows}),
        uncommitted_live_rows=len(all_rows)-len(rows),
        resume_cache_boundaries=resume_cache_boundaries,
        scope='All retained complete-episode scalar ledgers recomputed, not path replanning. '
              'complete_s includes optimizer/other-worker/validation waiting and is NOT deployment latency. '
              'Active workers are separate. At a restored episode boundary, physical cache deltas cover only '
              'post-load actions; preserved logical certification prefixes are separately checked against '
              'the retained interruption checkpoint under --checkpoints. Reset/load validation and any killed '
              'unlogged tail are outside these physical deltas, so this is not a total-process certification count.')
    return rows, result


def audit_evaluations(p, allow_incomplete):
    directories = [OUT/'validation'/f'ppo_{n}' for n in (8192,16384,32768)]
    directories += [OUT/'final_evaluation'/name for name in
                    ('controls_fixed_work','final_fixed_work','final_soft_time_0400ms')]
    result = {}
    for directory in directories:
        if not (directory/'results.json').exists():
            require(allow_incomplete, 'missing final evaluation: '+str(directory))
            continue
        request, rows = read(directory/'request.json'), read(directory/'results.json')
        require(request['cases'] == p['splits']['validation'] and request['config'] == p['config'],
                'evaluation input differs from protocol')
        methods = list(request['method_setup'])
        expected = {(i,m) for i in range(24) for m in methods}
        require({(r['case_index'],r['method']) for r in rows} == expected and len(rows)==len(expected),
                'missing/duplicate paired evaluation row')
        for row in rows:
            case = request['cases'][row['case_index']]
            require(row['seed']==case['seed'] and row['mode']==case['mode'], 'evaluation case mapping')
            require(row['status']=='complete' and row['valid'] is True, 'evaluation failure')
            require(finite(row) and row['J']<=row['anchor_J']+1e-7, 'evaluation finite/anchor')
            close(row['quality_gain_percent'],100*(row['anchor_J']-row['J'])/max(row['anchor_J'],1e-8),'evaluation quality')
            limit=request['time_limit_s']
            close(row['time_overrun_s'],max(0,row['complete_s']-limit) if limit is not None else 0,'time overrun')
            timing=row['timing']
            close(sum(v for k,v in timing.items() if v is not None and k!='complete_s'),row['complete_s'],'timing partition')
            path = directory/f"case_{row['case_index']:03d}_{row['method']}"/'evidence.json.gz'
            with gzip.open(path,'rt',encoding='utf-8') as handle:
                evidence=json.load(handle)
            require(evidence['row']==row, 'saved evidence differs from comparison row')
            plan=evidence['plan']
            require(plan['validation']['valid_model_plan'] and not plan['unserved_task_ids'],'saved plan status')
            close(plan['objective_length_plus_stops_m'],row['J'],'saved plan J')
            close(sum(plan['route_leg_costs_m']),plan['route_length_m'],'saved leg sum')
            close(plan['route_length_m']+len(plan['service_stops'])*plan['stop_penalty_equivalent_m'],row['J'],'L plus stop cost')
            if not row['shared_action_protocol']:
                require(row['total_reward'] is None and row['work_used'] is None,'fabricated original work return')
                continue
            state=evidence['state']
            require(state['terminated'] and len(state['transitions'])==row['steps'],'state terminal/steps')
            close(row['total_reward'], row['quality_gain_percent']-p['config']['lambda_work']*row['work_used']/p['config']['work_scale'],'evaluation return')
            close(sum(t['reward'] for t in state['transitions'])+sum(e['reward'] for e in state['events']),row['total_reward'],'transition plus event return')
            close(row['work_used'],state['counters']['menu_work']+state['counters']['proposal_work'],'evaluation work')
            close(state['total_reward'],row['total_reward'],'state reward')
            require(row['work_overrun']==max(0,row['work_used']-p['config']['work_budget']),'evaluation work overrun')
            for transition in state['transitions']:
                obs=transition['info']['sampled_observation']
                action=transition['action_id']
                require(obs['mask'][action] and obs['actions'][action]==transition['action'],'exact ordered legal action')
                info=transition['info']
                close(info['work_units'],info['menu_work_units']+info['proposal_work_units'],'step work partition')
                close(transition['reward'],info['quality_reward']-info['work_penalty']+info['shaping_reward'],'step reward components')
            actor=evidence['actor_records']
            require(len(actor)==row['actor_sample_count']==len(state['transitions']),'external event created action')
            if actor:
                close(sum(a['reward'] for a in actor),row['total_reward'],'external event exactly once')
                require(actor[-1]['terminated'],'last actor sample not terminal')
            else:
                require(row['metrics_only_zero_action'],'zero action episode lost')
        grouped={}
        for method in methods:
            own=[r for r in rows if r['method']==method]
            grouped[method]=dict(n=len(own),mean_J=statistics.mean(r['J'] for r in own),
                mean_return=None if own[0]['total_reward'] is None else statistics.mean(r['total_reward'] for r in own),
                median_complete_s=statistics.median(r['complete_s'] for r in own),
                mean_quality=statistics.mean(r['quality_gain_percent'] for r in own),
                root_STOP=sum(r['stop_at_root'] for r in own),fallback=sum(r['fallback'] for r in own),
                deadline_overruns=sum(r['time_overrun_s']>0 for r in own))
        result[directory.name]=dict(rows=len(rows),methods=grouped,results_sha256=sha(directory/'results.json'))
    return dict(datasets=result,scope='All saved final/stage JSON rows and evidence ledgers recomputed; this is not new path execution or timing. Only six explicitly identified cases are replayed with --replay.')


def audit_checkpoints(p, iterations, episodes, report):
    import torch
    from rc3_rl_g2.checkpoints import load_model, validate_training_state
    from rc3_rl_g2.prepared_cache import validate_worker_snapshot
    from rc3_rl_g2.tools.audit_teacher import audit_teacher
    torch.set_num_threads(1)
    initial=OUT/'training'/'bc_initial.pt'
    parent=sha(initial)
    model, base=load_model(initial,'cpu')
    require(base['updates']==400 and base['parent_bc_sha256'] is None,'initial BC metadata')
    identity=report['protocol']['protocol_sha256']
    interruption=report['interruption']
    resume_check=None
    if interruption['interrupted']:
        retained_path=ROOT/interruption['checkpoint']
        _,retained=load_model(retained_path,'cpu')
        validate_training_state(retained,p,parent,iterations[interruption['retained_boundary']//256-1])
        require(retained['protocol_sha256']==identity,'interrupted checkpoint protocol')
        preserved=[]
        for boundary in report['completed_episodes']['resume_cache_boundaries']:
            worker=retained['workers'][boundary['worker']]
            validate_worker_snapshot(worker,p['splits']['train'][boundary['worker']::8],p['config'],
                p['training_seed']+boundary['worker']*100003)
            require(worker['episode']==boundary['episode'],'resumed first episode ordinal')
            require(worker['env_state']['counters']['certification_calls']==boundary['preserved_logical_certifications'],
                    'restored certification prefix differs from checkpoint')
            boundary['checkpoint_prefix_verified']=True
            preserved.append(dict(worker=boundary['worker'],episode=worker['episode'],steps=worker['env_state']['steps'],
                logical_certifications=worker['env_state']['counters']['certification_calls'],
                old_prefix_physical_cache_breakdown_saved=False))
        require(len(preserved)==8,'not all resumed worker episode boundaries were audited')
        resume_check=dict(checkpoint_sha256=sha(retained_path),workers=preserved,
            source_and_protocol_verified=True,retained_prefix_count_verified=True,
            scope='Checkpoint-to-log restore prefixes verified. Certificate hits/misses before an active prefix '
                  'were not serialized; no physical breakdown is invented. Constructor reset/load preparation '
                  'is outside restored cumulative reset counters.')
    checks=[]
    for step in (512,1024,2048):
        path=OUT/'training'/f'bc_continued_{step:04d}.pt'
        _,payload=load_model(path,'cpu')
        require(payload['parent_bc_sha256']==parent and payload['protocol_sha256']==identity,'continued BC parent')
        require(payload['updates']==step,'continued BC updates')
        require(payload['teacher_initial_sha256']==sha(OUT/'teacher_initial'/'teacher.pkl') and
                payload['teacher_extra_sha256']==sha(OUT/'teacher_extra'/'teacher.pkl'),'BC teacher SHA')
        checks.append(dict(checkpoint=path.name,sha256=sha(path),parent_bc_sha256=parent,updates=step))
    final=None
    for step in (8192,16384,32768):
        path=OUT/'training'/f'ppo_{step:05d}.pt'
        _,payload=load_model(path,'cpu')
        log=iterations[step//256-1]
        validate_training_state(payload,p,parent,log)
        require(payload['protocol_sha256']==identity,'PPO protocol SHA')
        require(any(not torch.equal(base['model'][k],v) for k,v in payload['model'].items()),'PPO did not update')
        adam_steps=[int(state['step']) for state in payload['optimizer']['state'].values()]
        require(adam_steps and min(adam_steps)==max(adam_steps)==payload['optimizer_updates'],
                'Adam actual tensor step counters differ from log')
        checks.append(dict(checkpoint=path.name,sha256=sha(path),parent_bc_sha256=parent,
            actual_actions=step,optimizer_updates=payload['optimizer_updates'],parameters_changed=True,
            optimizer_parameter_states=len(adam_steps),adam_steps_min=min(adam_steps),adam_steps_max=max(adam_steps)))
        if step==32768:final=payload
    active=[]
    require(len(final['workers'])==8,'final worker count')
    for i, worker in enumerate(final['workers']):
        validate_worker_snapshot(worker,p['splits']['train'][i::8],p['config'],p['training_seed']+i*100003)
        state=worker['env_state']
        require(not state['terminated'] and state['finalization'] is None and state['returned_certificate'] is None,
                'collector boundary incorrectly finalized active worker')
        require(worker['episode']==report['completed_episodes']['episodes_by_worker'][str(i)]+1,'active episode ordinal')
        close(state['work_used'],state['counters']['menu_work']+state['counters']['proposal_work'],'active work')
        close(state['total_quality_reward'],100*(state['anchor_cost']-state['best_cost'])/max(state['anchor_cost'],1e-8),'active quality')
        close(state['total_work_penalty'],p['config']['lambda_work']*state['work_used']/p['config']['work_scale'],'active penalty')
        close(state['total_reward'],state['total_quality_reward']-state['total_work_penalty'],'active prefix reward')
        menu=state['menu']
        pending=0 if menu is None else menu['work_units']
        active.append(dict(worker=i,episode=worker['episode'],case=worker['active_case'],steps=state['steps'],
            charged_work=state['work_used'],pending_menu_work=pending,counters=state['counters'],
            prefix_reward=state['total_reward'],preparations=worker['preparations'],memory_reuses=worker['cache_reuses'],
            finalized=False))
    completed=report['completed_episodes']
    require(completed['completed_steps']+sum(w['steps'] for w in active)==32768,'complete plus active steps')
    nonstops=completed['counters']['proposal_calls']+sum(w['counters']['proposal_calls'] for w in active)
    require(32768-nonstops==report['ppo_iterations']['sampled_STOP_actions'],'STOP/action/proposal crosscheck')
    cold=sum(w['preparations'] for w in active)
    memory=sum(w['memory_reuses'] for w in active)
    started=completed['completed_episodes']+8
    audits={name:audit_teacher(OUT/name) for name in ('teacher_initial','teacher_extra')}
    return dict(checkpoints=checks,active_unfinished_workers=active,
        completed_plus_active_actions=32768,charged_work_completed_plus_active=completed['charged_work']+sum(w['charged_work'] for w in active),
        physically_built_pending_menu_work=sum(w['pending_menu_work'] for w in active),
        prepared_inputs=dict(retained_started_episodes=started,retained_counter_cold_builds=cold,
            retained_counter_memory_reuses=memory,retained_counter_disk_loads=started-cold-memory,
            excludes_resume_constructor_and_load_overhead=bool(interruption['interrupted']),
            excludes_unlogged_killed_tail=bool(interruption['interrupted'])),
        interruption_checkpoint=resume_check,
        teacher_all_raw_labels_recomputed=audits,
        scope='All checkpoint sources/finite tensors/parent identity and saved worker ledgers checked. Eight live episodes are kept unfinished; pending menu work is physical preparation, not yet debited reward. Metadata plus reviewed loader establishes shared BC parent; no unsaved initial PPO tensor snapshot is claimed.')


def native(value):
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    return value


def tree_close(actual, expected, path='value', atol=1e-7):
    actual,expected=native(actual),native(expected)
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and actual.keys()==expected.keys(),path+' keys')
        for key in expected:tree_close(actual[key],expected[key],path+'.'+key,atol)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),path+' length')
        for index,(a,b) in enumerate(zip(actual,expected)):tree_close(a,b,path+f'[{index}]',atol)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        close(actual,expected,path,atol)
    else:require(actual==expected,path+' differs')


def audit_replay(p):
    """Six predeclared fresh executions; two identical worker restorations."""
    import copy
    import time
    import torch
    from threadpoolctl import threadpool_limits
    from rc3_fast.scenes import make_complex_scene
    from rc3_rl_research.types import SearchConfig
    from rc3_rl_g2.adapter_g2 import ResearchAdapterG2
    from rc3_rl_g2.env_g2 import SearchEnvG2
    from rc3_rl_g2.graph_observation import build_policy_observation
    from rc3_rl_g2.checkpoints import load_model, capture_rng, restore_rng
    from rc3_rl_g2.evaluate_g2 import _model_identity
    torch.set_num_threads(1)
    threadpool_limits(1)
    config=SearchConfig(**p['config'])
    directory=OUT/'final_evaluation'/'final_fixed_work'
    checkpoints=dict(bc_frozen='bc_initial.pt',bc_continued_2048='bc_continued_2048.pt',ppo_final='ppo_32768.pt')
    models={}
    for method,name in checkpoints.items():
        models[method]=load_model(OUT/'training'/name,'cpu')[0].eval()
    results=[]
    started=time.perf_counter()
    for index in (0,1):
        for method in checkpoints:
            path=directory/f'case_{index:03d}_{method}'/'evidence.json.gz'
            with gzip.open(path,'rt',encoding='utf-8') as handle:evidence=json.load(handle)
            require(evidence['row']['model_identity']==_model_identity(models[method]),'replay model identity')
            adapter=ResearchAdapterG2.from_payload(evidence['adapter'])
            env=SearchEnvG2(adapter,config,evidence['initial_state']['seed'])
            env.load_state_dict(evidence['initial_state'])
            samples=[s for s in evidence['policy_samples'] if s['executed']]
            transitions=evidence['state']['transitions']
            require(len(samples)==len(transitions),'saved actor execution mapping')
            for transition,sample in zip(transitions,samples):
                observation=env.observe()
                tree_close(observation,transition['info']['sampled_observation'],'exact environment observation')
                raw=build_policy_observation(env)
                tree_close(raw,sample['observation'],'policy input reconstruction')
                action,logp,value=models[method].act([raw],deterministic=True)
                selected=int(action[0])
                require(selected==sample['action_id']==transition['action_id'],'greedy replay action')
                close(float(logp[0]),sample['old_logp'],'replay logp',atol=3e-5)
                close(float(value[0]),sample['old_value'],'replay value',atol=3e-5)
                _,reward,terminal,truncated,info=env.step(selected)
                close(reward,transition['reward'],'replayed transition reward')
                require(terminal==transition['terminated'] and truncated==transition['truncated'],'terminal replay')
                for key in ('work_units','menu_work_units','proposal_work_units','accepted','best_improved','failure_reason'):
                    require(info[key]==transition['info'][key], 'replayed info '+key)
            # Fixed-work runs should naturally finish; an actual saved exception
            # event is replayed as such, never an audit-imposed collector cut.
            for event in evidence['state']['events']:
                if not env.done:env.terminate(event['reason'])
            require(env.done,'saved fixed-work run did not terminate')
            certificate,finalization=env.finalize()
            tree_close(certificate.plan,evidence['plan'],'independent exported plan')
            tree_close(finalization,evidence['state']['finalization'],'finalization')
            close(env.total_reward,evidence['row']['total_reward'],'replay complete return')
            require(env.work_used==evidence['row']['work_used'],'replay work')
            results.append(dict(case_index=index,seed=evidence['row']['seed'],method=method,
                saved_evidence_sha256=sha(path),actual_actions=env.steps,J=certificate.cost,
                return_value=env.total_reward,work=env.work_used,validated=True,
                proposal_calls=env.counters['proposal_calls'],
                changed_proposals=sum(bool(t['info']['route_changed']) for t in env.transitions),
                cache_stats=adapter.cache_stats))
    _,payload=load_model(OUT/'training'/'ppo_32768.pt','cpu')
    worker=payload['workers'][0]
    original_rng=capture_rng('cpu')
    duplicate=[]
    for repeat in range(2):
        case=worker['active_case']
        scene=make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode'])
        adapter=ResearchAdapterG2.from_scene(scene)
        env=SearchEnvG2(adapter,config,worker['env_state']['seed'],record_transitions=False)
        env.load_state_dict(copy.deepcopy(worker['env_state']))
        restore_rng(payload['torch_rng'])
        trace=[]
        for _ in range(3):
            if env.done:break
            raw=build_policy_observation(env)
            action,logp,value=models['ppo_final'].act([raw],deterministic=False)
            selected=int(action[0]);semantic=env.observe()['actions'][selected]
            _,reward,terminal,truncated,info=env.step(selected)
            trace.append(dict(raw=native(raw),action_id=selected,action=semantic,logp=float(logp[0]),value=float(value[0]),
                reward=reward,terminal=terminal,truncated=truncated,work=env.work_used,
                current=list(env.current),best=list(env.best),memory=native(env.attempt_memory)))
        duplicate.append(dict(trace=trace,terminal=env.done,steps=env.steps,work=env.work_used,
            return_value=env.total_reward,cache_stats=adapter.cache_stats))
    restore_rng(original_rng)
    tree_close(duplicate[0]['trace'],duplicate[1]['trace'],'restored policy/environment sampled sequence')
    result=dict(saved_plan_replays=results,restored_worker=0,
        restored_case=worker['active_case'],restored_prefix_steps=worker['env_state']['steps'],
        duplicate_restorations=[{k:v for k,v in d.items() if k!='trace'} for d in duplicate],
        sampled_actions=[dict(action_id=t['action_id'],action=t['action'],reward=t['reward'],terminal=t['terminal']) for t in duplicate[0]['trace']],
        restore_sequences_identical=True,elapsed_audit_s=time.perf_counter()-started,
        scope='Six predeclared final fixed-work case/method runs physically reconstructed from saved inputs, policy observations/actions reproduced, all repairs and final validators rerun. Worker 0 checkpoint restored twice and sampled up to three real actions, stopping naturally; no other active worker finalized, no training counter modified. Audit time/cache queries are additional review costs, not deployment measurements.')
    (OUT/'replay_audit_trace.json').write_text(json.dumps(native(dict(result=result,worker_duplicate_traces=duplicate)),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT/'g2_audit.json')
    args = parser.parse_args()
    p, protocol_report = audit_protocol()
    metric_snapshot=lines(OUT/'training'/'ppo_iterations.jsonl',args.allow_incomplete)
    episode_snapshot=lines(OUT/'training'/'ppo_episodes.jsonl',args.allow_incomplete)
    uncommitted_iterations=0
    if args.allow_incomplete:
        counts=Counter(r['training_actions'] for r in episode_snapshot)
        prefix=[]
        for row in metric_snapshot:
            available=counts[row['actual_actions']]
            require(available<=row['completed_episodes'],'duplicate episode rows')
            if available<row['completed_episodes']:break
            prefix.append(row)
        uncommitted_iterations=len(metric_snapshot)-len(prefix)
        metric_snapshot=prefix
    iterations, iteration_report = audit_iterations(p,args.allow_incomplete,metric_snapshot)
    iteration_report['uncommitted_live_iterations']=uncommitted_iterations
    interruption_report=audit_interruption(p,iterations,episode_snapshot)
    episodes, episode_report = audit_episodes(p,iterations,args.allow_incomplete,episode_snapshot,interruption_report)
    report = dict(schema='rc3-g2-independent-audit-v1', status='live_partial' if args.allow_incomplete else 'complete',
        protocol=protocol_report, teacher=audit_teacher_metadata(p), teacher_root_diagnostics=teacher_root_diagnostics(), bc=audit_bc(p), rule_selection=audit_rule_selection(p),
        ppo_iterations=iteration_report, completed_episodes=episode_report,interruption=interruption_report,
        saved_evaluations=audit_evaluations(p,args.allow_incomplete),
        reviewer_scope='Separate audit agent reviewed the existing teacher-module audit and recomputed saved '
                       'evidence; inherited project context, not a zero-context external review.')
    if args.checkpoints:
        report['checkpoints'] = audit_checkpoints(p, iterations, episodes, report)
    if args.replay:
        require(args.checkpoints, '--replay requires --checkpoints')
        report['actual_replay'] = audit_replay(p)
    report['all_requested_checks_passed'] = True
    report['audit_script_sha256']=sha(__file__)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output),status=report['status'],actions=iteration_report['actual_actions'],
        complete_episodes=episode_report['completed_episodes'],checks_passed=True),ensure_ascii=False))


if __name__ == '__main__':
    main()
