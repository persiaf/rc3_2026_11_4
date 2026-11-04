"""Finite-lookahead actor supervision under G2 semantics, never PPO rollout.

Unknown legal actions retain an evidence mask of False and a harmless numeric
placeholder. Their logits must not enter the supervised normalization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

TEACHER_SCHEMA = 'rc3-g2-finite-return-teacher-v1'


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def semantic_action(action):
    """A semantic action, as opposed to its local position in an old menu."""
    return {key: action.get(key) for key in ('kind', 'start', 'end', 'forbidden_node')}


def queried_actions(observation, branch_limit=8):
    if type(branch_limit) is not int or branch_limit < 1:
        raise ValueError('branch_limit must be a positive integer including STOP')
    actions, mask = observation['actions'], observation['mask']
    if len(actions) != len(mask):
        raise ValueError('menu and legality mask disagree')
    stop = [i for i, (a, valid) in enumerate(zip(actions, mask))
            if valid and a['kind'] == 'stop']
    if len(stop) != 1:
        raise ValueError('an active teacher state requires one legal STOP')
    search = [i for i, (a, valid) in enumerate(zip(actions, mask))
              if valid and a['kind'] != 'stop']
    return search[:branch_limit-1] + stop


def make_labels(legality_mask, queried_returns, stop_id, tie_tolerance=.001):
    """Build evidence-only labels from highest *remaining* observed return."""
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError('tie tolerance must be finite and nonnegative')
    legal = np.asarray(legality_mask, dtype=bool)
    if legal.ndim != 1 or not len(legal):
        raise ValueError('nonempty one-dimensional legality mask required')
    q = np.zeros(len(legal), dtype=np.float64)
    evidence = np.zeros(len(legal), dtype=bool)
    for action_id, value in queried_returns.items():
        if type(action_id) is not int or not 0 <= action_id < len(legal) or not legal[action_id]:
            raise ValueError('teacher queried an illegal action')
        if not np.isfinite(value):
            raise ValueError('teacher return must be finite')
        evidence[action_id], q[action_id] = True, float(value)
    if not 0 <= stop_id < len(legal) or not evidence[stop_id]:
        raise ValueError('STOP must have observed evidence')
    best = float(q[evidence].max())
    ties = evidence & (q >= best-tie_tolerance)
    advantages = np.where(evidence, q-q[stop_id], 0.)
    return dict(legality_mask=legal, evidence_mask=evidence, q_values=q,
                ties=ties, advantages_vs_stop=advantages, stop_id=int(stop_id),
                best_observed_return=best, stop_return=float(q[stop_id]),
                tie_tolerance=float(tie_tolerance))


def assign_scene_route_weights(samples):
    """Equal scene mass, then equal current-route mass, then duplicate count.

    Different budget/memory states of one route keep distinct labels; only their
    combined training weight is controlled. A route is not mistaken for a full
    Markov state and the samples are never silently merged.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        groups[str(sample['scene_id'])][tuple(sample['current_route'])].append(sample)
    if not groups:
        return samples
    for routes in groups.values():
        for repeated in routes.values():
            weight = 1./(len(groups)*len(routes)*len(repeated))
            for sample in repeated:
                sample['weight'] = weight
                sample['route_repeat_count'] = len(repeated)
                sample['scene_route_count'] = len(routes)
    return samples


def finite_tree(env, policy_observer, *, depth=2, branch_limit=8,
                tie_tolerance=.001, scene_id='scene', log_query=None, log_menu=None):
    """Enumerate a bounded tree and make actor labels at every expanded node.

    Every branch uses the environment's own fork/step/termination. The best
    terminal total reward is converted to remaining return by subtracting the
    parent's accumulated reward, including depth-cut final charges/corrections.
    """
    if type(depth) is not int or not 1 <= depth <= 3:
        raise ValueError('teacher depth must be an integer in [1,3]')
    if branch_limit > 8:
        raise ValueError('first G2 teacher permits at most 8 branches including STOP')
    if env.done:
        return [], dict(queries=0, expanded_states=0, initial_terminal=True)
    samples, query_count, node_count = [], 0, 0
    physical = Counter()
    failures = Counter()
    before = env.counters['menu_builds']
    initial_obs = env.observe()
    initial_builds = env.counters['menu_builds']-before
    physical['menu_builds'] += initial_builds
    physical['menu_work_units'] += initial_obs['pending_menu_work'] if initial_builds else 0

    def visit(parent, remaining, path):
        nonlocal query_count, node_count
        observation = parent.observe()
        chosen = queried_actions(observation, branch_limit)
        policy_observation = policy_observer(parent)
        if list(map(bool, policy_observation['mask'])) != list(map(bool, observation['mask'])):
            raise ValueError('policy observation changed the teacher legality mask')
        if 'actions' in policy_observation and policy_observation['actions'] != observation['actions']:
            raise ValueError('policy observation reordered the teacher actions')
        node_id = node_count
        node_count += 1
        if log_menu:
            log_menu(dict(node_id=node_id, path=path, remaining_depth=remaining,
                          queried_action_ids=chosen, observation=observation))
        q_returns, q_paths = {}, {}
        prefix_reward = float(parent.total_reward)
        for action_id in chosen:
            child = parent.fork()
            old_calls = child.counters['certification_calls']
            old_builds = child.counters['menu_builds']
            next_obs, step_reward, terminated, _, info = child.step(action_id)
            physical['selected_edges'] += 1
            new_builds = child.counters['menu_builds']-old_builds
            physical['menu_builds'] += new_builds
            physical['menu_work_units'] += next_obs['pending_menu_work'] if new_builds else 0
            physical['proposal_calls'] += int(info.get('proposal') is not None)
            physical['proposal_work_units'] += int(info.get('proposal_work_units', 0))
            if info.get('failure_reason'):
                failures[info['failure_reason']] += 1
            query_id = query_count
            query_count += 1
            edge_path = path+[action_id]
            terminal_event_reward = 0.
            leaf = terminated or remaining == 1
            if leaf and not child.done:
                terminal_event_reward, _ = child.terminate('teacher_depth')
            # Count only this edge and its endpoint finalization; the recursive
            # branches below separately count their own actual calls.
            physical['certification_calls'] += child.counters['certification_calls']-old_calls
            if leaf:
                terminal_reward, best_path = float(child.total_reward), edge_path
            else:
                terminal_reward, best_path = visit(child, remaining-1, edge_path)
            q_returns[action_id] = terminal_reward-prefix_reward
            q_paths[action_id] = best_path
            if log_query:
                compact = {k: v for k, v in info.items() if k != 'sampled_observation'}
                log_query(dict(query_id=query_id, node_id=node_id, path=edge_path,
                    action_id=action_id, action=observation['actions'][action_id],
                    proposal_key=observation.get('proposal_keys', [None]*len(observation['mask']))[action_id],
                    remaining_depth=remaining, prefix_reward=prefix_reward,
                    step_reward=float(step_reward), terminal_event_reward=float(terminal_event_reward),
                    best_observed_terminal_reward=terminal_reward, remaining_return=q_returns[action_id],
                    best_descendant_path=best_path, leaf=leaf, current=list(child.current),
                    best=list(child.best), current_cost=child.current_cost, best_cost=child.best_cost,
                    work_used=child.work_used, total_reward=child.total_reward,
                    finalization=child.finalization, info=compact))
        stop_id = next(i for i in chosen if observation['actions'][i]['kind'] == 'stop')
        labels = make_labels(observation['mask'], q_returns, stop_id, tie_tolerance)
        signature = _digest(dict(problem_id=parent.adapter.problem_id,
                                 observation=observation, teacher_depth=remaining))
        sample = dict(schema=TEACHER_SCHEMA, scene_id=str(scene_id),
            problem_id=parent.adapter.problem_id, state_signature=signature,
            teacher_depth=remaining, path=path, current_route=list(parent.current),
            policy_observation=policy_observation, observation=policy_observation,
            environment_observation=observation,
            ordered_actions=[semantic_action(a) for a in observation['actions']],
            queried_action_ids=chosen, best_descendant_paths=q_paths,
            prefix_reward=prefix_reward, environment_schema=observation['schema'],
            config=asdict(parent.config), weight=0., **labels)
        samples.append(sample)
        best_action = min(q_returns, key=lambda i: (-q_returns[i], i))
        return prefix_reward+q_returns[best_action], q_paths[best_action]

    root_best, root_path = visit(env, depth, [])
    summary = dict(queries=query_count, expanded_states=node_count, initial_terminal=False,
        physical_oracle_work=dict(physical), failure_reasons=dict(failures),
        root_best_total_reward=root_best, root_best_path=root_path,
        teacher_depth=depth, branch_limit=branch_limit, tie_tolerance=tie_tolerance,
        terminal_semantics='Natural STOP/budget/horizon or charged external teacher_depth termination',
        scope='Finite queried-subset actor supervision; not horizon-16 optimal value or on-policy PPO data')
    return samples, summary


def _load_scene(case):
    if 'scene' in case:
        return case['scene']
    from rc3_fast.scenes import make_complex_scene
    return make_complex_scene(int(case.get('nx', 10)), int(case.get('ny', 10)),
                              int(case['seed']), case.get('mode', 'regular'))


def collect_teacher(cases, outdir, depth=2, branch_limit=8, *, config=None,
                    tie_tolerance=.001, adapter_factory=None,
                    environment_factory=None, policy_observer=None,
                    shard_index=0, shard_count=1):
    """Collect a fresh, version-bound G2 teacher, with complete query accounting.

    This trusted pickle is created locally. Never load arbitrary external pickle
    files. Cases are supplied by the preregistered caller; this module draws no
    new seeds and never silently replaces a failed case.
    """
    from rc3_rl_research.types import SearchConfig
    if adapter_factory is None:
        from .adapter_g2 import ResearchAdapterG2
        adapter_factory = ResearchAdapterG2.from_scene
    if environment_factory is None:
        from .env_g2 import SearchEnvG2
        environment_factory = SearchEnvG2
    if policy_observer is None:
        from .graph_observation import build_policy_observation
        policy_observer = build_policy_observation
    config = SearchConfig() if config is None else config
    cases = list(cases)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not 0 <= shard_index < shard_count:
        raise ValueError('invalid shard indices')
    if (outdir/'teacher.pkl').exists():
        raise ValueError('teacher output already exists; preserve its original evidence')
    source_files = ('env_g2.py', 'action_memory.py', 'adapter_g2.py',
                    'graph_observation.py', 'teacher_dataset.py')
    base = Path(__file__).parent
    hashes = {name: hashlib.sha256((base/name).read_bytes()).hexdigest()
              for name in source_files if (base/name).exists()}
    specification = dict(schema=TEACHER_SCHEMA, cases=cases, config=asdict(config),
        depth=depth, branch_limit=branch_limit, tie_tolerance=tie_tolerance,
        source_sha256=hashes, branch_rule='first 7 or fewer valid non-STOP semantic menu actions and STOP',
        weighting='equal scene mass, equal current-route mass within each scene, divided among repeated-route states',
        unknown_action_semantics='zero numeric placeholder, evidence_mask=False; not a negative sample',
        training_use='actor supervision only; no finite-depth critic targets or PPO reuse')
    spec_path = outdir/'collection_spec.json'
    spec_bytes = json.dumps(_jsonable(specification), ensure_ascii=False,
                           indent=2, allow_nan=False).encode('utf-8')
    # A single exclusive creation also allows concurrently launched disjoint
    # case shards, all of which must agree with the same full specification.
    try:
        with spec_path.open('xb') as handle:
            handle.write(spec_bytes)
    except FileExistsError:
        # The first process may still be completing the short metadata write.
        for retry in range(20):
            try:
                previous = json.loads(spec_path.read_text(encoding='utf-8'))
                break
            except json.JSONDecodeError:
                if retry == 19:
                    raise
                time.sleep(.05)
        if previous != _jsonable(specification):
            raise ValueError('teacher collection specification/source changed')
    specification_digest = _digest(specification)
    all_samples, case_summaries = [], []
    started = time.perf_counter()
    for index, case in enumerate(cases):
        if index % shard_count != shard_index:
            continue
        case_id = str(case.get('case_id', case.get('scene_id', index)))+':'+str(case['seed'])
        target = outdir/f'case_{index:04d}'
        if target.exists():
            raise ValueError(f'teacher case already exists: {target}; inspect before resuming')
        target.mkdir()
        tick = time.perf_counter()
        try:
            adapter = adapter_factory(_load_scene(case))
            env = environment_factory(adapter, config, int(case['seed']), record_transitions=False)
            _write_json(target/'input.json', adapter.payload())
            before_cache = dict(getattr(adapter, 'cache_stats', {}))
            with gzip.open(target/'queries.jsonl.gz', 'wt', encoding='utf-8') as queries, \
                    gzip.open(target/'menus.jsonl.gz', 'wt', encoding='utf-8') as menus:
                def append(handle, row):
                    handle.write(json.dumps(_jsonable(row), allow_nan=False)+'\n')
                samples, summary = finite_tree(env, policy_observer, depth=depth,
                    branch_limit=branch_limit, tie_tolerance=tie_tolerance,
                    scene_id=case_id, log_query=lambda row: append(queries, row),
                    log_menu=lambda row: append(menus, row))
            summary.update(case=case, status='complete', elapsed_s=time.perf_counter()-tick,
                           specification_sha256=specification_digest,
                           cache_before=before_cache, cache_after=dict(getattr(adapter, 'cache_stats', {})))
            with (target/'samples.pkl').open('wb') as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            all_samples.extend(samples)
        except Exception as exc:
            summary = dict(case=case, status='failed', error=f'{type(exc).__name__}: {exc}',
                           elapsed_s=time.perf_counter()-tick)
        case_summaries.append(summary)
        _write_json(target/'summary.json', summary)
        print(json.dumps(dict(teacher_case=index, status=summary['status'],
                             samples=summary.get('expanded_states'), error=summary.get('error'))), flush=True)
    if shard_count > 1:
        summary = dict(shard_index=shard_index, shard_count=shard_count,
            completed_cases=sum(r['status'] == 'complete' for r in case_summaries),
            failed_cases=[r for r in case_summaries if r['status'] != 'complete'],
            elapsed_s=time.perf_counter()-started, specification_sha256=specification_digest,
            scope='Disjoint case shard; use --merge after all shards complete')
        _write_json(outdir/f'shard_{shard_index:02d}.json', summary)
        if summary['failed_cases']:
            raise RuntimeError('teacher shard has failed preregistered cases')
        return summary
    return merge_teacher(outdir, elapsed_s=time.perf_counter()-started)


def merge_teacher(outdir, *, elapsed_s=None):
    """Merge completed disjoint case shards without performing new queries."""
    outdir = Path(outdir)
    if (outdir/'teacher.pkl').exists():
        raise ValueError('merged teacher already exists')
    specification = json.loads((outdir/'collection_spec.json').read_text(encoding='utf-8'))
    cases = specification['cases']
    all_samples, case_summaries = [], []
    for index, _ in enumerate(cases):
        target = outdir/f'case_{index:04d}'
        summary = json.loads((target/'summary.json').read_text(encoding='utf-8'))
        case_summaries.append(summary)
        if summary['status'] == 'complete':
            if summary['specification_sha256'] != _digest(specification):
                raise ValueError('case belongs to a different teacher specification')
            with (target/'samples.pkl').open('rb') as f:
                all_samples.extend(pickle.load(f))
    assign_scene_route_weights(all_samples)
    complete = [r for r in case_summaries if r['status'] == 'complete']
    metadata = dict(schema=TEACHER_SCHEMA, specification_sha256=_digest(specification),
        completed_cases=len(complete), planned_cases=len(cases), failed_cases=[r for r in case_summaries if r['status'] != 'complete'],
        samples=len(all_samples), unique_scenes=len({s['scene_id'] for s in all_samples}),
        distinct_routes=len({(s['scene_id'], tuple(s['current_route'])) for s in all_samples}),
        physical_oracle_work=dict(sum((Counter(r['physical_oracle_work']) for r in complete), Counter())),
        query_count=sum(r['queries'] for r in complete),
        elapsed_s=elapsed_s, summed_case_elapsed_s=sum(r['elapsed_s'] for r in case_summaries),
        weight_sum=sum(s['weight'] for s in all_samples),
        scope=specification['training_use'])
    if metadata['failed_cases']:
        _write_json(outdir/'summary.json', metadata)
        raise RuntimeError('teacher contains failed preregistered cases; no trainable artifact was written')
    with (outdir/'teacher.pkl').open('wb') as f:
        pickle.dump(dict(schema=TEACHER_SCHEMA, samples=all_samples, metadata=metadata), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    metadata['teacher_sha256'] = hashlib.sha256((outdir/'teacher.pkl').read_bytes()).hexdigest()
    _write_json(outdir/'summary.json', metadata)
    return metadata


def load_teacher(path):
    """Read only a locally generated, trusted teacher artifact."""
    with Path(path).open('rb') as f:
        payload = pickle.load(f)
    if payload.get('schema') != TEACHER_SCHEMA:
        raise ValueError('incompatible teacher schema')
    if payload.get('metadata', {}).get('failed_cases'):
        raise ValueError('failed teacher cases cannot enter supervised training')
    for sample in payload['samples']:
        legal = np.asarray(sample['legality_mask'], dtype=bool)
        evidence = np.asarray(sample['evidence_mask'], dtype=bool)
        ties = np.asarray(sample['ties'], dtype=bool)
        q = np.asarray(sample['q_values'], dtype=float)
        if not (legal.ndim == 1 and legal.shape == evidence.shape == ties.shape == q.shape):
            raise ValueError('teacher mask/label dimensions disagree')
        if not np.isfinite(q).all() or np.any(evidence & ~legal) or np.any(ties & ~evidence) or not ties.any():
            raise ValueError('invalid teacher evidence or tie labels')
        if not np.array_equal(legal, np.asarray(sample['observation']['mask'], dtype=bool)):
            raise ValueError('saved actor input and teacher legality differ')
        if not np.isfinite(sample['weight']) or sample['weight'] <= 0:
            raise ValueError('teacher sample weight must be finite and positive')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--split', help='optional dictionary key of a case list')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--branch-limit', type=int, default=8)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    if args.merge:
        print(json.dumps(merge_teacher(args.output)))
        return
    if args.cases is None:
        parser.error('--cases is required except with --merge')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    if args.split:
        cases = cases[args.split]
    print(json.dumps(collect_teacher(cases, args.output, args.depth, args.branch_limit,
                                    shard_index=args.shard_index, shard_count=args.shard_count)))


if __name__ == '__main__':
    main()
