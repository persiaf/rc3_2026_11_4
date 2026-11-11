"""R04-M1 metadata locking and exact saved-prefix Critic input restoration.

Importing this module performs no reads or numerical work. The owner must start
its durable cumulative budget before ``metadata_registry``. Selection inspects
only schedule/physical-parent identities and completion receipts, never route
costs or value errors. Feature restoration uses the frozen historical loader;
it never chooses an action, calls env.step, creates a task or produces H labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from rc3_critic_diagnostics_20261111 import data as saved

ROOT = Path(__file__).resolve().parents[1]
SEED = 11622026
SIZES = ((10, 10), (4, 3), (6, 6), (8, 8))
OLD_SPLIT = 'rc3_critic_diagnostics_20261111/CALIBRATION_SPLIT.json'
INDEX = 'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json'


def _nothing(kind, amount=1, **context):
    pass


def key(*parts):
    return hashlib.sha256('|'.join(map(str, (SEED,) + parts)).encode('utf-8')).hexdigest()


def metadata_registry(charge=None):
    """Lock 33 old fit parents and the exact proposed 16-parent hash reserve.

    Fit: 64 hash-ordered records in parent round-robin order (each parent gets
    one before any parent gets a second). Holdout: one hash-ordered record per
    fixed parent x size cell, at most 64. No missing cell/parent is replaced.
    ON/OFF are source trajectory roles, not newly evaluated Actor conditions.
    """
    charge = charge or _nothing
    metadata_sources = []

    def read(name):
        raw, identity = saved._bytes(name, charge)
        metadata_sources.append(identity)
        return json.loads(raw)

    previous = read(OLD_SPLIT)
    schedule = read(saved.SCHEDULE)
    references = read(saved.REFERENCES)
    index = read(INDEX)
    event_raw, event_identity = saved._bytes(saved.EVENTS, charge)
    metadata_sources.append(event_identity)
    completed = {}
    for line in event_raw.splitlines():
        if not line:
            continue
        event = json.loads(line)
        if event.get('event') == 'request_done':
            rid = event['request_id']
            if rid in completed:
                raise ValueError('Duplicate historical completed request: ' + rid)
            # Deliberately project identities only; no cost/error field is read.
            completed[rid] = dict(path=event['path'], sha256=event['artifact_sha256'])
    if schedule['seed'] != SEED or previous['seed'] != SEED:
        raise ValueError('Historical seed identity differs')
    fit = list(previous['fit_physical_parents'])
    old_holdout = list(previous['holdout_physical_parents'])
    old50 = set(fit) | set(old_holdout)
    if len(fit) != 33 or len(old_holdout) != 17 or len(old50) != 50:
        raise ValueError('Previous 33/17 physical-parent split differs')
    candidates = defaultdict(list)
    all_training_parents = set()
    missing_completion_ids = []
    for entry in schedule['training']:
        opportunity = entry['opportunity']
        if not 1 <= opportunity <= 128:
            continue
        phase = 'early_001_064' if opportunity <= 64 else 'late_065_128'
        for slot, spec in enumerate(entry['specs']):
            size = (spec['width'], spec['height'])
            if size not in SIZES:
                raise ValueError('Unexpected historical training window size')
            physical = references['geometry'][str(spec['map_seed'])]['physical_key']
            all_training_parents.add(physical)
            for method in ('ON', 'OFF'):
                request_id = f'train:{method}:{opportunity}:{slot}'
                if request_id not in completed:
                    missing_completion_ids.append(request_id)
                    continue
                source = completed[request_id]
                candidates[physical].append(dict(
                    trajectory_id=request_id, request_id=request_id,
                    request_path=source['path'], request_sha256=source['sha256'],
                    pair_id=f'pair_{opportunity:03}_{slot}', method=method, condition=method,
                    physical_key=physical, parent_id=physical, phase=phase,
                    width=size[0], height=size[1], opportunity=opportunity, slot=slot,
                    spec=dict(spec), selection_hash=key('R04-M1-trajectory', request_id)))
    if len(all_training_parents) != 96 or not old50 <= all_training_parents:
        raise ValueError('Expected 96 R03 training parents containing prior 50')
    outside = sorted(all_training_parents - old50,
                     key=lambda p: key('R04-validation', p))
    if len(outside) != 46:
        raise ValueError('Expected exactly 46 untouched offline calibration parents')
    holdout = outside[:16]
    if set(fit) & set(holdout):
        raise ValueError('Physical parent leakage')

    selected, missing_cells = [], []
    fit_order = sorted(fit, key=lambda p: key('R04-M1-fit-parent', p))
    for parent in fit_order:
        candidates[parent].sort(key=lambda r: r['selection_hash'])
    # One pass through every fixed parent before taking a second record.
    offset = 0
    while len(selected) < 64:
        added = 0
        for parent in fit_order:
            if offset < len(candidates[parent]):
                selected.append(dict(candidates[parent][offset], split='fit'))
                added += 1
                if len(selected) == 64:
                    break
        if not added:
            break
        offset += 1
    for parent in holdout:
        for width, height in SIZES:
            matches = [r for r in candidates[parent]
                       if (r['width'], r['height']) == (width, height)]
            if not matches:
                missing_cells.append(dict(parent_id=parent, width=width, height=height))
                continue
            selected.append(dict(min(matches, key=lambda r: r['selection_hash']),
                                 split='validation'))
    if len(selected) > 128 or len({r['trajectory_id'] for r in selected}) != len(selected):
        raise ValueError('Selected-trajectory cap or uniqueness violation')
    actual_fit = {r['parent_id'] for r in selected if r['split'] == 'fit'}
    actual_holdout = {r['parent_id'] for r in selected if r['split'] == 'validation'}
    coverage = []
    for split_name in ('fit', 'validation'):
        for width, height in SIZES:
            rows = [r for r in selected if r['split'] == split_name
                    and (r['width'], r['height']) == (width, height)]
            coverage.append(dict(split=split_name, width=width, height=height,
                                 trajectories=len(rows), parents=len({r['parent_id'] for r in rows}),
                                 phases=dict(Counter(r['phase'] for r in rows)),
                                 roles=dict(Counter(r['method'] for r in rows))))
    insufficient = []
    if actual_fit != set(fit):
        insufficient.append('A fixed fit physical parent has no selected complete trajectory')
    if actual_holdout != set(holdout):
        insufficient.append('A fixed reserve physical parent has no selected complete trajectory')
    for item in coverage:
        if item['split'] == 'validation' and item['parents'] < 3:
            insufficient.append(f"Validation {item['width']}x{item['height']} has fewer than 3 physical parents")
    if missing_completion_ids:
        # A missing historical record is retained as a failure, never silently
        # converted to an alternate completed trajectory.
        insufficient.append('Historical registered training completion receipt missing')

    dependencies = {r['path']: r['sha256'] for r in metadata_sources}
    for row in selected:
        dependencies[row['request_path']] = row['request_sha256']
        spec = row['spec']
        parent_key = f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        entry = index.get(parent_key, references['bank'].get(parent_key))
        if entry is None:
            raise ValueError('Selected saved input missing; no generation fallback')
        row['saved_input_files'] = dict(entry['files'])
        for value in entry['files'].values():
            prior = dependencies.setdefault(value['path'], value['sha256'])
            if prior != value['sha256']:
                raise ValueError('Contradictory source hashes for a saved dependency')
    return dict(
        schema='rc3-r04-m1-data-index-v1', seed=SEED,
        status='INSUFFICIENT' if insufficient else 'GO_METADATA',
        selection_uses_costs_or_value_errors=False,
        selection_rule=('fit: hash-ordered fixed33 parents, per-parent records ordered by '
                        'SHA256(11622026|R04-M1-trajectory|request_id), round-robin first64; '
                        'holdout: first16 of46 by SHA256(11622026|R04-validation|physical_key), '
                        'one lowest-hash record per fixed parent x size; missing cells never replaced'),
        fit_parent_order_rule='SHA256(11622026|R04-M1-fit-parent|physical_key)',
        fit_physical_parents=fit, holdout_physical_parents=holdout,
        previous_holdout_physical_parents=old_holdout,
        outside_previous50_physical_parents_in_hash_order=outside,
        all_R03_training_physical_parents=sorted(all_training_parents),
        holdout_scope='new offline calibration holdout; historically SL/DRL-seen R03 training parents',
        trajectories=selected, count=len(selected), maximum_trajectories=128,
        coverage=coverage, missing_parent_size_cells=missing_cells,
        missing_completion_request_ids=missing_completion_ids, insufficient_reasons=insufficient,
        source_metadata=metadata_sources, artifact_sha256=dependencies,
        feature_source='268 tensors were not persisted; exact saved-prefix reconstruction using frozen SL256; no new actions/task/navigation/H',
        label_source='existing complete saved L/K/H/RTG labels only; no newly evaluated terminal H')


def batch_schedule(registry):
    """256 identical 8-distinct-parent batches for C0/C1, fixed before fitting."""
    groups = {p: [] for p in registry['fit_physical_parents']}
    for row in registry['trajectories']:
        if row['split'] == 'fit':
            groups[row['parent_id']].append(row['trajectory_id'])
    if any(not ids for ids in groups.values()) or len(groups) != 33:
        raise ValueError('Cannot schedule missing fixed fitting parents')
    parents = sorted(groups, key=lambda p: key('R04-M1-batch-parent', p))
    for parent in parents:
        groups[parent].sort(key=lambda rid: key('R04-M1-batch-trajectory', rid))
    positions = {p: 0 for p in parents}
    batches = []
    for step in range(256):
        rows = []
        for j in range(8):
            parent = parents[(step * 8 + j) % len(parents)]
            rows.append(groups[parent][positions[parent] % len(groups[parent])])
            positions[parent] += 1
        batches.append(rows)
    return dict(schema='rc3-r04-m1-common-batch-schedule-v1', seed=SEED,
                steps=256, batch_size=8, distinct_parents_each_batch=8,
                parent_order=parents, batches=batches,
                rule='hash-ordered fit physical parents, circular groups of8, per-parent hash-ordered trajectory cycle; identical C0/C1')


def load_episode(row, actor, inputs, charge):
    """Restore only selected historical state prefixes; return Critic-only data.

    Caller supplies exact frozen SL256 and saved-input loader plus forbidden
    hooks on env.step/terminal/planning/teacher paths. The inherited routine
    verifies task/candidate IDs, saved logits, mask, exact prefixes and labels.
    No Actor correction checkpoint is read and no selected action is recomputed.
    """
    episode = saved.reconstruct_episode(row, actor, inputs, charge)
    features = episode['batch']['v'].detach().clone()
    if features.ndim != 2 or features.shape[1] != 268:
        raise ValueError('Saved-prefix Critic state dimension changed')
    return dict(v=features, targets=episode['targets'], task_count=episode['task_count'],
                parent_id=row['parent_id'], physical_key=row['physical_key'],
                trajectory_id=row['trajectory_id'], request_id=row['request_id'],
                split=row['split'], width=row['width'], height=row['height'],
                window_key=f"{row['width']}x{row['height']}",
                phase=row['phase'], stage=row['phase'], method=row['method'],
                opportunity=row['opportunity'], slot=row['slot'],
                cohort='saved_online_R03_training_parent',
                source=episode['source'], input_evidence=episode['input_evidence'],
                feature_source='exact_saved_prefix_frozen_SL256_reconstruction_no_env_step',
                original_tensor_persisted=False, saved_actions_reselected=False,
                original_V_old=episode['original_V_old'])
