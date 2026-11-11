"""Independent arithmetic path; reads only this task's exported state rows.

Does not call fitting, inference, production metrics, or the old large CSV audit.
"""
from collections import defaultdict
import csv
import gzip
import math
import numpy as np
from .common import HOME, read

def run(check):
    index = read(HOME/'STATE_INDEX.json')
    states, components = index['states'], index['components']
    systems = ('B-global0', 'B-cond0', 'CG0', 'CB0', 'C1_fixed128', 'CG128', 'CB128')
    n, c = len(states), len(components)
    predictions = {s: np.full((n, c), np.nan) for s in systems}
    targets = np.full((n, c), np.nan)
    seen = set()
    with gzip.open(HOME/'M2_STATE_PREDICTIONS.csv.gz', 'rt', encoding='utf-8', newline='') as f:
        for k, row in enumerate(csv.DictReader(f)):
            if k % 2048 == 0: check()
            s, i = row['system'], int(row['state_id'])
            assert s in predictions and 0 <= i < n and (s, i) not in seen
            seen.add((s, i))
            y = np.array([float(row['target_'+x]) for x in components])
            p = np.array([float(row['pred_'+x]) for x in components])
            assert np.isfinite(y).all() and np.isfinite(p).all()
            if np.isfinite(targets[i]).all(): assert np.array_equal(y, targets[i])
            targets[i], predictions[s][i] = y, p
    assert len(seen) == n*len(systems)
    parent_cells = {}
    checks, max_metric_error, max_interval_error = 0, 0., 0.
    def close(x, y):
        nonlocal checks
        checks += 1
        if not math.isclose(x, y, rel_tol=2e-10, abs_tol=2e-10):
            raise AssertionError(f'Independent numerical mismatch: {x} versus {y}')
    for s, p in predictions.items():
        assert np.allclose(p[:, 6], -(p[:, 3]+.75*p[:, 4]-p[:, 5]), atol=2e-5, rtol=2e-5)
        scale = np.array([a['task_count'] for a in states])[:, None]
        assert np.allclose(p[:, :3], p[:, 3:6]*scale, atol=1e-5, rtol=1e-6)
        assert (p[:, :2] >= 0).all()
        if s != 'C1_fixed128': assert (p[:, 3:5] >= 1e-6-1e-12).all()
    for a, b in (('CG0', 'B-global0'), ('CB0', 'B-cond0')):
        assert np.allclose(predictions[a], predictions[b], atol=5e-4, rtol=1e-6)
    assert np.allclose(targets[:, 6], -(targets[:, 3]+.75*targets[:, 4]-targets[:, 5]), atol=2e-5, rtol=2e-5)
    with (HOME/'M2_METRICS.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            key = row['system'], row['split'], row['window']
            if key not in parent_cells:
                s, split, window = key
                groups = defaultdict(list)
                for i, state in enumerate(states):
                    if state['split'] == split and (window == 'ALL' or state['window'] == window):
                        groups[state['parent_id']].append(i)
                cells = []
                for parent in sorted(groups):
                    ids = groups[parent]
                    d = predictions[s][ids]-targets[ids]
                    cells.append((len(ids), np.sum(d*d, axis=0), np.sum(d, axis=0), np.sum(np.abs(d), axis=0)))
                parent_cells[key] = cells
            cells = parent_cells[key]
            ci = components.index(row['component'])
            weights = [v[0] if row['aggregation'] == 'state_pooled' else 1 for v in cells]
            denom = math.fsum(weights)
            for column, offset in (('mse', 1), ('bias', 2), ('mae', 3)):
                actual = math.fsum(w*float(v[offset][ci])/v[0] for w, v in zip(weights, cells))/denom
                recorded = float(row[column]); close(actual, recorded)
                max_metric_error = max(max_metric_error, abs(actual-recorded))
            check()
    with (HOME/'PAIRED_PARENT_INTERVALS.csv').open(encoding='utf-8', newline='') as f:
        interval_rows = list(csv.DictReader(f))
    # A frequency-matrix implementation distinct from production direct resampling.
    cache = {}
    for row in interval_rows:
        window, agg, ci = row['window'], row['aggregation'], components.index(row['component'])
        a = parent_cells[(row['candidate'], 'validation', window)]
        b = parent_cells[(row['reference'], 'validation', window)]
        npar = len(a)
        if npar not in cache:
            draws = np.random.Generator(np.random.PCG64(11622026)).integers(npar, size=(20000, npar))
            frequencies = np.zeros((20000, npar), dtype=np.int64)
            for j in range(npar): frequencies[:, j] = np.count_nonzero(draws == j, axis=1)
            cache[npar] = frequencies
        f = cache[npar]
        count = np.array([x[0] for x in a], dtype=float)
        assert np.array_equal(count, [x[0] for x in b])
        x, y = np.array([x[1][ci] for x in a]), np.array([x[1][ci] for x in b])
        weight = count
        if agg == 'parent_equal': x, y, weight = x/count, y/count, np.ones(npar)
        # Sum by parent explicitly; avoids multi-thread BLAS overhead on tiny cells.
        sx = np.sum(f*x, axis=1); sy = np.sum(f*y, axis=1); sw = np.sum(f*weight, axis=1)
        lo, hi = np.percentile(sx/sy, [2.5, 97.5], method='linear')
        dl, dh = np.percentile((sx-sy)/sw, [2.5, 97.5], method='linear')
        for name, value in dict(ratio=x.sum()/y.sum(), ci95_low=lo, ci95_high=hi,
            mse_delta=(x-y).sum()/weight.sum(), delta_ci95_low=dl, delta_ci95_high=dh).items():
            close(float(row[name]), float(value)); max_interval_error = max(max_interval_error, abs(float(row[name])-value))
        check()
    decision = read(HOME/'M2_DECISION.json')
    cb = []
    for row in interval_rows:
        if row['candidate'] != 'CB128' or row['reference'] not in ('B-cond0', 'CG128'): continue
        if row['window'] == 'ALL' and row['component'] == 'value':
            cb.append(float(row['ratio']) <= .95 and float(row['ci95_high']) < 1)
        elif row['window'] == 'ALL' and row['component'] in ('normalized_L', 'normalized_K', 'normalized_H'):
            cb.append(float(row['ratio']) <= 1.10)
        elif row['window'] != 'ALL' and row['component'] == 'value':
            cb.append(int(row['n_parents']) >= 3 and float(row['ratio']) <= 1)
    assert len(cb) == 32 and all(cb) == decision['CB_residual_worth_keeping']
    log = list(csv.DictReader((HOME/'TRAINING_LOG.csv').open(encoding='utf-8', newline='')))
    schedule = read(HOME/'M2_PROTOCOL.json')['schedule_path']
    from .common import ROOT
    batches = read(ROOT/schedule)['batches']
    import json
    for s in ('CG', 'CB'):
        rows = [r for r in log if r['arm'] == s]
        assert [int(r['step']) for r in rows] == list(range(1, 129))
        assert all(json.loads(r['batch_ids']) == batches[int(r['step'])-1] for r in rows)
    old = read(HOME/'OLD_RESOURCE_SNAPSHOT.json')
    assert old['counts']['critic_optimizer_attempts'] == 256
    return dict(status='PASS', scope='new M2 output only; no old prediction CSV read',
        checks=checks, state_rows=len(seen), independent_gate_checks=len(cb),
        max_metric_absolute_difference=max_metric_error, max_interval_absolute_difference=float(max_interval_error),
        tolerance=dict(atol=2e-10, rtol=2e-10), bootstrap='independent frequency-matrix arithmetic',
        optimizer_attempts_verified=256, extra_optimizer_attempts=0)
