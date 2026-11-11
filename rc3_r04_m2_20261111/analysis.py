"""State-pooled and equal-parent metrics with paired parent bootstrap."""
import csv
import gzip
import numpy as np
from .common import HOME, put

COMPONENTS = ('raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit',
              'normalized_L', 'normalized_K', 'normalized_H', 'value')
AGGS = ('state_pooled', 'parent_equal')
PAIRS = (('CB128', 'B-cond0'), ('CB128', 'CG128'), ('CG128', 'B-global0'),
         ('CB128', 'C1_fixed128'), ('CG128', 'C1_fixed128'),
         ('B-cond0', 'B-global0'), ('CG128', 'B-cond0'))

def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def metrics(index, target, predictions):
    result, intervals = [], []
    for split in ('fit', 'validation'):
        for window in ('ALL', '10x10', '4x3', '6x6', '8x8'):
            ids = np.array([i for i, s in enumerate(index) if s['split'] == split and (window == 'ALL' or s['window'] == window)])
            if not len(ids): continue
            parents = sorted({index[i]['parent_id'] for i in ids})
            masks = [np.array([i for i in ids if index[i]['parent_id'] == p]) for p in parents]
            counts = np.array([len(g) for g in masks], dtype=np.float64)
            errors = {name: np.array([np.mean((p[g]-target[g])**2, axis=0) for g in masks]) for name, p in predictions.items()}
            for name, p in predictions.items():
                bias = np.array([np.mean(p[g]-target[g], axis=0) for g in masks])
                mae = np.array([np.mean(np.abs(p[g]-target[g]), axis=0) for g in masks])
                for agg in AGGS:
                    weights = counts if agg == 'state_pooled' else np.ones(len(masks))
                    for c, component in enumerate(COMPONENTS):
                        mse = float(np.average(errors[name][:, c], weights=weights))
                        b = float(np.average(bias[:, c], weights=weights))
                        pb = float(np.average(bias[:, c]**2, weights=weights))
                        result.append(dict(system=name, split=split, window=window, aggregation=agg, component=component,
                            n_states=len(ids), n_parents=len(parents), mse=mse,
                            mae=float(np.average(mae[:, c], weights=weights)), bias=b, bias_squared=b*b,
                            mean_parent_bias_squared=pb, within_parent_centered_error=max(0., mse-pb)))
            if split != 'validation': continue
            draws = np.random.Generator(np.random.PCG64(11622026)).integers(0, len(parents), size=(20000, len(parents)))
            for a, b in PAIRS:
                for agg in AGGS:
                    weights = counts if agg == 'state_pooled' else np.ones(len(masks))
                    for c, component in enumerate(COMPONENTS):
                        x, y = errors[a][:, c]*weights, errors[b][:, c]*weights
                        den = y[draws].sum(1)
                        if np.any(den <= 0): raise ValueError('Zero paired reference error')
                        ratios = x[draws].sum(1)/den
                        delta = (x-y)[draws].sum(1)/weights[draws].sum(1)
                        lo, hi = np.quantile(ratios, [.025, .975], method='linear')
                        dl, dh = np.quantile(delta, [.025, .975], method='linear')
                        intervals.append(dict(candidate=a, reference=b, window=window, aggregation=agg, component=component,
                            n_parents=len(parents), ratio=float(x.sum()/y.sum()), ci95_low=float(lo), ci95_high=float(hi),
                            mse_delta=float((x-y).sum()/weights.sum()), delta_ci95_low=float(dl), delta_ci95_high=float(dh)))
    return result, intervals

def decision(rows, intervals):
    def get(a, b, agg, comp='value', window='ALL'):
        return next(r for r in intervals if (r['candidate'], r['reference'], r['aggregation'], r['component'], r['window']) == (a, b, agg, comp, window))
    checks = []
    def add(name, passed, **details): checks.append(dict(check=name, passed=bool(passed), **details))
    for agg in AGGS:
        for ref in ('B-cond0', 'CG128'):
            r = get('CB128', ref, agg)
            add('CB_value_'+ref+'_'+agg, r['ratio'] <= .95 and r['ci95_high'] < 1, **r)
            for comp in ('normalized_L', 'normalized_K', 'normalized_H'):
                r = get('CB128', ref, agg, comp)
                add('CB_component_'+ref+'_'+agg+'_'+comp, r['ratio'] <= 1.10, **r)
            for window in ('10x10', '4x3', '6x6', '8x8'):
                r = get('CB128', ref, agg, window=window)
                add('CB_size_'+ref+'_'+agg+'_'+window, r['n_parents'] >= 3 and r['ratio'] <= 1, **r)
    cg_checks = []
    for agg in AGGS:
        r = get('CG128', 'B-global0', agg)
        cg_checks.append(dict(passed=r['ratio'] <= .95 and r['ci95_high'] < 1, **r))
    keep = all(r['passed'] for r in checks)
    return dict(schema='rc3-r04-m2-deletion-decision-v1', status='KEEP_CB_FOR_FUTURE_ONLINE_TEST' if keep else 'STOP_DELETE_CB_RESIDUAL',
        CB_checks=checks, CG_value_vs_own_anchor_checks=cg_checks,
        CB_residual_worth_keeping=keep, CG_has_incremental_value_evidence=all(r['passed'] for r in cg_checks),
        scope='exposed-holdout development mechanism screen; no new independent generalization confirmation',
        automatic_followup_authorized=False, M1_G3='FAIL_FROZEN', M1_step256='NOT_RUN_FROZEN',
        route_benefit='NOT_EVALUATED', joint_PPO='NOT_RUN_NOT_AUTHORIZED')

def save_predictions(index, target, predictions):
    put(HOME/'STATE_INDEX.json', dict(states=index, components=list(COMPONENTS)))
    with gzip.open(HOME/'M2_STATE_PREDICTIONS.csv.gz', 'wt', encoding='utf-8', newline='', compresslevel=1) as f:
        fields = ['system', 'state_id'] + ['target_'+c for c in COMPONENTS] + ['pred_'+c for c in COMPONENTS]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for name, prediction in predictions.items():
            for i in range(len(index)):
                w.writerow(dict(system=name, state_id=i, **{'target_'+c: float(target[i, j]) for j, c in enumerate(COMPONENTS)},
                    **{'pred_'+c: float(prediction[i, j]) for j, c in enumerate(COMPONENTS)}))
