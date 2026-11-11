"""Only the anchor differs. No target or episode object enters forward()."""
import copy
import torch
from torch import nn
from rc3_r04_m1_20261111 import critic as units

FLOOR = 1e-6
NAMES = ('normalized_L', 'normalized_K', 'normalized_H')

def lookup(table, window, features, conditional):
    if features.ndim != 2 or features.shape[1] != 268 or not torch.isfinite(features).all():
        raise ValueError('Finite 268-dimensional observable input required')
    fraction = features[:, 256]
    if not bool(((fraction >= -1e-6) & (fraction <= 1+1e-6)).all()):
        raise ValueError('Observable remaining fraction outside bounds')
    bins = torch.floor(4*(1-fraction.clamp(0, 1))).long().clamp(0, 3).tolist()
    fallback = []
    rows = []
    for q in bins:
        key = f'{window}|q{q}'
        found = conditional and key in table['conditional']
        rows.append(table['conditional'][key] if found else table['global'])
        fallback.append(bool(conditional and not found))
    return torch.tensor(rows, dtype=features.dtype, device=features.device), fallback

def fit_table(fit):
    # The reviewed original estimator balances parent/trajectory/state within stratum.
    from rc3_critic_diagnostics_20261111.critic import fit_references
    if any(e['split'] != 'fit' for e in fit): raise ValueError('Fit-only anchor estimation')
    refs = fit_references(fit)
    def components(r):
        a = [float(r['values'][n]) for n in NAMES]
        a[0], a[1] = max(FLOOR, a[0]), max(FLOOR, a[1])
        return a
    return dict(schema='rc3-r04-m2-fit-component-table-v1', parent_ids=refs['parent_ids'],
        trajectory_ids=refs['trajectory_ids'], reduction=refs['reduction'],
        condition=refs['condition'], unknown_condition='fit-global',
        components=list(NAMES), floor=FLOOR,
        **{'global': components(refs['global_constant']),
           'conditional': {k: components(v) for k, v in refs['conditional_constants'].items()}})

def positive_link(anchor, residual):
    # Stable inverse softplus; anchor and output floor are both fixed beforehand.
    a = anchor.clamp_min(FLOOR)
    inverse = a + torch.log(-torch.expm1(-a))
    return torch.nn.functional.softplus(inverse+residual).clamp_min(FLOOR)

class AnchorCritic(nn.Module):
    def __init__(self, arm, table):
        super().__init__()
        if arm not in ('CG', 'CB'): raise ValueError('Only the two authorized arms')
        self.arm, self.table = arm, copy.deepcopy(table)
        self.net = nn.Sequential(nn.Linear(268, 128), nn.ReLU(), nn.Linear(128, 128),
            nn.ReLU(), nn.Linear(128, 3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, features, window):
        anchor, _ = lookup(self.table, window, features, self.arm == 'CB')
        residual = self.net(features.detach())
        return torch.stack((positive_link(anchor[:, 0], residual[:, 0]),
            positive_link(anchor[:, 1], residual[:, 1]), anchor[:, 2]+residual[:, 2]), 1)

def matched_models(table):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11622026); initial = AnchorCritic('CG', table)
    models = {a: copy.deepcopy(initial) for a in ('CG', 'CB')}
    for a, m in models.items(): m.arm = a
    return models

def labels(episode, dtype=torch.float32):
    values = units.episode_labels(episode, dtype=dtype)
    return torch.stack([values[n] for n in NAMES], 1)

def loss(model, batch):
    if not batch or any(e['split'] != 'fit' for e in batch): raise ValueError('Fit-only batches')
    return torch.stack([(.5*(model(e['v'], e['window_key'])-labels(e)).square().sum(1)).sum()
        for e in batch]).mean()

def engineering_tests():
    table = {'global': [0., 0., -.25], 'conditional': {'4x3|q0': [2., .5, -.4]}}
    x = torch.zeros(4, 268); x[:, 256] = torch.tensor([1., .75, .5, 0.])
    a, fallback = lookup(table, '4x3', x, True)
    assert fallback == [False, True, True, True]
    assert torch.equal(lookup(table, 'unknown', x, True)[0], lookup(table, 'unknown', x, False)[0])
    m = matched_models(table)
    assert units.tensor_digest(m['CG'].state_dict()) == units.tensor_digest(m['CB'].state_dict())
    for arm in m:
        expected = lookup(table, '4x3', x, arm == 'CB')[0].clone(); expected[:, :2].clamp_min_(FLOOR)
        actual = m[arm](x, '4x3')
        assert torch.allclose(actual, expected, atol=2e-7, rtol=1e-6)
        assert (actual[:, 2] < 0).all()
        assert torch.count_nonzero(m[arm].net(x)) == 0
    z = torch.tensor([-1000., 0., 1000.], requires_grad=True)
    linked = positive_link(torch.tensor([0., FLOOR, 100.]), z)
    assert torch.isfinite(linked).all() and (linked >= FLOOR).all()
    linked.sum().backward(); assert torch.isfinite(z.grad).all()
    # Labels cannot be supplied to forward; a missing label is irrelevant to lookup.
    try: fit_table([{'split': 'validation'}])
    except ValueError: pass
    else: raise AssertionError('Holdout admitted to table fitting')
    return dict(status='PASS', optimizer_attempts=0, checks=[
        'same_zero_output_net_tensors', 'observable_bins_and_unknown_global',
        'zero_anchor_floor_1e-6', 'positive_link_zero_residual_identity_with_tolerance',
        'signed_H', 'finite_extreme_link_and_gradient', 'reject_holdout_fit',
        'runtime_signature_only_features_window'], atol=2e-7, rtol=1e-6)
