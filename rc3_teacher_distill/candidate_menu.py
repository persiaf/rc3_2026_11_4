"""Deployment-only deterministic candidates. Original navigation nodes are 1..M."""
import operator
import numpy as np
from rc3_nav_value.r1.policy import State


def generate(item, B, prefix=(), k=8):
    if isinstance(k, (bool, np.bool_)):
        raise ValueError('k must be a positive integer')
    try:
        k = operator.index(k)
    except TypeError as exc:
        raise ValueError('k must be a positive integer') from exc
    if k < 1:
        raise ValueError('k must be positive')
    s = State(item, prefix)
    if s.complete:
        return []
    ids = (np.flatnonzero(s.legal_mask())+1).tolist()
    ref = next((int(a) for a in B if int(a) in ids), None)
    if ref is None:
        raise ValueError('Frozen reference cannot complete prefix; no fallback')
    limit = min(k, len(ids))
    chosen = [ref]
    C = item.C[s.remaining]
    gains = C.sum(0)
    ratio = gains/(item.D[s.current, 1:]+.75)
    degree = C[:, np.asarray(ids)-1].sum(1)
    if (degree == 0).any():
        raise ValueError('Uncoverable remaining task')
    scarcity = (C / degree[:, None]).sum(0)

    def fill(order, target):
        for a in order:
            if len(chosen) >= min(limit, target):
                break
            if a not in chosen:
                chosen.append(a)

    fill(sorted(ids, key=lambda a: (-gains[a-1], item.D[s.current, a], a)), 2)
    fill(sorted(ids, key=lambda a: (-ratio[a-1], a)), 4)
    fill(sorted(ids, key=lambda a: (-scarcity[a-1], -gains[a-1], a)), 6)
    xy = np.asarray([[c['x'], c['y']] for c in item.base.scene['candidates']])
    while len(chosen) < limit:
        def spread(a):
            d = xy[np.asarray(chosen)-1]-xy[a-1]
            return (-float((d*d).sum(1).min()), a)
        chosen.append(min((a for a in ids if a not in chosen), key=spread))
    return chosen


def rule_action(item, menu, prefix=()):
    s = State(item, prefix)
    if not menu or any(not s.legal_mask()[a-1] for a in menu):
        raise ValueError('Expected a nonempty legal candidate menu')
    gain = item.C[s.remaining].sum(0)
    return min(menu, key=lambda a: (-gain[a-1]/(item.D[s.current, a]+.75), a))
