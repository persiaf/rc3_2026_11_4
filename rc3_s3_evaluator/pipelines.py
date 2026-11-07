"""Seven frozen pipelines. No optimizer, sampling, task embedding or route cache."""
import contextlib
import time
from unittest.mock import patch
import numpy as np
import torch

from .common import METHODS, H_METHODS, MODEL_ROLES, require, finite_number, digest, project_file, identity
from .inputs import prepare_view
from rc3_global_local import terminal
from rc3_global_local.environment import Environment
from rc3_global_local.policy import tensor_hash
from rc3_global_local_impl_review import endpoint_baselines as native
from rc3_nav_value.r1.oracle import _empty_internal, _instrument


def measured_callbacks(module, names, meter):
    stack = contextlib.ExitStack()
    def wrapper(fn, operation):
        def run(*args, **kw):
            with meter.span(operation):
                return fn(*args, **kw)
        return run
    for name, operation in names:
        stack.enter_context(patch.object(module, name, wrapper(getattr(module, name), operation)))
    return stack


def greedy(model, p, meter):
    # Callback deliberately returns raw objective: H is executed only after construction.
    env = Environment(p, lambda q, r: dict(q.metrics(r), raw_route=list(r), route=list(r)))
    with torch.inference_mode():
        with meter.span('features_and_global_local_encoding'):
            encoded = model.encode(p)
            meter.add('fresh_encodings')
        with meter.span('greedy_construction'):
            while not env.done:
                meter.check()
                logits = model.score(env, encoded)
                mask = torch.as_tensor(env.mask(), dtype=torch.bool)
                require(bool(mask.any()) and bool(torch.isfinite(logits[mask]).all()), 'No legal/finite policy action')
                require(bool(torch.isneginf(logits[~mask]).all()), 'Illegal action not masked')
                node = int(torch.argmax(logits).item()) + 1
                env.step(node)
                meter.add('greedy_actions')
    return list(env.result['raw_route']), env.trace


def evaluate(request, provider, case, pool, catalog, meter, remaining):
    method, seed = request['method'], request['seed']
    require(method in METHODS, 'Unregistered method')
    p, sources = prepare_view(provider, case, pool, request['kind'], meter)
    internal = _empty_internal()
    stats = dict(full_H_calls=0, export_attempts=0, exports=0, validation_attempts=0, validation_completed=0)
    trace, weight = [], None
    if method in METHODS[:2]:
        with meter.span('native_pipeline'), measured_callbacks(native,
                [('endpoint_export', 'export'), ('endpoint_validate', 'validation')], meter):
            plan, info = native.run_endpoint(p, 'fixed3' if method == METHODS[0] else 'detour16',
                time_limit_s=remaining(), reject_fallback=True, counters=stats)
        require(info['common_H_added'] is False, 'Native pipeline added common H')
        route = raw = list(info['local_route'])
        metrics = p.metrics(route)
        result = dict(metrics, raw_J=metrics['J'], raw_L=metrics['L'], raw_K=metrics['K'], plan=plan)
        internal = info['internal_search']
        raw_scope = 'native final route, including native internal search; no added common H'
    else:
        model, weight = catalog.load(seed, method, meter)
        before = tensor_hash(model)
        raw, trace = greedy(model, p, meter)
        with measured_callbacks(terminal, [('export', 'export'), ('validate', 'validation')], meter):
            if method in H_METHODS:
                with meter.span('common_H_pipeline'), _instrument(internal):
                    meter.add('common_H_invocations')
                    result = terminal.finish(p, raw, stats)
                route = list(result['route'])
            else:
                require(method == 'RL_raw', 'Unknown raw method')
                plan = terminal.export(p, raw, method)
                certificate = terminal.validate(p, plan)
                require(certificate['valid_model_plan'] and certificate['all_tasks_served'], 'Raw plan invalid')
                plan['validation'] = certificate
                route = list(raw)
                metrics = p.metrics(raw)
                result = dict(metrics, raw_J=metrics['J'], raw_L=metrics['L'], raw_K=metrics['K'], plan=plan)
        with meter.span('weight_immutability_verify'):
            require(before == tensor_hash(model) == weight['model_tensor_sha256'] and
                    digest(project_file(weight['path'])) == weight['checkpoint_sha256'], 'Inference modified weights')
            require(all(x.grad is None and not x.requires_grad for x in model.parameters()), 'Inference has gradients')
        raw_scope = 'single frozen greedy construction before common H'
    for name in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K'):
        result[name] = finite_number(result[name])
        require(result[name] >= 0, 'Negative route metric')
    require(result['plan']['validation']['valid_model_plan'], 'Uncertified plan')
    result['plan']['method'] = method
    require(meter.counts.get('common_H_invocations', 0) == int(method in H_METHODS), 'Method/common H mismatch')
    return dict(request=request, case_id=case['id'], group=case['group'], size=case['size'], endpoint=case['endpoint'],
        physical_key=case['physical_key'], source=sources, weight=weight, training_weight_role=MODEL_ROLES.get(method),
        raw_route=raw, route=route, raw_route_sha256=identity(raw), route_sha256=identity(route), trace=trace,
        metrics={k: result[k] for k in ('J','L','K','raw_J','raw_L','raw_K')}, raw_cost_scope=raw_scope,
        H_gain_J=result['raw_J']-result['J'], plan=result['plan'], pipeline_counters=stats,
        internal_search=internal, valid=True, counts=meter.counts, component_seconds=meter.seconds,
        timing_note='Nested inclusive components; do not add. Complete time includes durable result artifact write/hash; final ledger commit is charged to outer stage wall.')


def verify_determinism(rows):
    raw, final, pairs = {}, {}, {}
    for r in rows:
        q = r['request']; key = (r['case_id'], q['seed'], q['method'])
        value = (r['raw_route_sha256'], r['route_sha256'], r['metrics'])
        require(key not in final or final[key] == value, 'Frozen route/cost changed across cache/repeats')
        final[key] = value
        if q['method'] in ('RL_raw','RL_plus_H'):
            pair = (r['case_id'], q['seed'], q['kind'], q['repeat'])
            pairs.setdefault(pair, {})[q['method']] = r['raw_route_sha256']
            k = (r['case_id'], q['seed'])
            require(k not in raw or raw[k] == r['raw_route_sha256'], 'RL raw route mismatch')
            raw[k] = r['raw_route_sha256']
    require(all(set(v) == {'RL_raw','RL_plus_H'} and len(set(v.values())) == 1 for v in pairs.values()), 'Missing/mismatched RL pair')
    return dict(status='PASS', deterministic_case_seed_methods=len(final), RL_raw_plus_H_pairs=len(pairs))
