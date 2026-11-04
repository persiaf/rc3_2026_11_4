"""Small, isolated demonstrations of the uploaded adapter's API limitations.

These intentionally violate its documented invariant inputs. They do not show
that its valid static-Env cache computation is incorrect, and do not edit v4.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
from rc3_v3.environment import Env
from rc3_v4.features import observation, collate
from rc3_v4.test_features import counterexample
from rc3_v5.policy import load_policy


def errors(a, b):
    result = []
    for left, right in zip(a, b):
        finite = np.isfinite(left) & np.isfinite(right)
        result.append(float(np.abs(left[finite]-right[finite]).max()))
    return result


def main():
    source = Path('rc3_v4_review_import/rc3_v4_review_evidence/training/graph_embedding_cache.py')
    spec = importlib.util.spec_from_file_location('uploaded_adapter', source)
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    path = Path('rc3_v4/training/structured.npz')
    old = adapter.CachedGraphPolicy.load(path)
    env = Env(counterexample(.1), 8, [0, 1, 2, 3, 0]); base, mask = env.observe()
    with old.inference_session():
        old.outputs_env(env, base, mask)
        old.model.w['graph_pool.bias'] += .5
        stale = old.outputs_env(env, base, mask)
        fresh = old.model.outputs(collate([observation(env, base, mask)]))
    weight_error = errors(stale, fresh)
    if not any(x > 0 for x in weight_error): raise AssertionError('counterexample failed')
    current = load_policy(path)
    try:
        current.model.w['graph_pool.bias'] += .5
        raise AssertionError('new adapter allowed weight mutation')
    except ValueError:
        pass
    result = dict(
        scope='Intentional contract-violation demonstrations, not performance or model-quality evidence.',
        source=str(source),
        mutable_weight_example=dict(graph_pool_bias_added=.5, cache_hit=old.cache_hits,
            max_absolute_error_per_head=weight_error, original_all_heads_equal=False,
            v5_mutation_raises_ValueError=True),
        interpretation='The uploaded adapter assumes invariant weights but does not enforce it. '
            'The maintenance adapter rejects ordinary array or checkpoint configuration mutation. '
            'Both adapters require a new Env when F/D/map/task semantics change.'
    )
    target = Path(__file__).with_name('source_adapter_counterexamples.json')
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__': main()
