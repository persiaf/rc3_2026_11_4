"""Independent read-only audit of saved trainer acceptance and resume states.

No model construction, forward, backward, optimizer or terminal evaluation.
Paths in --pairs are repository-relative and must refer to this new study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASE = 'af8048149f6263e358f635385591950ab9a8f7c3'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def safe_study_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(HERE) or not path.is_file():
        raise ValueError('Expected saved acceptance evidence in new trainer study: ' + str(name))
    return path


def exact(a, b, path='state'):
    """Exact content/type comparison; no approximate resume endpoint test."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError('Tensor mismatch: ' + path)
        return a.numel(), 1
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or not np.array_equal(a, b):
            raise AssertionError('Array mismatch: ' + path)
        return a.size, 1
    if type(a) is not type(b):
        raise AssertionError('Type mismatch: ' + path)
    if isinstance(a, dict):
        if set(a) != set(b):
            raise AssertionError('Keys mismatch: ' + path)
        sums = [exact(a[k], b[k], path + '/' + str(k)) for k in a]
        return sum(x[0] for x in sums), sum(x[1] for x in sums)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError('Length mismatch: ' + path)
        sums = [exact(x, y, path + '/' + str(i)) for i, (x, y) in enumerate(zip(a, b))]
        return sum(x[0] for x in sums), sum(x[1] for x in sums)
    if a != b:
        raise AssertionError('Value mismatch: ' + path)
    return 1, 1


def load_saved(path):
    sidecar = path.with_suffix(path.suffix + '.json')
    r = read(sidecar)
    if digest(path) != r['sha256'] or path.stat().st_size != r['bytes']:
        raise AssertionError('Checkpoint differs from its saved integrity sidecar')
    s = torch.load(path, map_location='cpu', weights_only=False)
    h = hashlib.sha256()
    for name, tensor in sorted(s['model'].items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    if h.hexdigest() != s['model_tensor_sha256'] or h.hexdigest() != r['model_tensor_sha256']:
        raise AssertionError('Stored checkpoint tensor digest mismatch')
    return s, r


def audit(result_path, pairs_path):
    started = time.perf_counter(); cpu = time.process_time()
    result_path = safe_study_path(result_path)
    pairs_path = safe_study_path(pairs_path)
    result, pairs = read(result_path), read(pairs_path)
    if result.get('status') != 'PASS':
        raise AssertionError('Saved trainer acceptance has not passed')
    sources = result.get('source_sha256')
    if not isinstance(sources, dict) or not sources:
        raise AssertionError('Acceptance receipt must expose final source_sha256')
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise AssertionError('Acceptance source changed: ' + name)
    comparisons = {}
    for phase in ('S1', 'RL'):
        p = pairs[phase]
        a_path, b_path = safe_study_path(p['continuous']), safe_study_path(p['restored'])
        a, ar = load_saved(a_path); b, br = load_saved(b_path)
        fields = ('model', 'optimizer', 'streams', 'ledger', 'cursors', 'weight_version',
                  'stopped', 'model_training', 'config', 'terminal_state', 'global_rng')
        checks = {}
        for field in fields:
            elements, leaves = exact(a[field], b[field], phase + '/' + field)
            checks[field] = dict(exact=True, scalar_elements=elements, leaves=leaves)
        comparisons[phase] = dict(continuous=str(a_path.relative_to(ROOT)).replace('\\', '/'),
            restored=str(b_path.relative_to(ROOT)).replace('\\', '/'),
            continuous_sha256=ar['sha256'], restored_sha256=br['sha256'],
            tensor_sha256=a['model_tensor_sha256'], fields=checks,
            exclusions='Wall time/RSS, physical checkpoint bytes, and observational diagnostic timings are not required to match.')
    frozen = {}
    names = ['rc3_global_local/FUTURE_DESIGN_STATUS.json',
             'rc3_global_local/S2_S3_protocol_v2.json', 'rc3_global_local/future_seeds_v2.json',
             'rc3_global_local_impl_review/PROPOSED_PROTOCOL_V3.json',
             'rc3_global_local_impl_review/PROPOSED_SEEDS_V3.json']
    for name in names:
        old = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT)
        if old != (ROOT / name).read_bytes():
            raise AssertionError('Frozen protocol/registry/pointer bytes changed: ' + name)
        frozen[name] = digest(ROOT / name)
    return dict(schema='rc3-independent-saved-trainer-audit-v1', reviewer='/root/trainer_independent_review',
        status='PASS_SAVED_ACCEPTANCE_AND_EXACT_RESUME', base=BASE,
        acceptance_result=str(result_path.relative_to(ROOT)).replace('\\', '/'),
        acceptance_result_sha256=digest(result_path), resume_pairs=comparisons,
        source_hashes_checked=len(sources), frozen_protocol_registry_pointer_sha256=frozen,
        formal_S1_S2='REVISE_PENDING_FORMAL_PROTOCOL_APPROVAL_AND_EXECUTION_AUTHORITY',
        v3_activated=False,
        scientific_calls=dict(model_instantiation=0, forward=0, backward=0, optimizer_updates=0,
                              task_generation=0, navigation=0, teacher=0, full_H=0,
                              S3_target_queries=0, original60_semantic_reads=0),
        wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu,
        timing_scope='Saved JSON/tensor/source verification only, after Python/torch import; static code review not independently timed.',
        script_sha256=digest(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--output', default='rc3_global_local_trainer/INDEPENDENT_AUDIT.json')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(HERE):
        raise ValueError('Audit output must remain in new study')
    result = audit(args.result, args.pairs)
    if output.exists():
        history = read(output).get('verification_run_history', [])
    else:
        history = []
    history.append({k: result[k] for k in ('status', 'wall_s', 'cpu_s', 'acceptance_result_sha256', 'script_sha256')})
    result['verification_run_history'] = history
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    print(json.dumps({'status': result['status'], 'wall_s': result['wall_s'], 'source_hashes_checked': result['source_hashes_checked']}))


if __name__ == '__main__':
    main()
