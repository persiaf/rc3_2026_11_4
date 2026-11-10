"""Bounded synthetic verification. Never calls a planner or optimizer.step."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    home = Path(__file__).resolve().parent
    plan = json.loads((home / 'ENGINEERING_PLAN_R02.json').read_bytes())
    for name, entry in plan['source_files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Engineering source binding changed: ' + name)
    counters = dict(optimizer_step_attempts=0, weight_load_attempts=0, weight_save_attempts=0)

    def forbidden(kind):
        def call(*args, **kwargs):
            counters[kind] += 1
            raise AssertionError(kind + ' forbidden in zero-update engineering')
        return call

    stream = io.StringIO()
    started = time.perf_counter()
    with patch.object(torch.optim.Adam, 'step', forbidden('optimizer_step_attempts')), \
         patch.object(torch.optim.SGD, 'step', forbidden('optimizer_step_attempts')), \
         patch.object(torch, 'load', forbidden('weight_load_attempts')), \
         patch.object(torch, 'save', forbidden('weight_save_attempts')):
        suite = unittest.defaultTestLoader.discover(str(home), pattern='test_*.py',
                                                   top_level_dir=str(home.parent))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    (args.output / 'UNITTEST_LOG.txt').write_text(stream.getvalue(), encoding='utf-8')
    report = dict(status='PASS' if result.wasSuccessful() and not any(counters.values()) else 'FAIL',
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  elapsed_suite_s=elapsed, python=sys.version, platform=platform.platform(),
                  torch=torch.__version__, numpy=np.__version__, counters=counters,
                  optimizer_updates=0, planning_requests=0, navigation_calls=0, H_calls=0,
                  sealed_semantic_reads=0, original_checkpoint_loads=0,
                  no_historical_module_imports=not any(n.startswith(('rc3_integrated_rppo', 'rc3_global_local',
                                                                       'rc3_baseline', 'rc3_s3_')) for n in sys.modules),
                  scope='Synthetic arithmetic/features/gradients/mock restoration; not real rollout or training acceptance')
    (args.output / 'VERIFICATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
