"""Persistent, pre-debited engineering acceptance budget, not research training."""
import contextlib
import json
import os
from pathlib import Path
import time

import torch
from rc3_global_local.policy import GlobalLocalPolicy

HERE = Path(__file__).resolve().parent


def append(path, row):
    with Path(path).open('a', encoding='utf8', newline='\n') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


class AcceptanceBudget:
    def __init__(self, run, binding):
        self.path = HERE / 'ENGINEERING_JOURNAL.jsonl'
        self.run, self.binding = run, binding
        self.protocol = json.loads((HERE / 'ENGINEERING_VERIFICATION_PROTOCOL.json').read_text(encoding='utf8'))
        self.counts, self.prior_wall_s = {}, 0.
        if self.path.exists():
            events = [json.loads(x) for x in self.path.read_text(encoding='utf8').splitlines()]
            starts, ends = set(), set()
            for e in events:
                if e['event'] == 'run_start': starts.add(e['run'])
                if e['event'] == 'run_end':
                    ends.add(e['run']); self.prior_wall_s += e['wall_s']
                if e['event'] == 'charge':
                    k = e['counter']; self.counts[k] = self.counts.get(k, 0) + 1
            if starts != ends:
                raise RuntimeError('Prior interrupted acceptance has unclosed accounting; audit before any further calls')
        self.started = time.perf_counter()
        self.full_parent = False
        self.record('run_start', binding=binding)

    def record(self, event, **row):
        append(self.path, dict(event=event, run=self.run, elapsed_s=time.perf_counter()-self.started, **row))

    def check(self):
        if self.prior_wall_s + time.perf_counter()-self.started >= self.protocol['limits']['wall_s_cumulative_all_attempts']:
            raise TimeoutError('Cumulative registered engineering wall exhausted')

    def charge(self, key):
        self.check()
        limits = self.protocol['limits']
        cap = limits.get(key)
        if cap is not None and self.counts.get(key, 0) >= cap:
            raise RuntimeError('Registered engineering counter exhausted: ' + key)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.record('charge', counter=key, total=self.counts[key])

    def end(self, status):
        self.record('run_end', status=status, wall_s=time.perf_counter()-self.started, cumulative_counts=self.counts)

    @contextlib.contextmanager
    def instrument(self):
        original_step = torch.optim.Adam.step
        original_backward = torch.autograd.backward
        original_encode, original_score = GlobalLocalPolicy.encode, GlobalLocalPolicy.score
        budget = self
        def step(optimizer, *args, **kwargs):
            budget.charge('optimizer_step_attempts_total')
            budget.charge('saved_S0_full_parent_RL_batch_updates' if budget.full_parent else 'small_fixture_mock_terminal_optimizer_steps')
            result = original_step(optimizer, *args, **kwargs)
            budget.record('optimizer_step_completed', full_parent=budget.full_parent)
            return result
        def backward(*args, **kwargs):
            budget.charge('backward_calls')
            return original_backward(*args, **kwargs)
        def encode(model, *args, **kwargs):
            budget.charge('model_encode_calls')
            return original_encode(model, *args, **kwargs)
        def score(model, *args, **kwargs):
            budget.charge('model_score_calls')
            return original_score(model, *args, **kwargs)
        torch.optim.Adam.step, torch.autograd.backward = step, backward
        GlobalLocalPolicy.encode, GlobalLocalPolicy.score = encode, score
        try:
            yield
        finally:
            torch.optim.Adam.step, torch.autograd.backward = original_step, original_backward
            GlobalLocalPolicy.encode, GlobalLocalPolicy.score = original_encode, original_score
