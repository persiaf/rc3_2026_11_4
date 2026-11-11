"""Standard-library identities and append-only task accounting."""
import hashlib
import json
import os
from pathlib import Path
import time

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
BASE = '7aeab00d50793a3bd8f35adf99e74581e29350fb'
OLD = ROOT/'rc3_r04_m1_real_20261111/attempts/02'
RUN_ID = 'rc3-r04-m2-conditional-anchor-20261111-01'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def put(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

class Ledger:
    def __init__(self, started_m, started_w):
        self.started_m, self.started_w = started_m, started_w
        self.prior = read(HOME/'REGISTRATION_RECEIPT.json')['wall_s']
        self.counts = dict(critic_optimizer_attempts=0, critic_optimizer_completed=0,
            backward_calls=0, actor_optimizer_attempts=0, new_scenes=0,
            navigation_requests=0, teacher_requests=0, native_requests=0, H_requests=0,
            original60_reads=0, oldS3_reads=0, dataset_loads=0, base_model_loads=0,
            base_encoder_forwards=0, new_labels=0, restored_trajectories=0)
        self.arms = dict(CG=0, CB=0)
        self.status = 'RUNNING'
    def elapsed(self):
        return self.prior + max(time.monotonic()-self.started_m, time.time()-self.started_w)
    def check(self):
        if self.elapsed() >= 1750: raise TimeoutError('Cumulative M2 clock reached reserved stop boundary')
    def event(self, event, **fields):
        self.check()
        row = dict(event=event, elapsed_s=self.elapsed(), counts=dict(self.counts), **fields)
        with (HOME/'RESOURCE_EVENTS.jsonl').open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
    def admit_optimizer(self, arm, step):
        self.check()
        if arm not in self.arms or self.arms[arm] >= 128 or self.counts['critic_optimizer_attempts'] >= 256:
            raise RuntimeError('M2 optimizer cap reached')
        if step != self.arms[arm]+1: raise ValueError('Repeated/skipped optimizer cursor')
        self.arms[arm] += 1; self.counts['critic_optimizer_attempts'] += 1
        self.event('optimizer_attempt_debited_before_backward', arm=arm, step=step)
    def snapshot(self):
        return dict(status=self.status, counts=dict(self.counts), per_arm_attempts=dict(self.arms), wall_s=self.elapsed())
