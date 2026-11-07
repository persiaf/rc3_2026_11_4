from pathlib import Path
import contextlib
import datetime
import hashlib
import json
import math
import os
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
EXPERIMENT = 'rc3_v3_global_local_20261106_01'
BASELINE = 'fadfa492cd3f6e378ebe9530f83f27ac98b3c273'
PROTOCOL = '0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb'
REGISTRY = '75d110bef7e9954e9ace14198d262a940af6aacb17abad71cf83771ace8b363a'
TRAIN_SOURCE = '0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0'
H_ID = '5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14'
SEEDS = (11622026, 11622027, 11622028)
SIZES = ((4, 3), (6, 6), (8, 8))
METHODS = ('Fixed3_endpoint_adapter', 'Detour16_endpoint_adapter', 'untrained',
           'frozen_supervised', 'continued_SL', 'RL_raw', 'RL_plus_H')
MODEL_ROLES = {'untrained': 'untrained', 'frozen_supervised': 'SL256',
               'continued_SL': 'continuedSL512', 'RL_raw': 'RL512', 'RL_plus_H': 'RL512'}
H_METHODS = frozenset(('untrained', 'frozen_supervised', 'continued_SL', 'RL_plus_H'))
ENGINEERING_ID = 'S3_ENGINEERING_20261107_01'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf8')


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def reject(x):
        raise ValueError('Nonfinite JSON constant: ' + x)
    return json.loads(Path(path).read_text(encoding='utf8'), parse_constant=reject)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf8') + b'\n')
        f.flush()
        os.fsync(f.fileno())


def project_file(name):
    require(isinstance(name, str) and not Path(name).is_absolute(), 'Project-relative source path required')
    p = (ROOT / name).resolve()
    require(p.is_relative_to(ROOT) and p.is_file(), 'Missing/escaped source file: ' + name)
    return p


def label(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def exact_int(value, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high), 'Invalid integer')
    return value


def finite_number(value, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0), 'Nonfinite/invalid scalar result')
    return float(value)


class Meter:
    """Durable major-operation predebits; nested durations are never summed."""
    def __init__(self, record=lambda _: None, check=lambda: None):
        self.record, self.check = record, check
        self.counts, self.seconds = {}, {}

    def add(self, name, n=1):
        self.counts[name] = self.counts.get(name, 0) + n

    @contextlib.contextmanager
    def span(self, name):
        self.check()
        self.add(name + '_attempts')
        self.record(dict(event='operation_start', operation=name))
        started = time.perf_counter()
        try:
            yield
            self.add(name + '_completed')
        except BaseException:
            self.add(name + '_failed')
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.seconds[name] = self.seconds.get(name, 0.) + elapsed
            self.record(dict(event='operation_end', operation=name, elapsed_s=elapsed,
                             counts=dict(self.counts)))
        self.check()
