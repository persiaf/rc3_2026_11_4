"""Checksummed, non-overwriting update-boundary checkpoints and exact RNG resume."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from rc3_global_local.policy import tensor_hash
from .rng import NamedStreams


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _state_tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _plain(value):
    # Immutable execution mappings deliberately reject __setitem__ on unpickle;
    # only persist detached plain containers, arrays and tensor state here.
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _atomic_new(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Checkpoint evidence may not be overwritten: ' + str(path))
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # NTFS/POSIX hardlink creation is atomic and fails rather than replaces
        # an already existing target, including a competing writer's output.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(path, trainer, extra=None):
    if trainer.in_update:
        raise RuntimeError('Checkpoint only at complete update boundaries')
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.json')
    if sidecar.exists():
        raise FileExistsError(sidecar)
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    terminal_state = trainer.terminal.state_dict() if hasattr(trainer.terminal, 'state_dict') else None
    state = dict(schema='rc3-trainer-checkpoint-v1', config=_plain(trainer.config),
        model=_plain(trainer.model.state_dict()),
        optimizer=_plain(trainer.optimizer.state_dict()) if trainer.optimizer is not None else None,
        model_tensor_sha256=tensor_hash(trainer.model), model_training=trainer.model.training,
        streams=trainer.streams.state_dict(), ledger=_plain(trainer.ledger),
        cursors=_plain(trainer.cursors), weight_version=trainer.weight_version,
        stopped=trainer.stopped, last_diagnostics=_plain(trainer.last_diagnostics),
        terminal_state=_plain(terminal_state),
        global_rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                        torch=torch.get_rng_state(), cuda=cuda_rng),
        extra=_plain(extra or {}))
    _atomic_new(path, lambda stream: torch.save(state, stream))
    receipt = dict(schema='rc3-trainer-checkpoint-integrity-v1', sha256=file_hash(path),
                   bytes=path.stat().st_size, model_tensor_sha256=state['model_tensor_sha256'],
                   binding=_plain(trainer.config.get('binding', {})),
                   phase=trainer.phase, cursors=_plain(trainer.cursors),
                   weight_version=trainer.weight_version,
                   restore_boundary='Complete update boundary, never mid-gradient accumulation',
                   terminal_cache_persisted=terminal_state is not None)
    payload = (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')
    _atomic_new(sidecar, lambda stream: stream.write(payload))
    return receipt


def load_checkpoint(path, trainer, expected_binding=None):
    if trainer.in_update:
        raise RuntimeError('Cannot restore over active gradient accumulation')
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + '.json').read_text(encoding='utf8'))
    if receipt['schema'] != 'rc3-trainer-checkpoint-integrity-v1' or receipt['sha256'] != file_hash(path) or receipt['bytes'] != path.stat().st_size:
        raise ValueError('Checkpoint bytes do not match integrity receipt')
    wanted = trainer.config.get('binding', {}) if expected_binding is None else expected_binding
    if dict(receipt['binding']) != dict(wanted):
        raise ValueError('Checkpoint protocol/source/run binding differs')
    state = torch.load(path, map_location=next(trainer.model.parameters()).device, weights_only=False)
    if state['schema'] != 'rc3-trainer-checkpoint-v1':
        raise ValueError('Unsupported training checkpoint schema')
    if state['config'] != trainer.config:
        raise ValueError('Checkpoint trainer configuration differs, including phase')
    if state['config'].get('binding', {}) != receipt['binding']:
        raise ValueError('Checkpoint internal and sidecar bindings differ')
    # Validate all readily checkable science state BEFORE touching live weights,
    # optimizer momentum or global RNG. A damaged stream must not partially load
    # otherwise valid weights into another active model.
    expected_state = trainer.model.state_dict()
    if set(state['model']) != set(expected_state):
        raise ValueError('Checkpoint model parameter names differ')
    for name, tensor in state['model'].items():
        if tensor.shape != expected_state[name].shape or tensor.dtype != expected_state[name].dtype or not torch.isfinite(tensor).all():
            raise ValueError('Checkpoint parameter shape/dtype/value differs: ' + name)
    raw_tensor_hash = _state_tensor_hash(state['model'])
    if raw_tensor_hash != receipt['model_tensor_sha256'] or raw_tensor_hash != state['model_tensor_sha256']:
        raise ValueError('Checkpoint stored model hash differs before restoration')
    validated_streams = NamedStreams(trainer.streams.protocol_sha256, trainer.streams.optimizer_seed)
    validated_streams.load_state_dict(state['streams'])
    random.Random().setstate(state['global_rng']['python'])
    np.random.RandomState().set_state(state['global_rng']['numpy'])
    torch.Generator(device='cpu').set_state(state['global_rng']['torch'].cpu())
    if state['global_rng']['cuda'] is not None and not torch.cuda.is_available():
        raise ValueError('Cannot restore CUDA RNG without the saved device environment')
    terminal_state = state.get('terminal_state')
    if terminal_state is not None and not hasattr(trainer.terminal, 'load_state_dict'):
        raise ValueError('Terminal cache state exists but restore adapter is missing')
    if (state['optimizer'] is None) != (trainer.optimizer is None):
        raise ValueError('Frozen/optimizable checkpoint role differs')
    if terminal_state is not None and hasattr(trainer.terminal, 'validate_state_dict'):
        trainer.terminal.validate_state_dict(terminal_state)
    trainer.model.load_state_dict(state['model'], strict=True)
    if tensor_hash(trainer.model) != receipt['model_tensor_sha256'] or tensor_hash(trainer.model) != state['model_tensor_sha256']:
        raise ValueError('Model tensor hash differs after checkpoint restoration')
    if trainer.optimizer is not None:
        trainer.optimizer.load_state_dict(state['optimizer'])
    trainer.streams.load_state_dict(state['streams'])
    trainer.ledger.clear()
    trainer.ledger.update(copy.deepcopy(state['ledger']))
    trainer.cursors = copy.deepcopy(state['cursors'])
    trainer.weight_version = int(state['weight_version'])
    trainer.stopped = bool(state['stopped'])
    trainer.last_diagnostics = copy.deepcopy(state['last_diagnostics'])
    trainer.model.train(bool(state['model_training']))
    trainer.model.zero_grad(set_to_none=True)
    if terminal_state is not None:
        trainer.terminal.load_state_dict(terminal_state)
    random.setstate(state['global_rng']['python'])
    np.random.set_state(state['global_rng']['numpy'])
    torch.set_rng_state(state['global_rng']['torch'].cpu())
    if state['global_rng']['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Cannot restore CUDA RNG without the saved device environment')
        torch.cuda.set_rng_state_all(state['global_rng']['cuda'])
    return copy.deepcopy(state['extra'])
