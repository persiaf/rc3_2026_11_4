"""Named, serializable data and rollout streams; never reseed a shared row RNG."""
from __future__ import annotations

import copy
import hashlib
import json
import operator

import numpy as np
import torch


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be an integer, not bool')
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(name + ' must be an integer') from exc


def derive_seed(protocol_sha256, phase, optimizer_seed, update=-1, slot=-1, role='data'):
    key = [str(protocol_sha256), str(phase), _integer(optimizer_seed, 'optimizer seed'),
           _integer(update, 'update'), _integer(slot, 'instance slot'), str(role)]
    payload = json.dumps(key, ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf8')).digest()[:8], 'little') % (2**63)


class NamedStreams:
    def __init__(self, protocol_sha256, optimizer_seed):
        self.protocol_sha256 = str(protocol_sha256)
        self.optimizer_seed = _integer(optimizer_seed, 'optimizer seed')
        self._numpy = {}
        self._torch = {}
        self._torch_meta = {}

    def seed(self, phase, update=-1, slot=-1, role='data'):
        return derive_seed(self.protocol_sha256, phase, self.optimizer_seed, update, slot, role)

    def numpy(self, phase):
        phase = str(phase)
        if phase not in self._numpy:
            self._numpy[phase] = np.random.Generator(np.random.PCG64(self.seed(phase)))
        return self._numpy[phase]

    def torch(self, phase, update, slot, role, device='cpu'):
        update = _integer(update, 'update')
        slot = _integer(slot, 'instance slot')
        device = str(torch.device(device))
        key = json.dumps([str(phase), update, slot, str(role), device], separators=(',', ':'))
        if key not in self._torch:
            seed = self.seed(phase, update, slot, role)
            self._torch[key] = torch.Generator(device=device).manual_seed(seed)
            self._torch_meta[key] = dict(phase=str(phase), update=int(update), slot=int(slot), role=str(role), device=device, seed=seed)
        return self._torch[key]

    def state_dict(self):
        return dict(schema='rc3-named-streams-v1', protocol_sha256=self.protocol_sha256,
                    optimizer_seed=self.optimizer_seed,
                    numpy={k: copy.deepcopy(v.bit_generator.state) for k, v in self._numpy.items()},
                    torch={k: v.get_state().clone() for k, v in self._torch.items()},
                    torch_meta=copy.deepcopy(self._torch_meta))

    def load_state_dict(self, state):
        if state['schema'] != 'rc3-named-streams-v1' or state['protocol_sha256'] != self.protocol_sha256 or state['optimizer_seed'] != self.optimizer_seed:
            raise ValueError('RNG protocol/optimizer-seed binding differs')
        self._numpy = {}
        for name, value in state['numpy'].items():
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = copy.deepcopy(value)
            self._numpy[name] = rng
        self._torch = {}
        self._torch_meta = copy.deepcopy(state['torch_meta'])
        if set(state['torch']) != set(self._torch_meta):
            raise ValueError('Stored row RNG keys/metadata do not match')
        for key, value in state['torch'].items():
            meta = self._torch_meta[key]
            reconstructed = json.dumps([meta['phase'], meta['update'], meta['slot'], meta['role'], meta['device']], separators=(',', ':'))
            if reconstructed != key:
                raise ValueError('Stored row RNG name differs from its metadata')
            expected = self.seed(meta['phase'], meta['update'], meta['slot'], meta['role'])
            if meta['seed'] != expected:
                raise ValueError('Stored row RNG derivation differs')
            generator = torch.Generator(device=meta['device'])
            generator.set_state(value.cpu())
            self._torch[key] = generator

