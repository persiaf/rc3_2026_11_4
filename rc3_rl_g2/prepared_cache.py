"""Training-only prepared problems; non-pickle, checksummed, atomic storage.

This avoids recomputing navigation and Fixed3 after low-RAM worker eviction.
It is not used to report deployment speed: cold preparation is recorded, and
loading still independently certifies the stored anchor once. Final search
validation is unchanged. External service-checker inputs are not supported.
"""
from __future__ import annotations
from collections import OrderedDict
import copy
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import time
import uuid
import zipfile
import numpy as np
import scipy
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import Problem, route_objective
from rc3_improve.repair import route_is_valid
from rc3_rl_research.adapter import _immutable
from rc3_rl_research.types import Certificate
from .adapter_g2 import ResearchAdapterG2, validator_identity
from .env_g2 import SCHEMA, _all_finite, _number, _integer, _close

MAGIC=b'RC3G2PREP\x01'
CACHE_SCHEMA='rc3-g2-prepared-npz-v1'
MAX_BYTES=128*1024*1024
SOURCES=('rc3_baseline/baseline.py','rc3_baseline/validator.py','rc3_fast/scenes.py',
         'rc3_fast/solver.py','rc3_improve/repair.py','rc3_improve/environment.py',
         'rc3_v3/environment.py','rc3_v5/external.py','rc3_rl_research/adapter.py',
         'rc3_rl_research/types.py','rc3_rl_g2/adapter_g2.py','rc3_rl_g2/prepared_cache.py')


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')


def source_identity():
    root=Path(__file__).resolve().parents[1]
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCES}
    versions=dict(numpy=np.__version__,scipy=scipy.__version__)
    return hashlib.sha256(_json(dict(files=files,versions=versions,schema=CACHE_SCHEMA))).hexdigest()


def _case(case):
    if not isinstance(case,dict):raise ValueError('case must be a registered proxy scene dictionary')
    if any(key in case for key in ('bundle','planning','audit','service_checker','scene')):
        raise ValueError('prepared cache only accepts generated proxy training cases')
    for key in ('nx','ny','seed'):_integer(case[key],'case '+key)
    if min(case['nx'],case['ny'])<1 or case['nx']*case['ny']>100:
        raise ValueError('this G2 training cache is limited to at most 100 tables')
    if case['mode'] not in ('regular','mixed','blocked'):raise ValueError('unknown scene mode')
    _all_finite(case);_json(case)
    return copy.deepcopy(case)


def cache_path(case,cache_dir,identity=None):
    identity=source_identity() if identity is None else identity
    key=hashlib.sha256(_json(dict(case=_case(case),source_identity=identity))).hexdigest()
    return Path(cache_dir)/f'{key}.rc3prep'


def _pack(adapter,case,identity,cold_prepare_s):
    p=adapter.problem
    metadata=dict(schema=CACHE_SCHEMA,case=case,source_identity=identity,
        source=adapter.source,problem_id=adapter.problem_id,validator_version=adapter.validator_version,
        scene=p.scene,sparse=p.sparse,preprocessing_s=p.preprocessing_s,penalty=p.penalty,
        navigation={k:v for k,v in p.nav.items() if k not in ('costs','predecessors')},
        anchor=asdict(adapter.anchor),cold_prepare_s=cold_prepare_s)
    stream=io.BytesIO()
    np.savez_compressed(stream,F=p.F,arm_dist=p.arm_dist,costs=p.nav['costs'],
        predecessors=p.nav['predecessors'],metadata=np.frombuffer(_json(metadata),dtype=np.uint8))
    payload=stream.getvalue()
    header=_json(dict(schema=CACHE_SCHEMA,source_identity=identity,case=case,
                      payload_sha256=hashlib.sha256(payload).hexdigest(),payload_bytes=len(payload)))
    return MAGIC+struct.pack('<I',len(header))+header+payload


def _unpack(raw,case,identity):
    if len(raw)>MAX_BYTES or not raw.startswith(MAGIC):raise ValueError('invalid prepared-cache envelope')
    offset=len(MAGIC)
    if len(raw)<offset+4:raise ValueError('truncated prepared cache')
    n=struct.unpack('<I',raw[offset:offset+4])[0];offset+=4
    if n>64*1024 or len(raw)<offset+n:raise ValueError('invalid cache header size')
    header=json.loads(raw[offset:offset+n]);payload=raw[offset+n:]
    if (header.get('schema')!=CACHE_SCHEMA or header.get('source_identity')!=identity or header.get('case')!=case):
        raise ValueError('cache input/source version mismatch')
    if type(header.get('payload_bytes')) is not int or header['payload_bytes']!=len(payload):raise ValueError('cache payload size mismatch')
    if hashlib.sha256(payload).hexdigest()!=header.get('payload_sha256'):raise ValueError('cache checksum mismatch')
    # Inspect the safe array container before NumPy reads any member. No pickle
    # is accepted, even when a malicious file has a recomputed outer checksum.
    expected={'F.npy','arm_dist.npy','costs.npy','predecessors.npy','metadata.npy'}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members=archive.infolist()
        if {m.filename for m in members}!=expected or len(members)!=len(expected):raise ValueError('unexpected cache members')
        if sum(m.file_size for m in members)>MAX_BYTES:raise ValueError('oversized cache arrays')
    with np.load(io.BytesIO(payload),allow_pickle=False) as saved:
        arrays={key:saved[key] for key in ('F','arm_dist','costs','predecessors','metadata')}
    encoded=arrays.pop('metadata')
    if encoded.dtype!=np.uint8 or encoded.ndim!=1 or encoded.size>8*1024*1024:raise ValueError('invalid metadata array')
    meta=json.loads(encoded.tobytes());_all_finite(meta)
    if meta.get('schema')!=CACHE_SCHEMA or meta.get('case')!=case or meta.get('source_identity')!=identity:
        raise ValueError('metadata/source identity mismatch')
    if meta['validator_version']!=validator_identity():raise ValueError('cache validator version mismatch')
    for key in ('cold_prepare_s','preprocessing_s','penalty'):_number(meta[key],key,True)
    scene=meta['scene'];n_tasks=len(scene['tasks']);m=len(scene['candidates'])
    if m!=8*case['nx']*case['ny'] or len(scene['tables'])!=case['nx']*case['ny'] or n_tasks>3*len(scene['tables']):
        raise ValueError('cache scene dimensions disagree with case')
    # The source scene is also the content identity used by the frozen adapter.
    if meta['source']!={'kind':'proxy','scene':scene,'penalty':meta['penalty']}:
        raise ValueError('cached source and problem disagree')
    F,dist,D,pred=(arrays[k] for k in ('F','arm_dist','costs','predecessors'))
    nav=meta['navigation'];nodes=nav['node_count']
    _integer(nodes,'navigation node_count')
    if nodes<1 or nodes>2*m+(case['nx']+1)*(case['ny']+1)+10:raise ValueError('invalid navigation size')
    if F.dtype!=bool or F.shape!=(n_tasks,m,2):raise ValueError('invalid F matrix')
    if dist.dtype!=np.float64 or dist.shape!=F.shape or not np.isfinite(dist).all() or (dist<0).any():
        raise ValueError('invalid arm distances')
    if D.dtype!=np.float64 or D.shape!=(m+1,m+1) or np.isnan(D).any() or np.isneginf(D).any() or (D<0).any():
        raise ValueError('invalid navigation costs')
    if pred.dtype.kind!='i' or pred.shape!=(m+1,nodes) or ((pred<0)&(pred!=-9999)).any() or (pred>=nodes).any():
        raise ValueError('invalid navigation predecessors')
    if len(nav['coordinates'])!=nodes or len(nav['service_nodes'])!=m+1:
        raise ValueError('navigation indexing shape mismatch')
    coordinates=np.asarray(nav['coordinates'],dtype=float)
    if coordinates.shape!=(nodes,2) or not np.isfinite(coordinates).all():raise ValueError('invalid navigation coordinates')
    if any(type(v) is not int or not 0<=v<nodes for v in nav['service_nodes']):raise ValueError('invalid service node IDs')
    if len(meta['sparse'])!=n_tasks:raise ValueError('invalid sparse coverage count')
    expected_sparse=[dict(task_id=i,left=np.flatnonzero(F[i,:,0]).tolist(),right=np.flatnonzero(F[i,:,1]).tolist()) for i in range(n_tasks)]
    if expected_sparse!=meta['sparse']:raise ValueError('sparse and dense reachability disagree')
    nav.update(costs=_immutable(D),predecessors=_immutable(pred))
    problem=Problem(copy.deepcopy(scene),_immutable(F),_immutable(dist),meta['sparse'],nav,meta['preprocessing_s'],meta['penalty'])
    adapter=object.__new__(ResearchAdapterG2)
    adapter.source=copy.deepcopy(meta['source']);adapter.problem=problem;adapter.service_checker=None
    adapter.problem_id=meta['problem_id'];adapter.validator_version=meta['validator_version']
    adapter.certificate_cache_enabled=True;adapter.certificate_cache_capacity=64
    adapter._certificate_cache=OrderedDict()
    adapter.cache_stats=dict(hits=0,misses=0,bypasses=0,actual_certifications=0,certification_s=0.,cache_lookup_s=0.)
    if adapter.input_identity()!=adapter.problem_id:raise ValueError('prepared problem identity mismatch')
    anchor=meta['anchor'];_number(anchor['cost'],'cached anchor cost',True)
    route=anchor['route']
    if not isinstance(route,list) or any(type(v) is not int for v in route) or not route_is_valid(problem,route):
        raise ValueError('cached anchor route invalid')
    _close(route_objective(problem,route),anchor['cost'],'cached anchor objective')
    # Bound predecessor walks before the legacy expander, which assumes a
    # well-formed Dijkstra forest. Corrupted cycles must not hang a worker.
    for source,target in zip(route,route[1:]):
        a,b=nav['service_nodes'][source],nav['service_nodes'][target];seen=set()
        while b!=a:
            if b in seen:raise ValueError('cycle in cached anchor navigation')
            seen.add(b);b=int(pred[source,b])
            if b<0:raise ValueError('disconnected cached anchor navigation')
    verified=adapter.certify(route,method=anchor['plan'].get('method','g2_prepared_load'))
    if verified.route!=route or verified.plan!=anchor['plan']:raise ValueError('cached anchor certificate disagrees')
    _close(verified.cost,anchor['cost'],'cached anchor certificate cost')
    adapter.anchor=Certificate(route,anchor['plan'],anchor['cost'],anchor.get('diagnostics',{}))
    return adapter,meta


def prepare_case(case,cache_dir):
    """Return adapter, cost ledger. Invalid existing entries fail closed."""
    started=time.perf_counter();case=_case(case);identity=source_identity()
    directory=Path(cache_dir);directory.mkdir(parents=True,exist_ok=True)
    path=cache_path(case,directory,identity);lock=path.with_suffix('.lock')

    def load():
        tick=time.perf_counter();raw=path.read_bytes();adapter,meta=_unpack(raw,case,identity)
        return adapter,dict(cache_hit=True,cold_prepare_s=0.,cache_load_s=time.perf_counter()-tick,
            original_cold_prepare_s=meta['cold_prepare_s'],cache_write_s=0.,source_identity=identity,
            bytes=len(raw),path=str(path),total_s=time.perf_counter()-started,
            waiting_s=waited,scope='training prepared input reuse; final validation unchanged')

    waited=0.
    if path.exists():return load()
    deadline=time.perf_counter()+30.
    while True:
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);break
        except FileExistsError:
            if path.exists():return load()
            if time.perf_counter()>=deadline:raise TimeoutError('prepared cache writer lock timed out')
            time.sleep(.01);waited+=.01
    temporary=None
    try:
        if path.exists():return load()
        tick=time.perf_counter()
        adapter=ResearchAdapterG2.from_scene(make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode']))
        cold=time.perf_counter()-tick
        tick=time.perf_counter();data=_pack(adapter,case,identity,cold)
        temporary=directory/f'.{path.stem}.{uuid.uuid4().hex}.tmp'
        with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);temporary=None
        return adapter,dict(cache_hit=False,cold_prepare_s=cold,cache_load_s=0.,
            original_cold_prepare_s=cold,cache_write_s=time.perf_counter()-tick,source_identity=identity,
            bytes=len(data),path=str(path),total_s=time.perf_counter()-started,waiting_s=waited,
            scope='training prepared input reuse; final validation unchanged')
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def validate_worker_snapshot(state,cases,config,seed):
    """Validate worker metadata before the caller commits any live fields.

    This supplements (does not replace) SearchEnvG2.load_state_dict. Return an
    isolated copy; the caller may subsequently instantiate/verify the env and
    commit this metadata only after that verification also succeeds.
    """
    state=copy.deepcopy(state);_all_finite(state)
    if state.get('cases')!=cases or state.get('config')!=config or state.get('seed')!=seed:
        raise ValueError('worker checkpoint identity mismatch')
    for key in ('seed','position','episode','preparations','cache_reuses'):_integer(state[key],key)
    if not cases or state['episode']<1:raise ValueError('worker has no active episode')
    order=state['order']
    if not isinstance(order,list) or any(type(i) is not int for i in order) or sorted(order)!=list(range(len(cases))):
        raise ValueError('worker order is not an index permutation')
    if not 1<=state['position']<=len(order):raise ValueError('worker position out of range')
    if state['active_case']!=cases[order[state['position']-1]]:raise ValueError('worker active case/order mismatch')
    if state['position']!=(state['episode']-1)%len(cases)+1:raise ValueError('worker episode/position mismatch')
    env=state['env_state']
    if env.get('schema')!=SCHEMA or env.get('config')!=config:raise ValueError('worker environment schema/config mismatch')
    _integer(env['seed'],'environment seed')
    if env['seed']!=seed+state['episode']:raise ValueError('worker episode seed mismatch')
    _number(state['episode_reward'],'episode_reward');_number(env['total_reward'],'environment total_reward')
    _close(state['episode_reward'],env['total_reward'],'worker/environment reward')
    for key in ('prepare_s','feature_s','elapsed_episode_s'):_number(state[key],key,True)
    if max(state['prepare_s'],state['feature_s'])>state['elapsed_episode_s']+1e-8:
        raise ValueError('worker component time exceeds episode elapsed time')
    rng=np.random.default_rng();rng.bit_generator.state=state['rng_state']
    return state
