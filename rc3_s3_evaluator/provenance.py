"""Training checkpoint identity is separate from evaluation executable identity."""
from dataclasses import dataclass
from pathlib import Path
import platform
import sys

from .common import (ROOT, HERE, EXPERIMENT, PROTOCOL, REGISTRY, TRAIN_SOURCE, H_ID,
                     SEEDS, MODEL_ROLES, require, read, digest, project_file, identity, label)


def environment():
    import numpy as np
    import scipy
    import torch
    return dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                scipy=scipy.__version__, torch=torch.__version__, device='cpu',
                torch_threads=1, torch_interop_threads=1, deterministic_algorithms=True)


def configure_runtime():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def verify_training_anchors():
    paths = {'protocol': ROOT/'rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json',
             'registry': ROOT/'rc3_global_local_execution_v3/REGISTRY.json',
             'training_source': ROOT/'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json'}
    for key, expected in (('protocol', PROTOCOL), ('registry', REGISTRY), ('training_source', TRAIN_SOURCE)):
        require(digest(paths[key]) == expected, 'Frozen training anchor mismatch: ' + key)
    source = read(paths['training_source'])
    require(all(digest(project_file(n)) == h for n, h in source['source_sha256'].items()), 'Frozen training implementation changed')
    h = read(ROOT/'rc3_global_local_execution_v3/H_BINDING.json')
    require(h['H_identity'] == H_ID and identity(h['H_source_sha256']) == H_ID,
            'H identity changed')
    require(all(digest(project_file(n)) == v for n, v in h['H_source_sha256'].items()), 'H source changed')
    approval = read(ROOT/'rc3_global_local_execution_v3/authorizations/20261106_S1_A/PROTOCOL_APPROVAL.json')
    require(approval['status'] == 'APPROVED' and approval['approved_by'] == 'user', 'Original v3 approval absent')
    return source


def verified_geometry():
    """Read only task-free assets attested by the approved original layout review."""
    protocol = ROOT/'rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json'
    require(digest(protocol) == PROTOCOL, 'Wrong geometry evidence protocol')
    e = read(protocol)['layout_independence_gate']['evidence']
    gate_path, review_path = project_file(e['path']), project_file(e['independent_review'])
    require(digest(gate_path) == e['sha256'] and digest(review_path) == e['independent_sha256'],
            'Approved geometry gate/review bytes changed')
    gate, review = read(gate_path), read(review_path)
    require(gate['status'] == 'PASS', 'Geometry gate failed')
    files = {r['file']:r for r in review['files']}
    require(len(files) == len(gate['records']) == 111 and set(files) == {r['file'] for r in gate['records']}, 'Unattested geometry assets')
    records = []
    for row in gate['records']:
        attested = files[row['file']]
        path = project_file(row['file'])
        require(all(row[k] == attested[k] for k in ('role','label','map_seed','mode','bytes')) and
                path.stat().st_size == attested['bytes'] and digest(path) == attested['sha256'], 'Geometry source/asset changed')
        records.append(dict(row, sha256=attested['sha256']))
    return records


class ModelCatalog:
    def __init__(self, index_path=None, verify_files=True):
        self.path = index_path or HERE/'S3_MODEL_INDEX.json'
        self.index = read(self.path)
        require(self.index['schema'] == 'rc3-s3-fixed-model-index-v1' and self.index['experiment_id'] == EXPERIMENT,
                'Wrong model index')
        require(self.index['training_source_sha256'] == TRAIN_SOURCE and self.index['protocol_sha256'] == PROTOCOL,
                'Model training identity is not the approved S2 identity')
        self.models = self.index['models']
        require(set(self.models) == {str(s) for s in SEEDS}, 'Wrong fixed model seeds')
        for seed in SEEDS:
            require(set(self.models[str(seed)]) == {'untrained', 'SL256', 'continuedSL512', 'RL512'}, 'Wrong fixed endpoints')
            for role in self.models[str(seed)]:
                self.entry(seed, role, verify_files=verify_files)
        for name, sha in self.index['upstream_sha256'].items():
            require(digest(project_file(name)) == sha, 'Upstream completion/lineage changed')

    def entry(self, seed, role, verify_files=True):
        require(type(seed) is int and seed in SEEDS and role in self.models[str(seed)], 'Unknown seed/weight role')
        e = self.models[str(seed)][role]
        phase, cursor, name = {'untrained': ('S1', 0, 'untrained.pt'), 'SL256': ('S1', 256, 'update_0256.pt'),
                               'continuedSL512': ('continued_SL', 512, 'update_0512.pt'), 'RL512': ('RL', 512, 'update_0512.pt')}[role]
        expected = f'rc3_global_local_execution_v3/runs/{EXPERIMENT}/seed_{seed}/{phase}/{name}'
        require(e['path'] == expected and e['phase'] == phase and e['cursor'] == cursor, 'Intermediate/unrelated checkpoint refused')
        cp = project_file(e['path'])
        if verify_files:
            require(digest(cp) == e['checkpoint_sha256'] and cp.stat().st_size == e['checkpoint_bytes'], 'Checkpoint file changed')
            side, commit = cp.with_suffix('.pt.json'), cp.with_suffix('.pt.commit.json')
            require(digest(side) == e['sidecar_sha256'] and digest(commit) == e['commit_sha256'], 'Checkpoint metadata changed')
            m, c = read(side), read(commit)
            require(m['sha256'] == c['checkpoint_sha256'] == e['checkpoint_sha256'] and
                    c['sidecar_sha256'] == e['sidecar_sha256'] and m['weight_version'] == cursor and
                    m['cursors'][phase] == cursor and m['model_tensor_sha256'] == e['model_tensor_sha256'], 'Wrong endpoint/cursor/tensor identity')
            require(m['binding']['optimizer_seed'] == seed and m['binding']['phase'] == phase and
                    m['binding']['source_manifest_sha256'] == TRAIN_SOURCE and
                    m['binding']['protocol_sha256'] == PROTOCOL, 'Checkpoint training provenance differs')
        return e

    def load(self, seed, method, meter):
        import torch
        from rc3_global_local.policy import GlobalLocalPolicy, tensor_hash
        role = MODEL_ROLES[method]
        with meter.span('model_load_and_verify'):
            rng_before = torch.get_rng_state().clone()
            e = self.entry(seed, role)
            state = torch.load(project_file(e['path']), map_location='cpu', weights_only=False)
            require(state['model_tensor_sha256'] == e['model_tensor_sha256'], 'Checkpoint payload identity differs')
            # Meta materialization consumes no ambient random stream or new scientific initialization.
            with torch.device('meta'):
                model = GlobalLocalPolicy()
            model.load_state_dict(state['model'], strict=True, assign=True)
            del state  # No Adam object is created; optimizer/RNG payload is never resumed.
            model.eval().requires_grad_(False)
            require(tensor_hash(model) == e['model_tensor_sha256'], 'Loaded model tensor mismatch')
            require(all(not p.requires_grad and bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite/unfrozen weights')
            require(torch.equal(rng_before, torch.get_rng_state()), 'Frozen model load consumed the initialization RNG')
            meter.add('frozen_policy_materializations')
        return model, e


def verify_code_binding(path):
    b = read(path)
    require(b['schema'] == 'rc3-s3-code-binding-v1' and b['protocol_sha256'] == PROTOCOL and
            b['registry_sha256'] == REGISTRY and b['training_source_sha256'] == TRAIN_SOURCE and b['H_identity'] == H_ID,
            'Evaluation source binding differs')
    for n, h in b['source_sha256'].items():
        require(digest(project_file(n)) == h, 'Evaluation/dependency source changed: ' + n)
    require(b['environment'] == environment(), 'Evaluation runtime environment changed')
    require(digest(HERE/'S3_MODEL_INDEX.json') == b['model_index_sha256'], 'Model index changed')
    if b.get('revision') == 2:
        verified_geometry()
        for name, sha in b['fixed_data_sha256'].items():
            require(digest(project_file(name)) == sha, 'Frozen evaluator data changed: ' + name)
    return b


@dataclass(frozen=True)
class Permit:
    scope: str
    authority_sha256: str
    code_sha256: str
    model_index_sha256: str
    source_binding_sha256: str
    requests_per_seed: int
    wall_s_per_seed: int
    seeds: tuple
    registration: object

    def stable(self):
        return dict(scope=self.scope, authority_sha256=self.authority_sha256, code_sha256=self.code_sha256,
                    model_index_sha256=self.model_index_sha256, source_binding_sha256=self.source_binding_sha256,
                    requests_per_seed=self.requests_per_seed, wall_s_per_seed=self.wall_s_per_seed,
                    seeds=list(self.seeds), experiment_id=EXPERIMENT)


def admit_engineering(registration_path):
    p = Path(registration_path).resolve()
    require(p == HERE/'ENGINEERING_REGISTRATION.json', 'Only the canonical registered engineering experiment is admitted')
    r = read(p)
    require(r['scope'] == 'S3_ENGINEERING_ONLY' and r['status'] == 'REGISTERED_BEFORE_FIRST_REQUEST' and
            r['experiment_id'] == EXPERIMENT and r['engineering_id'] == 'S3_ENGINEERING_20261107_01' and
            r['complete_requests_max'] == 126 and r['wall_s_max'] == 1800 and r['model_seed'] == SEEDS[0], 'Wrong engineering authority/budgets')
    require(r['formal_S3_authorized'] is False and r['target_access'] == 'SAVED_DEV_ONLY', 'Engineering cannot authorize heldout data')
    code = HERE/'S3_CODE_BINDING.json'
    verify_code_binding(code)
    require(digest(code) == r['code_binding_sha256'], 'Engineering registration/code mismatch')
    require(len(r['cases']) == 6 and len({c['id'] for c in r['cases']}) == 6, 'Exactly six fixed old cases required')
    return Permit('engineering', digest(p), digest(code), digest(HERE/'S3_MODEL_INDEX.json'),
                  '', 126, 1800, (SEEDS[0],), r)


def admit_formal(authorization_path, source_approval_path):
    # No S3 declarations/task data are accessed before both independent authorities pass.
    bpath = HERE/'S3_SOURCE_BINDING.json'
    b = read(bpath)
    require(b['status'] == 'TECHNICAL_GO_NOT_S3_AUTHORIZATION' and b['training_source_sha256'] == TRAIN_SOURCE,
            'Evaluator has no finalized technical source binding')
    code_path = project_file(b['code_binding_path'])
    verify_code_binding(code_path)
    require(digest(code_path) == b['code_binding_sha256'], 'Final/code binding mismatch')
    for n, h in b['verification_sha256'].items():
        require(digest(project_file(n)) == h, 'Engineering verification changed')
    ap, sp = project_file(label(authorization_path)), project_file(label(source_approval_path))
    a, s = read(ap), read(sp)
    expected = dict(protocol_sha256=PROTOCOL, registry_sha256=REGISTRY, H_identity=H_ID,
                    evaluation_source_binding_sha256=digest(bpath), training_source_sha256=TRAIN_SOURCE,
                    model_index_sha256=digest(HERE/'S3_MODEL_INDEX.json'), experiment_id=EXPERIMENT)
    require(s.get('schema') == 'rc3-s3-evaluation-source-approval-v1' and s.get('status') == 'APPROVED' and
            s.get('approved_by') == 'user' and all(s.get(k) == v for k, v in expected.items()) and
            s.get('approval_source') and s.get('user_message_sha256'), 'Independent user evaluation-source approval missing/mismatched')
    require(a.get('schema') == 'rc3-s3-stage-authorization-v1' and a.get('status') == 'AUTHORIZED' and
            a.get('authorized_by') == 'user' and a.get('stage') == 'S3' and all(a.get(k) == v for k, v in expected.items()) and
            a.get('source_approval_sha256') == digest(sp) and a.get('policy_seeds') == list(SEEDS) and
            a.get('requests_per_seed_max') == 5292 and a.get('requests_all_seeds_max') == 15876 and
            a.get('wall_s_per_seed_max') == 14400 and a.get('optimizer_updates_max') == 0 and
            a.get('new_teacher_queries_max') == 0 and a.get('authorization_source') and a.get('user_message_sha256'),
            'Independent S3 stage authority missing/mismatched')
    require(a.get('S2_complete_sha256') == read(HERE/'S3_MODEL_INDEX.json')['S2_complete_sha256'], 'Wrong S2 predecessors')
    require(digest(project_file(s['approval_source'])) == s['user_message_sha256'] and
            digest(project_file(a['authorization_source'])) == a['user_message_sha256'], 'User approval/authorization source bytes differ')
    return Permit('formal', digest(ap), digest(code_path), digest(HERE/'S3_MODEL_INDEX.json'),
                  digest(bpath), 5292, 14400, SEEDS, None)
