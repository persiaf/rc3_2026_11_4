"""Exact saved-input allowlist for the new experiment; never generate data.

Old Inputs is intentionally not imported: its parent() fallback can materialize
tasks and writes into the closed run. Only previously saved train/selection,
development and explicitly hash-verified existing teacher inputs are exposed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = {
    'rc3_integrated_rppo/DATA_SPLITS.json': '56976323deaad88ab11c299e16fee4896f93ae7c0845e0e50662991482019c64',
    'rc3_integrated_rppo/DATA_REFERENCES.json': 'b2931877272214cdf7630f4bf450ec52bd20a52314006ad35d372b51b461ec8e',
    'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json': 'ff8b9f1421e3bc368dc10b66cd067e0fd7e924231031bf9e1f84d0b756dc9581',
    'rc3_global_local_execution_v3/REGISTRY.json': '75d110bef7e9954e9ace14198d262a940af6aacb17abad71cf83771ace8b363a',
    'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json': '0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0',
    'rc3_global_local_execution_v3/EXECUTION_SOURCE_BINDING.json': '435b1c8be95d6ec4f14f4aa87913a6eb3e937e3684eaadfabbad160b57ffe968',
    'rc3_navigation_compat_20261106/LEGACY_TEACHER_REPLAY_MANIFEST.json': 'c127b7ad7b7718cdab016d1366946be7aa24a5e6225055084211c5e47e131a83',
}
SPEC_FIELDS = ('map_seed', 'task_seed', 'mode', 'A_condition', 'width', 'height', 'x0', 'y0', 'endpoints')
CURRENT_PRODUCER_PATH = 'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json'
ORIGINAL_PRODUCER_PATH = 'rc3_global_local_execution_v3/EXECUTION_SOURCE_BINDING.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def producer_variant(receipt_source, current_source, original_source):
    """Match one EXACT producer map; no subset/filename/count-only aliases.

    Historical producer identity is separate from the currently checked loader
    source. The caller must additionally verify the original receipt/input's
    explicit legacy-replay record before consuming an original33 artifact.
    """
    if (not isinstance(current_source, dict) or not current_source
            or not isinstance(original_source, dict) or not original_source
            or current_source == original_source):
        raise ValueError('Distinct exact current and original producer maps are required')
    if not isinstance(receipt_source, dict):
        raise ValueError('Saved receipt producer must be an exact source map')
    if receipt_source == current_source:
        return 'current34'
    if receipt_source == original_source:
        return 'original33'
    raise ValueError('Saved receipt matches neither exact hash-pinned producer map')


def spec_key(spec):
    """Only fields used by the view constructor; reject bool/inexact IDs."""
    if not isinstance(spec, dict) or any(k not in spec for k in SPEC_FIELDS):
        raise ValueError('A complete explicitly registered saved-view spec is required')
    for k in ('map_seed', 'task_seed', 'A_condition', 'width', 'height', 'x0', 'y0'):
        if type(spec[k]) is not int:
            raise ValueError('Saved-view integer fields cannot be bool/float')
    if spec['mode'] not in ('mixed', 'blocked') or spec['endpoints'] not in ('closed_depot', 'open_first_last_connected'):
        raise ValueError('Unknown saved input convention')
    if spec['A_condition'] not in (0, 1, 2):
        raise ValueError('Unknown saved arm-allow condition')
    return tuple(spec[k] for k in SPEC_FIELDS)


def select_training_specs(batch):
    """Six fixed views: 3 full + one of each local size (50% full parents)."""
    selected = []
    for size, count in (((10, 10), 3), ((4, 3), 1), ((6, 6), 1), ((8, 8), 1)):
        matches = [s for s in batch if (s['width'], s['height']) == size]
        if len(matches) < count:
            raise ValueError('Historical schedule lacks the exact requested size prefix')
        selected.extend(copy.deepcopy(matches[:count]))
    return selected


def development_spec(row):
    w, h = row['size']
    center = row['position'] == 'center'
    if row['position'] not in ('corner', 'center'):
        raise ValueError('Unknown saved development window position')
    return dict(row, role='development', width=w, height=h,
                x0=(10-w)//2 if center else 0, y0=(10-h)//2 if center else 0,
                endpoints='open_first_last_connected' if center else 'closed_depot')


class SavedInputs:
    """No input/parent/task cache, task draws, navigation solves, or old writes.

    Every view call verifies exact persisted bytes and reconstructs snapshots.
    Input parse/provenance/nav binding therefore remain inside request timing.
    The allowlist is a data-access boundary, not formal execution permission.
    """
    def __init__(self):
        for path, sha in ANCHORS.items():
            if digest(ROOT/path) != sha:
                raise ValueError('Frozen saved-input anchor changed: '+path)
        self.splits = read(ROOT/'rc3_integrated_rppo/DATA_SPLITS.json')
        self.references = read(ROOT/'rc3_integrated_rppo/DATA_REFERENCES.json')
        self.index = read(ROOT/'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json')
        registry = read(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')
        current_binding = read(ROOT/CURRENT_PRODUCER_PATH)
        original_binding = read(ROOT/ORIGINAL_PRODUCER_PATH)
        self.producer = current_binding['source_sha256']
        self.original_producer = original_binding['source_sha256']
        science = current_binding['legacy_teacher_replay']['science_binding']
        if (len(self.producer) != 34 or len(self.original_producer) != 33
                or science['source_sha256'] != self.original_producer
                or science['source_manifest_sha256'] != ANCHORS[ORIGINAL_PRODUCER_PATH]
                or current_binding['prior_revision_path'] != ORIGINAL_PRODUCER_PATH
                or current_binding['prior_revision_sha256'] != ANCHORS[ORIGINAL_PRODUCER_PATH]):
            raise ValueError('Original producer is not the exact explicit historical lineage')
        from rc3_global_local_trainer.navigation_compat import LegacyTeacherReplay
        self.legacy_replay = LegacyTeacherReplay(current_binding)
        if (self.legacy_replay.manifest is None
                or self.legacy_replay.manifest['source_manifest_path'] != ORIGINAL_PRODUCER_PATH
                or self.legacy_replay.manifest['source_manifest_sha256'] != ANCHORS[ORIGINAL_PRODUCER_PATH]):
            raise ValueError('Original producer lacks its exact immutable replay anchors')
        # Validate CURRENT executable sources. Historical33 hashes describe the
        # original producer; they must never overwrite or masquerade as current34.
        for path, sha in self.producer.items():
            if digest(ROOT/path) != sha:
                raise ValueError('Saved input producer source changed: '+path)
        pool = self.references['teacher_pool_index']
        if digest(ROOT/pool['path']) != pool['sha256']:
            raise ValueError('Original teacher pool index changed')
        pool_meta = read(ROOT/pool['path'])
        if pool_meta['count'] != 256 or pool_meta['registry_sha256'] != ANCHORS['rc3_global_local_execution_v3/REGISTRY.json']:
            raise ValueError('Original 256-label pool has another registry')
        self.allowed = {}
        for seed in ('11622026',):
            for batch in self.splits['training'][seed][:256]:
                for spec in batch:
                    self.allowed[spec_key(spec)] = copy.deepcopy(spec)
        self.development_cases = []
        for row in registry['S2_development']:
            spec = development_spec(row)
            case = dict(id=f"development_{row['case']:03}", spec=spec, size=list(row['size']),
                        physical_key=self.references['geometry'][str(row['map_seed'])]['physical_key'],
                        group='independent_parent_exposed_development')
            self.development_cases.append(case)
            self.allowed[spec_key(spec)] = copy.deepcopy(spec)
        self.selection_cases = copy.deepcopy(self.splits['selection'])
        for case in self.selection_cases:
            case['group'] = 'trained_parent_new_tasks_exposed_selection'
            self.allowed[spec_key(case['spec'])] = copy.deepcopy(case['spec'])
        train_physical = {self.references['geometry'][str(row['map_seed'])]['physical_key']
                          for row in registry['S2_train']}
        development_physical = {case['physical_key'] for case in self.development_cases}
        selection_physical = {case['physical_key'] for case in self.selection_cases}
        if (len(train_physical) != 96 or len(development_physical) != 3
                or train_physical & development_physical
                or len(selection_physical) != 8 or not selection_physical <= train_physical):
            raise ValueError('Physical-parent independence/split identity differs')
        self.physical_groups = dict(training=sorted(train_physical),
                                    exposed_selection=sorted(selection_physical),
                                    independent_exposed_development=sorted(development_physical))
        self.evaluation_cases = self.selection_cases + self.development_cases
        self.engineering_cases = [dict(id='engineering_full10x10', spec=self.training(11622026, 1)[0])]
        self.engineering_cases.extend(copy.deepcopy(self.development_cases[i]) for i in (0, 2, 4))
        self.counts = dict(view_loads=0, teacher_loads=0, source_file_hashes=0,
                           task_generations=0, navigation_solves=0, original60_access=0, S3_access=0)

    def training(self, seed, opportunity):
        if type(seed) is not int or seed != 11622026 or type(opportunity) is not int or not 1 <= opportunity <= 256:
            raise ValueError('Only the explicitly proposed seed11622026 first256 prefix is accessible')
        return select_training_specs(self.splits['training'][str(seed)][opportunity-1])

    def teacher_indices(self, seed, opportunity):
        self.training(seed, opportunity)  # same fixed cursor/range checks
        return list(self.splits['SL_batches'][str(seed)][opportunity-1][:6])

    def _project_file(self, name):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('Saved source must have a project-relative path')
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Saved source escaped project or is missing')
        return path

    def view(self, spec):
        from rc3_fast.planner import map_key
        from rc3_global_local.problem import array_hash
        from rc3_global_local_impl_review.runtime_guards import snapshot_parent
        from rc3_global_local_trainer.navigation_compat import load_navigation_variants, select_navigation
        from rc3_s3_connectivity_20261109.scope import scoped_view

        key = spec_key(spec)
        if key not in self.allowed:
            raise ValueError('Saved view absent from the exact new allowlist; no generation fallback')
        saved_spec = self.allowed[key]
        parent_key = f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        entry = self.index.get(parent_key, self.references['bank'].get(parent_key))
        if entry is None:
            raise ValueError('Input was not already persisted; never generate or replace it')
        files = entry['files']
        paths = {}
        for name, record in files.items():
            path = self._project_file(record['path'])
            if digest(path) != record['sha256']:
                raise ValueError('Saved input source bytes changed: '+record['path'])
            paths[name] = path
            self.counts['source_file_hashes'] += 1
        scene = read(paths['scene'])
        with np.load(paths['arrays'], allow_pickle=False) as arrays:
            if set(arrays.files) != {'F', 'A'}:
                raise ValueError('Unknown saved reachability payload')
            F, A = arrays['F'].copy(), arrays['A'].copy()
        expected_parent = entry['parent_identity']
        producer_evidence = dict(variant='prepared_integrated_index',
                                source_manifest_path='rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json',
                                source_manifest_sha256=ANCHORS['rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json'])
        if 'receipt' in paths:
            receipt = read(paths['receipt'])
            variant = producer_variant(receipt['source_sha256'], self.producer, self.original_producer)
            if variant == 'original33':
                # This helper rechecks the exact receipt, scene, arrays, label,
                # navigation and navigation receipt against the hash-pinned
                # 256-record replay manifest. Unknown legacy folders fail.
                if self.legacy_replay.check_task(paths['receipt'].parent, receipt) is None:
                    raise ValueError('Original33 input has no explicit immutable replay record')
            source_path = ORIGINAL_PRODUCER_PATH if variant == 'original33' else CURRENT_PRODUCER_PATH
            producer_evidence = dict(variant=variant, source_manifest_path=source_path,
                                    source_manifest_sha256=ANCHORS[source_path],
                                    receipt_path=files['receipt']['path'],
                                    receipt_sha256=files['receipt']['sha256'],
                                    historical_source_not_upgraded=True)
            expected = {k: saved_spec[k] for k in ('map_seed', 'task_seed', 'mode', 'A_condition')}
            if any(receipt['spec'].get(k) != value for k, value in expected.items()):
                raise ValueError('Saved parent receipt task/physical spec differs')
            if (receipt['parent_identity'] != expected_parent
                    or receipt['scene_sha256'] != files['scene']['sha256']
                    or receipt['arrays_sha256'] != files['arrays']['sha256']
                    or receipt['map_key'] != map_key(scene)):
                raise ValueError('Saved parent receipt/source lineage differs')
        if scene.get('task_seed') != saved_spec['task_seed']:
            raise ValueError('Saved task identity differs')
        physical = self.references['geometry'][str(saved_spec['map_seed'])]['physical_key']
        if map_key(scene) != physical:
            raise ValueError('Saved physical layout differs')
        navigation = select_navigation(load_navigation_variants(scene, paths['navigation']),
                                       scene, F, A, expected_parent)
        parent = snapshot_parent(scene, F, A, navigation=navigation)
        if parent.identity != expected_parent:
            raise ValueError('Saved immutable parent identity differs')
        for name, value in (('F', F), ('A', A), ('D', parent.D)):
            if name+'_sha256' in entry and array_hash(value) != entry[name+'_sha256']:
                raise ValueError('Saved '+name+' array identity differs')
        view, evidence = scoped_view(parent, saved_spec)
        self.counts['view_loads'] += 1
        evidence.update(saved_parent_key=parent_key, saved_parent_identity=expected_parent,
                        physical_key=physical, input_source_files=copy.deepcopy(files),
                        input_producer=producer_evidence,
                        current_loader_reference=dict(path=CURRENT_PRODUCER_PATH, sha256=ANCHORS[CURRENT_PRODUCER_PATH]),
                        input_kind='hash_pinned_preexisting_only', global_task_ids=view.task_ids.tolist(),
                        global_candidate_ids=view.candidate_ids.tolist(),
                        F_sha256=array_hash(view.F), C_sha256=array_hash(view.C), D_sha256=array_hash(view.D))
        return view, evidence

    def teacher(self, index):
        if type(index) is not int or not 0 <= index < 256:
            raise ValueError('Teacher label index outside the frozen 256-label pool')
        source = self.references['teachers'][index]
        path = self._project_file(source['path'])
        if digest(path) != source['sha256']:
            raise ValueError('Saved teacher label bytes changed')
        label = read(path)
        if label['index'] != index:
            raise ValueError('Saved teacher index differs')
        spec = label['spec']
        self.allowed[spec_key(spec)] = copy.deepcopy(spec)
        problem, evidence = self.view(spec)
        if problem.identity != label['view_identity']:
            raise ValueError('Saved teacher view identity differs')
        actions = label['actions']
        if any(type(a) is not int or not 1 <= a <= len(problem.candidate_ids) for a in actions):
            raise ValueError('Saved teacher actions do not use local candidate IDs')
        self.counts['teacher_loads'] += 1
        evidence['teacher_source'] = copy.deepcopy(source)
        return problem, list(actions), evidence
