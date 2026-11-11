"""Reuse the strict saved-view loader with ONLY registered M1 training inputs.

No old S3 registry task descriptors or original60 cases are parsed. Registry
hash verification is byte maintenance only. No task/navigation generation path.
"""
import copy
from rc3_conservative_refinement import saved_inputs as si

def make_inputs(registry):
    for path,sha in si.ANCHORS.items():
        if si.digest(si.ROOT/path)!=sha: raise ValueError('Frozen loader anchor changed: '+path)
    inputs=si.SavedInputs.__new__(si.SavedInputs)
    inputs.references=si.read(si.ROOT/'rc3_integrated_rppo/DATA_REFERENCES.json')
    inputs.index=si.read(si.ROOT/'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json')
    current=si.read(si.ROOT/si.CURRENT_PRODUCER_PATH)
    original=si.read(si.ROOT/si.ORIGINAL_PRODUCER_PATH)
    inputs.producer=current['source_sha256'];inputs.original_producer=original['source_sha256']
    science=current['legacy_teacher_replay']['science_binding']
    if (len(inputs.producer)!=34 or len(inputs.original_producer)!=33 or
        science['source_sha256']!=inputs.original_producer or
        science['source_manifest_sha256']!=si.ANCHORS[si.ORIGINAL_PRODUCER_PATH] or
        current['prior_revision_path']!=si.ORIGINAL_PRODUCER_PATH or
        current['prior_revision_sha256']!=si.ANCHORS[si.ORIGINAL_PRODUCER_PATH]):
        raise ValueError('Exact old/current producer lineage differs')
    from rc3_global_local_trainer.navigation_compat import LegacyTeacherReplay
    inputs.legacy_replay=LegacyTeacherReplay(current)
    if (inputs.legacy_replay.manifest is None or
        inputs.legacy_replay.manifest['source_manifest_path']!=si.ORIGINAL_PRODUCER_PATH or
        inputs.legacy_replay.manifest['source_manifest_sha256']!=si.ANCHORS[si.ORIGINAL_PRODUCER_PATH]):
        raise ValueError('Exact legacy input provenance is absent')
    for path,sha in inputs.producer.items():
        if si.digest(si.ROOT/path)!=sha: raise ValueError('Loader producer source changed')
    inputs.allowed={si.spec_key(row['spec']):copy.deepcopy(row['spec']) for row in registry['trajectories']}
    inputs.counts=dict(view_loads=0,teacher_loads=0,source_file_hashes=0,
                       task_generations=0,navigation_solves=0,original60_access=0,S3_access=0)
    return inputs
