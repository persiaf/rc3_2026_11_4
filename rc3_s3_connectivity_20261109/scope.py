"""S3 target-scope contract r1. Whole-parent context remains intact."""
import numpy as np
from unittest.mock import patch
from rc3_s3_evaluator.common import require
from rc3_fast.planner import map_key
from rc3_global_local_impl_review.runtime_guards import snapshot_parent,extract_snapshot
from rc3_s3_evaluator.inputs import prepare_view as legacy_prepare_view

CONTRACT='S3_TARGET_SCOPE_R1_FULL_PARENT_DIAGNOSTIC_TARGET_SERVICE_HARD'


class TargetScopeInfeasible(ValueError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('Actual target tasks cannot all be served under unchanged F&A and endpoints')


def unservable(parent,task_ids,entry,exit_):
    ids=np.asarray(task_ids,dtype=np.int64)
    connected=np.isfinite(parent.D[entry,1:])&np.isfinite(parent.D[1:,exit_])
    covered=(parent.effective[ids].any(2)&connected[None,:]).any(1)
    return ids[~covered].tolist()


def scoped_view(parent,spec):
    """No case/method exceptions, no coordinate crop or candidate-table filter."""
    full_unservable=unservable(parent,list(range(len(parent.F))),0,0)
    if spec['endpoints']=='closed_depot':entry=exit_=0
    else:
        require(spec['endpoints']=='open_first_last_connected','Unknown endpoint convention')
        connected=np.flatnonzero(np.isfinite(parent.D[0,1:])&np.isfinite(parent.D[1:,0]))+1
        require(len(connected)>=2,'Open entry/exit unavailable; no replacement')
        entry,exit_=int(connected[0]),int(connected[-1])
    # Original extractor still validates bounds/IDs/endpoints and every actual
    # target after same-arm F&A AND endpoint filtering. It retains every legal
    # service candidate on the parent map, including candidates outside window.
    evidence=dict(input_contract=CONTRACT,full_parent_closed_depot_unservable_task_ids=full_unservable,
        full_parent_check='DIAGNOSTIC',target_check='HARD',entry=entry,exit=exit_)
    try:view=extract_snapshot(parent,spec['x0'],spec['y0'],spec['width'],spec['height'],entry,exit_)
    except ValueError as exc:
        if str(exc)!='Selected tasks cannot all be served under F&A and endpoints':raise
        tables={t['id'] for t in parent.scene['tables'] if spec['x0']<=t['ix']<spec['x0']+spec['width'] and spec['y0']<=t['iy']<spec['y0']+spec['height']}
        ids=[t['id'] for t in parent.scene['tasks'] if t['table_id'] in tables]
        evidence.update(target_task_ids=ids,target_unservable_task_ids=unservable(parent,ids,entry,exit_))
        raise TargetScopeInfeasible(evidence) from exc
    evidence.update(target_task_ids=view.task_ids.tolist(),target_unservable_task_ids=[],target_candidate_ids=view.candidate_ids.tolist(),
        boundary_candidate_ids=list(view.boundary_candidate_ids))
    return view,evidence


def prepare_view(provider,case,pool,kind,meter):
    scene,F,A,spec,source=provider.load(case,meter)
    with meter.span('navigation_access'):
        navigation=pool.get(scene,kind,meter)
        if 'navigation_costs_sha256' in source:
            from rc3_global_local.problem import array_hash
            require(array_hash(navigation.nav['costs'])==source['navigation_costs_sha256'],'Saved navigation costs differ')
    with meter.span('whole_parent_and_window'):
        require(scene['nx']==scene['ny']==10 and map_key(scene)==case['physical_key'],'Parent cropped/substituted')
        parent=snapshot_parent(scene,F,A,navigation=navigation)
        view,evidence=scoped_view(parent,spec)
        from rc3_global_local.problem import array_hash
        source.update(evaluation_parent_identity=parent.identity,view_identity=view.identity,
            F_sha256=array_hash(parent.F),A_sha256=array_hash(parent.A),D_sha256=array_hash(parent.D),
            global_task_ids=view.task_ids.tolist(),global_candidate_ids=view.candidate_ids.tolist(),
            full_parent_tables=len(scene['tables']),full_parent_obstacles=len(scene.get('obstacles',[])),
            full_parent_geometry=map_key(scene),scope_evidence=evidence,input_contract=CONTRACT)
    return view,source


def evaluate(*args,**kwargs):
    """Single-writer scoped dependency injection; original seven pipelines intact."""
    from rc3_s3_evaluator import pipelines
    require(pipelines.prepare_view is legacy_prepare_view,'Unexpected competing input adapter')
    with patch.object(pipelines,'prepare_view',prepare_view):
        result=pipelines.evaluate(*args,**kwargs)
    result['input_contract']=CONTRACT
    return result
