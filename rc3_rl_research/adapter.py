"""Read-only bridge to frozen geometry and independent plan validation."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import numpy as np

from rc3_fast.solver import prepare, route_objective, export_plan
from rc3_improve.environment import canonical_route
from rc3_improve.repair import route_is_valid
from rc3_v3.environment import reference_candidates
from .types import Certificate, SCHEMA


def _immutable(array):
    a=np.ascontiguousarray(array)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


class ResearchAdapter:
    """A fresh static problem per episode family; never mutate caller inputs."""
    @classmethod
    def from_scene(cls,scene,penalty=.75):
        source=dict(kind='proxy',scene=copy.deepcopy(scene),penalty=float(penalty))
        return cls(prepare(source['scene'],penalty=penalty),source)

    @classmethod
    def from_external(cls,bundle,penalty=.75,service_checker=None):
        from rc3_v5.external import prepare_external
        source=dict(kind='external',bundle=copy.deepcopy(bundle),penalty=float(penalty),
                    requires_service_checker=service_checker is not None)
        return cls(prepare_external(source['bundle'],penalty=penalty),source,service_checker)

    def __init__(self,problem,source,service_checker=None):
        if not math.isfinite(problem.penalty) or problem.penalty<0:
            raise ValueError('invalid penalty')
        self.source=copy.deepcopy(source)
        self.problem=copy.deepcopy(problem)
        self.problem.F=_immutable(self.problem.F)
        self.problem.arm_dist=_immutable(self.problem.arm_dist)
        self.problem.nav['costs']=_immutable(self.problem.nav['costs'])
        self.service_checker=service_checker
        self.problem_id=self.input_identity()
        candidates,failures=reference_candidates(self.problem)
        self.anchor=None
        for _,recipe,route in candidates:
            try:
                self.anchor=self.certify(route,'rl_fixed3_anchor')
                self.anchor.diagnostics.update(recipe=recipe,reference_failures=failures)
                break
            except Exception as error:
                failures.append(dict(stage='certify_reference',recipe=recipe,error=str(error)))
        if self.anchor is None:
            raise ValueError('no certified Fixed3 anchor: '+str(failures))

    def input_identity(self):
        h=hashlib.sha256(json.dumps(self.source,sort_keys=True,separators=(',',':'),
                                   ensure_ascii=False,allow_nan=False).encode('utf-8'))
        for a in (self.problem.F,self.problem.nav['costs']):
            h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
        return h.hexdigest()

    def objective(self,route):
        return float(route_objective(self.problem,route))

    def certify(self,route,method='rl_search'):
        route=list(map(int,canonical_route(self.problem,route)))
        if not route_is_valid(self.problem,route):
            raise ValueError('invalid full route under common reachability and navigation')
        if self.source['kind']=='external':
            from rc3_v5.external import export_external
            plan=export_external(self.problem,route,self.source['bundle'],method,self.service_checker)
        else:
            plan=export_plan(self.problem,route,method)
        if plan.get('validation',{}).get('valid_model_plan') is not True:
            raise ValueError('exported plan lacks successful independent validation')
        cost=self.objective(route)
        if not math.isclose(cost,float(plan['objective_length_plus_stops_m']),rel_tol=1e-9,abs_tol=1e-7):
            raise ValueError('abstract route and exported plan cost disagree')
        if [v-1 for v in route[1:-1]]!=plan['visit_order_candidate_ids']:
            raise ValueError('canonical route and exported service order disagree')
        return Certificate(route,plan,cost,dict(problem_id=self.problem_id,
                           scope='2D static navigation and supplied reachability; no physical IK'))

    def payload(self):
        return dict(schema=SCHEMA,source=copy.deepcopy(self.source),problem_id=self.problem_id,
                    anchor_route=list(self.anchor.route),anchor_cost=self.anchor.cost)

    @classmethod
    def from_payload(cls,payload,service_checker=None):
        if payload['schema']!=SCHEMA:raise ValueError('unknown replay schema')
        s=payload['source']
        if s.get('kind') not in ('proxy','external'):
            raise ValueError('unknown replay source kind')
        anchor_cost=payload['anchor_cost']
        if type(anchor_cost) not in (int,float) or not math.isfinite(anchor_cost) or anchor_cost<0:
            raise ValueError('saved anchor cost must be finite and nonnegative')
        anchor_route=payload['anchor_route']
        if not isinstance(anchor_route,list) or any(type(node) is not int for node in anchor_route):
            raise ValueError('saved anchor route must contain integer node IDs')
        if s.get('requires_service_checker') and service_checker is None:
            raise ValueError('replay requires the original external service checker')
        obj=(cls.from_scene(s['scene'],s['penalty']) if s['kind']=='proxy' else
             cls.from_external(s['bundle'],s['penalty'],service_checker))
        if obj.problem_id!=payload['problem_id']:raise ValueError('input identity changed')
        if obj.anchor.route!=anchor_route or abs(obj.anchor.cost-anchor_cost)>1e-8:
            raise ValueError('certified anchor changed')
        return obj
