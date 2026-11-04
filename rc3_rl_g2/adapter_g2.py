"""Certified static-input adapter with exact-route acceptance certificate reuse."""
from __future__ import annotations
import copy
from collections import OrderedDict
import hashlib
from pathlib import Path
import time
from rc3_rl_research.adapter import ResearchAdapter


def validator_identity():
    root=Path(__file__).resolve().parents[1]
    names=('rc3_baseline/validator.py','rc3_fast/solver.py','rc3_v5/external.py',
           'rc3_rl_research/adapter.py','rc3_rl_g2/adapter_g2.py')
    h=hashlib.sha256()
    for name in names:h.update(name.encode());h.update((root/name).read_bytes())
    return h.hexdigest()


class ResearchAdapterG2(ResearchAdapter):
    def __init__(self,problem,source,service_checker=None,*,certificate_cache=True):
        self.certificate_cache_enabled=bool(certificate_cache) and service_checker is None
        self.validator_version=validator_identity()
        self._certificate_cache=OrderedDict()
        self.certificate_cache_capacity=64
        self.cache_stats=dict(hits=0,misses=0,bypasses=0,actual_certifications=0,
                              certification_s=0.,cache_lookup_s=0.)
        super().__init__(problem,source,service_checker)

    def certify(self,route,method='rl_search'):
        """Only accept-stage exact-route evidence is reused; final/load bypass."""
        started=time.perf_counter()
        route=list(route)
        cacheable=self.certificate_cache_enabled and method in ('rl_search_accept','g2_search_accept')
        key=(getattr(self,'problem_id',None),self.validator_version,tuple(route))
        if cacheable and key in self._certificate_cache:
            self._certificate_cache.move_to_end(key)
            certificate=copy.deepcopy(self._certificate_cache[key])
            certificate.plan['method']=method
            certificate.diagnostics.update(certificate_cache_hit=True,validator_version=self.validator_version)
            self.cache_stats['hits']+=1
            self.cache_stats['cache_lookup_s']+=time.perf_counter()-started
            return certificate
        self.cache_stats['misses' if cacheable else 'bypasses']+=1
        self.cache_stats['actual_certifications']+=1
        try:certificate=super().certify(route,method)
        finally:self.cache_stats['certification_s']+=time.perf_counter()-started
        certificate.diagnostics.update(certificate_cache_hit=False,validator_version=self.validator_version)
        # A validated anchor/final/load may safely populate acceptance evidence.
        if self.certificate_cache_enabled:
            actual_key=(self.problem_id,self.validator_version,tuple(certificate.route))
            self._certificate_cache[actual_key]=copy.deepcopy(certificate)
            self._certificate_cache.move_to_end(actual_key)
            while len(self._certificate_cache)>self.certificate_cache_capacity:
                self._certificate_cache.popitem(last=False)
        return certificate

    def payload(self):
        value=super().payload()
        value.update(adapter_schema='rc3-g2-adapter-v1',validator_version=self.validator_version,
                     certificate_cache=self.certificate_cache_enabled)
        return value

    @classmethod
    def from_payload(cls,payload,service_checker=None):
        if payload.get('adapter_schema')!='rc3-g2-adapter-v1':raise ValueError('unknown G2 adapter schema')
        if payload.get('validator_version')!=validator_identity():raise ValueError('validator implementation changed')
        result=super().from_payload(payload,service_checker)
        if type(payload.get('certificate_cache')) is not bool:
            raise ValueError('certificate cache mode must be bool')
        result.certificate_cache_enabled=payload['certificate_cache'] and service_checker is None
        if not result.certificate_cache_enabled:result._certificate_cache.clear()
        return result
