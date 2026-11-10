"""Hash-bound bank/splits, scope_r1 targets, original physical D kept available."""
from collections import OrderedDict
import json
import numpy as np
from .common import *
from rc3_fast.planner import map_key
from rc3_global_local.problem import array_hash
from rc3_global_local_impl_review.runtime_guards import snapshot_parent
from rc3_global_local_trainer.navigation_compat import load_navigation_variants,select_navigation,save_navigation_artifact
from rc3_s3_connectivity_20261109.scope import scoped_view


class Inputs:
    def __init__(self,log,clock):
        self.log,self.clock=log,clock;self.nav=OrderedDict();self.parents=OrderedDict();self.counts={}
        self.declared=read(HOME/'DATA_SPLITS.json');self.ref=read(HOME/'DATA_REFERENCES.json')
        p=OUT/'inputs/INDEX.json';self.index=read(p) if p.exists() else {}
        sealed={e['key']:e['entry'] for e in log.events if e['event']=='input_frozen'}
        require(all(sealed.get(k)==v for k,v in self.index.items()),'Input index not bound to original freeze event')
        if set(sealed)!=set(self.index):self.index=sealed;self.save_index()
    def inc(self,k,n=1):self.counts[k]=self.counts.get(k,0)+n
    def bind_nav(self,scene,path,F=None,A=None,parent_id=None):
        key=map_key(scene)
        if key not in self.nav:
            self.inc('saved_navigation_loads');self.nav[key]=load_navigation_variants(scene,path)
            if len(self.nav)>4:self.nav.popitem(last=False)
        else:self.inc('navigation_only_hits');self.nav.move_to_end(key)
        return select_navigation(self.nav[key],scene,F,A,parent_id)
    def save_index(self):
        b=canonical(plain(self.index));p=OUT/'inputs/INDEX.json';p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix('.pending');require(not tmp.exists(),'Unclosed input index pending');tmp.write_bytes(b)
        import os
        os.replace(tmp,p)
    def parent(self,spec):
        self.clock.check();key=f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        if key in self.parents:self.inc('parent_memory_hits');self.parents.move_to_end(key);return self.parents[key]
        if key in self.index:
            e=self.index[key];files=e['files']
            require(all(digest(ROOT/v['path'])==v['sha256'] for v in files.values()),'Prepared input bytes changed')
            scene=read(ROOT/files['scene']['path'])
            with np.load(ROOT/files['arrays']['path'],allow_pickle=False) as z:F,A=z['F'].copy(),z['A'].copy()
            nav=self.bind_nav(scene,ROOT/files['navigation']['path'],F,A,e['parent_identity'])
        else:
            original=self.ref['bank'].get(key)
            if original is not None:
                for n,v in original['files'].items():require(digest(ROOT/v['path'])==v['sha256'],'Original source changed')
                files=original['files'];scene=read(ROOT/files['scene']['path'])
                with np.load(ROOT/files['arrays']['path'],allow_pickle=False) as z:F,A=z['F'].copy(),z['A'].copy()
                r=read(ROOT/files['receipt']['path']);nav=self.bind_nav(scene,ROOT/files['navigation']['path'],F,A,r['parent_identity'])
                self.inc('original_bank_loads')
            else:
                require(spec.get('source_family') in ('new','missing_bank','unseen'),'Unregistered missing bank input')
                folder=OUT/'inputs'/key;folder.mkdir(parents=True,exist_ok=True)
                require(not any(folder.iterdir()),'Partial new input cannot regenerate')
                if spec['source_family']=='unseen':
                    from rc3_fast.scenes import make_complex_scene
                    layout=make_complex_scene(10,10,spec['map_seed'],spec['mode']);self.inc('new_physical_scene_generations')
                else:
                    src=self.ref['geometry'][str(spec['map_seed'])];require(digest(ROOT/src['scene_file'])==src['scene_sha256'],'Geometry source changed')
                    layout=read(ROOT/src['scene_file'])
                from rc3_constructive.problem import resample_tasks
                from rc3_fast.solver import vectorized_feasibility
                from rc3_global_local.run_s0 import make_allow
                scene=resample_tasks(layout,spec['task_seed']);F,_,_=vectorized_feasibility(scene);A=make_allow(F,spec['A_condition'])
                self.inc('new_task_draws');self.inc('proxy_FA_generations')
                write_new(folder/'scene.json',scene)
                with (folder/'arrays.npz').open('xb') as f:np.savez_compressed(f,F=F,A=A)
                if spec['source_family']=='unseen':
                    require(map_key(scene) not in self.ref['known_physical_keys'],'Unseen physical identity collision; no replacement')
                    require(not any(e['physical_key']==map_key(scene) and not k.startswith(str(spec['map_seed'])+'_') for k,e in self.index.items()),
                        'Two declared unseen seeds alias the same physical layout; no replacement')
                    from rc3_baseline.baseline import navigation
                    import time
                    t=time.perf_counter();self.inc('new_navigation_attempts');n=navigation(scene)
                    path=folder/'navigation.npz';save_navigation_artifact(scene,n,path,time.perf_counter()-t);self.inc('new_navigation_completed')
                else:path=ROOT/self.ref['geometry'][str(spec['map_seed'])]['navigation_file']
                nav=self.bind_nav(scene,path)
                files={n:dict(path=label(p),sha256=digest(p)) for n,p in [('scene',folder/'scene.json'),('arrays',folder/'arrays.npz'),('navigation',path),('navigation_receipt',path.with_suffix('.json'))]}
            parent=snapshot_parent(scene,F,A,navigation=nav)
            self.index[key]=dict(files=files,parent_identity=parent.identity,physical_key=map_key(scene),F_sha256=array_hash(F),A_sha256=array_hash(A),D_sha256=array_hash(parent.D),navigation_content=nav.content_sha256)
            self.log(dict(event='input_frozen',key=key,entry=self.index[key]))
            self.save_index()
        require(scene.get('task_seed')==spec['task_seed'],'Wrong task seed/source')
        if spec['source_family']!='unseen':require(map_key(scene)==self.ref['geometry'][str(spec['map_seed'])]['physical_key'],'Wrong physical geometry')
        parent=snapshot_parent(scene,F,A,navigation=nav)
        require(parent.identity==self.index[key]['parent_identity'],'Input identity reload mismatch')
        self.parents[key]=parent
        if len(self.parents)>4:self.parents.popitem(last=False)
        return parent
    def view(self,spec):
        parent=self.parent(spec);view,evidence=scoped_view(parent,spec);self.inc('target_view_checks')
        return view,evidence
    def preflight(self):
        specs=[]
        for seed in SEEDS:
            for r in self.declared['training'][str(seed)]+self.declared['warm'][str(seed)]:specs.extend(r)
        specs += [c['spec'] for split in ('selection','promotion') for c in self.declared[split]]
        # The input check is method independent; preserve every failure, never resample.
        unique={identity(s):s for s in specs};grouped={}
        for s in unique.values():grouped.setdefault((s['map_seed'],s['task_seed'],s['A_condition']),[]).append(s)
        checked=0;empty=0;physical=set()
        for _,ss in sorted(grouped.items()):
            parent=self.parent(ss[0]);physical.add(map_key(parent.scene))
            for s in ss:
                v,e=scoped_view(parent,s);checked+=1;empty+=int(len(v.C)==0)
            self.clock.check()
        write_new(HOME/'INPUT_FEASIBILITY.json',dict(status='PASS_TRAIN_SELECTION_PROMOTION',unique_declared_views=checked,
            physical_layouts=len(physical),empty_targets=empty,final_confirmation_materialized=False,counts=self.counts,
            source_binding_sha256=digest(source_path()),original60_access=0))


def base_state(seed):
    e=read(HOME/'DATA_REFERENCES.json')['models'][str(seed)]
    require(e['phase']=='S1' and e['cursor']==256 and e['weight_version']==256,'Not fixed SL256 endpoint')
    for key,sha in [('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')]:require(digest(ROOT/e[key])==e[sha],'SL upstream boundary receipt changed')
    side=read(ROOT/e['sidecar_path']);require(side['sha256']==e['checkpoint_sha256'] and side['model_tensor_sha256']==e['model_tensor_sha256']
        and side['binding']['registry_sha256']==REGISTRY and side['binding']['H_sha256']==H_ID,'SL source/H/registry lineage mismatch')
    require(digest(ROOT/e['path'])==e['checkpoint_sha256'],'SL model source changed')
    p=torch.load(ROOT/e['path'],map_location='cpu',weights_only=False)
    require(tensor_digest(p['model'])==e['model_tensor_sha256'],'Not exact seed-specific SL256')
    return p['model'],e
