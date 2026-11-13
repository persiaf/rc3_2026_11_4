"""Hash-bound saved plans and arrays. No historical runtime is imported."""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from .contracts import ProblemSnapshot,RecordedRoute,array_hash

ROOT=Path(__file__).resolve().parent.parent
HOME=Path(__file__).resolve().parent

class SavedEvidence:
    def __init__(self,root=ROOT):self.root=Path(root).resolve();self.reads={};self.json_cache={};self.input_cache={};self.archives=0
    def file(self,name,expected=None):
        path=(self.root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(self.root) or not path.is_file():raise ValueError('Unknown saved evidence path')
        if name in self.reads:
            if expected and self.reads[name]['sha256']!=expected:raise ValueError('Inconsistent source hash')
            return path
        tick=time.perf_counter();body=path.read_bytes();actual=hashlib.sha256(body).hexdigest()
        if expected and actual!=expected:raise ValueError('Saved source bytes changed: '+name)
        self.reads[name]=dict(sha256=actual,bytes=len(body),read_hash_s=time.perf_counter()-tick)
        return path
    def json(self,name,expected=None):
        path=self.file(name,expected)
        if name not in self.json_cache:
            data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
            self.json_cache[name]=json.loads(data.decode('utf-8-sig'))
        return self.json_cache[name]
    def archived(self,ref):
        # Exact already-saved member. Do not hash/scan the rest of a large journal.
        name=ref['path'];allowed=('rc3_d5_revision_20261112/evaluations/development/records/','rc3_d6_20261113/evaluations/development/records/','rc3_r07_20261113/rollouts/')
        if not name.startswith(allowed):raise ValueError('Archive is outside exposed analysis scope')
        path=(self.root/name).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Archive path escape')
        with path.open('rb') as f:f.seek(ref['offset']);body=gzip.decompress(f.read(ref['compressed_bytes']))
        if hashlib.sha256(body).hexdigest()!=ref['raw_sha256']:raise ValueError('Member hash mismatch')
        self.archives+=1
        return json.loads(body)
    def snapshot(self,record):
        if record['spec']['role'] not in ('train','selection','development'):raise ValueError('Not training or exposed development')
        sc=record['scope'];key=(record['problem_identity'],sc['D_sha256'])
        if key in self.input_cache:return self.input_cache[key]
        src=record['input_sources'];files={k:self.file(e['path'],e['sha256']) for k,e in src.items()}
        scene=self.json(src['scene']['path'],src['scene']['sha256']);tasks=tuple(sc['global_task_ids']);cids=tuple(sc['global_candidate_ids'])
        if tuple(record['plan']['target_task_ids_global'])!=tasks:raise ValueError('Task identity mismatch')
        with np.load(files['arrays'],allow_pickle=False) as z:F=z['F'][np.ix_(tasks,cids,range(2))];A=z['A'][np.ix_(tasks,cids,range(2))]
        entry,exit=record['plan']['entry_global_node'],record['plan']['exit_global_node'];nodes=np.asarray(cids,dtype=int)+1;matches=[]
        with np.load(files['navigation'],allow_pickle=False) as z:
            for name in z.files:
                if not name.endswith('costs'):continue
                allD=z[name]
                if allD.ndim!=2:continue
                D=np.zeros((len(cids)+1,len(cids)+1),np.float64);D[0,0]=allD[entry,exit];D[0,1:]=allD[entry,nodes];D[1:,0]=allD[nodes,exit];D[1:,1:]=allD[np.ix_(nodes,nodes)]
                if array_hash(D)==sc['D_sha256']:matches.append(D)
        if not matches or any(not np.array_equal(matches[0],d) for d in matches[1:]):raise ValueError('No unique exact saved navigation matrix')
        if array_hash(F&A)!=sc['F_sha256'] or array_hash((F&A).any(2))!=sc['C_sha256']:raise ValueError('Same-arm F/A source mismatch')
        poses=tuple(tuple(float(scene['candidates'][j][k]) for k in ('x','y','theta','h')) for j in cids)
        p=ProblemSnapshot(record['problem_identity'],record['parent_identity'],record['physical_key'],tasks,cids,F,A,matches[0],entry,exit,poses)
        self.input_cache[key]=p;return p

def recorded_route(plan,evidence_id):
    if not plan['validation']['valid_model_plan'] or plan['unserved_task_ids']:raise ValueError('Saved plan lacks complete certificate')
    return RecordedRoute(plan['problem_identity'],tuple(plan['visit_order_candidate_ids']),float(plan['objective_length_plus_stops_m']),float(plan['route_length_m']),len(plan['visit_order_candidate_ids']),evidence_id)
