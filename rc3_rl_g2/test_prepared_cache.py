"""Prepared training input cache and worker metadata validation regressions."""
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rc3_rl_research.types import SearchConfig
from .env_g2 import SearchEnvG2
from .baselines import make_policy
from .prepared_cache import (prepare_case,source_identity,cache_path,_unpack,_pack,MAGIC,
                             validate_worker_snapshot)

ROOT=Path(__file__).resolve().parent/'outputs'/'prepared_cache_tests'


def run(adapter):
    env=SearchEnvG2(adapter,SearchConfig(horizon=4),seed=7,record_transitions=False)
    policy=make_policy('detour_memory');actions=[]
    while not env.done:
        obs=env.observe();a=policy.select(obs,env.rng);actions.append(obs['actions'][a])
        _,_,_,_,info=env.step(a);policy.update(info)
    cert,_=env.finalize()
    return dict(route=cert.route,plan=cert.plan,J=cert.cost,reward=env.total_reward,
                work=env.work_used,actions=actions)


class PreparedCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ROOT.mkdir(parents=True,exist_ok=True)
        cls.base=dict(nx=2,ny=2,seed=202611039,mode='regular')

    def directory(self):
        return tempfile.TemporaryDirectory(dir=ROOT)

    def test_all_modes_and_hundred_tables_exact_policy_equivalence(self):
        receipt=[]
        for i,(mode,n) in enumerate([('regular',2),('mixed',2),('blocked',3),('mixed',10)]):
            case=dict(nx=n,ny=n,seed=202611040+i,mode=mode)
            with self.directory() as d:
                fresh,first=prepare_case(case,d);cached,second=prepare_case(case,d)
                self.assertFalse(first['cache_hit']);self.assertTrue(second['cache_hit'])
                self.assertEqual(fresh.problem_id,cached.problem_id)
                for field in ('F','arm_dist'):
                    self.assertTrue(np.array_equal(getattr(fresh.problem,field),getattr(cached.problem,field)))
                    with self.assertRaises(ValueError):getattr(cached.problem,field).setflags(write=True)
                with self.assertRaises(ValueError):cached.problem.nav['costs'].setflags(write=True)
                self.assertEqual(run(fresh),run(cached))
                receipt.append(dict(case=case,cold=first,loaded=second,equivalent=True))
        (ROOT/'equivalence_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')

    def test_corruption_and_wrong_source_fail_closed(self):
        with self.directory() as d:
            _,stats=prepare_case(self.base,d);p=Path(stats['path']);original=p.read_bytes()
            changed=bytearray(original);changed[-10]^=1;p.write_bytes(changed)
            with self.assertRaisesRegex(ValueError,'checksum'):prepare_case(self.base,d)
            p.write_bytes(original)
            with self.assertRaisesRegex(ValueError,'version'):
                _unpack(original,self.base,'wrong-source')

    def test_nan_metadata_and_arrays_are_rejected_even_with_fresh_checksum(self):
        with self.directory() as d:
            adapter,stats=prepare_case(self.base,d);raw=Path(stats['path']).read_bytes()
            offset=len(MAGIC);size=struct.unpack('<I',raw[offset:offset+4])[0];offset+=4
            header=json.loads(raw[offset:offset+size]);payload=raw[offset+size:]
            with np.load(io.BytesIO(payload),allow_pickle=False) as src:arrays={k:src[k] for k in src.files}
            for mutation in ('nan_anchor','nan_D','object_array'):
                changed={k:v.copy() for k,v in arrays.items()}
                if mutation=='nan_anchor':
                    meta=json.loads(changed['metadata'].tobytes());meta['anchor']['cost']=float('nan')
                    changed['metadata']=np.frombuffer(json.dumps(meta,allow_nan=True).encode(),dtype=np.uint8)
                elif mutation=='nan_D':changed['costs'][0,0]=float('nan')
                else:changed['F']=np.asarray([{'malicious':'pickle object not accepted'}],dtype=object)
                stream=io.BytesIO();np.savez_compressed(stream,**changed);new=stream.getvalue()
                h=copy.deepcopy(header);h['payload_sha256']=hashlib.sha256(new).hexdigest();h['payload_bytes']=len(new)
                encoded=json.dumps(h).encode();entry=MAGIC+struct.pack('<I',len(encoded))+encoded+new
                Path(stats['path']).write_bytes(entry)
                with self.assertRaises(ValueError):prepare_case(self.base,d)

    def test_noncacheable_external_case_rejected(self):
        with self.directory() as d:
            with self.assertRaises(ValueError):prepare_case(dict(self.base,service_checker='not-supported'),d)

    def test_simultaneous_writers_publish_one_complete_cache(self):
        with self.directory() as d:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda _:prepare_case(self.base,d),range(2)))
            self.assertEqual(sorted(r[1]['cache_hit'] for r in results),[False,True])
            self.assertEqual(results[0][0].problem_id,results[1][0].problem_id)
            self.assertEqual(len(list(Path(d).glob('*.rc3prep'))),1)
            self.assertFalse(list(Path(d).glob('*.lock')))
            self.assertFalse(list(Path(d).glob('*.tmp')))

    def snapshot(self):
        with self.directory() as d:adapter,_=prepare_case(self.base,d)
        cases=[self.base];config=asdict(SearchConfig());seed=123
        env=SearchEnvG2(adapter,SearchConfig(**config),seed=seed+1,record_transitions=False)
        env.observe()
        state=dict(cases=cases,config=config,seed=seed,order=[0],position=1,
                   rng_state=np.random.default_rng(seed).bit_generator.state,episode=1,episode_reward=0.,
                   active_case=self.base,env_state=env.state_dict(),prepare_s=.1,feature_s=.01,
                   elapsed_episode_s=.2,preparations=1,cache_reuses=0)
        return state,cases,config,seed

    def test_worker_metadata_validates_before_any_commit(self):
        state,cases,config,seed=self.snapshot();original=copy.deepcopy(state)
        validated=validate_worker_snapshot(state,cases,config,seed)
        self.assertEqual(validated,state);self.assertIsNot(validated,state)
        mutations=[('episode_reward',float('nan')),('episode_reward',True),('episode_reward',1.),
                   ('position',0),('position',True),('order',[1]),('prepare_s',float('inf')),
                   ('elapsed_episode_s',-.1),('preparations',-1)]
        for key,value in mutations:
            bad=copy.deepcopy(state);bad[key]=value
            with self.assertRaises(ValueError):validate_worker_snapshot(bad,cases,config,seed)
        bad=copy.deepcopy(state);bad['rng_state']['bit_generator']='invalid'
        with self.assertRaises(ValueError):validate_worker_snapshot(bad,cases,config,seed)
        self.assertEqual(state,original)


if __name__=='__main__':unittest.main()
