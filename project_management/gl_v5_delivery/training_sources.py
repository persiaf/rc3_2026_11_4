"""Append explicit caller identities for model-independent saved H plans."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];HOME=ROOT/'rc3_gl_v5_20261109'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,m):
    if not x:raise ValueError(m)


def main():
    folder=HOME/'runs/SCIENCE/G128';events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
    before={e['update']:e['weight_tensor_before'] for e in events if e['event']=='update_start'}
    need(read(folder/'COMPLETE.json')['ledger']['terminal']['RL']['terminal_cache_hits']==0,'This exact run source join expects no sample cache hits')
    records=[]
    for u in range(1,129):
        d=read(folder/'updates'/f'update_{u:04}.json')
        parameters=folder/('checkpoints/update_0000.pt' if u==1 else f'gradients_and_weights/update_{u-1:04}.pt')
        need(parameters.exists(),'Actual sampling parameters missing')
        param_path=parameters.relative_to(ROOT).as_posix();param_sha=sha(parameters)
        for slot,i in enumerate(d['instances']):
            for row in i['rows']:
                n=row['terminal_request'];p=folder/'terminal_plans'/f'RL_{n:06}.json';h=read(p)
                need(h['problem']['identity']==i['problem'] and h['result']['raw_route']==row['raw_route'] and h['result']['J']==row['J'],'Terminal/trajectory identity conflict')
                records.append(dict(query_kind='G_sample',request=n,study_method='G128',checkpoint_role='G128',
                    training_run=folder.relative_to(ROOT).as_posix(),actual_parameter_version=u-1,fixed_endpoint_update=128,
                    tensor_sha256=before[u],parameter_artifact=param_path,parameter_artifact_sha256=param_sha,
                    actor_file=None,actor_sha256=None,actor_note='Training uses current mutable parameters; no inference actor exists at this intermediate version. Exact full parameter artifact and tensor identity supplied instead.',
                    update=u,slot=slot,trajectory_role=row['role'],problem=i['problem'],
                    terminal_artifact=p.relative_to(ROOT).as_posix(),terminal_artifact_sha256=sha(p)))
            p=ROOT/i['reference']['artifact_path'];x=read(p)
            need(x['problem']==i['problem'] and sha(p)==i['reference']['artifact_sha256'],'Wrong reference input/artifact')
            records.append(dict(query_kind='frozen_SL_reference',request=x['request'],**x['model_identity'],update=u,slot=slot,
                actual_parameter_version=256,problem=i['problem'],terminal_artifact=p.relative_to(ROOT).as_posix(),terminal_artifact_sha256=sha(p)))
    need(sum(x['query_kind']=='G_sample' for x in records)==1536 and sum(x['query_kind']=='frozen_SL_reference' for x in records)==512,'Wrong identity index counts')
    output=HOME/'TRAINING_REQUEST_IDENTITIES.json'
    with output.open('x',encoding='utf8') as f:
        json.dump(dict(scope='Append-only explicit caller identity for all2048 paid training requests; no raw plan/old metadata rewrite',
            notes='H terminal plans are functions of input/raw route/H, so their own schema is model-independent. This index binds each caller to the actual pre-update G parameters or frozen SL actor. Final432 evaluations already contain explicit actor identities in every row.',
            records=records,new_scientific_calls=0,original_results_changed=0),f,ensure_ascii=False,indent=2);f.write('\n')
    print(dict(training_request_identities=len(records),source_index_sha256=sha(output)))


if __name__=='__main__':main()
