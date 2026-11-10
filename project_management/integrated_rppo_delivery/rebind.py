"""Append a task-authorized semantic-preserving engineering source revision."""
import argparse
import json
import subprocess
from pathlib import Path
from rc3_integrated_rppo.common import HOME,ROOT,OUT,EXP,read,write_new,digest,label,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--reason',required=True);p.add_argument('--artifact',action='append',default=[]);a=p.parse_args()
    pointer=HOME/'EXECUTION_SOURCE.json';old=read(pointer) if pointer.exists() else dict(binding_path=label(HOME/'SOURCE_BINDING.json'),binding_sha256=digest(HOME/'SOURCE_BINDING.json'))
    prior=read(ROOT/old['binding_path']);require_same={n:h for n,h in prior['artifacts'].items()}
    for n,h in require_same.items():
        if digest(ROOT/n)!=h:raise ValueError('Protocol/data authorization artifact changed: '+n)
    sources=prior['source_sha256'].copy();sources.update({label(f):digest(f) for f in HOME.glob('*.py')})
    changed={n:dict(before=prior['source_sha256'].get(n),after=h) for n,h in sources.items() if prior['source_sha256'].get(n)!=h}
    artifacts=prior['artifacts'].copy()
    for name in a.artifact:
        f=(ROOT/name).resolve()
        if not f.is_relative_to(HOME):raise ValueError('Artifact outside new study')
        artifacts[label(f)]=digest(f)
    binding=HOME/f'SOURCE_BINDING_{a.revision}.json';b=dict(prior,source_sha256=sources,artifacts=artifacts,utc=now(),prior_binding=old,engineering_revision=a.revision)
    if (HOME/'GPU_ENGINEERING_REGISTRATION.json').exists():b['environment']=dict(prior['environment'],training_head='cuda_float32',frozen_base_and_critic='cpu_float32',deployment='cpu_float32_D1',TF32=False)
    write_new(binding,b)
    receipt=dict(experiment=EXP,status='TASK_AUTHORIZED_SEMANTIC_PRESERVING_FIX',revision=a.revision,reason=a.reason,
        prior_binding=old,new_binding=dict(path=label(binding),sha256=digest(binding)),changed_sources=changed,
        protocol_changed=False,data_schedule_changed=False,algorithm_changed=False,old_frozen_source_retrieval='Git '+subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf8').strip(),
        authorization='Explicit integrated user task authorizes small engineering corrections in scope; no claim user pre-read generated hash',
        original_authorization_sha256=digest(HOME/'AUTHORIZATION.json'),utc=now())
    write_new(HOME/f'SOURCE_REVISION_{a.revision}.json',receipt)
    if pointer.exists():write_new(HOME/f'EXECUTION_SOURCE_BEFORE_{a.revision}.json',old)
    pointer.write_bytes((json.dumps(dict(binding_path=label(binding),binding_sha256=digest(binding),revision_receipt=label(HOME/f'SOURCE_REVISION_{a.revision}.json'),
        revision_receipt_sha256=digest(HOME/f'SOURCE_REVISION_{a.revision}.json')),ensure_ascii=False,indent=2)+'\n').encode('utf8'))
    print(digest(binding))


if __name__=='__main__':main()
