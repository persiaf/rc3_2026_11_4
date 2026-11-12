"""One pre-execution correction; prior source registration is retained verbatim."""
import ast,time,json,hashlib
from .common import ROOT,HOME,BASE,RUN,read,put,sha

def main():
    start=time.perf_counter();r=read(HOME/'RESOURCE_ACCOUNTING.json')
    if r['counts']['actor_optimizer_attempts']!=5 or r['counts']['scientific_optimizer_attempts']!=0:raise RuntimeError('Pre-optimizer reseal cannot replace an executed scientific identity')
    old=HOME/'source_revisions/pre_optimizer02';b=read(old/'SCIENTIFIC_SOURCE_BINDING.json')
    for f in HOME.glob('*.py'):
        ast.parse(f.read_text(encoding='utf8'));b['source_sha256'][f.relative_to(ROOT).as_posix()]=sha(f)
    payload=dict(run_id=RUN,base_commit=BASE,revision_artifacts=b['revision_artifacts'],source_sha256=b['source_sha256'],environment=b['environment'])
    ident=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    put(HOME/'SCIENTIFIC_IDENTITY.json',dict(identity=ident,**payload,stable_identity_excludes_downstream_admission_receipts=True),True)
    b.update(scientific_identity=ident,pre_execution_revision='03_worker_and_CLI_admission_consistent',previous_unexecuted_binding_sha256=sha(old/'SCIENTIFIC_SOURCE_BINDING.json'))
    b['artifact_sha256'][str((HOME/'SCIENTIFIC_IDENTITY.json').relative_to(ROOT)).replace('\\','/')]=sha(HOME/'SCIENTIFIC_IDENTITY.json')
    put(HOME/'SCIENTIFIC_SOURCE_BINDING.json',b,True)
    a=read(HOME/'SCIENTIFIC_AUTHORIZATION.json');a.update(scientific_identity=ident,source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING.json'));put(HOME/'SCIENTIFIC_AUTHORIZATION.json',a,True)
    elapsed=time.perf_counter()-start;r['wall']['total_s']+=elapsed;r['wall']['engineering_s']+=elapsed;put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    put(HOME/'PRE_OPTIMIZER_SOURCE_CORRECTION.json',dict(reason='Static review before any new optimizer: atomic checkpoint pointer, full stream timing, consistent worker/CLI admission checks',
        source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING.json'),scientific_identity=ident,old_registration_preserved=['source_revisions/pre_optimizer01','source_revisions/pre_optimizer02'],
        Actor_attempts_added=0,planning_requests_added=0,wall_s=elapsed,model_loss_optimizer_schedule_H_gates_unchanged=True),True)
    print('Production source02 sealed before all new optimizer attempts; earlier registration retained; recipe and gates unchanged.')

if __name__=='__main__':main()
