"""Single explicit recovery of the pre-compute missing-torch import failure.

No registered algorithm, source, configuration, data, or cap is changed. The
same journal continues, original failure bytes are archived, and its elapsed
time is deducted from the same 3600s allowance. Not a general failed-fit replay.
"""
import time
ENTRY=time.perf_counter()
import json
import os
import sys
from pathlib import Path
from .ledger import Budget,canonical,digest,write_new
from . import run

HOME=Path(__file__).resolve().parent


class RecoveryBudget(Budget):
    def __init__(self,folder,registration,*,started):
        self.folder=Path(folder);self.path=self.folder/'events.jsonl'
        prior_path=self.folder/'RESOURCE_ACCOUNTING.json'
        prior=json.loads(prior_path.read_text(encoding='utf-8'))
        receipt=json.loads((HOME/'ENVIRONMENT_RECOVERY_RECEIPT.json').read_text(encoding='utf-8'))
        if (prior['status']!='FAILED' or prior['error']['type']!='ModuleNotFoundError'
                or prior['error']['message']!="No module named 'torch'"
                or any(prior['counts'].values()) or prior['events']!=2
                or digest(prior_path)!=receipt['prior_resource_sha256']
                or digest(__file__)!=receipt['recovery_launcher_sha256']
                or digest(HOME/'PREREGISTRATION.json')!=receipt['registration_sha256']
                or prior['caps']!=registration['caps']):
            raise RuntimeError('Not the exact zero-compute environmental failure; replay refused')
        events=[json.loads(x) for x in self.path.read_bytes().splitlines()]
        tail='0'*64
        import hashlib
        for i,e in enumerate(events):
            body={k:v for k,v in e.items() if k!='sha256'}
            if e['sequence']!=i or e['previous']!=tail or hashlib.sha256(canonical(body)).hexdigest()!=e['sha256']:
                raise RuntimeError('Original journal prefix changed')
            tail=e['sha256']
        if len(events)!=2 or tail!=prior['journal_tail']:
            raise RuntimeError('Dirty work cannot be replayed')
        archive=HOME/'attempts';archive.mkdir(exist_ok=True)
        with (archive/'ATTEMPT01_RESOURCE_ACCOUNTING.json').open('xb') as f:
            f.write(prior_path.read_bytes())
        write_new(archive/'RECOVERY_ACTIVATED.json',dict(receipt_sha256=digest(HOME/'ENVIRONMENT_RECOVERY_RECEIPT.json')))
        self.resource=archive/'ATTEMPT02_RESOURCE_ACCOUNTING.json'
        self.started=ENTRY-prior['wall_s']
        self.caps=registration['caps'];self.counts=dict(prior['counts'])
        self.timings=dict(prior['completed_component_wall_s']);self.tail=tail;self.sequence=len(events)
        self.registration_sha256=prior['registration_sha256']
        self.emit(event='explicit_zero_compute_environment_recovery',
            receipt_sha256=digest(HOME/'ENVIRONMENT_RECOVERY_RECEIPT.json'),
            prior_failed_wall_s=prior['wall_s'],prior_failed_counts=prior['counts'],
            source_algorithm_configuration_changed=False,resource_reset=False)


def main():
    receipt=json.loads((HOME/'ENVIRONMENT_RECOVERY_RECEIPT.json').read_text(encoding='utf-8'))
    if Path(sys.executable).resolve()!=(HOME.parent/receipt['runtime_python']).resolve():
        raise RuntimeError('Use the exact registered virtual-environment interpreter')
    run.Budget=RecoveryBudget
    try:
        run.main()
    finally:
        attempt=HOME/'attempts/ATTEMPT02_RESOURCE_ACCOUNTING.json'
        if attempt.is_file():
            final=json.loads(attempt.read_text(encoding='utf-8'))
            final['prior_failed_attempt']='attempts/ATTEMPT01_RESOURCE_ACCOUNTING.json'
            final['prior_failure_wall_included']=True
            final['recovery_receipt_sha256']=digest(HOME/'ENVIRONMENT_RECOVERY_RECEIPT.json')
            temporary=HOME/'RESOURCE_ACCOUNTING.pending'
            with temporary.open('xb') as f:
                f.write(canonical(final)+b'\n');f.flush();os.fsync(f.fileno())
            os.replace(temporary,HOME/'RESOURCE_ACCOUNTING.json')


if __name__=='__main__':main()
