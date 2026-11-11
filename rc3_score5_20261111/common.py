from pathlib import Path
from rc3_frozen_reference_pg_20261111.common import read,put,sha,append,archive,csvrow
from rc3_frozen_reference_pg_20261111 import common as prior
HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
BASE='2f6dc749ba3533d18f12254dd2ba50e1d52bcc40'
ROLES=('Score5-PG','Score5-SL')
METHODS=('FrozenSL','Score5-SL','Score5-PG','Fixed3','Detour16')
def admit(stage,authorization=None):
    b=read(HOME/'SOURCE_BINDING.json')
    if sha(HOME/'PROTOCOL.json')!=b['protocol_sha256'] or sha(HOME/'SCHEDULE.json')!=b['schedule_sha256']:raise ValueError('Frozen protocol/schedule differs')
    if read(HOME/'IMPLEMENTATION_REVIEW.json')['status']!='GO':raise PermissionError('Engineering implementation review not GO')
    if stage=='engineering':p=read(HOME/'ENGINEERING_AUTHORIZATION.json')
    else:
        if authorization is None:raise PermissionError('Formal science NOT_AUTHORIZED; explicit separate source-bound approval receipt required')
        p=read(authorization)
        if not (HOME/'engineering/CHECKS.json').exists() or read(HOME/'engineering/CHECKS.json')['status']!='PASS':raise PermissionError('Engineering acceptance required')
    if (p.get('authorized') is not True or p.get('stage')!=stage or p.get('source_binding_sha256')!=sha(HOME/'SOURCE_BINDING.json')
            or p.get('protocol_sha256')!=sha(HOME/'PROTOCOL.json') or p.get('schedule_sha256')!=sha(HOME/'SCHEDULE.json')):raise PermissionError('Stage/source authority mismatch')
    return p
class Ledger(prior.Ledger):
    def __init__(self,stage,m,w):
        prior.HOME=HOME;super().__init__(stage,m,w)
        self.prior=read(HOME/'REGISTRATION_RECEIPT.json')['charged_wall_s'] if stage=='engineering' else 0.
        self.counts.update(cache_rebuilds=0,teacher_label_replays=0,backward_calls=0)
