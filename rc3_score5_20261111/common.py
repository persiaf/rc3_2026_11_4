from pathlib import Path
from rc3_frozen_reference_pg_20261111.common import read,put,sha,append,archive,csvrow
from rc3_frozen_reference_pg_20261111 import common as prior
HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
BASE='2f6dc749ba3533d18f12254dd2ba50e1d52bcc40'
ROLES=('Score5-PG','Score5-SL')
METHODS=('FrozenSL','Score5-SL','Score5-PG','Fixed3','Detour16')
BINDING=HOME/'SOURCE_BINDING_R02.json'
def admit(stage,authorization=None):
    b=read(BINDING)
    if sha(HOME/'PROTOCOL.json')!=b['protocol_sha256'] or sha(HOME/'SCHEDULE.json')!=b['schedule_sha256']:raise ValueError('Frozen protocol/schedule differs')
    review=read(HOME/'IMPLEMENTATION_REVIEW_R02.json')
    if review['status']!='GO' or review['source_binding_sha256']!=sha(BINDING):raise PermissionError('Engineering implementation review not GO')
    if stage=='engineering':p=read(HOME/'ENGINEERING_AUTHORIZATION_R02.json')
    else:
        if authorization is None:raise PermissionError('Formal science NOT_AUTHORIZED; explicit separate source-bound approval receipt required')
        p=read(authorization)
        if not (HOME/'ENGINEERING_ACCEPTANCE.json').exists() or read(HOME/'ENGINEERING_ACCEPTANCE.json')['status']!='PASS':raise PermissionError('Engineering acceptance required')
    if (p.get('authorized') is not True or p.get('stage')!=stage or p.get('source_binding_sha256')!=sha(BINDING)
            or p.get('protocol_sha256')!=sha(HOME/'PROTOCOL.json') or p.get('schedule_sha256')!=sha(HOME/'SCHEDULE.json')):raise PermissionError('Stage/source authority mismatch')
    return p
class Ledger(prior.Ledger):
    def __init__(self,stage,m,w):
        self.stage=stage;self.path=HOME/('engineering_r02' if stage=='engineering' else stage);self.path.mkdir(exist_ok=False)
        self.start_m=m;self.start_w=w;self.prior=read(HOME/'REPAIR_REGISTRATION_R02.json')['cumulative_prior_wall_s'] if stage=='engineering' else 0.
        self.cap=read(HOME/'PROTOCOL.json')['caps'][stage]
        self.counts=dict(requests=0,requests_completed=0,requests_failed=0,actor_optimizer_attempts=0,actor_optimizer_completed=0,
            critic_optimizer_attempts=0,teacher_replays=0,new_teacher_requests=0,new_scenes=0,navigation_solves=0,original60_reads=0,oldS3_reads=0,
            full_H_calls=0,native_requests=0,feature_states=0,feature_s=0.,input_prepare_s=0.,request_io_s=0.,
            internal_search_counts={},internal_search_inclusive_s={},internal_search_exclusive_s={})
        self.ids=set();self.status='RUNNING'
        self.counts.update(cache_rebuilds=0,teacher_label_replays=0,backward_calls=0)
