import hashlib
import json
from pathlib import Path
from rc3_frozen_reference_pg_20261111.common import put,read,sha,append,archive,csvrow
from rc3_frozen_reference_pg_20261111 import common as prior

HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
BASE='664de37687c4db603169a8dd0c105c27481cd07a'
RATES=(3e-5,3e-4,1e-3)
TABLE='rc3_r04_m2_20261111/attempts/02/FIT_ONLY_COMPONENT_ANCHORS.json'

class Ledger(prior.Ledger):
    def __init__(self,stage,start_m,start_w):
        prior.HOME=HOME
        super().__init__(stage,start_m,start_w)
        if stage=='positive':self.prior=read(HOME/'REGISTRATION_RECEIPT.json')['wall_s']
        self.counts.update(existing_teacher_label_reads=0,restored_teacher_states=0,
            witness_parameterizations=0,witness_verified=0,diagnostic_optimizer_attempts=0)

def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def identity(v):return hashlib.sha256(canonical(v)).hexdigest()
