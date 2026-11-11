from pathlib import Path
from rc3_frozen_reference_pg_20261111.common import read,put,sha,archive,csvrow
from rc3_frozen_reference_pg_20261111 import common as prior
HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
BASE='32c42f95297250629dc990270a689ff138cc5cbc'
DATA='rc3_actor_positive_scale_20261111/positive/SELECTED_TEACHER_STATES.pt'
class Ledger(prior.Ledger):
    def __init__(self,m,w):
        prior.HOME=HOME
        super().__init__('diagnostic',m,w)
        self.prior=read(HOME/'REGISTRATION_RECEIPT.json')['charged_wall_s']
        self.counts.update(saved_state_file_loads=0,fixed_states=0,shared_construction_attempts=0,
                           diagnostic_target_presentations=0,guard_rejections=0,exact_rollbacks=0)
