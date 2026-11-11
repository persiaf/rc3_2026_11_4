from pathlib import Path
from rc3_frozen_reference_pg_20261111.common import read,put,sha,archive,csvrow
from rc3_frozen_reference_pg_20261111 import common as prior
HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
BASE='6250a3f75e7e36ae81c2aadc4346c7c55f580173'
DATA='rc3_actor_positive_scale_20261111/positive/SELECTED_TEACHER_STATES.pt'
WITNESS='rc3_shared_a1_rank_20261111/diagnostic/SHARED_WITNESS.diagnostic.pt'
class Ledger(prior.Ledger):
    def __init__(self,m,w):
        prior.HOME=HOME
        super().__init__('diagnostic',m,w)
        self.prior=read(HOME/'REGISTRATION_RECEIPT.json')['charged_wall_s']
        self.counts.update(subproblem_optimizations=0,cutting_rounds=0,dual_projected_iterations=0,
            exact_Jacobian_rows=0,all_constraint_JVP_checks=0,actor_parameter_candidate_checks=0,
            actor_applied_updates=0,actor_rejected_candidates=0,exact_restores=0,
            path_parameter_assignments=0,saved_state_file_loads=0,shared_witness_loads_after_optimization=0)
