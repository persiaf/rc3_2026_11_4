import time
START_M,START_W=time.monotonic(),time.time()
import ast
import subprocess
from .common import HOME,ROOT,BASE,DATA,WITNESS,read,put,sha
def register():
    if (HOME/'SOURCE_BINDING.json').exists():raise FileExistsError('One registration, no resource reset')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    prior=read(ROOT/'rc3_shared_a1_rank_20261111/SOURCE_BINDING.json')
    sources=dict(prior['source_sha256']);artifacts=dict(prior['artifact_sha256'])
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'));sources[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in [DATA,WITNESS,'rc3_shared_a1_rank_20261111/JOINT_FEASIBILITY.json','rc3_shared_a1_rank_20261111/RANK_STAGE_STATUS.json',
        'rc3_shared_a1_rank_20261111/RESOURCE_ACCOUNTING.json','rc3_actor_positive_scale_20261111/FEASIBLE_RANK_WITNESSES.json']:
        artifacts[name]=sha(ROOT/name)
    assert artifacts[DATA]==prior['artifact_sha256'][DATA]
    protocol=dict(schema='fixed-A1-joint-constrained-optimization-v1',base_commit=BASE,seed=11622026,
        goal='Large complex parent DRL training; frozen Actor quickly completes local full constrained planning',
        fixed_data=dict(path=DATA,sha256=artifacts[DATA],states=32,order='Exact saved order, features206/teacher/mask/candidates unchanged',
            initial_tensor_sha256='0fe392e9ecce47abbc957ad9942369ffd647a2b5f427bab7db5d642b55ca0d27',no_teacher_pool_scan_or_reselection=True),
        actor=dict(architecture=[206,64,64,1],original_trainable_parameters=17473,all_parameters_included=True,distances='ON',amplitude=.25,
            initial='Exact original saved zero-output initial_correction; no prior fitted/constructed parameters',no_state_ID_or_lookup=True),
        loss='Original float32 mean32(relu(1e-4 + max_other_legal_logit - teacher_logit)); evaluated with original arithmetic order from exact actual logits',
        local_problem=dict(margin='m_ij=z_teacher-z_j for EVERY other legal candidate',
            primal='min_d,s 0.001/2*||d||_2^2 + mean32(s_i), subject m_ij+J_ij*d+s_i>=1e-4 for all legal alternatives, s_i>=0',
            dual='max_alpha b*alpha - ||J^T alpha||^2/(2*0.001); alpha>=0, each state sum(alpha)<=1/32; d=J^Talpha/0.001',
            regularization=1e-3,solver='Single float64 projected FISTA on exact active-row Gram, fixed adaptive restart; no optimizer sweep/dependency install',
            dual_iterations_max_per_cutting_round=5000,duality_gap_absolute_tolerance=1e-6,gap_check_every=25,
            cutting_rounds_max=16,initial_constraints='Current worst legal alternative of each original state, first-index tie',
            separation='Exact all-original-parameter active margin gradient rows; AD Jacobian-vector product for ALL legal margins. Add largest positive violation per state, fixed order; no candidate truncation',
            full_constraint_violation_tolerance=1e-6,complete='Duality gap<=1e-6 and all legal linearized constraints max violation<=1e-6',
            incomplete='Any dual/cut/numerical separation limit => INCOMPLETE, no direction trial, stop branch; never claim full convergence',
            Jacobian_rank='Effective SVD rank of ACTIVE gradient rows via Gram, threshold max(1e-10,sigma_max*1e-6); report scope/condition, no full-Jacobian rank claim',
            conflict='Initial worst-state gradient cosine/negative pairs descriptive only, not a unique-cause identification'),
        step=dict(outer_max=32,candidate_checks_max=192,backtracking=[1.,.5,.25,.125,.0625,.03125],
            direction='Use all original17473 coordinates, no norm clipping, no hidden freeze; fixed QP regularization only',
            accept='Actual original hinge falls by>1e-9 AND actual full-legal original incremental guard, finite, mask and abs(delta)<=.25000001',
            reject='Exact restore of original float32 parameter SHA after every rejection; charge each actual parameter assignment/candidate before checking',
            predicted='Float64 linearized mean slack under same margin1e-4; not substituted for actual float32 hinge',
            terminal=['32 outer subproblems','all32 strict-top1 and actual hinge<=1e-7','no predicted decrease>1e-9','all six candidates rejected','INCOMPLETE/source/nonfinite/budget stop'],
            quality='Only actual final retained model, never best intermediate: >=26/32 strict margins>1e-5, all applied actual guard checks, no INCOMPLETE/input/source failure',
            acceptance_semantics='Backtracking trials are not Adam128 acceptance opportunities; report actual rates without rewriting historical Adam gates or claiming equal compute'),
        independent_path=dict(positions_max=22,execution='ONLY AFTER optimizer final model saved. Witness bytes first read here; no witness initial/target/search-direction access in optimizer',
            positions='21 original->shared hidden linear-interpolation positions k/20,k0..20 with final layer always zero, then full shared output at position22',
            checks='All parameters finite, actual float32 forwards and original guard on all32 states each position; optimized model SHA unchanged',learning=False,not_a_formal_training_path=True),
        guard=dict(max_step_kl=.001,max_step_probability_change=.02,max_base_total_variation=.1),
        caps=dict(diagnostic=dict(requests=0,actor_optimizer_attempts=192,critic_optimizer_attempts=0,wall_s=1300)),
        total_wall_cap_s=1800,review_wall_reserve_s=500,pre_registration_source_read_charged_upper_s=10,
        accounting='Count outer QP optimizations, every dual projected iteration, exact Jacobian rows/full JVP scans, candidate parameter checks/accepted updates/restores, isolated path assignments and wall/compute separately. Never call this optimizer0.',
        wall_scope='Source reads/registration, fixed data/provenance, Jacobians/QP/candidate forwards/path/fit and full independent review charged. Code authorship/prose/Git-LFS maintenance separate; no old huge CSV audit.',
        history='Keep sharedPASS32/32,rank18/32FAIL,CE14/32 and all previous resources; all models diagnostic_only forbidden deploy/formal initialization',
        stop='Last mechanism screen on these32 states. Failure/INCOMPLETE ends branch, no new losses/configurations/extra learning/online DRL. Success may justify a separately authorized online complete-return test only.',
        zero='Critic,new scenes,navigation,new teacher,native,H,complete planning requests,original60/oldS3 task reads all0')
    put(HOME/'PROTOCOL.json',protocol);artifacts[(HOME/'PROTOCOL.json').relative_to(ROOT).as_posix()]=sha(HOME/'PROTOCOL.json')
    put(HOME/'SOURCE_BINDING.json',dict(schema='shared-constraint-opt-source-v1',base_commit=BASE,source_sha256=sources,artifact_sha256=artifacts,
        environment=prior['environment'],protocol_sha256=sha(HOME/'PROTOCOL.json')))
    for name in ['PROTOCOL_APPROVAL.json','DIAGNOSTIC_AUTHORIZATION.json']:
        put(HOME/name,dict(approved=True,authorized=True,authority='Explicit current user task',scope='Final32-state shared QP diagnostic plus isolated optional path; no online/route request',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')))
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'REGISTRATION_RECEIPT.json',dict(status='FROZEN_BEFORE_FIRST_SCIENTIFIC_COMPUTATION',registration_s=elapsed,
        source_read_charged_upper_s=10,charged_wall_s=elapsed+10,actor_candidates=0,QP_solves=0,requests=0))
    print('Exact protocol/data/source registered before first scientific computation')
def launch():
    if (HOME/'diagnostic_control').exists():raise FileExistsError('No repeated entry/resource reset')
    b=read(HOME/'SOURCE_BINDING.json')
    for name in ['__init__.py','supervisor.py']:
        p=ROOT/'rc3_r04_m1_recovery_20261111'/name;assert sha(p)==b['source_sha256'][p.relative_to(ROOT).as_posix()]
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior=read(HOME/'REGISTRATION_RECEIPT.json')['charged_wall_s']
    receipt=launch_supervised([str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker'],HOME/'diagnostic_control',
        'shared-constraint-opt-20261111-diagnostic-01',wall_limit_s=1300-prior-2,closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,
        entry_monotonic=START_M,entry_wall=START_W,worker_heartbeat_path=HOME/'diagnostic/WORKER_HEARTBEAT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'ENTRY_CONSUMPTION.json',dict(status=receipt['status'],entry_wall_s=elapsed,source_and_registration_charged_s=prior,core_charged_wall_s=elapsed+prior,
        root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print(dict(status=receipt['status'],core_charged_wall_s=elapsed+prior));raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args();register() if a.action=='register' else launch()
