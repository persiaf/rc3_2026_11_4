"""One pre-registered synthetic process-control suite. No optimizer/model I/O."""
import time
ENTRY=time.perf_counter()
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .supervisor import launch_supervised,process_exited,_write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True)
    parser.add_argument('--run-id',required=True);args=parser.parse_args()
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    results=[]
    def execute(label,mode,*,fault=False):
        if time.perf_counter()-ENTRY>270:raise RuntimeError('Engineering closeout reserve reached')
        folder=root/label
        command=[sys.executable,'-B','-X','utf8','-m','rc3_r04_m1_recovery_20261111.synthetic_worker',
            '--mode',mode,'--output-dir',str(folder)]
        receipt=launch_supervised(command,folder,args.run_id+'-'+label,
            wall_limit_s=8.,closeout_reserve_s=4.,termination_tolerance_s=2.,inject_receipt_failure=fault)
        expected='COMPLETE' if mode=='normal' else ('WORKER_FAILED' if mode=='receipt_failure' else 'TIMEOUT_TERMINATED')
        ok=(receipt['status']==expected and receipt['root_exit_confirmed']
            and receipt['job_active_processes_at_end']==0 and receipt['assigned_before_resume']
            and not receipt['execution_wall_violation'])
        if receipt['timeout_triggered']:
            ok=ok and receipt['termination_elapsed_s']<=2. and receipt['actual_deadline_overshoot_s']<=2.
        extra={}
        if mode=='child_tree':
            child=json.loads((folder/'OWNED_CHILD.json').read_text(encoding='utf-8'))
            extra['known_child_pid']=child['pid'];extra['known_child_exit_confirmed']=process_exited(child['pid'])
            ok=ok and extra['known_child_exit_confirmed']
        if mode=='unresponsive':ok=ok and (folder/'IGNORING_COOPERATIVE_EXIT.json').is_file()
        if fault:ok=ok and receipt['receipt_write_errors'] and Path(receipt['receipt_path']).is_file()
        row=dict(case=label,status='GO' if ok else 'FAIL',expected_supervisor_status=expected,
                 receipt=receipt,**extra);results.append(row)
        if not ok:raise RuntimeError('Synthetic engineering case failed: '+label)
        return command,folder,receipt
    status='FAIL';error=None
    try:
        command,folder,receipt=execute('normal','normal')
        duplicate=False;before=time.perf_counter()
        try:launch_supervised(command,folder,args.run_id+'-normal',8.,4.,2.)
        except FileExistsError:duplicate=True
        results.append(dict(case='duplicate_run_id',status='GO' if duplicate else 'FAIL',
            repeated_process_spawned=False if duplicate else None,elapsed_s=time.perf_counter()-before))
        if not duplicate:raise RuntimeError('Duplicate run_id was not refused')
        execute('blocked','block')
        execute('unresponsive','unresponsive')
        sibling_folder=root/'independent_synthetic_sibling'
        sibling=subprocess.Popen([sys.executable,'-B','-X','utf8','-m',
            'rc3_r04_m1_recovery_20261111.synthetic_worker','--mode','block',
            '--output-dir',str(sibling_folder)],creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            execute('child_tree','child_tree')
            survived=sibling.poll() is None
            results[-1].update(independent_synthetic_sibling_pid=sibling.pid,
                               independent_synthetic_sibling_survived=survived)
            if not survived:
                results[-1]['status']='FAIL'
                raise RuntimeError('Private test Job termination affected its independent synthetic sibling')
        finally:
            if sibling.poll() is None:sibling.terminate()
            sibling.wait(timeout=2.)
        execute('worker_receipt_failure','receipt_failure')
        execute('supervisor_receipt_failure','normal',fault=True)
        status='GO'
    except BaseException as exc:error=repr(exc)
    result=dict(schema='rc3-m1-supervisor-engineering-results-v1',status=status,error=error,
        actual_suite_wall_s=time.perf_counter()-ENTRY,cap_s=300,termination_tolerance_s=2.,
        cases=results,actor_optimizer_attempts=0,critic_optimizer_attempts=0,model_forwards=0,
        new_scene_navigation_teacher_native_H=0,old_attempt_modified=False,
        scope='Synthetic process interruption/reclamation only; not Critic fitting or G2 Actor/Adam transaction acceptance')
    _write(root/'ENGINEERING_TEST_RESULTS.json',result,exclusive=True)
    print(json.dumps(dict(status=status,wall_s=result['actual_suite_wall_s'],cases=len(results),error=error)),flush=True)
    raise SystemExit(0 if status=='GO' else 1)

if __name__=='__main__':main()
