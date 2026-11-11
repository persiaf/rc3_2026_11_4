"""Windows process-tree deadline supervisor. Standard library only.

The worker starts suspended, enters a private kill-on-close Job Object, then
runs. No process enumeration or PID-pattern kill is used. This independent
process starts its clock before the worker can import Torch or other modules.
"""
from __future__ import annotations
import time
MODULE_ENTRY_MONOTONIC = time.monotonic()
MODULE_ENTRY_WALL = time.time()
import argparse
import ctypes
from ctypes import wintypes as w
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import queue
import subprocess
import sys
import threading
import traceback

POLL_S = .05
HEARTBEAT_S = .5
DEFAULT_TERMINATION_TOLERANCE_S = 5.
STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESTDHANDLES = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
WAIT_OBJECT_0, WAIT_TIMEOUT = 0, 258
KILL_EXIT_CODE = 0xE0420001


def utc(): return datetime.now(timezone.utc).isoformat()


def _write(path, value, exclusive=False):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if exclusive:
        with Path(path).open('x', encoding='utf-8', newline='\n') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
    else:
        temp = Path(path).with_suffix(Path(path).suffix+'.tmp')
        with temp.open('w', encoding='utf-8', newline='\n') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        temp.replace(path)


class STARTUPINFO(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('lpReserved', w.LPWSTR), ('lpDesktop', w.LPWSTR),
        ('lpTitle', w.LPWSTR), ('dwX', w.DWORD), ('dwY', w.DWORD),
        ('dwXSize', w.DWORD), ('dwYSize', w.DWORD), ('dwXCountChars', w.DWORD),
        ('dwYCountChars', w.DWORD), ('dwFillAttribute', w.DWORD), ('dwFlags', w.DWORD),
        ('wShowWindow', w.WORD), ('cbReserved2', w.WORD), ('lpReserved2', ctypes.c_void_p),
        ('hStdInput', w.HANDLE), ('hStdOutput', w.HANDLE), ('hStdError', w.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [('hProcess', w.HANDLE), ('hThread', w.HANDLE),
                ('dwProcessId', w.DWORD), ('dwThreadId', w.DWORD)]


class BASIC_LIMIT(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', w.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', w.DWORD), ('Affinity', ctypes.c_size_t),
                ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount',
        'WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BASIC_LIMIT), ('IoInfo', IO_COUNTERS),
        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class BASIC_ACCOUNTING(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in ('TotalUserTime','TotalKernelTime',
        'ThisPeriodTotalUserTime','ThisPeriodTotalKernelTime')] + [
        ('TotalPageFaultCount', w.DWORD), ('TotalProcesses', w.DWORD),
        ('ActiveProcesses', w.DWORD), ('TotalTerminatedProcesses', w.DWORD)]


def _kernel():
    if os.name != 'nt': raise RuntimeError('This registered supervisor requires Windows Job Objects')
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    def declare(name, args, result):
        function = getattr(k,name);function.argtypes=args;function.restype=result
    declare('CreateJobObjectW', [ctypes.c_void_p,w.LPCWSTR], w.HANDLE)
    declare('SetInformationJobObject',[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD],w.BOOL)
    declare('QueryInformationJobObject',[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.c_void_p],w.BOOL)
    declare('AssignProcessToJobObject',[w.HANDLE,w.HANDLE],w.BOOL)
    declare('TerminateJobObject',[w.HANDLE,w.UINT],w.BOOL)
    declare('CreateProcessW',[w.LPCWSTR,w.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,w.BOOL,
        w.DWORD,ctypes.c_void_p,w.LPCWSTR,ctypes.POINTER(STARTUPINFO),ctypes.POINTER(PROCESS_INFORMATION)],w.BOOL)
    declare('ResumeThread',[w.HANDLE],w.DWORD)
    declare('WaitForSingleObject',[w.HANDLE,w.DWORD],w.DWORD)
    declare('GetExitCodeProcess',[w.HANDLE,ctypes.POINTER(w.DWORD)],w.BOOL)
    declare('CloseHandle',[w.HANDLE],w.BOOL)
    declare('OpenProcess',[w.DWORD,w.BOOL,w.DWORD],w.HANDLE)
    declare('TerminateProcess',[w.HANDLE,w.UINT],w.BOOL)
    declare('SetThreadExecutionState',[w.DWORD],w.DWORD)
    return k


def _require(ok, operation):
    if not ok: raise ctypes.WinError(ctypes.get_last_error(), operation)


def process_exited(pid):
    """Check one known synthetic experiment PID, never enumerate processes."""
    k=_kernel();h=k.OpenProcess(0x00100000|0x1000,False,int(pid))
    if not h:
        error=ctypes.get_last_error()
        if error==87: return True  # This PID no longer names a process.
        raise ctypes.WinError(error, 'OpenProcess for owned synthetic PID')
    try:return k.WaitForSingleObject(h,0)==WAIT_OBJECT_0
    finally:k.CloseHandle(h)


def launch_supervised(command, output_dir, run_id, wall_limit_s,
                      closeout_reserve_s=10., termination_tolerance_s=DEFAULT_TERMINATION_TOLERANCE_S,
                      cwd=None, env=None, inject_receipt_failure=False,
                      entry_monotonic=None, entry_wall=None,
                      worker_heartbeat_path=None):
    """Run exactly once, returning an independent process/resource receipt.

    `wall_limit_s` includes spawn, worker, termination and receipt writing.
    Forced termination starts at wall_limit-closeout_reserve, not after the
    advertised maximum. max(monotonic,UTC wall) retains host-suspend intervals.
    Duplicate run IDs raise FileExistsError before another process is spawned.
    """
    start_m = time.monotonic() if entry_monotonic is None else entry_monotonic
    start_w = time.time() if entry_wall is None else entry_wall
    elapsed=lambda:max(time.monotonic()-start_m,time.time()-start_w)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',run_id): raise ValueError('Invalid exact run_id')
    if not 0 < closeout_reserve_s < wall_limit_s or termination_tolerance_s<=0:
        raise ValueError('Positive deadline, closeout reserve and termination tolerance required')
    if not isinstance(command,list) or not command or any(not isinstance(x,str) for x in command):
        raise ValueError('Explicit command argument list required')
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    claims=STAGE/'run_claims';claims.mkdir(exist_ok=True)
    claim=claims/(run_id+'.json')
    _write(claim,dict(run_id=run_id,created_utc=utc(),supervisor_pid=os.getpid(),
        command=command,output_dir=str(output_dir),wall_limit_s=wall_limit_s,
        closeout_reserve_s=closeout_reserve_s,termination_tolerance_s=termination_tolerance_s),exclusive=True)
    receipt=dict(schema='rc3-process-tree-supervision-v1',run_id=run_id,supervisor_pid=os.getpid(),
        command=command,start_utc=datetime.fromtimestamp(start_w,timezone.utc).isoformat(),
        wall_limit_s=float(wall_limit_s),closeout_reserve_s=float(closeout_reserve_s),
        force_termination_at_s=float(wall_limit_s-closeout_reserve_s),
        termination_tolerance_s=float(termination_tolerance_s),poll_interval_s=POLL_S,
        root_pid=None,worker_exit_code=None,root_exit_confirmed=False,job_active_processes_at_end=None,
        assigned_before_resume=False,timeout_triggered=False,job_termination_requested=False,
        status='SUPERVISOR_FAILED',events=[],receipt_write_errors=[],worker_last_heartbeat=None,
        arbitrary_process_enumeration=False,unrelated_process_termination=False)
    def event(kind,**details):
        receipt['events'].append(dict(event=kind,elapsed_s=elapsed(),utc=utc(),**details))
    k=None;job=None;pi=PROCESS_INFORMATION();out=None;stdin=None;assigned=False;execution_state=None
    terminal_started=None;exit_confirmed_at=None;root_code=None;last_beat=-1.
    heartbeat_queue=queue.Queue(maxsize=1);heartbeat_stop=threading.Event();heartbeat_thread=None
    def heartbeat_writer():
        # Disk/log/worker-heartbeat I/O must never run on the deadline monitor
        # thread. A stuck writer cannot prevent TerminateJobObject below.
        while not heartbeat_stop.is_set():
            try:heartbeat=heartbeat_queue.get(timeout=.1)
            except queue.Empty:continue
            worker_heartbeat=Path(worker_heartbeat_path) if worker_heartbeat_path else output_dir/'WORKER_HEARTBEAT.json'
            try:
                if worker_heartbeat.is_file():
                    saved=json.loads(worker_heartbeat.read_text(encoding='utf-8-sig'))
                    receipt['worker_last_heartbeat']=saved
                    heartbeat['worker_last_heartbeat']=saved
                _write(output_dir/'SUPERVISOR_HEARTBEAT.json',heartbeat)
            except (OSError,ValueError) as exc:
                if len(receipt['receipt_write_errors'])<32:
                    receipt['receipt_write_errors'].append('heartbeat: '+str(exc))
    try:
        k=_kernel()
        execution_state=k.SetThreadExecutionState(0x80000000|0x00000001)
        receipt['temporary_system_sleep_prevention_requested']=bool(execution_state)
        receipt['sleep_prevention_scope']='Temporary thread ES_SYSTEM_REQUIRED; does not override explicit user/system suspension or alter power settings'
        name='RC3_M1_'+hashlib.sha256((str(output_dir)+'|'+run_id).encode()).hexdigest()[:32]
        ctypes.set_last_error(0)
        job=k.CreateJobObjectW(None,name);_require(job,'CreateJobObjectW')
        if ctypes.get_last_error()==183: raise RuntimeError('Private Job Object already exists')
        limits=EXTENDED_LIMIT();limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        _require(k.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)),'SetInformationJobObject')
        event('private_job_created',job_name=name)
        import msvcrt
        out=(output_dir/'WORKER_STDOUT.log').open('xb',buffering=0)
        stdin=open(os.devnull,'rb',buffering=0)
        out_h=msvcrt.get_osfhandle(out.fileno());in_h=msvcrt.get_osfhandle(stdin.fileno())
        os.set_handle_inheritable(out_h,True);os.set_handle_inheritable(in_h,True)
        startup=STARTUPINFO();startup.cb=ctypes.sizeof(startup);startup.dwFlags=STARTF_USESTDHANDLES
        startup.hStdInput=in_h;startup.hStdOutput=out_h;startup.hStdError=out_h
        cmd=ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        merged=dict(os.environ)
        if env is not None:merged.update(env)
        merged.update(RC3_SUPERVISOR_START_UNIX=str(start_w),
            RC3_SUPERVISOR_START_MONOTONIC=str(start_m),RC3_SUPERVISOR_RUN_ID=run_id,
            RC3_SUPERVISOR_WALL_LIMIT=str(wall_limit_s),RC3_SUPERVISOR_OUTPUT_DIR=str(output_dir))
        env_buffer=ctypes.create_unicode_buffer('\0'.join(f'{k}={v}' for k,v in sorted(merged.items()))+'\0\0')
        flags=CREATE_SUSPENDED|CREATE_NO_WINDOW|CREATE_UNICODE_ENVIRONMENT
        _require(k.CreateProcessW(None,cmd,None,None,True,flags,
            ctypes.cast(env_buffer,ctypes.c_void_p) if env_buffer is not None else None,
            str(Path(cwd or ROOT).resolve()),ctypes.byref(startup),ctypes.byref(pi)),'CreateProcessW suspended')
        receipt['root_pid']=int(pi.dwProcessId);event('worker_created_suspended',pid=int(pi.dwProcessId))
        os.set_handle_inheritable(out_h,False);os.set_handle_inheritable(in_h,False)
        _require(k.AssignProcessToJobObject(job,pi.hProcess),'AssignProcessToJobObject before resume')
        assigned=True;receipt['assigned_before_resume']=True
        if elapsed()>=receipt['force_termination_at_s']: raise TimeoutError('Deadline reached before worker resume')
        if k.ResumeThread(pi.hThread)==0xFFFFFFFF:raise ctypes.WinError(ctypes.get_last_error(),'ResumeThread')
        k.CloseHandle(pi.hThread);pi.hThread=None
        event('worker_resumed')
        heartbeat_thread=threading.Thread(target=heartbeat_writer,name='supervisor-heartbeat-io',daemon=True)
        heartbeat_thread.start()
        while True:
            now=elapsed();info=BASIC_ACCOUNTING()
            _require(k.QueryInformationJobObject(job,1,ctypes.byref(info),ctypes.sizeof(info),None),'QueryInformationJobObject')
            root_exited=k.WaitForSingleObject(pi.hProcess,0)==WAIT_OBJECT_0
            if root_exited and root_code is None:
                code=w.DWORD();_require(k.GetExitCodeProcess(pi.hProcess,ctypes.byref(code)),'GetExitCodeProcess')
                root_code=int(code.value);event('root_exit_observed',exit_code=root_code,active_processes=int(info.ActiveProcesses))
            if root_exited and info.ActiveProcesses==0:
                receipt['root_exit_confirmed']=True;receipt['job_active_processes_at_end']=0
                exit_confirmed_at=elapsed();event('entire_experiment_tree_exit_confirmed')
                break
            if terminal_started is None and (now>=receipt['force_termination_at_s'] or root_exited):
                receipt['timeout_triggered']=now>=receipt['force_termination_at_s']
                terminal_started=now
                event('job_termination_requested',reason='deadline' if receipt['timeout_triggered'] else 'root_exited_descendants_remain')
                _require(k.TerminateJobObject(job,KILL_EXIT_CODE),'TerminateJobObject private experiment')
                receipt['job_termination_requested']=True
            if terminal_started is not None and now-terminal_started>termination_tolerance_s:
                raise RuntimeError('Experiment tree did not exit within declared termination tolerance')
            if now-last_beat>=HEARTBEAT_S:
                last_beat=now
                heartbeat=dict(run_id=run_id,supervisor_pid=os.getpid(),root_pid=int(pi.dwProcessId),
                    elapsed_s=now,utc=utc(),active_experiment_processes=int(info.ActiveProcesses),
                    worker_last_heartbeat=receipt['worker_last_heartbeat'])
                try:heartbeat_queue.put_nowait(heartbeat)
                except queue.Full:
                    try:heartbeat_queue.get_nowait()
                    except queue.Empty:pass
                    try:heartbeat_queue.put_nowait(heartbeat)
                    except queue.Full:pass
            time.sleep(POLL_S)
        receipt['worker_exit_code']=root_code
        receipt['status']='TIMEOUT_TERMINATED' if receipt['timeout_triggered'] else ('COMPLETE' if root_code==0 else 'WORKER_FAILED')
    except BaseException as exc:
        receipt['error']=repr(exc);receipt['traceback']=traceback.format_exc();event('supervisor_error',error=repr(exc))
        if k is not None and pi.hProcess:
            try:
                if assigned and job:
                    _require(k.TerminateJobObject(job,KILL_EXIT_CODE),'Exception cleanup private Job Object')
                    receipt['job_termination_requested']=True
                else:
                    # This process was created suspended and never allowed to
                    # run; it has no descendants. Terminate only its owned handle.
                    _require(k.TerminateProcess(pi.hProcess,KILL_EXIT_CODE),'Cleanup unresumed owned worker')
                stop=time.perf_counter()
                while time.perf_counter()-stop<termination_tolerance_s:
                    root_dead=k.WaitForSingleObject(pi.hProcess,0)==WAIT_OBJECT_0
                    active=0
                    if assigned and job:
                        info=BASIC_ACCOUNTING();_require(k.QueryInformationJobObject(job,1,ctypes.byref(info),ctypes.sizeof(info),None),'Cleanup query')
                        active=int(info.ActiveProcesses)
                    if root_dead and active==0:
                        receipt['root_exit_confirmed']=True;receipt['job_active_processes_at_end']=0
                        exit_confirmed_at=elapsed();break
                    time.sleep(POLL_S)
            except BaseException as cleanup_error:receipt['cleanup_error']=repr(cleanup_error)
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=.1)
            receipt['heartbeat_writer_still_blocked_at_close']=heartbeat_thread.is_alive()
        if k is not None:
            if pi.hThread:k.CloseHandle(pi.hThread)
            if pi.hProcess:k.CloseHandle(pi.hProcess)
            if job:k.CloseHandle(job)  # Kill-on-close is a final fail-safe.
            if execution_state:
                receipt['temporary_system_sleep_prevention_restored']=bool(k.SetThreadExecutionState(execution_state))
        if out is not None:out.close()
        if stdin is not None:stdin.close()
    receipt.update(termination_started_s=terminal_started,entire_tree_exit_confirmed_s=exit_confirmed_at,
        termination_elapsed_s=None if terminal_started is None or exit_confirmed_at is None else exit_confirmed_at-terminal_started,
        actual_deadline_overshoot_s=max(0.,(exit_confirmed_at or elapsed())-receipt['force_termination_at_s']),
        monotonic_elapsed_s=time.monotonic()-start_m,utc_wall_elapsed_s=time.time()-start_w,
        elapsed_s=elapsed(),finish_utc=utc())
    receipt['execution_wall_violation']=receipt['elapsed_s']>wall_limit_s
    if receipt['execution_wall_violation']:receipt['status']='EXECUTION_WALL_VIOLATION'
    primary=output_dir/'SUPERVISOR_RECEIPT.json'
    fallback=claims/(run_id+'.terminal.json')
    try:
        if inject_receipt_failure:raise OSError('Registered synthetic primary receipt-write failure')
        _write(primary,receipt,exclusive=True);receipt['receipt_path']=str(primary)
    except OSError as exc:
        receipt['receipt_write_errors'].append('primary: '+str(exc))
        receipt['receipt_path']=str(fallback)
    # The terminal receipt outside the worker output directory is independent
    # of worker cleanup/receipt failures, and includes the first write cost.
    receipt.update(elapsed_s=elapsed(),monotonic_elapsed_s=time.monotonic()-start_m,
                   utc_wall_elapsed_s=time.time()-start_w,finish_utc=utc())
    receipt['execution_wall_violation']=receipt['elapsed_s']>wall_limit_s
    if receipt['execution_wall_violation']:receipt['status']='EXECUTION_WALL_VIOLATION'
    _write(fallback,receipt,exclusive=True)
    if receipt['receipt_path']==str(primary):_write(primary,receipt)
    # Detect a host pause or blocked fsync even during terminal publication.
    # This is after worker/tree exit; keep the full observed time, never clamp
    # it to the advertised cap. The receipt cannot include its own final byte
    # write duration, so the separately observed post-write boundary is explicit.
    receipt['post_write_observed_elapsed_s']=elapsed()
    if receipt['post_write_observed_elapsed_s']>wall_limit_s:
        receipt.update(elapsed_s=receipt['post_write_observed_elapsed_s'],
            execution_wall_violation=True,status='EXECUTION_WALL_VIOLATION',finish_utc=utc())
        _write(fallback,receipt)
        if receipt['receipt_path']==str(primary):_write(primary,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker-module',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wall-limit-s',type=float,required=True)
    parser.add_argument('--closeout-reserve-s',type=float,default=10.)
    parser.add_argument('--termination-tolerance-s',type=float,default=DEFAULT_TERMINATION_TOLERANCE_S)
    parser.add_argument('--worker-heartbeat')
    parser.add_argument('worker_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    extra=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
    command=[sys.executable,'-B','-X','utf8','-m',args.worker_module]+extra
    receipt=launch_supervised(command,args.output_dir,args.run_id,args.wall_limit_s,
        args.closeout_reserve_s,args.termination_tolerance_s,
        entry_monotonic=MODULE_ENTRY_MONOTONIC,entry_wall=MODULE_ENTRY_WALL,
        worker_heartbeat_path=args.worker_heartbeat)
    print(json.dumps(dict(status=receipt['status'],run_id=args.run_id,elapsed_s=receipt['elapsed_s'],
        root_exit_confirmed=receipt['root_exit_confirmed'],job_active_processes_at_end=receipt['job_active_processes_at_end'],
        receipt_path=receipt['receipt_path'])),flush=True)
    raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)


if __name__=='__main__':main()
