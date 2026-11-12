"""Bounded real Windows read-sharing test; no ML or planning operations."""
import os,time,threading,ctypes,json
from .common import HOME,put,read,Budget

def main():
    protocol=read(HOME/'R03_IO_REPAIR_REGISTRATION.json')
    if protocol['io_test']['wall_limit_s']!=10:raise ValueError('IO-test bound changed')
    budget=Budget('engineering','io-acceptance-01');start=time.perf_counter()
    path=HOME/'acceptance/io_repair/READ_SHARED_ATOMIC.json';put(path,dict(value='before'))
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    k.CreateFileW.restype=ctypes.c_void_p;k.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=k.CreateFileW(str(path),0x80000000,3,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    closed=threading.Event()
    def release():
        time.sleep(.1);k.CloseHandle(handle);closed.set()
    thread=threading.Thread(target=release);thread.start()
    try:put(path,dict(value='after'),True)
    finally:thread.join(timeout=2)
    if not closed.is_set() or read(path)!={'value':'after'}:raise ValueError('Atomic write after shared-reader release failed')
    if time.perf_counter()-start>=10:raise TimeoutError('IO acceptance exceeded registered bound')
    retries=[json.loads(line) for line in (HOME/'ATOMIC_IO_RETRIES.jsonl').read_text(encoding='utf8').splitlines()]
    if not any(e['path']==path.relative_to(HOME.parent).as_posix() and e['winerror'] in (5,32,33) for e in retries):raise ValueError('Real read-sharing conflict was not observed')
    result=dict(status='PASS',real_windows_read_share_conflict_observed=True,bounded_same_write_retry_succeeded=True,
        no_permission_or_ACL_bypass=True,elapsed_s=time.perf_counter()-start,new_requests=0,new_Actor=0,new_forward=0,new_H=0)
    put(HOME/'R03_IO_ACCEPTANCE.json',result);budget.close();print(result)

if __name__=='__main__':main()
