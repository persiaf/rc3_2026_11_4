"""Stop only app-generated readonly diff jobs for THIS repo; never the worker."""
import json
from pathlib import Path
import psutil
import argparse
import time
from rc3_integrated_rppo.common import ROOT,HOME,write_new,now


def clean(orphans_only=False):
    chosen=[];errors=[]
    for p in psutil.process_iter(['pid','name']):
        try:
            if p.info['name'].lower() not in ('git.exe','git'):continue
            args=p.cmdline()
            if not all(v in args for v in ('diff','--no-ext-diff','--raw','--numstat','diff.autoRefreshIndex=false')):continue
            if Path(p.cwd()).resolve()!=ROOT.resolve():continue
            if orphans_only and psutil.pid_exists(p.ppid()):continue
            children=p.children(recursive=True)
            if any(c.name().lower() not in ('git.exe','git','git-lfs.exe','git-lfs') for c in children):continue
            item=dict(pid=p.pid,kind='Codex-app readonly working-tree diff',private_bytes=p.memory_info().private,children=[c.pid for c in children])
            for c in reversed(children):
                try:c.terminate()
                except psutil.Error:pass
            p.terminate();chosen.append(item)
        except (psutil.Error,OSError) as e:errors.append(type(e).__name__)
    exclude=ROOT/'.git/info/exclude';old=exclude.read_bytes() if exclude.exists() else b''
    line=b'/rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/'
    if line not in old:
        exclude.parent.mkdir(parents=True,exist_ok=True)
        with exclude.open('ab') as f:f.write(b'\n# Temporary live-run UI scan exclusion; immutable evidence explicitly force-added and published.\n'+line+b'\n')
    if not chosen:return dict(terminated=0,errors=errors)
    receipt=dict(utc=now(),purpose='Memory pressure from repeated background readonly Git/LFS diff processes',terminated_readonly_diff_jobs=chosen,orphans_only=orphans_only,
        process_match='git.exe, exact readonly diff flags, repository cwd; only Git/LFS descendants; no Python/optimizer/planner process touched',
        errors=errors,raw_research_files_written=0,optimizer_or_budget_reset=False,
        local_scan_exclusion='New live run only; Git force-add verified immutable evidence. Remove temporary line after complete staging.',
        journal_storage='New live journal binary ordinary Git; original committed P0 LFS versions retained. Models/episode archives remain LFS.')
    target=HOME/('GIT_CONTENTION_RECEIPT_'+now().replace(':','').replace('+','_')+'.json');write_new(target,receipt)
    result=dict(terminated=len(chosen),released_private_bytes_estimate=sum(e['private_bytes'] for e in chosen),receipt=str(target),errors=errors)
    print(json.dumps(result),flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch-worker',type=int);a=parser.parse_args()
    if a.watch_worker is None:clean();return
    worker=psutil.Process(a.watch_worker);created=worker.create_time();args=worker.cmdline()
    if 'rc3_integrated_rppo.runner' not in args or '--worker' not in args:raise ValueError('Not this task scientific worker')
    started=time.monotonic()
    while time.monotonic()-started<57600:
        try:
            if not worker.is_running() or worker.create_time()!=created:break
        except psutil.Error:break
        clean(orphans_only=True);time.sleep(10)
    print('SCOPED_READONLY_ORPHAN_MONITOR_STOPPED',flush=True)


if __name__=='__main__':main()
