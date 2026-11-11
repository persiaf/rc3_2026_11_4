"""Standard-library synthetic timeout fixtures; no science/model imports."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True)
    parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    folder=Path(args.output_dir);folder.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        (folder/name).write_text(json.dumps(value)+'\n',encoding='utf-8')
    write('WORKER_STARTED.json',dict(pid=os.getpid(),mode=args.mode,wall=time.time()))
    if args.mode=='normal':
        time.sleep(.1);write('WORKER_FINISHED.json',dict(status='COMPLETE',pid=os.getpid()));return
    if args.mode=='receipt_failure':
        try:raise RuntimeError('Registered synthetic worker failure')
        finally:
            target=folder/'unwritable_receipt';target.mkdir()
            target.write_text('cannot write to directory',encoding='utf-8')
    if args.mode=='child_tree':
        child=subprocess.Popen([sys.executable,'-B','-X','utf8','-m',__spec__.name,
            '--mode','block','--output-dir',str(folder/'child')],creationflags=subprocess.CREATE_NO_WINDOW)
        write('OWNED_CHILD.json',dict(pid=child.pid,parent_pid=os.getpid()))
    if args.mode=='unresponsive':
        for name in ('SIGINT','SIGTERM','SIGBREAK'):
            if hasattr(signal,name):signal.signal(getattr(signal,name),signal.SIG_IGN)
        write('IGNORING_COOPERATIVE_EXIT.json',dict(pid=os.getpid(),signals_ignored=True))
    if args.mode not in ('block','child_tree','unresponsive'):raise ValueError('Unknown fixture')
    while True:time.sleep(60)

if __name__=='__main__':main()
