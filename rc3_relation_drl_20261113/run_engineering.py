"""Use the existing Windows deadline supervisor; no visible helper window."""
import argparse,os,sys,time
from .common import ROOT,HOME,read,put

def main(part):
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    ledger=HOME/'engineering/RESOURCE_ACCOUNTING.json'
    prior=read(ledger)['elapsed_s'] if ledger.exists() else 0.
    supervised=sum(read(p)['elapsed_s'] for p in (HOME/'engineering/controls').glob('*/SUPERVISOR_RECEIPT.json'))
    prior=max(prior,supervised)
    controls=HOME/'engineering/controls'/part
    if controls.exists():raise ValueError('This supervised part already attempted; preserve it and explicitly register any repair')
    remaining=7200-prior
    if remaining<30:raise TimeoutError('No engineering wall remaining')
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    command=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.engineering',part]
    receipt=launch_supervised(command,controls,'relation-M1-'+part,wall_limit_s=remaining,closeout_reserve_s=10,
        termination_tolerance_s=2,cwd=ROOT,env=env,entry_monotonic=time.monotonic(),entry_wall=time.time())
    print(receipt,flush=True)
    if receipt['status']!='COMPLETE':raise RuntimeError('Engineering worker failure retained')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['first','second']);main(p.parse_args().part)
