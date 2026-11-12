import os,time
from .common import ROOT,HOME,read,put,sha
def main():
    if read(HOME/'STATUS.json').get('active_job'):raise RuntimeError('Scientific job is active')
    if (HOME/'FIGURE_RENDER_RECEIPT.json').exists():raise FileExistsError('Keep delivered figures; do not overwrite')
    binding=read(HOME/'FIGURE_SOURCE_BINDING.json')
    for name,value in binding['source_sha256'].items():
        if sha(HOME/name)!=value:raise ValueError('Figure source changed after registration')
    r=read(HOME/'RESOURCE_ACCOUNTING.json');remaining=min(7200-r['wall']['reproduction_s'],345600-r['wall']['total_s'])
    if remaining<30:raise TimeoutError('Original reproduction budget')
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    command=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.figures.generate_all']
    receipt=launch_supervised(command,HOME/'postprocess_controls/figures-01','d5-revision-figures-01',wall_limit_s=remaining-2,
        closeout_reserve_s=20,termination_tolerance_s=2,cwd=ROOT,env=os.environ.copy(),entry_monotonic=time.monotonic(),entry_wall=time.time())
    now=read(HOME/'RESOURCE_ACCOUNTING.json')
    if now['counts']!=r['counts']:raise RuntimeError('Plotting changed scientific counters')
    now['wall']['total_s']+=receipt['elapsed_s'];now['wall']['reproduction_s']+=receipt['elapsed_s'];now['jobs']['post-figures-01']=dict(pool='reproduction',wall_s=receipt['elapsed_s'],status=receipt['status'])
    put(HOME/'RESOURCE_ACCOUNTING.json',now,True);put(HOME/'FIGURE_RENDER_RECEIPT.json',dict(status=receipt['status'],wall_s=receipt['elapsed_s'],new_forward_optimizer_or_planning=0))
    print(dict(status=receipt['status'],wall_s=receipt['elapsed_s']))
    if receipt['status']!='COMPLETE':raise RuntimeError('Figure failure preserved')
if __name__=='__main__':main()
