"""Supervised saved-output audit/table commands, with non-refundable wall."""
import sys,time,os,argparse
from .common import ROOT,HOME,read,put,sha

def main(module):
    if read(HOME/'STATUS.json').get('active_job'):raise RuntimeError('Scientific worker remains active')
    binding=read(HOME/'POSTPROCESSING_SOURCE_BINDING_R02.json')
    for name,value in binding['source_sha256'].items():
        if sha(HOME/name)!=value:raise ValueError('Saved-output source changed after registration')
    resource=read(HOME/'RESOURCE_ACCOUNTING.json');remaining=min(7200-resource['wall']['reproduction_s'],345600-resource['wall']['total_s'])
    if remaining<30:raise TimeoutError('Original reproduction pool exhausted')
    controls=HOME/'postprocess_controls';controls.mkdir(exist_ok=True)
    job=module+'-'+str(len(list(controls.glob(module+'-*')))+1).zfill(2)
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    command=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.'+module]
    put(controls/(job+'.REGISTRATION.json'),dict(module=module,source_sha256=binding['source_sha256'][module+'.py'],new_forward_optimizer_or_planning=0,
        remaining_original_pool_s=remaining,scientific_source_or_models_changed=False))
    receipt=launch_supervised(command,controls/job,'d5-revision-post-'+job,wall_limit_s=remaining-2,closeout_reserve_s=20,
        termination_tolerance_s=2,cwd=ROOT,env=os.environ.copy(),entry_monotonic=time.monotonic(),entry_wall=time.time())
    current=read(HOME/'RESOURCE_ACCOUNTING.json')
    if current['counts']!=resource['counts']:raise RuntimeError('Read-only postprocessing changed scientific consumption counters')
    elapsed=receipt['elapsed_s'];current['wall']['reproduction_s']+=elapsed;current['wall']['total_s']+=elapsed
    current['jobs']['post-'+job]=dict(pool='reproduction',wall_s=elapsed,status=receipt['status']);put(HOME/'RESOURCE_ACCOUNTING.json',current,True)
    print(dict(job=job,status=receipt['status'],actual_full_process_tree_wall_s=elapsed),flush=True)
    if receipt['status']!='COMPLETE':raise RuntimeError('Postprocessing failure preserved in actual supervisor receipt: '+job)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('module',choices=['audit_saved_results','build_saved_tables']);main(p.parse_args().module)
