import time
START_M,START_W=time.monotonic(),time.time()
import argparse
import csv
import traceback
from .common import HOME,ROOT,Ledger,read,put,sha,csvrow

def main(stage):
    ledger=Ledger(stage,START_M,START_W)
    try:
        from .bootstrap import setup
        trace,inputs=setup(ledger)
        if stage=='positive':
            # Only saved-prefix restoration and CE. Any accidental route/H/native
            # dispatch fails here; no terminal callback is reached by the loader.
            from rc3_conservative_refinement import runtime as rt
            from rc3_global_local import terminal
            from rc3_constructive import baselines
            from rc3_global_local_impl_review import endpoint_baselines
            def forbidden(*a,**kw):raise RuntimeError('Planning/native/H forbidden in Actor positive control')
            rt.rollout=rt.sl_route=rt.plan_request=terminal.finish=baselines.postprocess=endpoint_baselines.run_endpoint=forbidden
            from .positive import run
            chosen=run(ledger,trace,inputs)
            if chosen is None:
                d=read(HOME/'TRIGGER_DECISION.json')
                put(HOME/'ONLINE_STAGE_STATUS.json',dict(status='NOT_RUN',reason=d['reason'],selected_learning_rate=None,
                    route_metrics=None,actual_online_execution=False))
                csvrow(HOME/'CONDITIONAL_ONLINE_RESULTS.csv',dict(stage_status='NOT_RUN',endpoint='',size='',candidate='',control='',metric='',physical_parents='',relative_saving='',ci95_low='',ci95_high='',reason=d['reason']))
        else:
            d=read(HOME/'TRIGGER_DECISION.json')
            if not d['triggered']:raise ValueError('Online conditional authority not triggered')
            from .online import run
            from .audit import online
            run(ledger,trace,inputs);online(ledger)
            put(HOME/'ONLINE_STAGE_STATUS.json',dict(status='COMPLETE',selected_learning_rate=d['selected_learning_rate'],
                decision=read(ledger.path/'FINAL_DECISION.json')['status']))
        trace.check_loaded_modules();put(ledger.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary())
        ledger.status='COMPLETE';ledger.event('stage_complete',counts=ledger.counts,input_loader_counts=inputs.counts)
        put(ledger.path/'RESOURCE_ACCOUNTING.json',dict(ledger.snapshot(),input_loader_counts=inputs.counts))
        print({'stage':stage,'status':'COMPLETE','wall_s':ledger.elapsed(),'counts':ledger.counts},flush=True)
    except BaseException as e:
        ledger.status='FAILED_PRESERVED'
        put(ledger.path/'FAILURE.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),resource=ledger.snapshot()))
        put(ledger.path/'RESOURCE_ACCOUNTING.json',ledger.snapshot())
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('positive','online'));main(p.parse_args().stage)
