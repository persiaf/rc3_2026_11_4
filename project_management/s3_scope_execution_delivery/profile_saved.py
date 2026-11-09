"""Descriptive component clocks from saved requests; no new inference or planning."""
import csv
import time
from collections import defaultdict
import numpy as np
from rc3_s3_evaluator.common import ROOT,read,write_new,digest


def profile():
    started=time.perf_counter();out=ROOT/'rc3_s3_scope_results_20261109/results'
    with (out/'REQUEST_METRICS.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json');groups=defaultdict(list)
    for r in rows:groups[r['method'],r['kind']].append(r)
    result=[]
    fields=('model_load_and_verify_s','features_and_global_local_encoding_s','greedy_construction_s',
            'common_H_pipeline_s','native_pipeline_s','cold_navigation_s','navigation_access_s',
            'whole_parent_and_window_s','validation_s','export_s','artifact_write_s','complete_time_s')
    for (m,kind),rr in sorted(groups.items()):
        values={}
        for k in fields:
            v=[float(r[k]) for r in rr if r.get(k)]
            values[k]=dict(observed=len(v),median=float(np.median(v)) if v else None,
                           p95=float(np.quantile(v,.95)) if v else None)
        result.append(dict(method=m,kind=kind,requests=len(rr),components=values))
    write_new(out/'SAVED_COMPONENT_PROFILE.json',dict(components=result,
        checkpoint_bytes={s:{role:v['checkpoint_bytes'] for role,v in models.items()} for s,models in index['models'].items()},
        interpretation='Descriptive measured clocks, nested spans and separate medians must not be summed. '
            'Missing/non-applicable spans remain null. Model loading includes full checkpoint identity validation and '
            'torch.load; no counterfactual stripped-checkpoint latency or new deployment result is inferred.',
        new_scientific_calls=0,wall_s=time.perf_counter()-started,source_helper_sha256=digest(__file__)))
    print('Saved component profile; zero new scientific calls')


if __name__=='__main__':profile()
