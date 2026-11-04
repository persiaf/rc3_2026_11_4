"""Check vectorization is byte-compatible with all frozen structured examples."""
import pickle
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json
from rc3_fast.solver import prepare
from .features import local_graphs

def main():
    with Path('rc3_v4/data/data.pkl').open('rb') as f:data=pickle.load(f)
    states=0;arrays=0;maximum=0.;identical=True
    for sid,scene in enumerate(data['scenes']['train']):
        problem=prepare(scene)
        for row in [r for r in data['states'] if r['scene_id']==sid]:
            graphs=local_graphs(problem,list(map(int,row['route'])))
            for old,new in zip(row['graphs'],graphs):
                for key in old:
                    error=float(np.max(abs(old[key].astype(float)-new[key].astype(float))))
                    maximum=max(maximum,error);identical &= np.array_equal(old[key],new[key]);arrays+=1
                    if error>1e-7:raise AssertionError('feature semantics changed')
            states+=1
    out=dict(passed=True,states=states,arrays=arrays,bit_identical=bool(identical),max_absolute_error=maximum)
    write_json(Path('rc3_v4/feature_equivalence.json'),out);print(out,flush=True)

if __name__=='__main__':main()
