"""Offline STOP audit on labeled training states; not checkpoint selection."""
import pickle
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json
from .features import collate
from .model import load_policy


def main():
    with Path('rc3_v4/data/data.pkl').open('rb') as f:data=pickle.load(f)
    report=[]
    for variant in ('control','structured'):
        model=load_policy(Path('rc3_v4/training')/(variant+'.npz'))
        counts=dict(states=0,zero_gain_states=0,stop_predictions=0,false_stops=0,missed_zero_gain_stops=0)
        with model.inference_session():
            for row in data['states']:
                logits=model.outputs(collate([row]))[0][0]
                stop=bool(logits[-1]>=np.log(9.))
                zero=not bool((row['gains'][row['mask']]>1e-7).any())
                counts['states']+=1;counts['zero_gain_states']+=int(zero)
                counts['stop_predictions']+=int(stop);counts['false_stops']+=int(stop and not zero)
                counts['missed_zero_gain_stops']+=int(zero and not stop)
        report.append(dict(variant=variant,**counts))
    output=dict(scope='Training-state diagnostic only; no independent STOP generalization claim; threshold .9 fixed before test.',results=report)
    write_json(Path('rc3_v4/training/stop_diagnostic.json'),output);print(output,flush=True)


if __name__=='__main__':main()
