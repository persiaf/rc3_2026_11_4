"""Small tabular alternative to the neural selector; validation-only model selection."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

from rc3_baseline.baseline import write_json
from .features import FEATURE_NAMES
from .selector import Selector
from .solver import RECIPES


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trees',type=int,default=64)
    parser.add_argument('--leaves',nargs='+',type=int,default=[8,24,64])
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    args.output.mkdir(parents=True,exist_ok=True)
    train=np.load(args.data/'train.npz',allow_pickle=False)
    val=np.load(args.data/'validation.npz',allow_pickle=False)
    mean,std=train['x'].mean(0),train['x'].std(0).clip(.01)
    x,xv=(train['x']-mean)/std,(val['x']-mean)/std
    actions=[list(a) for k in (1,3) for a in itertools.combinations(range(13),k)]
    costs=np.column_stack([train['costs'][:,a].min(1) for a in actions])
    best_cost=train['costs'].min(1,keepdims=True)
    targets=(costs-best_cost)/np.maximum(best_cost,1e-8)/.05
    best_key,best_model,best_record=None,None,None
    records=[]
    started=time.perf_counter()
    for leaf in args.leaves:
        for seed in args.seeds:
            model=RandomForestRegressor(n_estimators=args.trees,min_samples_leaf=leaf,max_depth=10,
                                        max_features=.8,random_state=seed,n_jobs=1)
            model.fit(x,targets)
            prediction=model.predict(xv)
            stats={}
            for k in (1,3):
                ids=[i for i,a in enumerate(actions) if len(a)==k]
                choices=np.asarray([actions[i] for i in ids])[prediction[:,ids].argmin(1)]
                chosen=np.take_along_axis(val['costs'],choices,axis=1).min(1)
                stats[f'top{k}_regret_pct']=float(np.mean((chosen-val['costs'].min(1))/np.maximum(val['costs'].min(1),1e-8)*100))
            record=dict(seed=seed,min_samples_leaf=leaf,metrics=stats)
            records.append(record)
            key=(stats['top3_regret_pct'],stats['top1_regret_pct'])
            if best_key is None or key<best_key:
                best_key,best_model,best_record=key,model,record
            print(json.dumps(record),flush=True)
    fixed={}
    relative=train['costs']/np.maximum(train['costs'].min(1,keepdims=True),1e-8)
    for k in (1,3):
        fixed[f'top{k}']=list(min(itertools.combinations(range(13),k),key=lambda ids:relative[:,ids].min(1).mean()))
    size=max(e.tree_.node_count for e in best_model.estimators_)
    shape=(args.trees,size)
    arrays={k:np.zeros(shape,dtype=np.int64 if k in ('left','right','feature') else np.float64)
            for k in ('left','right','feature','threshold')}
    arrays['value']=np.zeros((*shape,len(actions)),dtype=np.float32)
    for i,estimator in enumerate(best_model.estimators_):
        tree=estimator.tree_
        for key,name in (('left','children_left'),('right','children_right'),('feature','feature'),('threshold','threshold')):
            arrays[key][i,:tree.node_count]=getattr(tree,name)
        arrays['value'][i,:tree.node_count]=tree.value[:,:,0]
    metadata=dict(architecture='random_forest',members=0,feature_names=FEATURE_NAMES,recipes=RECIPES,
                  action_sets=actions,max_depth=max(e.tree_.max_depth for e in best_model.estimators_),
                  trees=args.trees,selected=best_record,fixed_portfolios=fixed,penalty=.75,search_passes=2,
                  target='relative portfolio cost (best of selected recipes)',
                  checkpoint_criterion='validation top3 regret; tie-break top1')
    np.savez_compressed(args.output/'selector.npz',**arrays,mean=mean,std=std,
                        metadata=np.array(json.dumps(metadata)))
    portable=Selector(args.output/'selector.npz')
    error=float(np.abs(portable.predict(val['x'])-best_model.predict(xv)).max())
    if error>2e-5:
        raise AssertionError(f'forest export parity failed: {error}')
    summary=dict(status='complete',selected=best_record,candidates=records,training_s=time.perf_counter()-started,
                 sklearn=sklearn.__version__,numpy=np.__version__,export_max_abs_error=error,
                 training_instances=len(x),validation_instances=len(xv),metadata=metadata,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 data_hashes={f:hashlib.sha256((args.data/f).read_bytes()).hexdigest() for f in ('train.npz','validation.npz')})
    write_json(args.output/'training_summary.json',summary)
    print(json.dumps(dict(selected=best_record,training_s=summary['training_s'],export_max_abs_error=error)),flush=True)


if __name__=='__main__':
    main()
