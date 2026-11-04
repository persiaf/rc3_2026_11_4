"""Train cost-regression selectors using train/validation only; export NumPy weights."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn

from rc3_baseline.baseline import write_json, write_csv
from .features import FEATURE_NAMES
from .selector import Selector
from .solver import RECIPES


def fixed_portfolio(costs, k):
    normalized = costs/np.maximum(costs.min(1, keepdims=True), 1e-8)
    return list(min(itertools.combinations(range(len(RECIPES)), k),
                    key=lambda ids: normalized[:, ids].min(1).mean()))


def metrics(predictions, costs, action_sets=None):
    order = np.argsort(predictions, axis=1, kind='stable')
    best = costs.min(1)
    result = {}
    for k in (1, 3):
        if action_sets is None:
            selected = np.take_along_axis(costs, order[:, :k], axis=1).min(1)
        else:
            ids = [i for i, a in enumerate(action_sets) if len(a) == k]
            actions = np.asarray([action_sets[i] for i in ids])
            chosen = actions[predictions[:, ids].argmin(1)]
            selected = np.take_along_axis(costs, chosen, axis=1).min(1)
        result[f'top{k}_regret_pct'] = float(np.mean((selected-best)/np.maximum(best,1e-8)*100))
        result[f'top{k}_objective'] = float(selected.mean())
    return result


def network(hidden, outputs=len(RECIPES)):
    return nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


def export_numpy(path, states, mean, std, metadata):
    arrays = dict(mean=mean.astype(np.float32), std=std.astype(np.float32),
                  metadata=np.array(json.dumps(dict(metadata, members=len(states)), ensure_ascii=False)))
    for k, state in enumerate(states):
        for i, layer in enumerate((0, 2, 4)):
            arrays[f'm{k}_w{i}'] = state[f'{layer}.weight'].cpu().numpy()
            arrays[f'm{k}_b{i}'] = state[f'{layer}.bias'].cpu().numpy()
    np.savez_compressed(path, **arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--portfolio-targets', action='store_true', help='predict costs of all singletons and 3-recipe portfolios directly')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch size must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['feature_names'] != FEATURE_NAMES or manifest['recipes'] != RECIPES:
        raise ValueError('incomplete or incompatible dataset')
    # Intentionally never load test.npz or test_cases.json during training/selection.
    train = np.load(args.data/'train.npz', allow_pickle=False)
    val = np.load(args.data/'validation.npz', allow_pickle=False)
    mean, std = train['x'].mean(0), train['x'].std(0).clip(.01)
    x = torch.tensor((train['x']-mean)/std, device=args.device)
    xv = torch.tensor((val['x']-mean)/std, device=args.device)
    action_sets = ([list(a) for k in (1,3) for a in itertools.combinations(range(len(RECIPES)),k)]
                   if args.portfolio_targets else None)
    costs = (np.column_stack([train['costs'][:, a].min(1) for a in action_sets])
             if action_sets else train['costs'])
    # Direct portfolio targets train the actual best-of-k decision, rather than
    # assuming the k lowest independent expectations form a diverse portfolio.
    best_cost = train['costs'].min(1, keepdims=True)
    target = ((costs-best_cost)/np.maximum(best_cost, 1e-8)/.05).astype(np.float32)
    target = torch.tensor(target, device=args.device)
    groups = [list(range(13)),list(range(13,len(action_sets)))] if action_sets else [list(range(13))]
    soft_targets = [torch.softmax(-target[:, group]/.6, dim=-1) for group in groups]
    fixed = {f'top{k}': fixed_portfolio(train['costs'], k) for k in (1, 3)}
    history, candidates, states = [], [], []
    started = time.perf_counter()
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = network(args.hidden, target.shape[1]).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate*.05)
        best_key, best_state, best_predictions, best_epoch = None, None, None, None
        for epoch in range(args.epochs+1):
            loss_value = 0.
            if epoch:
                model.train()
                for ids in torch.randperm(len(x), device=args.device).split(args.batch_size):
                    predictions = model(x[ids])
                    mse = sum(nn.functional.smooth_l1_loss(predictions[:, group], target[ids][:, group]) for group in groups)/len(groups)
                    ranking = sum(-(soft[ids]*torch.log_softmax(-predictions[:, group]/.6, dim=-1)).sum(-1).mean()
                                  for group, soft in zip(groups,soft_targets))/len(groups)
                    loss = mse+.2*ranking
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    loss_value += float(loss.detach())*len(ids)/len(x)
                scheduler.step()
            model.eval()
            with torch.no_grad():
                predictions = model(xv).cpu().numpy()
            stats = metrics(predictions, val['costs'], action_sets)
            # Predeclared checkpoint criterion: top-3 regret, with top-1 as tie breaker.
            key = (stats['top3_regret_pct'], stats['top1_regret_pct'])
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_predictions = predictions.copy()
            history.append(dict(seed=seed, epoch=epoch, loss=loss_value if epoch else None,
                                elapsed_s=time.perf_counter()-started, **stats))
            if epoch % 100 == 0:
                print(f'seed={seed} epoch={epoch}/{args.epochs} loss={loss_value:.4f} '
                      f'val regret top1={stats["top1_regret_pct"]:.3f}% top3={stats["top3_regret_pct"]:.3f}%', flush=True)
        states.append(best_state)
        candidates.append(dict(name=f'seed{seed}', member_ids=[len(states)-1], epoch=best_epoch,
                               metrics=metrics(best_predictions, val['costs'], action_sets), predictions=best_predictions))
        torch.save(dict(state_dict=best_state, hidden=args.hidden, seed=seed, epoch=best_epoch,
                        mean=mean.tolist(), std=std.tolist(), feature_names=FEATURE_NAMES, recipes=RECIPES,
                        action_sets=action_sets),
                   args.output/f'seed{seed}_best.pt')
    ensemble = np.mean([c['predictions'] for c in candidates], axis=0)
    candidates.append(dict(name='ensemble', member_ids=list(range(len(states))), epoch=None,
                           metrics=metrics(ensemble,val['costs'],action_sets), predictions=ensemble))
    chosen = min(candidates, key=lambda c: (c['metrics']['top3_regret_pct'], c['metrics']['top1_regret_pct']))
    metadata = dict(feature_names=FEATURE_NAMES, recipes=RECIPES, selected=chosen['name'],
                    selected_member_ids=chosen['member_ids'], fixed_portfolios=fixed,
                    checkpoint_criterion='validation top3 relative regret; tie-break top1',
                    target='relative recipe cost; lower is better', penalty=.75, search_passes=2)
    if action_sets:
        metadata.update(action_sets=action_sets, target='relative portfolio cost (best of the selected recipes); lower is better')
    export_numpy(args.output/'selector.npz',[states[i] for i in chosen['member_ids']],mean,std,metadata)
    portable = Selector(args.output/'selector.npz').predict(val['x'])
    max_error = float(np.abs(portable-chosen['predictions']).max())
    if max_error > 2e-5:
        raise AssertionError(f'NumPy export parity failed: {max_error}')
    fixed_validation = {key: float(np.mean((val['costs'][:,ids].min(1)-val['costs'].min(1))/np.maximum(val['costs'].min(1), 1e-8)*100))
                        for key, ids in fixed.items()}
    summary = dict(status='complete', selected=chosen['name'], epochs_per_seed=args.epochs, seeds=args.seeds,
                   training_instances=len(x), validation_instances=len(xv), device=args.device,
                   gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                   python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   hidden=args.hidden, optimizer_updates=args.epochs*int(np.ceil(len(x)/args.batch_size))*len(args.seeds),
                   training_s=time.perf_counter()-started, fixed_portfolios=fixed,
                   fixed_validation_regret_pct=fixed_validation, export_max_abs_error=max_error,
                   candidates=[{k:v for k,v in c.items() if k!='predictions'} for c in candidates],
                   data_hashes={f:hashlib.sha256((args.data/f).read_bytes()).hexdigest()
                                for f in ('manifest.json','train.npz','validation.npz')},
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), metadata=metadata)
    write_json(args.output/'training_summary.json',summary)
    write_csv(args.output/'history.csv',history)
    print(json.dumps({k:summary[k] for k in ('selected','optimizer_updates','training_s','fixed_validation_regret_pct','export_max_abs_error')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
