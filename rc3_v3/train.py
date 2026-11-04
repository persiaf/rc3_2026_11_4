"""Multi-state gain supervision with actual-route validation and three seeds."""
import argparse
import copy
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from rc3_fast.solver import prepare, route_objective
from rc3_baseline.baseline import write_json, write_csv
from .environment import Env, FEATURE_NAMES, FEATURE_SCHEMA
from .model import batch_states, make_policy, save_policy, export_numpy, select_action


def loss_for(model, rows, gain_scale, pos_weight, device):
    x, mask = batch_states([(r['features'], r['mask']) for r in rows], device)
    gain = torch.zeros(mask.shape, device=device)
    for i, row in enumerate(rows):
        gain[i, :len(row['gains'])] = torch.as_tensor(row['gains'], device=device)
    gain = gain.clamp(min=0)*mask
    positive = (gain > 1e-7).float()
    weights = gain.sqrt()
    sums = weights.sum(1)
    target = weights/sums.clamp(min=1e-8)[:, None]
    logits, predicted, binary = model(x, mask)
    logp = torch.log_softmax(logits[:, :-1], -1)
    per_state = -(target*torch.where(mask, logp, 0.)).sum(1)
    ranking = per_state.sum()/(sums > 1e-8).sum().clamp(min=1)
    # Global no-improvement decision, trained without class reweighting. Its
    # output is not in the window softmax: many good windows cannot dilute it.
    stop_loss = F.binary_cross_entropy_with_logits(logits[:, -1], (sums <= 1e-8).float())
    regression = F.smooth_l1_loss(predicted[mask], (gain/gain_scale)[mask], reduction='none')
    regression = (regression*(1+4*positive[mask])).mean()
    classification = F.binary_cross_entropy_with_logits(binary[mask], positive[mask], pos_weight=torch.tensor(pos_weight, device=device))
    loss = ranking+regression+.2*classification+stop_loss
    return loss, dict(rank=float(ranking.detach()), regression=float(regression.detach()), positive=float(classification.detach()), stop=float(stop_loss.detach()))


def validate(model, problems, initial_routes, horizon=8):
    envs = [Env(p, horizon=horizon, initial_route=r) for p, r in zip(problems, initial_routes)]
    stopped = np.zeros(len(envs), bool)
    model.eval()
    while True:
        states = [e.observe() for e in envs]
        active = [i for i, e in enumerate(envs) if not stopped[i] and not e.done and states[i][1].any()]
        if not active:
            break
        x, mask = batch_states([states[i] for i in active], model.mean.device)
        with torch.no_grad():
            logits = model(x, mask)[0].cpu().numpy()
        for row, i in enumerate(active):
            n = len(states[i][1])
            action = select_action(np.r_[logits[row,:n],logits[row,-1]],states[i][1],model.stop_rule,model.stop_threshold)
            if action is None:
                stopped[i] = True
            else:
                envs[i].step(int(action))
    gains = [100*(e.reference_cost-route_objective(e.problem, e.route))/max(e.reference_cost, 1e-8) for e in envs]
    return dict(mean_gain_pct=float(np.mean(gains)), mean_calls=float(np.mean([e.steps for e in envs])),
        gains_pct=gains, stopped=int(stopped.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    p.add_argument('--architectures', nargs='+', choices=['attention', 'mlp'], default=['attention', 'mlp'])
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with (a.data/'data.pkl').open('rb') as f:
        data = pickle.load(f)  # Only our locally generated cache, never uploaded pickle.
    if data['schema'] != FEATURE_SCHEMA or data['feature_names'] != FEATURE_NAMES:
        raise ValueError('training data schema mismatch')
    rows = [r for r in data['states'] if r['mask'].any()]
    flat = np.concatenate([r['features'][r['mask']] for r in rows])
    mean, std = flat.mean(0), flat.std(0).clip(.05)
    gains = np.concatenate([r['gains'][r['mask']] for r in rows])
    positive = gains > 1e-7
    scale = max(float(np.quantile(gains[positive], .9)), .05) if positive.any() else .05
    pos_weight = min(float((~positive).sum()/max(positive.sum(), 1)), 8.)
    started = time.perf_counter()
    validation = [prepare(s) for s in data['scenes']['validation']]
    validation_routes = data['initial_routes']['validation']
    history, runs, total_updates = [], [], 0
    for architecture in a.architectures:
        for seed in a.seeds:
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            model = make_policy(mean, std, architecture).to(a.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            best, best_state, best_epoch, best_stats = -np.inf, None, None, None
            for epoch in range(a.epochs+1):
                losses = []
                if epoch:
                    model.train()
                    for ids in np.array_split(rng.permutation(len(rows)), int(np.ceil(len(rows)/a.batch_size))):
                        loss, parts = loss_for(model, [rows[i] for i in ids], scale, pos_weight, a.device)
                        if not torch.isfinite(loss):
                            raise ValueError('nonfinite training loss')
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                        total_updates += 1
                if epoch % 10 == 0 or epoch == a.epochs:
                    stats = validate(model, validation, validation_routes)
                    score = stats['mean_gain_pct']
                    if score > best+1e-10 or (abs(score-best) <= 1e-10 and stats['mean_calls'] < best_stats['mean_calls']):
                        best, best_epoch, best_stats = score, epoch, stats
                        best_state = copy.deepcopy(model.state_dict())
                    row = dict(architecture=architecture, seed=seed, epoch=epoch,
                        loss=float(np.mean(losses)) if losses else None, val_gain_pct=score,
                        val_calls=stats['mean_calls'], val_stops=stats['stopped'],
                        optimizer_updates=total_updates, elapsed_s=time.perf_counter()-started)
                    history.append(row)
                    write_csv(a.output/'history.csv', history)
                    print(f'{architecture} seed={seed} epoch={epoch} val={score:.4f}% calls={stats["mean_calls"]:.2f} best={best:.4f}%', flush=True)
            model.load_state_dict(best_state)
            name = f'{architecture}_seed{seed}.pt'
            save_policy(a.output/name, model, seed=seed, epoch=best_epoch, horizon=8, gain_scale=scale,
                val_gain_pct=best, val_calls=best_stats['mean_calls'])
            export_numpy(a.output/name, (a.output/name).with_suffix('.npz'))
            runs.append(dict(architecture=architecture, seed=seed, checkpoint=name, epoch=best_epoch,
                val_gain_pct=best, val_calls=best_stats['mean_calls'], validation=best_stats,
                parameter_count=sum(v.numel() for v in model.parameters())))
            write_json(a.output/'progress.json', runs)
    chosen = max(runs, key=lambda r: (r['val_gain_pct'], -r['val_calls']))
    shutil.copy2(a.output/chosen['checkpoint'], a.output/'best.pt')
    shutil.copy2((a.output/chosen['checkpoint']).with_suffix('.npz'), a.output/'best.npz')
    summary = dict(status='complete', schema=FEATURE_SCHEMA, phase='gain_supervised', runs=runs,
        selected=chosen, training_states=len(rows), all_zero_states=sum(not (r['gains'][r['mask']]>1e-7).any() for r in rows),
        training_s=time.perf_counter()-started, total_optimizer_updates=total_updates,
        training_instances=len(data['scenes']['train']), validation_instances=len(validation),
        gain_scale=scale, positive_loss_weight=pos_weight, stop_rule='binary_threshold',stop_threshold=.9, torch=torch.__version__,
        device=torch.cuda.get_device_name() if a.device=='cuda' else a.device,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        data_sha256=hashlib.sha256((a.data/'data.pkl').read_bytes()).hexdigest(),
        checkpoints={n: hashlib.sha256((a.output/n).read_bytes()).hexdigest() for n in ('best.pt','best.npz')})
    write_json(a.output/'training_summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
