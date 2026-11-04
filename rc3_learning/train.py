"""Behavior cloning followed by REINFORCE with a frozen greedy rollout baseline."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
from scipy.stats import ttest_rel
import torch

from rc3_baseline import baseline
from .data import collate, export_instance, make_instance
from .env import ReachabilityCoverRoutingEnv
from .evaluate import benchmark, greedy_costs, load_policy, synchronize
from .model import AttentionPolicy, rollout


def checkpoint(path, policy, args, scene_config, phase, epoch, val_cost, optimizer=None):
    data = dict(model_config=policy.config, model_state=policy.state_dict(),
                scene_config=scene_config, stop_penalty=args.stop_penalty,
                phase=phase, epoch=epoch, validation_mean_cost=val_cost,
                seed=args.seed, format_version=1)
    if optimizer is not None:
        data['optimizer_state'] = optimizer.state_dict()
    torch.save(data, path)


def train_epoch(policy, frozen, env, instances, optimizer, args, rng, device, phase):
    policy.train()
    order = rng.permutation(len(instances))
    losses, norms, updates = [], [], 0
    for start in range(0, len(order), args.batch_size):
        batch = collate([instances[i] for i in order[start:start+args.batch_size]], device)
        optimizer.zero_grad(set_to_none=True)
        result = rollout(policy, env, batch, 'teacher' if phase == 'bc' else 'sample')
        if not result['state']['completed'].all() or result['state']['failed'].any():
            raise RuntimeError('training episode did not complete; refusing to optimize a partial route')
        if phase == 'bc':
            loss = -result['log_likelihood'].sum()/result['state']['service_count'].sum().clamp_min(1)
        else:
            with torch.no_grad():
                reference = rollout(frozen, env, batch, 'greedy')
            advantage = (result['normalized_cost']-reference['normalized_cost']).detach()
            # Cost minimization: positive (worse) advantage decreases sampled probability.
            loss = (advantage*result['log_likelihood']).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        if not loss.requires_grad:  # Entire batch consists of empty-task scenes.
            continue
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        norms.append(float(norm.detach()))
        updates += 1
    return dict(loss=float(np.mean(losses)) if losses else 0., updates=updates,
                max_gradient_norm=max(norms, default=0.))


def baseline_update_decision(candidate, reference):
    difference = np.asarray(candidate)-np.asarray(reference)
    if len(difference) < 2 or difference.mean() >= -1e-8:
        return False, None
    if np.max(np.abs(difference-difference.mean())) < 1e-10:
        pvalue = 0.0
    else:
        pvalue = float(ttest_rel(candidate, reference, alternative='less').pvalue)
    return bool(math.isfinite(pvalue) and pvalue < 0.05), pvalue if math.isfinite(pvalue) else None


def save_training_plot(history, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output/'mplcache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bc = [r for r in history if r['phase'] == 'bc']
    axes[0].plot([r['epoch'] for r in bc], [r['loss'] for r in bc], marker='o')
    axes[0].set(xlabel='BC epoch', ylabel='Teacher cross-entropy per decision', title='Imitation training')
    axes[1].plot(range(len(history)), [r['val_ratio_to_teacher'] for r in history], marker='.')
    axes[1].axhline(1., color='gray', linestyle='--', label='Replayed teacher')
    axes[1].axvline(len(bc), color='#bb7722', linestyle=':', label='BC / RL boundary')
    axes[1].set(xlabel='Validation checkpoint (initial, BC, RL)', ylabel='Mean cost / teacher cost',
                title='Held-out validation; lower is better')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output/'training_curves.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-scales', nargs='+', type=baseline.scale_argument, default=[(2, 2), (3, 2)])
    parser.add_argument('--eval-scales', nargs='+', type=baseline.scale_argument, default=[(2, 2), (3, 2), (4, 3)])
    parser.add_argument('--train-instances', type=baseline.positive_int, default=64)
    parser.add_argument('--val-instances', type=baseline.positive_int, default=16)
    parser.add_argument('--test-per-scale', type=baseline.positive_int, default=4)
    parser.add_argument('--bc-epochs', type=int, default=15)
    parser.add_argument('--rl-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=baseline.positive_int, default=8)
    parser.add_argument('--embed-dim', type=baseline.positive_int, default=128)
    parser.add_argument('--layers', type=baseline.positive_int, default=2)
    parser.add_argument('--heads', type=baseline.positive_int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--stop-penalty', type=float, default=0.75)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--init-checkpoint', type=Path, help='initialize weights, with a fresh optimizer')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.seed < 0 or min(args.bc_epochs, args.rl_epochs) < 0 or args.bc_epochs+args.rl_epochs == 0:
        parser.error('seed/epochs must be nonnegative; at least one training epoch is required')
    if args.embed_dim % args.heads or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('embed_dim must be divisible by heads; lr must be finite and positive')
    if not math.isfinite(args.stop_penalty) or args.stop_penalty < 0:
        parser.error('stop penalty must be finite and nonnegative')
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
        if not isinstance(config, dict) or set(config)-{'geometry', 'robot'}:
            raise ValueError('config supports only geometry and robot')
        geometry, robot = baseline.scene_parameters(config.get('geometry'), config.get('robot'))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    scene_config = dict(geometry=geometry, robot=robot)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available; use --device cpu')
    started = datetime.now(timezone(timedelta(hours=8)))
    output = (args.output or Path(__file__).parent/f'run_{started:%Y%m%d_%H%M%S}').resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('output must be a new or empty directory')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    manifest = dict(status='running', started_at=started.isoformat(), device=device,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    scene_config=scene_config, seed=args.seed,
                    framework='RL4COEnvBase + custom PyTorch attention/BC/REINFORCE loop',
                    source_sha256={str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder in (Path(__file__).parent, Path(__file__).parent.parent/'rc3_baseline')
                                   for p in sorted(folder.glob('*.py'))})
    baseline.write_json(output/'run_manifest.json', manifest)
    baseline.write_json(output/'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
                        device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                        packages={name: version(name) for name in ('torch', 'rl4co', 'torchrl', 'tensordict', 'numpy', 'scipy', 'matplotlib')}))
    total_instances = args.train_instances+args.val_instances+args.test_per_scale*len(args.eval_scales)
    seeds = iter(rng.choice(2**31-1, size=total_instances, replace=False).tolist())
    splits, split_records = {}, {}
    for split, count, scales in (('train', args.train_instances, args.train_scales),
                                  ('validation', args.val_instances, args.train_scales),
                                  ('test', args.test_per_scale*len(args.eval_scales), args.eval_scales)):
        items, records = [], []
        folder = output/'teachers'/split
        folder.mkdir(parents=True)
        for index in range(count):
            nx, ny = scales[index % len(scales)] if split != 'test' else scales[index//args.test_per_scale]
            seed = next(seeds)
            item = make_instance(nx, ny, seed, penalty=args.stop_penalty, **scene_config)
            items.append(item)
            records.append(dict(nx=nx, ny=ny, seed=seed, tasks=len(item.scene['tasks']),
                                original_stops=item.teacher_original_stops,
                                replayed_stops=len(item.teacher['visit_order_candidate_ids']),
                                teacher_cost=item.teacher['objective_length_plus_stops_m']))
            baseline.write_json(folder/f'scene_plan_{nx}x{ny}_seed{seed}.json',
                                export_instance(item, {'teacher_replayed': item.teacher}))
        splits[split], split_records[split] = items, records
        print(f'{split}: {count} instances; {sum(r["original_stops"]-r["replayed_stops"] for r in records)} redundant teacher stops removed', flush=True)
    baseline.write_json(output/'dataset_manifest.json', split_records)
    env = ReachabilityCoverRoutingEnv(device, seed=args.seed)
    if args.init_checkpoint:
        policy, loaded = load_policy(args.init_checkpoint, device)
        if loaded['scene_config'] != scene_config or loaded['stop_penalty'] != args.stop_penalty:
            parser.error('initial checkpoint parameters differ from this run')
    else:
        policy = AttentionPolicy(args.embed_dim, args.layers, args.heads).to(device)
    initial_weights = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    validation = splits['validation']
    ref = np.asarray([i.teacher['objective_length_plus_stops_m'] for i in validation]).clip(1.)
    val_costs = greedy_costs(policy, env, validation, args.batch_size, device)
    best_mean = float(val_costs.mean())
    history = [dict(phase='initial', epoch=0, loss=None, updates=0, max_gradient_norm=0.,
                    val_cost=best_mean, val_ratio_to_teacher=float((val_costs/ref).mean()),
                    baseline_updated=False, baseline_pvalue=None, epoch_s=0.)]
    checkpoint(output/'best.pt', policy, args, scene_config, 'initial', 0, best_mean)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    frozen = None
    phase_updates = dict(bc=0, rl=0)
    rl_start_weights = None
    for phase, epochs in (('bc', args.bc_epochs), ('rl', args.rl_epochs)):
        if phase == 'rl':
            best, _ = load_policy(output/'best.pt', device)
            policy.load_state_dict(best.state_dict())
            checkpoint(output/'bc.pt', policy, args, scene_config, 'bc_best_validation', args.bc_epochs, best_mean)
            rl_start_weights = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
            frozen = copy.deepcopy(policy).eval()
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
            baseline_costs = greedy_costs(frozen, env, validation, args.batch_size, device)
            optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
        for epoch in range(1, epochs+1):
            synchronize(device)
            start = time.perf_counter()
            info = train_epoch(policy, frozen, env, splits['train'], optimizer, args, rng, device, phase)
            phase_updates[phase] += info['updates']
            val_costs = greedy_costs(policy, env, validation, args.batch_size, device)
            mean = float(val_costs.mean())
            updated, pvalue = False, None
            if phase == 'rl':
                updated, pvalue = baseline_update_decision(val_costs/ref, baseline_costs/ref)
                if updated:
                    frozen.load_state_dict(policy.state_dict())
                    baseline_costs = val_costs.copy()
            if mean < best_mean-1e-7:
                best_mean = mean
                checkpoint(output/'best.pt', policy, args, scene_config, phase, epoch, mean)
            synchronize(device)
            record = dict(phase=phase, epoch=epoch, **info, val_cost=mean,
                          val_ratio_to_teacher=float((val_costs/ref).mean()),
                          baseline_updated=updated, baseline_pvalue=pvalue, epoch_s=time.perf_counter()-start)
            history.append(record)
            baseline.write_json(output/'history.json', history)
            checkpoint(output/'last.pt', policy, args, scene_config, phase, epoch, mean, optimizer)
            print(f'{phase} {epoch}/{epochs}: loss={info["loss"]:.5f} val_cost={mean:.3f} '
                  f'ratio={record["val_ratio_to_teacher"]:.4f} updates={info["updates"]} '
                  f'baseline_update={updated} seconds={record["epoch_s"]:.2f}', flush=True)
    rl_changed = any(not torch.equal(value.detach().cpu(), rl_start_weights[key]) for key, value in policy.state_dict().items())
    parameters_changed = any(not torch.equal(value.detach().cpu(), initial_weights[key]) for key, value in policy.state_dict().items())
    # Only validation selects the checkpoint. Test policy predictions begin here.
    best, saved = load_policy(output/'best.pt', device)
    reloaded = greedy_costs(best, env, validation, args.batch_size, device)
    if not np.isclose(reloaded.mean(), saved['validation_mean_cost'], atol=2e-4):
        raise RuntimeError('checkpoint reload changed greedy validation cost')
    report = benchmark(best, splits['test'], output/'evaluation', device, not args.no_plots)
    baseline.write_csv(output/'history.csv', history)
    if not args.no_plots:
        save_training_plot(history, output)
    manifest.update(status='complete' if report['all_valid'] and report['all_tasks_served'] else 'failed',
                    finished_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    parameters_changed=parameters_changed, rl_parameters_changed=rl_changed,
                    optimizer_updates=phase_updates, selected_phase=saved['phase'], selected_epoch=saved['epoch'],
                    best_validation_cost=saved['validation_mean_cost'], initial_validation_cost=history[0]['val_cost'],
                    checkpoint_reload_verified=True, evaluation_plans=report['plans'],
                    all_valid=report['all_valid'], all_tasks_served=report['all_tasks_served'],
                    peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if device == 'cuda' else None)
    baseline.write_json(output/'run_manifest.json', manifest)
    print(f'COMPLETE: {output}; selected {saved["phase"]} epoch {saved["epoch"]}; '
          f'test plans={report["plans"]}, valid={report["all_valid"]}, covered={report["all_tasks_served"]}', flush=True)
    return 0 if manifest['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
