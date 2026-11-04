"""Post-training fit on existing teacher states; zero new environment queries.

This is supervision-set diagnosis, not generalization or policy rollout
evaluation. Run only after the final matched comparisons have completed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ('Post hoc inference on the already queried supervised teacher states, '
         'not a new generalization set. Near-best means best among the queried '
         'finite-depth suffixes; unknown legal actions have no quality label. '
         'No environment step, repair oracle, finalization or new teacher query is performed.')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_metrics(logits, sample):
    """Pure array diagnostics; logits may be padded only beyond this state."""
    legal = np.asarray(sample['legality_mask'], dtype=bool)
    evidence = np.asarray(sample['evidence_mask'], dtype=bool)
    ties = np.asarray(sample['ties'], dtype=bool)
    q = np.asarray(sample['q_values'], dtype=float)
    n = len(legal)
    logits = np.asarray(logits, dtype=float)[:n]
    if logits.shape != (n,) or evidence.shape != (n,) or ties.shape != (n,) or q.shape != (n,):
        raise ValueError('sample/logit shape mismatch')
    if not legal.any() or not evidence.any() or not ties.any() or (evidence & ~legal).any() or (ties & ~evidence).any():
        raise ValueError('invalid legal/evidence/tie masks')
    if not np.isfinite(logits[legal]).all() or not np.isneginf(logits[~legal]).all():
        raise ValueError('model legal/padding mask differs from teacher state')
    probabilities = np.zeros(n, dtype=float)
    shifted = logits[legal]-logits[legal].max()
    probabilities[legal] = np.exp(shifted)/np.exp(shifted).sum()
    selected = int(np.argmax(logits))
    observed_logits = np.where(evidence, logits, -np.inf)
    evidence_selected = int(np.argmax(observed_logits))
    unknown = legal & ~evidence
    stop = int(sample['stop_id'])
    if not 0 <= stop < n or not evidence[stop]:
        raise ValueError('STOP must have teacher evidence')
    evidence_shift = observed_logits[evidence]-observed_logits[evidence].max()
    normalizer = observed_logits[evidence].max()+np.log(np.exp(evidence_shift).sum())
    return dict(full_greedy_action=selected, evidence_greedy_action=evidence_selected,
        legal_actions=int(legal.sum()), evidence_actions=int(evidence.sum()), unknown_legal_actions=int(unknown.sum()),
        near_best_actions=int(ties.sum()),
        full_legal_greedy_observed_near_best_hit=bool(ties[selected]),
        evidence_greedy_near_best_hit=bool(ties[evidence_selected]),
        full_greedy_unknown=bool(unknown[selected]),
        unknown_probability_mass=float(probabilities[unknown].sum()),
        observed_near_best_probability_mass=float(probabilities[ties].sum()),
        evidence_probability_mass=float(probabilities[evidence].sum()),
        full_greedy_stop=selected==stop, evidence_greedy_stop=evidence_selected==stop,
        stop_probability=float(probabilities[stop]), stop_teacher_near_best=bool(ties[stop]),
        evidence_normalized_tie_cross_entropy=float(np.mean(normalizer-logits[ties])),
        observed_full_greedy_Q_gap=(float(q[evidence].max()-q[selected]) if evidence[selected] else None),
        evidence_greedy_Q_gap=float(q[evidence].max()-q[evidence_selected]),
        full_greedy_quality_known=bool(evidence[selected]))


MEAN_FIELDS = ('full_legal_greedy_observed_near_best_hit', 'evidence_greedy_near_best_hit',
    'full_greedy_unknown', 'unknown_probability_mass', 'observed_near_best_probability_mass',
    'evidence_probability_mass', 'full_greedy_stop', 'evidence_greedy_stop',
    'stop_probability', 'stop_teacher_near_best', 'evidence_normalized_tie_cross_entropy',
    'observed_full_greedy_Q_gap', 'evidence_greedy_Q_gap', 'full_greedy_quality_known')


def summarize(rows):
    groups = []
    for model in sorted({r['model'] for r in rows}):
        for split in ('initial', 'extra', 'combined'):
            for level in ('root', 'all'):
                own = [r for r in rows if r['model']==model and (split=='combined' or r['split']==split)
                       and (level=='all' or r['is_root'])]
                if not own:
                    continue
                item = dict(model=model, split=split, level=level, samples=len(own),
                    scenes=len({(r['split'],r['scene_id']) for r in own}),
                    depth_counts={str(d):sum(r['teacher_depth']==d for r in own) for d in sorted({r['teacher_depth'] for r in own})},
                    metrics={})
                for field in MEAN_FIELDS:
                    known = [r for r in own if r[field] is not None]
                    item['metrics'][field] = dict(n=len(known), unknown=len(own)-len(known),
                        sample_mean=float(np.mean([r[field] for r in known])) if known else None,
                        teacher_weighted_mean=float(np.average([r[field] for r in known],
                            weights=[r['analysis_weight'] for r in known])) if known else None)
                groups.append(item)
    return dict(scope=SCOPE, groups=groups,
        interpretation=dict(full_hit='Unknown chosen actions count as no observed-near-best hit, not evidence that their quality is poor.',
            evidence_hit='Argmax restricted to queried evidence; separates ranking on known labels from probability allocated outside the queried subset.',
            Q_gap='Only queried choices have an observed Q gap. Unknown quality is null, never filled with zero.',
            weighting='Within each group, renormalize equal-scene/equal-route teacher weights. Combined split masses are 64:32, preserving equal scene mass.',
            all_states='Includes roots at depth 2 and descendants at depth 1; root-only groups isolate a common depth.',
            probabilities='Probabilities use all legal actions; evidence cross-entropy renormalizes only queried actions as during BC.'))


def run(output):
    # All inexpensive readiness checks happen before importing Torch or models.
    final = ROOT/'outputs/final_evaluation/summary.json'
    if not final.exists():
        raise ValueError('complete final matched evaluation before teacher fit diagnosis')
    evaluation = _read(final)
    if set(evaluation['datasets']) != {'controls_fixed_work','final_fixed_work','final_soft_time_0400ms'}:
        raise ValueError('required final comparison directories are incomplete')
    completion = _read(ROOT/'outputs/training/ppo_receipt.json')
    if not completion.get('completed') or completion['actual_actions'] != 32768:
        raise ValueError('formal PPO must be complete')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists() or (output/'samples.csv').exists():
        raise ValueError('diagnostic exists; retain its evidence and use a fresh directory')
    paths = dict(bc_frozen=ROOT/'outputs/training/bc_initial.pt',
                 bc_continued_2048=ROOT/'outputs/training/bc_continued_2048.pt',
                 ppo_final=ROOT/'outputs/training/ppo_32768.pt')
    teacher_paths = {split: ROOT/f'outputs/teacher_{split}/teacher.pkl' for split in ('initial','extra')}
    teacher_hashes = {split:_sha(path) for split,path in teacher_paths.items()}
    for split, path in teacher_paths.items():
        declared = _read(path.parent/'summary.json')
        audited = _read(path.parent/'audit_receipt.json')
        if teacher_hashes[split] != declared['teacher_sha256'] or teacher_hashes[split] != audited['teacher_sha256'] or not audited['all_passed']:
            raise ValueError('trusted teacher identity or prior label audit differs')
    import torch
    from rc3_rl_g2.checkpoints import load_model
    from rc3_rl_g2.teacher_dataset import load_teacher
    from rc3_rl_g2.train_ppo import configure
    protocol = _read(ROOT/'outputs/protocol.json')
    configure(protocol['training_seed'])
    samples = []
    for split, path in teacher_paths.items():
        own = load_teacher(path)['samples']
        for index, sample in enumerate(own):
            samples.append((split,index,sample))
    scene_counts = {split:len({s['scene_id'] for name,_,s in samples if name==split}) for split in teacher_paths}
    manifest = dict(schema='rc3-g2-teacher-fit-v1', scope=SCOPE, batch_size=8, device='cpu',
        tool_sha256=_sha(__file__), checkpoint_sha256={name:_sha(path) for name,path in paths.items()},
        teacher_sha256=teacher_hashes, final_comparison_summary_sha256=_sha(final),
        protocol_sha256=_sha(ROOT/'outputs/protocol.json'), new_environment_queries=0,
        expected_rows=3*len(samples))
    (output/'request.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rows=[];timings={}
    for name,path in paths.items():
        started=time.perf_counter();model,payload=load_model(path,'cpu');model.eval()
        if payload['protocol_sha256'] != manifest['protocol_sha256']:
            raise ValueError('model protocol identity differs')
        if name!='bc_frozen' and payload['parent_bc_sha256'] != manifest['checkpoint_sha256']['bc_frozen']:
            raise ValueError('model BC parent identity differs')
        with torch.inference_mode():
            for offset in range(0,len(samples),8):
                batch=samples[offset:offset+8]
                logits,_=model([s['observation'] for _,_,s in batch])
                values=logits.cpu().numpy()
                for j,(split,index,sample) in enumerate(batch):
                    rows.append(dict(model=name,split=split,sample_index=index,scene_id=sample['scene_id'],
                        state_signature=sample['state_signature'],path=json.dumps(sample['path']),
                        teacher_depth=sample['teacher_depth'],is_root=len(sample['path'])==0,
                        original_teacher_weight=sample['weight'],
                        analysis_weight=sample['weight']*scene_counts[split]/sum(scene_counts.values()),
                        **sample_metrics(values[j],sample)))
        timings[name]=time.perf_counter()-started
        del model,payload
    result=summarize(rows)
    result.update(schema=manifest['schema'],rows=len(rows),request=manifest,
        diagnostic_wall_s=timings,timing_scope='Diagnostic loading plus inference; not deployment planning latency.')
    with (output/'samples.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/teacher_fit_diagnostic')
    args=parser.parse_args();result=run(args.output)
    print(json.dumps(dict(output=str(args.output),rows=result['rows'],new_environment_queries=0,scope=SCOPE)))


if __name__=='__main__':
    main()
