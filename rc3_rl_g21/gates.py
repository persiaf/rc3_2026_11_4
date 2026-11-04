"""Evaluate preregistered full-action opportunity criteria on saved diagnostics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_rows(diagnostic):
    """Check finite-label accounting before using saved rows for a gate.

    This verifies internal consistency, not their agreement with a model or
    teacher. audit_results performs the separate teacher-backed verification.
    """
    seen=set()
    for row in diagnostic['rows']:
        identity=(row['scene_id'],row['sample_index'])
        if identity in seen:raise ValueError('Duplicate diagnostic row')
        seen.add(identity)
        for field in ('is_root','known_positive','clear_stop','full_greedy_unknown',
                      'full_greedy_stop','full_known_positive_continue'):
            if type(row[field]) is not bool:raise ValueError('Boolean diagnostic flag required')
        if row['teacher_depth']!=(2 if row['is_root'] else 1):raise ValueError('Teacher depth mismatch')
        delta=row['delta']
        if not isinstance(delta,(int,float)) or not math.isfinite(delta):raise ValueError('Finite delta required')
        if row['known_positive']!=(delta>.001) or row['clear_stop']!=(delta<=-.02):
            raise ValueError('Diagnostic label threshold mismatch')
        unknown=row['full_greedy_unknown'];stop=row['full_greedy_stop'];gain=row['full_selected_gain']
        if unknown:
            if stop or row['full_known_positive_continue'] or gain is not None:
                raise ValueError('Unknown selected action has no observed quality')
            if row['observed_full_greedy_Q_gap'] is not None:
                raise ValueError('Unknown selected action requires null Q gap')
        else:
            if gain is None or not math.isfinite(gain):raise ValueError('Observed selected gain required')
            if stop and not math.isclose(gain,0.,abs_tol=1e-12):raise ValueError('STOP gain must be zero')
            if row['full_known_positive_continue']!=(not stop and gain>.001):
                raise ValueError('Known positive continue flag mismatch')
            regret=row['observed_full_greedy_Q_gap']
            if regret is None or not math.isfinite(regret) or regret < -1e-12:
                raise ValueError('Observed Q gap must be finite and nonnegative')

def evaluate(initial,extra,criteria):
    validate_rows(initial);validate_rows(extra)
    roots=[r for r in initial['rows'] if r['is_root']]
    positive=[r for r in roots if r['known_positive']]
    clear_stop=[r for r in roots if r['clear_stop']]
    other_positive=[r for r in extra['rows'] if r['is_root'] and r['known_positive']]
    def metric(rows,key):
        return dict(count=sum(bool(r[key]) for r in rows),n=len(rows),fraction=sum(bool(r[key]) for r in rows)/len(rows) if rows else None)
    metrics=dict(initial_positive_full_known=metric(positive,'full_known_positive_continue'),
        extra_positive_full_known=metric(other_positive,'full_known_positive_continue'),
        initial_clear_stop_full_stop=metric(clear_stop,'full_greedy_stop'),
        initial_root_full_unknown=metric(roots,'full_greedy_unknown'))
    checks={}
    for name,value in metrics.items():
        suffix='_max_fraction' if name.endswith('unknown') else '_min_fraction'
        threshold=criteria[name+suffix]
        if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:
            raise ValueError('Gate fraction threshold must be finite in [0,1]')
        fraction=value['fraction']
        checks[name]=fraction is not None and (fraction<=threshold if suffix=='_max_fraction' else fraction>=threshold)
    return dict(passed=all(checks.values()),checks=checks,metrics=metrics,
        scope='Previously queried finite-depth development labels. Unknown actions are unevaluated, not bad. No new generalization or planner quality claim.')

def inspect(run):
    p=read(ROOT/'outputs/protocol.json');protocol_sha=sha256(ROOT/'outputs/protocol.json')
    if protocol_sha!=(ROOT/'outputs/protocol.sha256').read_text().strip():raise ValueError('Protocol hash mismatch')
    if not read(ROOT/'outputs/toy16/receipt.json')['gate']['passed']:raise ValueError('Toy prerequisite failed')
    folder=ROOT/'outputs'/run;receipt=read(folder/'receipt.json')
    if receipt['protocol_sha256']!=protocol_sha or not receipt['completed']:raise ValueError('Run identity mismatch')
    if receipt['checkpoint_sha256']!=sha256(folder/'model.pt'):raise ValueError('Checkpoint changed')
    primary=folder/f"diagnostic_{receipt['updates']:04d}.json";extra=folder/'extra_existing_teacher_diagnostic.json'
    result=evaluate(read(primary),read(extra),p['opportunity_gate'])
    result.update(run=run,protocol_sha256=protocol_sha,checkpoint_path=(folder/'model.pt').relative_to(ROOT.parent).as_posix(),
        checkpoint_sha256=receipt['checkpoint_sha256'],receipt_sha256=sha256(folder/'receipt.json'),
        diagnostic_sha256=sha256(primary),extra_diagnostic_sha256=sha256(extra),new_environment_queries=0)
    write(folder/'opportunity_gate.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run');parser.add_argument('--authorize',action='store_true')
    args=parser.parse_args();result=inspect(args.run)
    if args.authorize:
        if not result['passed']:raise ValueError('Opportunity criteria failed; interactions remain disabled')
        destination=ROOT/'outputs/opportunity_gate.json'
        if destination.exists():raise FileExistsError('Existing opportunity decision must be preserved')
        write(destination,result)
    import json
    print(json.dumps(result,ensure_ascii=False))
