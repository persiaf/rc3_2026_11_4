"""Read-only arithmetic audit of saved evidence; no planner/model imports or timing runs."""
from pathlib import Path
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
INPUTS = {}
CHECKS = []

def track(path):
    p = ROOT / path
    INPUTS[str(p.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def read(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))

def check(label, value):
    CHECKS.append({'check': label, 'passed': bool(value)})

def close(a, b):
    return bool(np.allclose(a, b, rtol=1e-10, atol=1e-9))

def paired(values):
    v = np.asarray(values, dtype=float)
    samples = np.random.default_rng(731203).integers(0, len(v), size=(20000, len(v)))
    return {'scenes': len(v), 'mean_reduction_pct': float(v.mean()),
            'ci95_pct': np.quantile(v[samples].mean(1), [.025, .975]).tolist(),
            'wins': int((v > 1e-9).sum()), 'ties': int((abs(v) <= 1e-9).sum()),
            'losses': int((v < -1e-9).sum())}

def saved_cost(plan):
    pts = plan['route_polyline']
    length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    return length + plan['stop_penalty_equivalent_m'] * len(plan['service_stops'])

def supplement(folder, protocol_path):
    manifest, summary = read(folder + '/manifest.json'), read(folder + '/summary.json')
    protocol = read(protocol_path)
    for name, expected in manifest['frozen_source_sha256'].items():
        relative = name.replace('\\', '/')
        check(f'{folder}: frozen hash {relative}', hashlib.sha256(track(relative).read_bytes()).hexdigest() == expected)
    grouped = {}
    valid = bounds = recalc_costs = 0
    for p in sorted((ROOT / folder / 'plans').glob('scene_plan_*.json')):
        data = read(p.relative_to(ROOT))
        m = data['metadata']
        key = (m['seed'], m['nx'], m['ny'], m['mode'])
        grouped[key] = {}
        for method in ('detour_wall400', 'v3_mlp_wall400'):
            rr = [(k, r) for k, r in data['runs'].items() if r['config']['id'] == method]
            check(f'{folder}/{key}/{method}: three repeats', len(rr) == 3)
            rs = [r for _, r in rr]
            for k, r in rr:
                valid += bool(r['independent_validation']['valid_model_plan'])
                bounds += r['objective_m'] <= r['reference_objective_m'] + 1e-8
                recalc_costs += close(saved_cost(data['plans'][k]), r['objective_m'])
                check(f'{folder}/{key}/{k}: budget', r['config']['seconds'] == .4 and r['config']['horizon'] == 256)
            grouped[key][method] = {'cost': float(np.mean([r['objective_m'] for r in rs])),
                                   'reference': float(np.mean([r['reference_objective_m'] for r in rs])),
                                   'time_ms': float(np.median([1000*r['planning_s'] for r in rs])),
                                   'runs': rs}
    methods = {}
    for method in ('detour_wall400', 'v3_mlp_wall400'):
        ss = [grouped[k][method] for k in sorted(grouped)]
        rs = [r for s in ss for r in s['runs']]
        times = [1000*r['planning_s'] for r in rs]
        methods[method] = {'scenes': len(ss), 'runs': len(rs),
            'mean_objective_m': float(np.mean([s['cost'] for s in ss])),
            'mean_reduction_vs_fixed3_pct': float(np.mean([100*(s['reference']-s['cost'])/s['reference'] for s in ss])),
            'median_scene_planning_ms': float(np.median([s['time_ms'] for s in ss])),
            'p95_all_run_planning_ms': float(np.quantile(times, .95)),
            'mean_dp_calls': float(np.mean([r['dp_calls'] for r in rs])),
            'deadline_exceeded_runs': sum(r['planning_s'] > .4 for r in rs),
            'fallback_runs': sum(r['fallback_used'] for r in rs)}
        for field, v in methods[method].items():
            check(f'{folder}/{method}/{field}: summary', close(v, summary['methods'][method][field]))
            check(f'{folder}/{method}/{field}: protocol audit', close(v, protocol['methods'][method][field]))
    values = []
    for key in sorted(grouped):
        d, m = grouped[key]['detour_wall400']['cost'], grouped[key]['v3_mlp_wall400']['cost']
        values.append(100*(d-m)/d)
    pair = paired(values)
    for field, v in pair.items():
        check(f'{folder}/paired/{field}: summary', close(v, summary['paired_mlp_vs_detour'][field]))
        check(f'{folder}/paired/{field}: protocol audit', close(v, protocol['paired'][field]))
    check(folder + ': all saved objectives reconstructed', recalc_costs == 108)
    check(folder + ': valid flags and cost bounds', valid == bounds == 108)
    return {'platform': manifest['platform'], 'scene_keys': list(sorted(grouped)),
            'methods': methods, 'paired': pair, 'valid_saved_flags': valid,
            'recomputed_cost_bounds': bounds, 'recomputed_polyline_objectives': recalc_costs}

native = supplement('rc3_v5/mlp_budget_windows', 'rc3_v5/native_protocol_audit.json')
uploaded = supplement('rc3_v4_review_import/rc3_v4_review_evidence/mlp_budget_supplement', 'rc3_v5/uploaded_protocol_audit.json')
check('platform supplements reuse exactly the same scenes', native['scene_keys'] == uploaded['scene_keys'])

cache = {}
for phase, expected in [('equivalence', 36), ('benchmark', 108)]:
    folder = f'rc3_v5/cache_results/{phase}'
    summary = read(folder + '/summary.json')
    manifest = read(folder + '/manifest.json')
    for name, expected_hash in manifest['source_sha256'].items():
        relative = name.replace('\\', '/')
        check(f'{folder}: frozen hash {relative}', hashlib.sha256(track(relative).read_bytes()).hexdigest() == expected_hash)
    by_scene = []
    count = 0
    for p in sorted((ROOT / folder / 'plans').glob('scene_plan_*.json')):
        d = read(p.relative_to(ROOT))
        plans = list(d['plans'].values())
        check(f'{folder}/{p.stem}: complete plan equality', all(x == plans[0] for x in plans))
        count += len(plans)
        if phase == 'benchmark':
            r = [x['record'] for x in d['runs'].values()]
            ss = {m: {k: float(np.median([x[k] for x in r if x['method'] == m]))
                      for k in ('planning_ms', 'model_ms', 'graph_ms', 'inference_ms')}
                  for m in ('original', 'cached')}
            by_scene.append(ss)
    check(folder + ': correct saved-plan denominator', count == expected == summary['saved_plans'])
    cache[phase] = {'scenes': 18, 'saved_plans': count, 'head_checks': summary['head_checks']}
    if phase == 'benchmark':
        stats = {m: {k: float(np.median([s[m][k] for s in by_scene]))
                     for k in ('planning_ms', 'model_ms', 'graph_ms', 'inference_ms')}
                 for m in ('original', 'cached')}
        for m, vals in stats.items():
            for k, v in vals.items(): check(f'cache/{m}/{k}', close(v, summary['summary'][m][k]))
        reductions = [(1-s['cached']['planning_ms']/s['original']['planning_ms'])*100 for s in by_scene]
        ratio = stats['original']['planning_ms']/stats['cached']['planning_ms']
        paired_reduction = float(np.median(reductions))
        cache[phase].update({'summary': stats, 'ratio_of_group_medians': ratio,
            'reduction_from_group_medians_pct': (1-1/ratio)*100,
            'median_paired_reduction_pct': paired_reduction,
            'faster_scenes': sum(x > 0 for x in reductions), 'slower_scenes': sum(x < 0 for x in reductions)})
        check('cache paired reduction', close(paired_reduction, summary['median_paired_reduction_pct']))
heads = read('rc3_v5/cache_results/equivalence/head_checks.json')
check('144 saved head-check records', len(heads) == 144)
check('saved head and action equality assertions', all(h['all_heads_bitwise_equal'] and h['max_output_error'] == 0 and h['original_action'] == h['cached_action'] for h in heads))
cache['head_check_scope'] = 'Recount of saved assertions; this read-only audit does not run inference again.'

v4_manifest = read('rc3_v4/results/manifest.json')
v4_summary = read('rc3_v4/results/summary.json')
old_pairs = []
for p in sorted((ROOT / 'rc3_v4/results/plans').glob('scene_plan_*.json')):
    d = read(p.relative_to(ROOT)); m = d['metadata']
    if m['nx']*m['ny'] != 100: continue
    costs = {method: float(np.mean([r['objective_m'] for r in d['runs'].values() if r['config']['id'] == method])) for method in ['v3_mlp8','detour8']}
    old_pairs.append(((m['seed'],m['nx'],m['ny'],m['mode']),100*(costs['detour8']-costs['v3_mlp8'])/costs['detour8']))
old_mlp = paired([x[1] for x in sorted(old_pairs)])
csvpath = track('rc3_v4_review_import/rc3_v4_review_evidence/experiments/paired_comparisons.csv')
csvrows = list(csv.DictReader(csvpath.open(encoding='utf-8-sig')))
old_claim = next(x for x in csvrows if x['group']=='main_100' and x['config']=='v3_mlp8' and x['reference']=='detour8')
check('v4 descriptive MLP cost mean', close(old_mlp['mean_reduction_pct'], float(old_claim['mean_reduction_pct'])))
check('v4 descriptive MLP CI', close(old_mlp['ci95_pct'], [float(old_claim['ci_low']),float(old_claim['ci_high'])]))
main = {x['config']:x for x in v4_summary['summaries'] if x['group']=='main_100'}
structured_control = next(x for x in v4_summary['comparisons'] if x['group']=='main_100' and x['config']=='structured8' and x['reference']=='control8')
ratio = main['structured8']['median_planning_ms']/main['detour8']['median_planning_ms']
check('v4 frozen time gate', v4_manifest['acceptance']['max_median_time_ratio_to_detour']==1.25)
check('v4 original adoption failure preserved', ratio>1.25 and structured_control['ci95_pct'][0]<0)
v4 = {'acceptance': v4_manifest['acceptance'], 'original_structured_detour_time_ratio': ratio,
      'structured_vs_control_main100_fixed8': structured_control, 'legacy_fixed8_descriptive_comparison': old_mlp}

small = read('rc3_v4/small_results/summary.json')
rows = list(csv.DictReader(track('rc3_v4/small_results/metrics.csv').open(encoding='utf-8-sig')))
small_nonempty = {}
for method in ['fixed3','detour8','table_order','fixed_set_exact_order']:
    rr = [r for r in rows if r['method']==method and float(r['penalty'])==.75 and r['bound_status']=='optimal' and int(r['tasks'])>0]
    small_nonempty[method] = {'scenes':len(rr), 'mean_gap_pct':float(np.mean([float(r['optimal_gap_pct']) for r in rr])), 'optimal':sum(float(r['optimal_gap_pct'])<=1e-7 for r in rr)}
check('small finite optimum counts', small['scene_count']==27 and small['scenario_penalty_count']==45 and small['exact_status_counts']=={'optimal':44,'unsupported':1})
check('small 25-case nonempty denominator', all(x['scenes']==25 for x in small_nonempty.values()))
training = read('rc3_v4/training/summary.json')
dataset = read('rc3_v4/data/manifest.json')
check('v4 data denominators', dataset['train_scenes']==96 and dataset['validation_scenes']==18 and dataset['test_scenes']==39)
check('v4 training updates', training['states']==384 and len(training['runs'])==6 and training['config']['epochs']==80 and training['updates']==23040)

inventory = read('rc3_v5/robot_assets/asset_inventory.json')
preflight = read('rc3_v5/robot_assets/preflight_report.json')
meshes = list((ROOT/'rc3_v5/robot_assets/simulation/Aloha/meshes').glob('*.STL'))
meshbytes = sum(track(p.relative_to(ROOT)).stat().st_size for p in meshes)
urdfs = [track(p).read_bytes().replace(b'\r\n', b'\n') for p in ['rc3_v5/robot_assets/simulation/Aloha/urdf/Aloha.urdf','rc3_v5/robot_assets/official_reference/Aloha/urdf/Aloha.urdf']]
jointroot = ET.fromstring(urdfs[0])
bounded = [j for j in jointroot.findall('joint') if j.attrib['type'] in ('revolute','prismatic')]
all_zero = all(all(float(j.find('limit').attrib[k])==0 for k in ['lower','upper','velocity','effort']) for j in bounded)
check('robot meshes', len(meshes)==19 and meshbytes==15108946)
check('normalized URDF equality', urdfs[0]==urdfs[1])
check('13 zero-limit bounded joints', len(bounded)==13 and all_zero)
check('physical stage remains blocked', inventory['status']=='assets_verified' and preflight['status']=='blocked_inputs' and len(preflight['blocked_inputs'])==135 and not preflight['physical_ik_verified'] and not preflight['collision_verified'])
robot = {'meshes':len(meshes),'mesh_bytes':meshbytes,'bounded_zero_limit_joints':len(bounded),'urdf_normalized_equal':urdfs[0]==urdfs[1], 'input_blockers':len(preflight['blocked_inputs']),'status':preflight['status'],'physical_ik_verified':preflight['physical_ik_verified'],'collision_verified':preflight['collision_verified']}

external = read('rc3_v5/external_results/SUMMARY.json')
verification = read('rc3_v5/verification/summary.json')
check('external test counts', external['tests_run']==28 and external['original_external_tests']==9 and external['imported_common_arm_regressions']==11 and external['v5_common_coverage_and_guard_regressions']==8 and external['cli_checks_passed']==5)
check('external counterexample value', close(external['canonicalization_v5_objective'],12.8))
check('integration counts', verification['tests']['run']==54 and verification['tests']['all_passed'] and not verification['torch_imported'])
log = track('rc3_v5/v4_regression.log').read_text(encoding='utf-8-sig')
check('v4 regression saved log', 'Ran 40 tests' in log and log.rstrip().endswith('OK'))
for doc in ['rc3_v5/RESULTS.md','rc3_v5/README.md','rc3_v5/EVIDENCE_MAP.md']: track(doc)
unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in INPUTS.items())
check('all audit input files unchanged during arithmetic audit',unchanged)
result = {'all_passed':all(x['passed'] for x in CHECKS), 'checks':len(CHECKS), 'failures':[x for x in CHECKS if not x['passed']],
    'native':native,'uploaded':uploaded,'cache':cache,'v4':v4,'small_nonempty':small_nonempty,'robot':robot,
    'test_counts':{'external':28,'external_cli':5,'v5':54,'v4_saved_log':40},
    'input_sha256':INPUTS,'audit_scope':'Read-only saved-data arithmetic and evidence-status audit. No training, model inference, planning, benchmark timing or physical validation was executed.'}
(OUT/'numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'all_passed':result['all_passed'],'checks':len(CHECKS),'failures':result['failures'], 'native':native['paired'],'uploaded':uploaded['paired'],'cache':cache['benchmark'],'v4_legacy':old_mlp,'robot':robot},ensure_ascii=False,indent=2))
