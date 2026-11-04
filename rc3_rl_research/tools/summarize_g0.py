"""Descriptive tables from frozen diagnostics; never search or select new states."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    cases=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'outputs/g0').glob('case_*/summary.json'))]
    rows=[]
    for s in cases:
        case=s['case'];payload=json.loads((ROOT/f"outputs/g0/case_{case['case_id']:02d}/selected_replay.json").read_text(encoding='utf-8'))
        state=payload['state']
        rows.append(dict(case_id=case['case_id'],seed=case['seed'],mode=case['mode'],tasks=s['task_count'],
            anchor_J=s['anchor_cost'],best_one_step_J=s['root_one_step_best'],best_depth3_J=s['best_depth3_cost'],
            worse_first_beats_one_step=s['worse_first_beats_one_step'],ranking_reversal=s['ranking_reversal'],
            worse_first_beats_monotone_first=s['worse_first_beats_all_monotone_first'],
            selected_return=state['total_reward'],selected_work=state['work_used'],
            selected_J=state['returned_certificate']['cost']))
    out=ROOT/'outputs/g0/cases.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    all_failures=sum((Counter(s['failures']) for s in cases),Counter())
    before=json.loads((ROOT/'outputs/development_case00_before_replay_fixes/summary.json').read_text(encoding='utf-8'))
    totals=dict(formal_cases=len(cases),formal_oracle_queries=sum(s['queries'] for s in cases),
        formal_oracle_work=dict(sum((Counter(s['physical_oracle_work']) for s in cases),Counter())),
        post_tree_selected_replays=dict(sum((Counter(s['verification_replay']) for s in cases),Counter())),
        pre_freeze_debug_case00=dict(query_count=before['queries'],physical_oracle_work=before['physical_oracle_work'],
                                    selected_replay=before['verification_replay']),
        failed_or_rejected_queries=dict(all_failures),accepted=sum(s['accepted'] for s in cases),
        accepted_worse=sum(s['accepted_worse'] for s in cases),
        global_pruning_outside_window=sum(s['globally_pruned_retained'] for s in cases),
        scope='Oracle work, selected-sequence verification and pre-freeze debug are separate. Additional G1 tests, final delivery replay and independent saved-record audits have separate receipts; their executions are not extra independent scenes.')
    (ROOT/'outputs/execution_accounting.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(totals,ensure_ascii=False))

if __name__=='__main__':main()
