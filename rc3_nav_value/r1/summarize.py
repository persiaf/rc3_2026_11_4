"""Summarize saved R1 evidence only; never generate routes or oracle calls."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .analyze import analyze_files

OUT=Path(__file__).resolve().parent


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def rows(name):return [json.loads(s) for s in (OUT/name).read_text(encoding='utf-8-sig').splitlines() if s.strip()] if (OUT/name).exists() else []
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(name,data,fields):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(data)


def main():
    analysis=analyze_files(OUT/'quality.jsonl',OUT/'audit.jsonl',OUT/'labels_round.jsonl');write('analysis.json',analysis)
    quality=rows('quality.jsonl');audit=rows('audit.jsonl');training=rows('training.jsonl')
    csv_write('per_request.csv',quality,['phase','method','case','cachemode','rep','J','L','K','valid','total_s',
        'preparation_s','decision_s','oracle_s','H_s','native_s','export_validation_s','independent_validation_s',
        'measured_request_wall_s','bookkeeping_and_timer_residual_s','call_id'])
    tables=[]
    for phase,methods in [('initial',['warm_base','warm_nav','old128','prior','gBH','fixed3','detour16']),('round2',['A','B','C','D'])]:
        for method in methods:
            q=analysis['quality'][phase][method];r=dict(method=method,phase=phase,**q)
            r['all_feasible']=q.get('feasible_count')==q.get('cases') if q.get('status')=='complete' else None
            for mode in ('cold','hot'):
                t=analysis['timing'][mode].get(method,{})
                for k in ('median_s','p95_s','raw_call_p95_s'):r[mode+'_'+k]=t.get(k)
            tables.append(r)
    csv_write('quality_time.csv',tables,['method','phase','mean_J','mean_L','mean_K','feasible_rate','all_feasible',
        'cold_median_s','cold_p95_s','cold_raw_call_p95_s','hot_median_s','hot_p95_s','hot_raw_call_p95_s'])
    wide=[]
    for case in range(32):
        r={'case':case}
        for q in quality:
            if q['case']==case and q['phase']!='timing':r[q['phase']+'_'+q['method']+'_J']=q['J']
        wide.append(r)
    csv_write('per_scene.csv',wide,['case']+sorted(set().union(*(r.keys() for r in wide))-{'case'}))
    learning={}
    for tag in sorted({r['tag'] for r in training}):
        subset=[r for r in training if r['tag']==tag]
        learning[tag]=dict(updates=len(subset),first_loss=subset[0]['loss'],final_loss=subset[-1]['loss'],
            first128_mean_loss=float(np.mean([r['loss'] for r in subset[:128]])),last128_mean_loss=float(np.mean([r['loss'] for r in subset[-128:]])))
    fit={}
    for group in 'ABCD':
        fit[group]={}
        for rnd in (1,2):
            subset=[r for r in audit if r['group']==group and r['round']==rnd]
            errors=[];ranks=[];best_hits=[];regrets=[]
            for r in subset:
                p=np.array(r['queried_prediction']);t=np.array(r['queried_target']);errors.extend(((p-t)**2).tolist())
                i=int(p.argmax());best_hits.append(t[i]>=t.max()-1e-7);regrets.append(float(t.max()-t[i]))
                for a in range(len(t)):
                    for b in range(a):
                        if abs(t[a]-t[b])>1e-8:ranks.append(np.sign(p[a]-p[b])==np.sign(t[a]-t[b]))
            fit[group][str(rnd)]=dict(states=len(subset),queried_action_mse=float(np.mean(errors)) if errors else None,
                queried_pair_sign_accuracy=float(np.mean(ranks)) if ranks else None,queried_top1_hit_rate=float(np.mean(best_hits)) if best_hits else None,
                mean_queried_top1_advantage_regret=float(np.mean(regrets)) if regrets else None)
    write('training_summary.json',dict(training=learning,queried_fit_on_seen_training_states=fit,
        interpretation='Queried-only fit and ranks do not establish all-menu route quality or independent-state generalization.'))
    print(json.dumps(dict(status=analysis['status'],research_gate=analysis['research_gate'],deployment_gate=analysis['deployment_gate']),ensure_ascii=False))


if __name__=='__main__':main()
