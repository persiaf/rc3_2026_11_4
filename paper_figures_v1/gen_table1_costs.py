"""Export the fixed saved case's comparison table without changing its plans."""
import csv
from shared import *

NAMES={'fixed3_rep0':'Fixed3','detour8_rep0':'Detour8','v3_mlp8_rep0':'旧 MLP8',
       'control8_rep0':'Control8','structured8_rep0':'Structured8'}


def main():
    entry=display('first_main100_fixed_call_case');rows=[]
    for item in entry['plans']:
        plan=resolve(item['plan']);run=resolve(entry['run_configurations'][item['label']])
        value=cost_summary(plan)
        assert run['kind']=='calls' and run['seconds'] is None
        rows.append(dict(method=NAMES[item['label']],saved_label=item['label'],repair_budget=run['horizon'],
                         seed=entry['seed'],repeat=0,**value))
    with (HERE/'TABLE_1_costs.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    header=['\\begin{table}[htbp]','\\centering',
            '\\caption{固定百桌场景的保存方案成本。种子1565649625，统一取rep0；Fixed3为无修复初始参照，其余方法采用固定8次修复。$J=L+0.75K$为等效路程成本。}',
            '\\label{tab:fixed-main100-costs}','\\begin{tabular}{lrrrr}','\\toprule',
            '方法 & 修复预算 & $L$ / m & $K$ & $J$ / 等效 m \\\\','\\midrule']
    for row in rows:
        header.append(f"{row['method']} & {row['repair_budget']} & {row['L_m']:.3f} & {row['K']} & {row['J_equivalent_m']:.3f} " + chr(92)*2)
    header+=['\\bottomrule','\\end{tabular}','\\end{table}']
    (HERE/'TABLE_1_costs.tex').write_text('\n'.join(header)+'\n',encoding='utf-8')
    md=['|方法|修复预算|L / m|K|J / 等效m|','|---|---:|---:|---:|---:|']
    md.extend(f"|{r['method']}|{r['repair_budget']}|{r['L_m']:.3f}|{r['K']}|{r['J_equivalent_m']:.3f}|" for r in rows)
    (HERE/'TABLE_1_costs.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    write_json('table1_metadata.json',dict(scene=entry['scene'],rows=rows,
               scope='Same frozen case and rep0; per-instance illustration, not aggregate superiority.'))


if __name__=='__main__':main()
