"""Build an offline figure gallery and table from the generated artifacts."""
from pathlib import Path
import csv
import html
import re

HERE=Path(__file__).resolve().parent


def main():
    captions=(HERE/'CAPTIONS.md').read_text(encoding='utf-8')
    chunks=re.split(r'^## ',captions,flags=re.M)[1:]
    sections={c.split('\n',1)[0]:c.split('\n',1)[1].strip() for c in chunks}
    figures=[('图1：场景、任务与输出','fig1_scene'),('图2：共同约束的必要性','fig2_constraints'),
             ('图3：固定百桌场景的五方法对照','fig3_routes100'),('图3补充：共同中心区域','fig3_zoom')]
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结规划记录 · 论文图表</title>',
           '<style>body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#202630;max-width:1100px;margin:40px auto;padding:0 24px}figure{margin:24px 0 60px}img{max-width:100%;height:auto;border:1px solid #ddd}a{color:#14568d}figcaption{font-size:15px}table{border-collapse:collapse;width:100%}td,th{padding:8px 14px;border-bottom:1px solid #ccc;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#f2f4f6;padding:14px 18px}h1{font-size:28px}h2{font-size:22px}@media print{figure{break-before:page}img{max-height:70vh;object-fit:contain}a{color:#222}}</style>',
           '<h1>冻结规划记录的论文图表</h1><p class="note">来自固定JSON记录；没有重新训练、规划或测时。原v4/v5/v5.1保持冻结。图表说明的是代理模型与合成约束，真实标定和IK暂缓。</p>',
           '<p><a href="README.md">使用说明</a> · <a href="CAPTIONS.md">图注与论证衔接</a> · <a href="latex_includes.tex">LaTeX插入段</a></p>']
    for heading,name in figures:
        caption=sections[heading].split('\n\n')[0].replace('**','')
        parts += [f'<figure><h2>{html.escape(heading)}</h2>',
                  f'<p><a href="{name}.pdf">PDF矢量</a> · <a href="{name}.svg">SVG</a> · <a href="{name}.png">PNG</a></p>',
                  f'<img src="{name}.png" alt="{html.escape(heading)}"><figcaption>{html.escape(caption)}</figcaption></figure>']
    with (HERE/'TABLE_1_costs.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    parts += ['<h2>表1：同一百桌场景的成本</h2><table><thead><tr><th>方法</th><th>修复预算</th><th>L / m</th><th>K</th><th>J / 等效m</th></tr></thead><tbody>']
    for row in rows:
        cells=[row['method'],row['repair_budget'],f"{float(row['L_m']):.3f}",row['K'],f"{float(row['J_equivalent_m']):.3f}"]
        parts.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    parts += ['</tbody></table><p>同一seed=1565649625、同一rep0。J=L+0.75K，不是实测任务时间；本单例不用于推断总体排名。</p>',
              '<p><a href="TABLE_1_costs.csv">CSV数据</a> · <a href="TABLE_1_costs.tex">LaTeX表</a></p></html>']
    (HERE/'index.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':main()
