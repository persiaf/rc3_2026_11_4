"""Regenerate plots from bundled records only; no robot/planner/model imports."""
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    scripts=['gen_fig1_scene.py','gen_fig2_constraints.py','gen_fig3_routes100.py',
             'gen_fig3_zoom.py','gen_table1_costs.py','build_gallery.py','check_exports.py']
    records=[]
    for name in scripts:
        proc=subprocess.run([sys.executable,'-B',str(HERE/name)],cwd=HERE,capture_output=True,
                            text=True,encoding='utf-8',errors='replace')
        records.append({'script':name,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
        if proc.returncode or 'glyph' in proc.stderr.lower():
            raise RuntimeError(json.dumps(records[-1],ensure_ascii=True))
    (HERE/'generation_log.json').write_text(json.dumps({'all_passed':True,'runs':records,
        'planner_or_model_executed':False,'scope':'Plot regeneration, not an experiment.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'all_passed':True,'scripts':len(scripts),'new_experiment':False}))


if __name__=='__main__':main()
