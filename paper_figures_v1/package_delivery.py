"""Package figures and frozen source copies; verify standalone regeneration."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE.parent/'rc3_paper_figures_v1.zip')
    p.add_argument('--smoke-directory',type=Path,default=HERE.parent/'rc3_paper_figures_v1_smoke')
    a=p.parse_args();dest=a.smoke_directory.resolve()
    if dest.exists():raise ValueError('choose a fresh relocation directory')
    records=[]
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name=='MANIFEST.json':continue
        records.append({'path':path.relative_to(HERE.parent).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())})
    manifest={'schema':'rc3-frozen-paper-figures-v1','scope':'Plots/tables from saved records only, not a new experiment.',
              'files':records,'formats':['vector PDF','editable-text SVG','300dpi PNG'],'planner_executed':False}
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for record in records:z.write(HERE.parent/record['path'],record['path'])
        z.write(HERE/'MANIFEST.json','paper_figures_v1/MANIFEST.json')
    with zipfile.ZipFile(a.output) as z:
        assert z.testzip() is None
        for record in records:assert sha(z.read(record['path']))==record['sha256']
        for info in z.infolist():
            name=PurePosixPath(info.filename)
            assert not name.is_absolute() and '..' not in name.parts and ':' not in str(name) and '\\' not in str(name)
            target=dest.joinpath(*name.parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    script=dest/'paper_figures_v1/generate_all.py'
    proc=subprocess.run([sys.executable,'-B',str(script)],cwd=dest,env=env,capture_output=True,
                        text=True,encoding='utf-8',errors='replace')
    assert proc.returncode==0,proc.stdout+proc.stderr
    expected=['fig1_metadata.json','fig2_metadata.json','fig3_metadata.json','fig3_zoom_metadata.json','table1_metadata.json']
    checks=[]
    for name in expected:
        left=json.loads((HERE/name).read_text(encoding='utf-8'))
        right=json.loads((dest/'paper_figures_v1'/name).read_text(encoding='utf-8'))
        assert left==right,name;checks.append({'metadata':name,'equal':True})
    for record in records:
        if '/sources/' in record['path'] or record['path'].endswith('.py'):
            assert sha((dest/record['path']).read_bytes())==record['sha256']
    for name in ('fig1_scene','fig2_constraints','fig3_routes100','fig3_zoom'):
        for suffix in ('pdf','svg','png'):assert (dest/'paper_figures_v1'/f'{name}.{suffix}').stat().st_size>0
    receipt={'path':str(a.output),'files':len(records)+1,'manifest_payloads':len(records),'bytes':a.output.stat().st_size,
        'sha256':sha(a.output.read_bytes()),'crc_passed':True,'payload_hashes_passed':True,
        'relocated_regeneration_passed':True,'metadata_checks':checks,
        'regeneration_stdout':proc.stdout,'regeneration_stderr':proc.stderr,
        'scope':'Regenerated figures from packaged source copies, without original workspace imports or new plans.'}
    a.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('metadata_checks','regeneration_stdout','regeneration_stderr')}))


if __name__=='__main__':main()
