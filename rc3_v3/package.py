"""Package sources, frozen training data, weights and every raw evaluation plan."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('rc3_v3_delivery.zip'))
    args=parser.parse_args()
    if args.output.exists():raise ValueError('choose a new package name; preserve earlier deliveries')
    root=Path.cwd();files=[]
    for folder in ('rc3_baseline','rc3_fast','rc3_improve'):
        files.extend((root/folder).glob('*.py'))
    for p in (root/'rc3_v3').rglob('*'):
        if p.is_file() and not any(part in ('__pycache__','.venv') for part in p.parts) and p.suffix!='.pyc':
            files.append(p)
    for name in ('EXPERIMENT_PLAN_20261102_v7.md','EXPERIMENT_TRACKER_20261102_v7.md'):
        p=root/'paper'/name
        if p.exists():files.append(p)
    files=sorted(set(files))
    manifest=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    with ZipFile(args.output,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(root).as_posix())
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(dict(files=manifest,
            scope='Includes complete v3 raw results and compact train data; prior-generation source dependencies only. No environment binaries. Original v2 user files remain outside this new package.'),ensure_ascii=False,indent=2))
    with ZipFile(args.output) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'ZIP CRC error: {bad}')
    info=dict(path=str(args.output.resolve()),files=len(files)+1,bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),zip_crc_ok=True)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
