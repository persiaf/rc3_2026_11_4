import json,csv,gzip,hashlib
from pathlib import Path
EXP=Path(__file__).resolve().parents[1];REPO=EXP.parent
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def report(name):return read(EXP/'evaluations/development'/(name+'.json'))
def table(name):
    with (EXP/'tables'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def record(row):
    ref=row['record']
    with (REPO/ref['path']).open('rb') as f:f.seek(ref['offset']);raw=gzip.decompress(f.read(ref['compressed_bytes']))
    if hashlib.sha256(raw).hexdigest()!=ref['raw_sha256']:raise ValueError('Saved plot record identity')
    return json.loads(raw)
