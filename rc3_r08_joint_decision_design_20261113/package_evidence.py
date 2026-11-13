"""Lossless archival of THIS new derived analysis, not a new experiment."""
import gzip,hashlib,json,re
from .adapters import HOME

def main():
    source=(HOME/'ROUTE_STRUCTURE_EVIDENCE.json').resolve();target=HOME/'ROUTE_STRUCTURE_EVIDENCE.json.gz'
    if source.parent!=HOME.resolve() or source.name!='ROUTE_STRUCTURE_EVIDENCE.json':raise ValueError('Wrong intermediate path')
    raw=source.read_bytes();packed=gzip.compress(raw,compresslevel=6,mtime=0)
    if gzip.decompress(packed)!=raw:raise ValueError('Lossless archive failure')
    with target.open('xb') as f:f.write(packed)
    receipt=dict(status='PASS',source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),
        archive=target.name,archive_bytes=len(packed),archive_sha256=hashlib.sha256(packed).hexdigest(),
        byte_exact_decompression_verified=True,scope='Only new derived R08-A structure evidence; no historical input changed',
        uncompressed_intermediate_removed_after_exact_preservation=True)
    (HOME/'STRUCTURE_ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    # Exact newly-created duplicate only. Original teacher/reference/route files remain.
    source.unlink()
    for p in HOME.glob('*.md'):
        s=p.read_text(encoding='utf-8');s=re.sub(r'ROUTE_STRUCTURE_EVIDENCE\.json(?!\.gz)','ROUTE_STRUCTURE_EVIDENCE.json.gz',s);p.write_text(s,encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
