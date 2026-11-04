"""Retrieve only frozen URDF-referenced meshes; verify upstream Git blob SHA-1."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def one(entry):
    out = ROOT / 'simulation' / 'Aloha' / 'meshes' / entry['name']
    receipt = {'path': str(out.relative_to(ROOT)), 'url': entry['download_url'],
               'expected_git_blob_sha1': entry['sha'], 'expected_bytes': entry['size']}
    try:
        data = out.read_bytes() if out.exists() else urllib.request.urlopen(entry['download_url'], timeout=35).read(20_000_000)
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != entry['size'] or blob != entry['sha']:
            raise ValueError('upstream size/blob hash mismatch')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        receipt.update(status='verified', sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc))
    return receipt


if __name__ == '__main__':
    entries = json.loads((ROOT / 'upstream_listings.json').read_text(encoding='utf-8'))[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(one, entries))
    (ROOT / 'mesh_fetch_receipt.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    print(json.dumps({'verified': sum(r['status'] == 'verified' for r in receipts), 'total': len(receipts),
                      'failures': [r for r in receipts if r['status'] != 'verified']}))
