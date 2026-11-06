"""Verify original LFS working-tree bytes; no project/scientific imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = '057830d584b6c71c79ef5118087c681c98135fe2'
MAINTENANCE = {'AGENTS.md', 'README.md', 'PROJECT_VERSION.json'}


def main():
    start = time.perf_counter()
    raw = subprocess.check_output(['git', 'show', BASE + ':project_management/GIT_MANIFEST.json'], cwd=ROOT)
    old = json.loads(raw)
    failed, changed_metadata = [], []
    for name, expected in old['files'].items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            raise ValueError(name)
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                h.update(block)
        actual = {'sha256': h.hexdigest(), 'bytes': p.stat().st_size}
        if actual != expected:
            (changed_metadata if name in MAINTENANCE else failed).append(name)
    result = {'schema': 'rc3-preflight-frozen-working-tree-verification-v1', 'base': BASE,
              'base_manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'files_checked': len(old['files']), 'bytes_checked': old['total_bytes'],
              'allowed_maintenance_metadata': sorted(MAINTENANCE),
              'changed_maintenance_metadata': changed_metadata, 'scientific_file_failures': failed,
              'all_frozen_scientific_bytes_passed': not failed,
              'wall_s': time.perf_counter() - start,
              'scope': 'Byte-level reads only, including sealed assets; no deserialization, scene/task construction, planning, training or test queries.'}
    dest = ROOT / 'rc3_global_local_impl_review/FROZEN_BYTE_CHECK.json'
    if dest.exists():
        raise RuntimeError('Receipt already exists; do not overwrite')
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=True))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
