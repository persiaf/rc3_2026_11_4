"""Update only this recovery review's entries and the repository README."""
import json
import re
from pathlib import Path
from .persistence import read, sha

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def main():
    target = ROOT / 'project_management/GIT_MANIFEST.json'
    text = target.read_bytes().decode('utf8')
    original = json.loads(text)
    paths = [ROOT / 'README.md', *[p for p in HOME.rglob('*') if p.is_file()]]
    changes, additions = set(), []
    for path in sorted(paths):
        if '__pycache__' in path.parts or '.engineering_tmp' in path.parts or path.suffix == '.pending':
            raise ValueError('Temporary file remains: ' + str(path))
        key = path.relative_to(ROOT).as_posix()
        value = dict(sha256=sha(path), bytes=path.stat().st_size)
        if original['files'].get(key) == value:
            continue
        block = '    ' + json.dumps(key, ensure_ascii=False) + ': ' + json.dumps(value, indent=2).replace('\n', '\n    ')
        if key in original['files']:
            pattern = r'(?m)^    ' + re.escape(json.dumps(key, ensure_ascii=False)) + r': \{[\s\S]*?^    \}(,?)\r?\n'
            text, n = re.subn(pattern, lambda m: block + m[1] + '\n', text, count=1)
            assert n == 1
        else:
            additions.append(block + ',\n')
        changes.add(key)
    if additions:
        text, n = re.subn(r'  "files": \{\r?\n', lambda _: '  "files": {\n' + ''.join(additions), text, count=1)
        assert n == 1
    metadata = dict(base_commit='527c31c0911796d01c47723264f97ce80e31b0e0',
        branch='codex/rc3-m2a-persistence-recovery-20261114', entry=HOME.name + '/README.md',
        status=read(HOME / 'CURRENT_STAGE_STATUS.json'), historical_files_unchanged=True,
        user_files_excluded=True, main_merged=False)
    key = 'relation_m2a_recovery_review_20261114'
    block = '  "' + key + '": ' + json.dumps(metadata, ensure_ascii=False, indent=2).replace('\n', '\n  ') + ',\n'
    if key in original:
        text, n = re.subn(r'(?m)^  "' + key + r'": \{[\s\S]*?^  \},\r?\n', lambda _: block, text, count=1)
        assert n == 1
    else:
        at = text.index('\n') + 1
        text = text[:at] + block + text[at:]
    entries = json.loads(text)['files']
    assert all(entries[k] == v for k, v in original['files'].items() if k not in changes)
    for key, value in [('file_count', len(entries)), ('total_bytes', sum(x['bytes'] for x in entries.values()))]:
        text, n = re.subn(r'("' + key + r'": )\d+', lambda m: m[1] + str(value), text, count=1)
        assert n == 1
    text, n = re.subn(r'("version": )"[^"]+"', r'\g<1>"2026.11.14-m2a-persistence-recovery-review"', text, count=1)
    assert n == 1
    target.write_bytes(text.encode())
    print(dict(changed_entries=len(changes), new_entries=len(additions), file_count=len(entries)))


if __name__ == '__main__':
    main()
