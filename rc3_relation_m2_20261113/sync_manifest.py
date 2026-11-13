"""Update only this M2 delivery and run paths in the existing Git manifest."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def main():
    target = ROOT/'project_management/GIT_MANIFEST.json'
    text = target.read_bytes().decode('utf8')
    old = json.loads(text)
    run = ROOT/'rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113'
    paths = [ROOT/'README.md', *[p for p in HOME.rglob('*') if p.is_file()]]
    if run.exists():
        paths += [p for p in run.rglob('*') if p.is_file()]
    additions = []
    changed = []
    for path in sorted(paths):
        if path.suffix in ('.pending', '.pyc') or '__pycache__' in path.parts:
            raise RuntimeError('Unclosed or temporary delivery file: '+str(path))
        key = path.relative_to(ROOT).as_posix()
        value = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
        if old['files'].get(key) == value:
            continue
        block = '    '+json.dumps(key, ensure_ascii=False)+': '+json.dumps(value, indent=2).replace('\n', '\n    ')
        if key in old['files']:
            pattern = r'(?m)^    '+re.escape(json.dumps(key, ensure_ascii=False))+r': \{[\s\S]*?^    \}(,?)\r?\n'
            text, n = re.subn(pattern, lambda match: block+match[1]+'\n', text, count=1)
            assert n == 1, key
            changed.append(key)
        else:
            additions.append(block+',\n')
    if additions:
        text, n = re.subn(r'  "files": \{\r?\n', lambda _: '  "files": {\n'+''.join(additions), text, count=1)
        assert n == 1
    status = json.loads((HOME/'CURRENT_STAGE_STATUS.json').read_text(encoding='utf8'))
    meta = dict(base_commit='9c878928e7f619bce25ddd00792757431ca705a1',
        branch='codex/rc3-relation-m2-20261113', entry='rc3_relation_m2_20261113/README.md',
        status=status, old_protocol_and_source_unchanged=True, user_files_excluded=True, main_merged=False)
    block = '  "relation_m2_science_20261113": '+json.dumps(meta, ensure_ascii=False, indent=2).replace('\n', '\n  ')+',\n'
    if 'relation_m2_science_20261113' in old:
        text, n = re.subn(r'(?m)^  "relation_m2_science_20261113": \{[\s\S]*?^  \},\r?\n', lambda _: block, text, count=1)
        assert n == 1
    else:
        pos = text.index('\n')+1
        text = text[:pos]+block+text[pos:]
    entries = json.loads(text)['files']
    for key, value in [('file_count', len(entries)), ('total_bytes', sum(x['bytes'] for x in entries.values()))]:
        text, n = re.subn(r'("'+key+r'": )\d+', lambda match: match[1]+str(value), text, count=1)
        assert n == 1
    text, n = re.subn(r'("version": )"[^"]+"', r'\g<1>"2026.11.13-relation-m2-science"', text, count=1)
    assert n == 1
    assert all(entries[k] == value for k, value in old['files'].items() if k not in changed)
    target.write_bytes(text.encode())
    print(dict(new_entries=len(additions), changed_entries=len(changed), total_entries=len(entries)))


if __name__ == '__main__':
    main()
