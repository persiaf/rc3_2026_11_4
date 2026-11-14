"""Static delivery checks only; not another engineering suite or model test."""
import ast
import json
import re
import subprocess
from pathlib import Path
from .persistence import read, sha, digest

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def main():
    originals = read(HOME / 'HISTORICAL_BINDING.json')['files']
    for name, expected in originals.items():
        assert sha(ROOT / name) == expected, name
    binding = read(HOME / 'NEW_SOURCE_BINDING.json')
    for name, expected in binding['files'].items():
        assert sha(ROOT / name) == expected, name
    protocol = read(HOME / 'RECOVERY_PROTOCOL_DRAFT.json')
    status = read(HOME / 'CURRENT_STAGE_STATUS.json')
    assert status['proposal_file_sha256'] == sha(HOME / 'RECOVERY_PROTOCOL_DRAFT.json')
    assert status['proposal_canonical_sha256'] == digest(protocol)
    refs = []
    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                assert sha(ROOT / value['path']) == value['sha256'], value['path']
                refs.append(value['path'])
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
    walk(protocol)
    rows = read(HOME / 'RECOVERY_SCHEDULE.json')['rows']
    new_ids = [rid for row in rows for rid in row['request_ids']]
    assert len(new_ids) == len(set(new_ids)) == 2192
    assert [r['update'] for r in rows] == list(range(239, 513))
    for row in rows:
        assert row['optimizer_id'] == f'init_{row["update"]:04}'
        assert row['request_ids'] == [f'recovery_v1_init_{row["update"]:04}_teacher_{s}' for s in range(8)]
    assert 1910 + len(new_ids) == 4102 and 238 + len(rows) == 512
    assert abs(2221.312 + 12178.688 - 14400) < 1e-10
    assert not (ROOT / protocol['canonical_recovery_folder']).exists()
    old_init = ROOT / 'rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init'
    assert not (old_init / 'COMPLETE.json').exists() and not (old_init.parent / 'paired').exists()
    python_files = list(HOME.glob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(encoding='utf8'))
    json_files = list(HOME.rglob('*.json'))
    for path in json_files:
        read(path)
    links = []
    for path in HOME.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf8')):
            if '://' in target or target.startswith('#'):
                continue
            assert (path.parent / target.split('#', 1)[0]).exists(), (path.name, target)
            links.append(target)
    subprocess.run(['git', 'diff', '--quiet', '527c31c0911796d01c47723264f97ce80e31b0e0', '--',
                    'rc3_relation_drl_20261113', 'rc3_relation_m2_20261113'], cwd=ROOT, check=True)
    result = dict(status='PASS_STATIC_DELIVERY_CHECKS', historical_files_unchanged=len(originals),
        source_binding_files=106, protocol_references_checked=len(refs), Python_AST_files=len(python_files),
        JSON_files=len(json_files), relative_markdown_links=len(links), new_request_ids=2192,
        original_model_and_reports_untouched=True, original_SL512_COMPLETE_absent=True,
        recovery_run_directory_absent=True, paired_directory_absent=True,
        scientific_optimizer_calls=0, formal_teacher_replays=0, new_model_tests=0,
        validation_note='Fresh static hashes/metadata checks after edits; no replay of the engineering suite')
    (HOME / 'DELIVERY_CHECKS.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
