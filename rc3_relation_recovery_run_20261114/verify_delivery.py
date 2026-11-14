"""Static final delivery checks; reuse the closed-run tensor/source audit."""
import ast
import csv
import json
import re
from pathlib import Path
from rc3_relation_recovery_20261114.persistence import read, sha

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def main():
    audit = read(HOME / 'CLOSED_RUN_AUDIT.json')
    assert audit['status'] == 'PASS_CLOSED_RECOVERY_AUDIT'
    status = read(HOME / 'CURRENT_STAGE_STATUS.json')
    assert status['status'] == 'COMPLETE_RECOVERED' and status['execution_closed']
    mapping = read(HOME / 'M2B_INITIALIZATION_MAPPING.json')
    assert mapping['M2b_execution_authorized'] is False
    assert mapping['stream_initialization']['separately_fresh_Adam_required'] is True
    assert mapping['stream_initialization']['RelationRL'] == mapping['stream_initialization']['RelationSL'] == audit['checkpoint']['export']
    assert mapping['recovered_completion'] == audit['checkpoint']['receipt']
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
            assert (path.parent / target.split('#', 1)[0]).exists(), (path, target)
            links.append(target)
    counts = {}
    for name, expected in [('TEACHER_REQUESTS.csv', 2192), ('TRAINING_UPDATES.csv', 274)]:
        with (HOME / 'analysis' / name).open(encoding='utf8', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == expected and len({r['id'] for r in rows}) == expected
        counts[name] = len(rows)
    proposal = read(ROOT / 'rc3_relation_recovery_20261114/RECOVERY_PROTOCOL_DRAFT.json')
    run = ROOT / proposal['canonical_recovery_folder']
    files = [p for p in run.rglob('*') if p.is_file()]
    assert not any(p.suffix in ('.pending', '.pyc') for p in files)
    weights = []
    for path in sorted(run.glob('*.pt')):
        weights.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size))
    assert len(weights) == 3
    assert not (run.parent / 'COMPLETE.json').exists()
    assert not (run.parent.parent / 'paired').exists()
    result = dict(status='PASS_DELIVERY_CHECKS_ONLY', scientific_stage_status='COMPLETE_RECOVERED',
        Python_AST_files=len(python_files), JSON_metadata_files=len(json_files), markdown_links=len(links),
        CSV_rows=counts, raw_run_files=len(files), raw_run_bytes=sum(p.stat().st_size for p in files),
        LFS_weight_objects=weights, LFS_weight_bytes=sum(x['bytes'] for x in weights),
        published_snapshot_count=len(list((run/'published').glob('*.json'))),
        closed_run_tensor_source_audit_reused=True, additional_model_forwards=0,
        additional_optimizer_calls=0, additional_teacher_replays=0, M2b_calls=0,
        original_COMPLETE_absent=True, original_TECHNICAL_INCOMPLETE_preserved=True)
    (HOME / 'DELIVERY_CHECKS.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
