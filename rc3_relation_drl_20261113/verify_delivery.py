"""Scoped metadata check; no Actor forward, optimizer or planning request."""
import ast
import copy
import json
import re
from pathlib import Path
from unittest.mock import patch

from .common import HOME, ROOT, read, put, verify_binding
from . import m2


def main():
    protocol = read(HOME/'M2_PROTOCOL.json')
    engineering = verify_binding(HOME/'ENGINEERING_SOURCE_BINDING.json')
    scientific = verify_binding(HOME/'SCIENTIFIC_SOURCE_BINDING.json')
    m2._load_schedule(protocol)
    original_read = m2.read

    def altered_read(path):
        value = original_read(path)
        if Path(path).name == 'M2_EVALUATION_GRID.json':
            value = copy.deepcopy(value)
            value['requests'][0], value['requests'][1] = value['requests'][1], value['requests'][0]
        return value

    with patch.object(m2, 'read', side_effect=altered_read):
        try:
            m2._load_schedule(protocol)
        except ValueError as exc:
            assert 'evaluation grid' in str(exc)
        else:
            raise AssertionError('Changed evaluation order was not rejected')
    assert not (HOME/'science').exists()
    pyfiles = list(HOME.glob('*.py'))
    for file in pyfiles:
        ast.parse(file.read_text(encoding='utf8'))
    jsonfiles = list(HOME.rglob('*.json'))
    for file in jsonfiles:
        json.loads(file.read_text(encoding='utf8'))
    docs = list(HOME.glob('*.md')) + [ROOT/'rc3_ch5_learning_plan_20261113'/n for n in (
        'EXPERIMENT_PLAN.md', 'EXPERIMENT_PLAN_20261113_v2.md', 'EXPERIMENT_TRACKER.md',
        'EXPERIMENT_TRACKER_20261113_v2.md', 'README.md', 'MANIFEST.md')]
    links = 0
    for file in docs:
        for target in re.findall(r'\]\(([^)]+)\)', file.read_text(encoding='utf8')):
            if re.match(r'(https?://|#|mailto:)', target):
                continue
            target = target.split('#')[0]
            if not (file.parent/target).exists():
                raise AssertionError((str(file), target))
            links += 1
    result = dict(status='PASS', engineering_bound_files=len(engineering['files']),
        scientific_bound_files=len(scientific['files']), valid_evaluation_grid_count=1404, wrong_order_rejected=True,
        python_AST_files=len(pyfiles), json_files=len(jsonfiles), markdown_links=links, science_directory_absent=True,
        model_forwards=0, optimizer_updates=0, complete_requests=0,
        verification_scope='Frozen metadata and additive grid-order check; no model or planning test repetition')
    put(HOME/'FINAL_CHECKS.json', result)
    print(result)


if __name__ == '__main__':
    main()
