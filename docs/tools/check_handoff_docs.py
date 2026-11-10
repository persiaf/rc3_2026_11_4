"""Check authored handoff links and selected immutable bytes; no task dataset reads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    authored = [root / 'README.md', root / 'CONTRIBUTING.md']
    authored += [p for p in (root / 'docs').rglob('*.md')
                 if 'handoff_reference' not in p.parts]
    errors, links = [], 0
    for path in authored:
        content = path.read_text(encoding='utf-8')
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            links += 1
            resolved = path.parent / unquote(target.split('#')[0])
            if not resolved.exists():
                errors.append(f'{path.relative_to(root)}: missing {target}')
    identities = json.loads((root / 'docs/evidence/FROZEN_IDENTITIES.json').read_text(encoding='utf-8'))
    for item in identities['files']:
        if hashlib.sha256((root / item['path']).read_bytes()).hexdigest() != item['sha256']:
            errors.append(f'Frozen bytes changed: {item["path"]}')
    integration = json.loads((root / 'docs/evidence/SOURCE_INTEGRATION.json').read_text(encoding='utf-8'))
    original = subprocess.check_output(['git', 'show', identities['base_commit'] + ':README.md'], cwd=root)
    if original != (root / 'README_HISTORY.md').read_bytes():
        errors.append('Original README was not preserved byte for byte')
    for item in integration['selected_members']:
        if hashlib.sha256((root / item['repository_path']).read_bytes()).hexdigest() != item['sha256']:
            errors.append(f'Extracted source changed: {item["repository_path"]}')
    archive = root / integration['archive_path']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != integration['archive_sha256']:
        errors.append('Handoff ZIP changed')
    allowed = {'README.md', '.gitignore', 'CONTRIBUTING.md', 'README_HISTORY.md',
               'project_management/GIT_MANIFEST.json', integration['archive_path']}
    changed = subprocess.check_output(['git', '-c', 'core.quotepath=false', 'diff', '--name-only', identities['base_commit']],
                                      cwd=root, text=True, encoding='utf-8').splitlines()
    for name in changed:
        if name not in allowed and not name.startswith(('docs/', '.github/')):
            errors.append(f'Outside maintenance scope: {name}')
    manifest = json.loads((root / 'project_management/GIT_MANIFEST.json').read_bytes())
    old_manifest = json.loads(subprocess.check_output(
        ['git', 'show', identities['base_commit'] + ':project_management/GIT_MANIFEST.json'], cwd=root))
    checked_manifest_paths = 0
    for name in changed:
        if name in ('project_management/GIT_MANIFEST.json', 'docs/evidence/DOCUMENT_CHECKS.json'):
            continue
        payload = (root / name).read_bytes()
        expected = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        if manifest['files'].get(name) != expected:
            errors.append(f'Manifest mismatch: {name}')
        checked_manifest_paths += 1
    for name, entry in old_manifest['files'].items():
        if name not in changed and manifest['files'].get(name) != entry:
            errors.append(f'Historical manifest entry changed: {name}')
    whitespace = subprocess.run(['git', '-c', 'core.whitespace=cr-at-eol', 'diff', '--cached', '--check'],
                                cwd=root, capture_output=True, text=True, encoding='utf-8')
    if whitespace.returncode:
        errors.append('CRLF-aware whitespace check failed: ' + whitespace.stdout[:2000])
    report = dict(status='PASS' if not errors else 'FAIL', authored_markdown_files=len(authored),
                  local_links_checked=links, frozen_hashes_checked=len(identities['files']),
                  extracted_members_checked=len(integration['selected_members']),
                  original_readme_byte_identical=True if original == (root / 'README_HISTORY.md').read_bytes() else False,
                  changed_tracked_paths=changed, errors=errors,
                  task_manifest_paths_checked=checked_manifest_paths,
                  historical_manifest_entries_preserved=True if not any('Historical manifest' in e for e in errors) else False,
                  whitespace_check='git -c core.whitespace=cr-at-eol diff --cached --check',
                  whitespace_exit_code=whitespace.returncode,
                  source_excerpt_links='Preserved ZIP documents use original ZIP-relative paths; excluded from authored-link check',
                  sealed_task_semantic_reads=0)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
