"""Verify a published anchor and record delivery; never starts scientific work.

Run only after a successful normal Git push of the complete report anchor.
The receipt is committed in a child of the verified anchor, avoiding self-hashes.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BRANCH = 'codex/global-local-preflight'
BASE_MAIN = '5d5fe25754ffa032f97937a3fb54f15901c2fb7d'


def call(argv):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='strict')
    if p.returncode:
        raise RuntimeError(json.dumps(dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)))
    return dict(argv=argv, returncode=0, stdout=p.stdout, stderr=p.stderr)


def write(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published-commit', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    receipt_path = HERE / 'PUBLICATION_RECEIPT.json'
    if receipt_path.exists():
        raise FileExistsError('Publication receipt already exists; do not overwrite the original verification')
    head = call(['git', 'rev-parse', 'HEAD'])['stdout'].strip()
    branch = call(['git', 'branch', '--show-current'])['stdout'].strip()
    if head != args.published_commit or branch != BRANCH:
        raise ValueError('Local published anchor/branch mismatch')
    remote_result = call(['git', '-c', 'http.version=HTTP/1.1', 'ls-remote', 'origin',
                          'refs/heads/' + BRANCH, 'refs/heads/main'])
    remote = dict(line.split()[::-1] for line in remote_result['stdout'].splitlines())
    if remote.get('refs/heads/' + BRANCH) != head or remote.get('refs/heads/main') != BASE_MAIN:
        raise ValueError('Live remote research HEAD mismatch or unexpected main change')
    fsck_started = time.perf_counter()
    fsck = call(['git', 'lfs', 'fsck', '--objects', 'HEAD'])
    fsck_wall = time.perf_counter() - fsck_started
    lfs_status = call(['git', '-c', 'core.quotepath=false', 'lfs', 'status'])
    if not call(['git', 'diff', '--cached', '--name-only'])['stdout'].strip() == '':
        raise ValueError('Unexpected staged content while recording publication')
    anchor_report = HERE / 'EXPERIMENT_REPORT.md'
    source_path = ROOT / 'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json'
    receipt = dict(schema='rc3-s2-github-publication-receipt-v1',
        verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        branch=BRANCH, local_verified_published_anchor=head,
        remote_verified_published_anchor=remote['refs/heads/' + BRANCH],
        remote_main=remote['refs/heads/main'], main_merged=False,
        push_complete_for_anchor=True, live_remote_verification=remote_result,
        LFS_integrity_verification=fsck, LFS_fsck_wall_s=fsck_wall,
        LFS_status_after_anchor_push=lfs_status,
        complete_report_sha256=hashlib.sha256(anchor_report.read_bytes()).hexdigest(),
        approved_source_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        optimizer_or_query_or_S3_calls_added=0,
        measured_publication_verification_wall_s=time.perf_counter()-started,
        receipt_commit_scope='This receipt is saved in a child commit of the already-published anchor. Its own child SHA must be checked independently after the final normal push; no self-referential hash is asserted.')
    audit_path = HERE / 'GIT_VERSION_AUDIT.json'
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    audit.update(schema='rc3-s2-published-git-version-audit-v1', branch_push_complete=True,
        remote_branch_HEAD_live_verified=True, remote_synchronized_claim=True,
        verified_published_anchor=head, actual_remote_research_HEAD_at_verification=remote['refs/heads/' + BRANCH],
        actual_remote_main_HEAD_at_verification=remote['refs/heads/main'],
        main_merged=False, publication_receipt='rc3_s2_results_20261107/PUBLICATION_RECEIPT.json',
        LFS_objects_fsck_passed=True,
        remote_verification_scope=receipt['receipt_commit_scope'])
    version_path = ROOT / 'PROJECT_VERSION.json'
    version_raw = version_path.read_bytes()
    version = json.loads(version_raw.decode('utf8'))
    # Replace only the final S2 current-state field; preserve every older object byte.
    marker = b'"branch_push_complete": false'
    position = version_raw.rfind(marker)
    if position < 0:
        raise ValueError('Missing new S2 push-state marker')
    updated_version = version_raw[:position] + version_raw[position:].replace(marker, b'"branch_push_complete": true', 1)
    parsed_updated = json.loads(updated_version.decode('utf8'))
    if version == parsed_updated:
        raise ValueError('No S2 publication state changed')
    delivery_path = HERE / 'DELIVERY_PROGRESS.json'
    delivery = json.loads(delivery_path.read_text(encoding='utf8'))
    delivery.update(branch_push_complete=True, verified_published_anchor=head,
                    publication_receipt='rc3_s2_results_20261107/PUBLICATION_RECEIPT.json',
                    scope=receipt['receipt_commit_scope'])
    write(receipt_path, receipt, True)
    write(audit_path, audit)
    version_path.write_bytes(updated_version)
    write(delivery_path, delivery)
    for p in (ROOT / 'README.md', ROOT / 'rc3_global_local_execution_v3/README.md'):
        data = p.read_bytes()
        for old, new in (('发布收据（待实际推送后生成）', '已核验发布收据'),
                         ('待推送后生成的发布收据', '已核验发布收据')):
            data = data.replace(old.encode('utf8'), new.encode('utf8'))
        p.write_bytes(data)
    print(json.dumps(dict(status='PUBLISHED_ANCHOR_VERIFIED_LFS_PASS', anchor=head,
                          remote=remote, receipt=str(receipt_path.relative_to(ROOT))), ensure_ascii=False))


if __name__ == '__main__':
    main()
