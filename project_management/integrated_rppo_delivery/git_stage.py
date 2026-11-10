"""Explicitly stage this study including locally scan-excluded scientific data."""
import json
import subprocess
from pathlib import Path
from rc3_integrated_rppo.common import *


def command(args):return subprocess.check_output(['git',*args],cwd=ROOT,encoding='utf8').strip()


def lfs(ref):
    d=json.loads(command(['lfs','ls-files','--json',ref]));return {e['name']:e['oid'] for e in d['files']}


def main():
    require(read(HOME/'FINAL_REPORT_COMPLETE.json')['state']['scientific']=='REVISE_P2_NOT_PASSED','Final reviewed state missing')
    existing=[n for n in command(['diff','--cached','--name-only','-z']).split('\0') if n]
    audit_path=HOME/'GIT_VERSION_AUDIT.json'
    if existing:
        allowed_root={'.gitattributes','.gitignore','README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json','RC3_Integrated_DRL_Plan_and_Codex_Task_v1.md'}
        require(audit_path.exists() and read(audit_path)['local_pre_payload_head']==command(['rev-parse','HEAD'])
            and all(n in allowed_root or n.startswith(('rc3_integrated_rppo/','project_management/integrated_rppo_delivery/','paper/review-traces/experiment-result-to-claim/20261110_run01/')) for n in existing),
            'Do not mix existing staged user edits')
    before=command(['rev-parse','HEAD']);old=lfs(BASE);current=lfs('HEAD')
    require(all(current.get(k)==v for k,v in old.items()),'Original LFS identity changed')
    require(command(['merge-base',BASE,'HEAD'])==BASE,'Not an independent descendant of requested baseline')
    remote=command(['-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/codex/rc3-integrated-rppo-v1']).split()[0]
    record=dict(status='PAYLOAD_READY_FINAL_REMOTE_VERIFICATION_PENDING',branch=command(['branch','--show-current']),baseline=BASE,
        local_pre_payload_head=before,actual_remote_pre_payload_head=remote,origin=command(['remote','get-url','origin']),
        old_LFS_files=len(old),old_LFS_paths_OIDs_preserved=True,source_binding_sha256=digest(source_path()),protocol_sha256=digest(protocol_path()),
        old_evidence_audit_sha256=digest(HOME/'CLOSEOUT_INTEGRITY_AUDIT.json'),independent_review_sha256=digest(HOME/'CLAIM_REVIEW.json'),
        original60_semantic_reads=0,old_two_missing_retried=0,force_push=False,main_merged=False,
        user_files='Unrelated modified thesis Word and untracked drafts/Office locks excluded; never stage them',
        local_runtime_scan_exclusion='Force-add only declared study evidence; lease files omitted. Remove temporary exclude after all evidence staged.',utc=now())
    if audit_path.exists():require(read(audit_path)['local_pre_payload_head']==before,'Publication audit belongs to another state')
    else:write_new(audit_path,record)
    from .metadata import main as metadata
    metadata('CLOSED_REVIEWED_P2_NOT_PASSED')
    names={'.gitattributes','.gitignore','README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json','RC3_Integrated_DRL_Plan_and_Codex_Task_v1.md'}
    for folder in (HOME,ROOT/'project_management/integrated_rppo_delivery',ROOT/'paper/review-traces/experiment-result-to-claim/20261110_run01'):
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('.lease','.pyc','.pyo')):names.add(label(p))
    ordered=sorted(names)
    for i in range(0,len(ordered),40):subprocess.run(['git','add','-f','--',*ordered[i:i+40]],cwd=ROOT,check=True)
    staged=[n for n in command(['diff','--cached','--name-only','-z']).split('\0') if n]
    require(all(n in names for n in staged),'Unrelated staged file detected')
    subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=ROOT,check=True)
    # Everything is now tracked/staged; remove this one temporary local scan
    # exclusion without changing any other user's local ignore instructions.
    exclude=ROOT/'.git/info/exclude';raw=exclude.read_bytes()
    marker=b'# Temporary live-run UI scan exclusion; immutable evidence explicitly force-added and published.\n'
    line=b'/rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/\n'
    exclude.write_bytes(raw.replace(marker,b'').replace(line,b''))
    print(json.dumps(dict(staged_files=len([s for s in staged if s]),old_LFS_files=len(old),baseline=BASE,only_study_files=True)))


if __name__=='__main__':main()
