"""Read-only frozen-source audit. No Torch, planning, training or benchmarks."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe_path(base,name):
    path=(base/name).resolve()
    if not path.is_relative_to(base.resolve()):raise ValueError('manifest path escapes its root')
    return path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def audit(phase='source_snapshot'):
    before=read(ROOT/'outputs/freeze_before.json')
    report=dict(schema='rc3-g2-freeze-audit-v1',utc=datetime.now(timezone.utc).isoformat(),phase=phase,
        scope='Archive/source/payload identity only. No old experiments, training, tests or performance evaluations rerun.',
        freeze_before_sha256=digest(ROOT/'outputs/freeze_before.json'))
    archive_checks=[]
    for name,expected in before['archive_sha256'].items():
        path=safe_path(WORKSPACE,name)
        actual=digest(path) if path.exists() else None
        archive_checks.append(dict(path=name,expected=expected,actual=actual,matches=actual==expected))
    report['archives']=archive_checks
    g01=WORKSPACE/'rc3_rl_research';manifest_path=g01/'PACKAGE_MANIFEST.json'
    manifest=read(manifest_path);entries=manifest['files']
    checks=[];seen=set()
    for entry in entries:
        name=entry['path'];path=safe_path(g01,name)
        if name in seen:raise ValueError('duplicate G01 manifest path')
        seen.add(name);actual=digest(path) if path.exists() else None
        size=path.stat().st_size if path.exists() else None
        checks.append(dict(path=name,sha256_matches=actual==entry['sha256'],
                           bytes_matches=size==entry['bytes']))
    with zipfile.ZipFile(g01/'outputs/rc3_rl_research_g01.zip') as archive:
        archived_manifest=archive.read('rc3_rl_research/PACKAGE_MANIFEST.json')
    manifest_same=archived_manifest==manifest_path.read_bytes()
    report['g01']=dict(payload_count=len(entries),expected_payload_count=before['g01_payloads'],
        matches=sum(c['sha256_matches'] and c['bytes_matches'] for c in checks),
        manifest_matches_frozen_archive=manifest_same,
        manifest_sha256=digest(manifest_path),checks=checks)
    formal_path=ROOT/'outputs/formal_source_manifest.json';formal=read(formal_path)
    source_checks=[]
    for name,expected in formal['source_sha256'].items():
        actual=digest(safe_path(ROOT,name))
        source_checks.append(dict(path=name,expected=expected,actual=actual,matches=actual==expected))
    report['g2_formal']=dict(manifest_sha256=digest(formal_path),source_count=len(source_checks),
                             matches=sum(c['matches'] for c in source_checks),checks=source_checks)
    protocol_path=ROOT/'outputs/protocol.json';protocol=read(protocol_path)
    protocol_sha=digest(protocol_path)
    recorded=(ROOT/'outputs/protocol.sha256').read_text().split()[0]
    report['protocol']=dict(actual=protocol_sha,recorded=recorded,matches=protocol_sha==recorded)
    specs=[]
    for folder,filename,source_key,split in (
        ('teacher_initial','collection_spec.json','source_sha256','teacher_initial'),
        ('teacher_extra','collection_spec.json','source_sha256','teacher_extra'),
        ('rule_development','selection_protocol.json','source_hashes','rule_dev')):
        spec_path=ROOT/'outputs'/folder/filename;spec=read(spec_path)
        binding=[]
        for name,expected in spec[source_key].items():
            actual=digest(safe_path(ROOT,name))
            binding.append(dict(path=name,expected=expected,actual=actual,
                                matches=actual==expected,
                                matches_formal=expected==formal['source_sha256'].get(name)))
        entry=dict(path=f'outputs/{folder}/{filename}',sha256=digest(spec_path),
            case_count=len(spec['cases']),cases_match_protocol=spec['cases']==protocol['splits'][split],
            config_matches_protocol=spec['config']==protocol['config'],sources=binding)
        if folder=='rule_development':
            selection=read(ROOT/'outputs/rule_development/selected_ils.json')
            entry.update(protocol_sha_matches=spec['protocol_sha256']==protocol_sha,
                selected_rule_binds_spec=selection['selection_protocol_sha256']==digest(spec_path),
                validation_unused=selection['validation_used'] is False and spec['validation_used'] is False,
                selection_frozen=selection['frozen'] is True)
        specs.append(entry)
    report['formal_teacher_and_rule_specs']=specs
    report['excluded_from_current_source_binding']=[
        'outputs/teacher_debug_v1_bound_rejection/ (interrupted historical teacher version)',
        'outputs/pipeline_before_resume_guards/mlp* (historical numerical probe)',
        'outputs/pipeline_before_resume_guards/graph* (historical numerical probe)',
        'outputs/training/ (live checkpoints/logs; result integrity is audited separately)',
        'outputs/prepared_train/ (rebuildable large training cache; excluded from delivery ZIP)',
        'documentation, tools, test fixtures and receipts not listed among the 18 bound production files']
    report['all_passed']=(all(c['matches'] for c in archive_checks)
        and len(entries)==before['g01_payloads']==279 and report['g01']['matches']==len(entries)
        and manifest_same and before['g01_matches'] is True
        and len(source_checks)==18 and formal.get('formal_training_started') is True
        and all(c['matches'] for c in source_checks)
        and protocol_sha==recorded
        and all(s['cases_match_protocol'] and s['config_matches_protocol']
                and all(v['matches'] and v['matches_formal'] for v in s['sources']) for s in specs)
        and all(s.get(k,True) for s in specs for k in
                ('protocol_sha_matches','selected_rule_binds_spec','validation_unused','selection_frozen')))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/freeze_after.json')
    p.add_argument('--phase',default='source_snapshot')
    a=p.parse_args();report=audit(a.phase)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_passed=report['all_passed'],phase=a.phase,
        archive_checks=len(report['archives']),g01_matches=report['g01']['matches'],
        formal_source_matches=report['g2_formal']['matches'],
        bound_specs=len(report['formal_teacher_and_rule_specs'])),ensure_ascii=False))
    if not report['all_passed']:raise SystemExit(2)


if __name__=='__main__':main()
