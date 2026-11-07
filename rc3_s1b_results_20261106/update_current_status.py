"""Advance mutable project pointers; leave frozen candidate/evidence untouched."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rc3_s1b_results_20261106.audit_s1b import HERE,RUN,read,digest,SH,PR,AH

VERSION='2026.11.06-s1b-three-seed-complete'

def main():
    audit=read(HERE/'MECHANICAL_AUDIT.json');diag=read(HERE/'TEACHER_STATE_DIAGNOSTICS.json')
    review_path=HERE/'INDEPENDENT_REVIEW.json'
    auth='rc3_global_local_execution_v3/authorizations/20261106_S1_B/'
    p=ROOT/'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json'
    d=read(p)
    d.update(status='APPROVED_V3_S1_A_AND_S1_B_COMPLETE_S2_NOT_AUTHORIZED',
        original_S1_A_source_manifest_sha256='435b1c8be95d6ec4f14f4aa87913a6eb3e937e3684eaadfabbad160b57ffe968',
        source_manifest='rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json',source_manifest_sha256=SH,
        source_revision_approval=auth+'SOURCE_REVISION_APPROVAL.json',source_revision_approval_sha256=PR,
        stage_authorization=auth+'S1_B_AUTHORIZATION.json',stage_authorization_sha256=AH,
        authorized_stages=['S1-A','S1-B'],S1_B_authorized=True,S2_authorized=False,S3_authorized=False,
        S1_B_S2_S3_authorized=False,S1_B_S2_S3_authorized_scope='Conjunction: S2 and S3 remain unauthorized; S1-B was explicitly authorized and is complete.',
        downstream_S1_B='COMPLETE_ALL3_FIXED_ENDPOINTS',S1_B_runtime='COMPLETE',S1_B_scientific_optimizer_updates=768,
        S1_B_report='rc3_s1b_results_20261106/README.md',S1_B_mechanical_audit='PASS_17450_CHECKS',
        S1_B_independent_review='rc3_s1b_results_20261106/INDEPENDENT_REVIEW.md' if review_path.exists() else 'RUNNING',
        S1_B_completed_seeds=[s['seed'] for s in audit['seeds']],S1_B_new_scene_navigation_native_H_teacher=0,
        S1_B_completion_sha256={str(s['seed']):s['completion_sha256'] for s in audit['seeds']},
        S2_technical='GO_SOURCE_LINEAGE_AND_NUMERICAL_STABILITY',S2_runtime='NOT_STARTED',S3_runtime='NOT_STARTED',
        original60='SEALED',S3_target_tasks='SEALED',
        approved_source_bytes_changed=False,
        pending_source_revision=None,
        approved_source_revision=dict(path='rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json',sha256=SH,
            approved=True,approval_receipt=auth+'SOURCE_REVISION_APPROVAL.json',approval_receipt_sha256=PR),
        working_tree_execution_sources='APPROVED_COMPATIBILITY_REVISION_0f0ffd2; exact independent S1-B receipt consumed for all3 seeds; S2 requires its own authorization.',
        result_report='rc3_s1b_results_20261106/README.md')
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    p=ROOT/'PROJECT_VERSION.json';raw=p.read_bytes()
    old=read(p)['version'].encode();raw=raw.replace(b'"version": "'+old+b'"',b'"version": "'+VERSION.encode()+b'"',1)
    current=dict(status='S1_B_COMPLETE_ALL3_SEEDS',experiment_id='rc3_v3_global_local_20261106_01',
        branch='codex/global-local-preflight',protocol_approved=True,source_revision_approved=True,
        approved_source_manifest_sha256=SH,S1_B_authorized=True,scientific_optimizer_updates=768,
        new_teacher_scene_navigation_native_H=0,S1_A_original_bytes_preserved=True,
        report='rc3_s1b_results_20261106/README.md',active_status='rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json',
        S2_technical='GO',S2_authorized=False,S2_executed=False,S3_authorized=False,S3_executed=False,
        old60='SEALED',main_merged=False,
        historical_metadata_scope='Earlier per-phase objects retain their original-turn status; current execution authority and status are this record and ACTIVE_EXECUTION_STATUS.')
    key='S1_B_scientific_execution'
    if key not in read(p):
        end=raw.rfind(b'}');tail=raw[end+1:]
        item=json.dumps({key:current},ensure_ascii=False,indent=2)[1:-1].strip('\n')
        raw=raw[:end].rstrip()+b',\n'+item.encode('utf8')+b'\n}'+tail
        p.write_bytes(raw)
    text=f'''<!-- S1B_CURRENT_BEGIN -->
**最新执行状态：S1-B三个种子已完成，共768次正式监督更新，S2技术GO但未授权/未执行。** 全256训练教师保存状态上，CE从3.8087/3.7942/3.5688降至1.9483/1.9656/1.9130，Top1为38.56%/38.60%/39.40%；这是部分教师模仿，不能宣称完整路线或DRL收益。新增scene/nav/native/H/teacher均0，原S1-A1482文件字节未变，S3/原60继续封存。见[实际训练权重、曲线与审查](rc3_s1b_results_20261106/README.md)和[当前准确授权状态](rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json)。旧源码批准及原协议字节保留，新0f0ffd2源码批准与S1-B收据分别归档。

以下历史阶段段落按发生顺序保留，各阶段“当时未批准/未训练”等描述不代表当前状态；当前以上述新记录为准。
<!-- S1B_CURRENT_END -->
'''
    p=ROOT/'README.md';raw=p.read_bytes().replace(b'2026.11.06-navigation-compatibility-pass',VERSION.encode())
    if b'<!-- S1B_CURRENT_BEGIN -->' not in raw:
        marker='当前验证范围'.encode('utf8');pos=raw.index(marker)
        raw=raw[:pos]+text.replace('\n','\r\n').encode('utf8')+b'\r\n'+raw[pos:]
        p.write_bytes(raw)
    p=ROOT/'rc3_global_local_execution_v3/README.md';raw=p.read_bytes()
    if b'<!-- S1B_CURRENT_BEGIN -->' not in raw:
        head=('<!-- S1B_CURRENT_BEGIN -->\n**当前：v3及0f0ffd2兼容源码已经明确批准，S1-A/S1-B实际完成。** '
              'S1-B三seed各256更新，S2仍未授权/未启动，S3与原60继续封存。'
              '见[当前活动执行记录](ACTIVE_EXECUTION_STATUS.json)、[本轮真实收据](authorizations/20261106_S1_B/)及'
              '[三种子训练实际结果](../rc3_s1b_results_20261106/README.md)。'
              '下文为候选定稿时的历史状态，候选原文件中的PENDING不覆盖其后独立批准收据。\n<!-- S1B_CURRENT_END -->\n\n')
        first=raw.index(b'\n')+1
        raw=raw[:first]+b'\n'+head.encode('utf8')+raw[first:];p.write_bytes(raw)
    print('Current execution pointers updated; frozen protocol/source/data unchanged.')

if __name__=='__main__':main()
