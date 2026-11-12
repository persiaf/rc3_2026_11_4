"""Scoped delivery metadata; never scan protected task data or refresh user files."""
from pathlib import Path
import json,hashlib,subprocess,datetime,sys

H=Path(__file__).resolve().parent;ROOT=H.parent
TRACE=ROOT/'paper/review-traces/experiment-result-to-claim/20261112_d5_revision_run01'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

now=datetime.datetime.now(datetime.timezone.utc)
start=datetime.datetime.fromisoformat(load(H/'controls/science-03/SUPERVISOR_RECEIPT.json')['finish_utc'])
review_start=(TRACE/'REVIEW_PROMPT.md').stat().st_mtime
review_end=max((TRACE/n).stat().st_mtime for n in ['REVIEWER_RAW.md','FIGURE_REVIEW.md','FIGURE_REVIEW.json','TRACE_MANIFEST.json'])
parallel_review=max(0,review_end-review_start)
calendar=(now-start).total_seconds()
r=load(H/'RESOURCE_ACCOUNTING.json');raw=load(H/'RESOURCE_SCIENTIFIC_FINAL_RAW.json')
post_jobs=sum(v['wall_s'] for k,v in r['jobs'].items() if k.startswith('post-'))
final='--final' in sys.argv
# Includes all root-side audit, rendering, reads, writing, Git and waiting since
# the scientific tree exit. Already measured post-jobs are included once.
tail_upper=120 if final else 0
reproduction=max(post_jobs,calendar)+parallel_review+tail_upper
assert reproduction<=7200 and raw['wall']['total_s']+reproduction<=345600
assert r['counts']==raw['counts'], 'Delivery must not change scientific counters'
r['wall']['reproduction_s']=reproduction;r['wall']['total_s']=raw['wall']['total_s']+reproduction
r['delivery_accounting']=dict(status='CLOSED_WITH_BOUNDED_METADATA_TAIL' if final else 'PUBLICATION_IN_PROGRESS',
    scientific_tree_finish_utc=start.isoformat(),root_closeout_through_utc=now.isoformat(),
    root_full_calendar_s=calendar,measured_post_jobs_s=post_jobs,
    root_other_closeout_s=max(0,calendar-post_jobs),parallel_reviewer_window_s=parallel_review,
    reviewer_window_kind='Conservative prompt-to-final-trace full wall, including idle; charged in addition to root interval',
    final_metadata_tail_charged_upper_bound_s=tail_upper,
    final_metadata_tail_rule='Final metadata commit/push/readback must finish within120s; additional work requires another actual accounting receipt' if final else None,
    not_scientific_compute=True,requests_unchanged=True,
    earliest_unmeasured_reads='UNKNOWN preserved; no false zero',
    note='Scientific raw ledger retained; data is a subset of training wall, not an additional pool.')
save(H/'RESOURCE_ACCOUNTING.json',r)
receipt=H/('DELIVERY_ACCOUNTING_FINAL.json' if final else 'DELIVERY_ACCOUNTING_BEFORE_PUBLICATION.json')
if not receipt.exists():save(receipt,dict(wall=r['wall'],accounting=r['delivery_accounting'],global_limit_s=345600,reproduction_limit_s=7200))

p=ROOT/'README.md'
block='''<!-- D5_REVISION_CURRENT_BEGIN -->
## 当前：完整教师与科学驱动完成，D5固定P1负结果收口

[本轮交付](rc3_d5_revision_20261112/README.md) · [完整任务结果](rc3_d5_revision_20261112/EXPERIMENT_REPORT.md) · [给老师的短材料](rc3_d5_revision_20261112/TEACHER_DECISION_BRIEF.md)。新增96条完整10×10强教师覆盖96个训练父布局，旧256标签原字节不变；PG和SL-U逐任务同日程，各完成4096更新。固定终点H后J父配对收益对FrozenSL为−2.6431%、对匹配SL-U为−3.7371%，SEL8/DEV3均负，按原门槛NO_DEVELOPMENT_SIGNAL→P6；P2—P5均NOT_RUN。累计33954请求、Actor8202（科学8192、工程10）、Critic0；449149项保存输出复核及独立角色主张审查完成。实际模型、逐请求、raw/H分量、时间/资源和恢复证据完整保存；旧v1技术停止、封存数据、历史权重与用户文件保持，不合并main。
<!-- D5_REVISION_CURRENT_END -->

'''
# README had no user edit at admission. Preserve its mixed historical line
# endings byte-for-byte rather than normalizing old sections during maintenance.
old_readme=subprocess.check_output(['git','show','20478a4dde2888f86abb9526c9d39ab71a27ee52:README.md'],cwd=ROOT)
p.write_bytes(block.encode('utf-8').replace(b'\n',b'\r\n')+old_readme)

specs=[H.name,'paper/review-traces/experiment-result-to-claim/20261112_d5_revision_run01',
       'rc3_r04_m1_recovery_20261111/run_claims/d5-data-revision-20261112-r1-*',
       'rc3_r04_m1_recovery_20261111/run_claims/d5-revision-*']
out=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--',*specs],cwd=ROOT)
paths=sorted({v.decode('utf-8') for v in out.split(b'\0') if v})
manifest_path=ROOT/'project_management/GIT_MANIFEST.json';m=load(manifest_path)
scope=paths+['README.md','.gitattributes']
for rel in scope:
    p=ROOT/rel
    if p.is_file():m['files'][rel]=dict(sha256=sha(p),bytes=p.stat().st_size)
m['version']='2026.11.12-d5-data-revision-P1-closeout'
m['file_count']=len(m['files']);m['total_bytes']=sum(v['bytes'] for v in m['files'].values())
m['d5_data_revision']=dict(branch='codex/rc3-d5-data-revision-20261112',base_commit='20478a4dde2888f86abb9526c9d39ab71a27ee52',
    run_id='d5-data-revision-20261112-r1',scientific_source_commit='f31cf3a71',new_full_teachers=96,physical_training_parents=96,
    old256_bytes_unchanged=True,PG4096=True,SL_U4096=True,same_task_schedule=True,P1='NO_DEVELOPMENT_SIGNAL',
    P2_P5='NOT_RUN',report=H.name+'/EXPERIMENT_REPORT.md',resource=H.name+'/RESOURCE_ACCOUNTING.json',
    publication='PENDING_REMOTE_VERIFICATION',user_Word_entries_not_refreshed=True,main_merged=False)
if (H/'PUBLICATION_RECEIPT.json').exists():
    q=load(H/'PUBLICATION_RECEIPT.json')
    m['d5_data_revision'].update(publication=q['status'],payload_commit=q['payload_commit'],actual_new_LFS_objects=q['actual_new_LFS_objects'],actual_new_LFS_bytes=q['actual_new_LFS_bytes'],receipt=H.name+'/PUBLICATION_RECEIPT.json')
with manifest_path.open('w',encoding='utf-8',newline='') as f:f.write(json.dumps(m,ensure_ascii=False,indent=2).replace('\n','\r\n')+'\r\n')
print(json.dumps(dict(status='SCOPED_METADATA_UPDATED',scope_files=len(scope),reproduction_charged_s=reproduction,total_charged_s=r['wall']['total_s'],final=final),ensure_ascii=False))
