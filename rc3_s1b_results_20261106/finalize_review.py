"""Persist independent claim/figure review traces, never run science."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def md(p,s):p.write_text(s.strip()+'\n',encoding='utf8',newline='\n')

def main():
    r=read(HERE/'INDEPENDENT_REVIEW.json');a=read(HERE/'MECHANICAL_AUDIT.json')
    assert r['status']=='PASS_WITH_SCOPE_LIMITS' and r['technical_S2']['verdict']=='GO'
    assert all(s['data_stream_seed']==t['data_seed'] for s,t in zip(a['seeds'],r['per_seed_training']))
    day=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d')
    trace=HERE/'paper/review-traces/experiment-result-to-claim'/f'{day}_run01'
    trace.mkdir(parents=True,exist_ok=True)
    prompt='''Review the actual completed three-seed S1-B at approved source0f0ffd2 and protocol0c5bc7d. Independently check raw receipts/768updates/27checkpoints/Adam/RNG/originalS1Abytes; reconstruct batch RNG. Recompute all49752savedstate/1536route aggregates and verify plots/captions. No models, optimizers, scene/nav/teacher/H queries or repeat production replay. Evaluate completedS1-B and partial training-prefix imitation separately from the still unproven fixed large-parentDRL→frozenlocal complete planning objective. Report supports/unsupported/missing/revisedclaim/confidence and S2 technical versus authority status. Do not start S2/S3 or original60 tests. Process startups/receipt closures not independently instrumented; preserve measured clock scopes. Paper claim audit unavailable means writing verdict provisional.'''
    md(trace/'REVIEW_PROMPT.md',prompt)
    (trace/'raw_response.md').write_bytes((HERE/'INDEPENDENT_REVIEW.md').read_bytes())
    (trace/'raw_response.json').write_bytes((HERE/'INDEPENDENT_REVIEW.json').read_bytes())
    fields=['claim_supported','stage_claim_supported','what_results_support','what_results_dont_support','missing_evidence',
        'suggested_claim_revision','next_experiments_needed','confidence','paper_claim_audit','writing_verdict','routing_action']
    write(trace/'parsed_verdict.json',{k:r[k] for k in fields})
    write(trace/'TRACE_STATUS.json',dict(reviewer='Independent delegated agent',pending_external_review=False,
        formal_paper_claim_audit=False,writing_provisional=True,source_review_sha256=digest(HERE/'INDEPENDENT_REVIEW.json'),
        new_experiments_started=False,S2_authorized=False,S3_authorized=False))
    md(trace/'routing_action.md','Confirm completedS1-B and partial training-teacher-prefix imitation only. Keep fixedDRL objective unproven; registered S2 requires separate real user stage authority. No automatic supplementary experiment, retraining, S3 or old60 access.')
    figures=HERE/'paper/review-traces/paper-figure'/f'{day}_run01';figures.mkdir(parents=True,exist_ok=True)
    md(figures/'REVIEW_PROMPT.md','Verify the two data-generated S1-B learning/endpoint plots and captions against raw logs/CSV, inspect PNGs and vector PDFs. Confirm denominators, smoothing, fixed endpoints, seed styling and narrow claim limits. No science rerun.')
    (figures/'raw_response.md').write_bytes((HERE/'INDEPENDENT_REVIEW.md').read_bytes())
    write(figures/'verdict.json',dict(status='PASS',independent_review_sha256=digest(HERE/'INDEPENDENT_REVIEW.json'),
        curves_checked=768,states_checked=49752,PNG_visual_review=True,PDF_vector_review=True,
        captions='Training teachers, teacher forcing, fixed endpoints and no route/DRL claim explicit',required_changes=[]))
    md(HERE/'RESULT_TO_CLAIM.md',f'''# 结果到结论

**阶段窄主张得到支持；整体DRL研究主张仍为partial。S2技术GO，当前无S2执行授权。**

独立审查271,538断言通过。三seed各256次真实batch4优化完成；固定256训练教师、每端点8292保存状态上的路线宏CE与状态微Top1均改善，原教师证据及精确源/权重/Adam/RNG谱系保持。可据此描述“部分训练教师状态模仿”和后续S2的准确初始化来源。

最终Top1仅38.56%–39.40%，不能称为可靠复制完整教师策略。没有执行预测替代动作，没有新的模型路线J/L/K、完整规划时间、独立可行率或未见场景结果，更没有DRL增量。代理约束证据不等于真实机械臂IK/关节轨迹认证。

后续固定目标不变：新S2阶段明确授权后，按原v3从各自SL256分叉RL与匹配对照；S3及原60仍封存。本轮没有追加任何实验补齐论文主张。

置信度high仅适用于本阶段数值/谱系及描述性训练前缀学习证据。无`paper/PAPER_CLAIM_AUDIT.json`，论文写作结论标注`provisional — no paper claim audit available`，不冒充正式论文主张审计。

详细[独立审查](INDEPENDENT_REVIEW.md)、[机器可读判定](INDEPENDENT_REVIEW.json)、[完整审查轨迹]({trace.relative_to(HERE).as_posix()}/)。
''')
    md(HERE/'EXPERIMENT_RESULT_TO_CLAIM.md','本阶段`claim_supported=partial`，`stage_claim_supported=yes`。详细判定、支持范围、证据缺口和后续权限见[RESULT_TO_CLAIM.md](RESULT_TO_CLAIM.md)。固定DRL目标不变，未授权阶段未执行。')
    md(HERE/'FIGURE_QA.md','''# 训练图表验收

PASS。执行者及独立审查者均查看两张PNG；PDF保留矢量字体，无栅格截图替代数据图。独立重算全部768原始曲线点/尾随均值与6端点汇总，图与日志/CSV一致。

图1展示实际batch CE（浅色原始、深色最多16点尾随均值）及裁剪前梯度对数轴/阈值1；图2使用固定全部训练教师，左为路线宏CE，右为状态微Top1，0–100%纵轴。三种子颜色和线型区分，图例与标签可读，未选最佳seed或checkpoint。

说明保留训练集/teacher forcing/固定端点/未验证路线和DRL收益的边界。复现脚本、CSV、PNG/PDF和LaTeX include均在figures/。审查原文见INDEPENDENT_REVIEW.md，独立审查不是正式论文数字/引用审计。
''')
    resource=read(HERE/'RESOURCE_ACCOUNTING.json')
    resource['independent_review']=r['independent_verification']
    resource['figure_generation']=dict(plot_training_tool_wall_s=3.3081704,plot_teacher_agreement_tool_wall_s=3.1887195,
        executed_concurrently=True,outer_batch_tool_wall_s=4.2,optimizer_updates=0,teacher_queries=0,
        scope='Read-only plots from saved logs and fixed endpoint metrics; overlapping durations are not summed')
    resource['review_serialization_correction']=r.get('report_serialization_correction')
    write(HERE/'RESOURCE_ACCOUNTING.json',resource)
    sources={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts}
    write(HERE/'ANALYSIS_SOURCE_BINDING.json',dict(schema='rc3-s1b-analysis-only-source-binding-v1',source_sha256=sources,
        production_source_manifest='rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json',
        production_source_manifest_sha256='0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0',
        scope='Additional data/plot/review/publication analysis only; approved production source/algorithm/protocol unchanged; no new scientific execution authority.'))
    raw=subprocess.check_output(['git','diff','--name-only','-z','ffcfcabc4ccc0b071a368b5e6e49684be3de4d64','HEAD','--',
        'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01',
        'rc3_global_local_execution_v3/authorizations/20261106_S1_B'],cwd=ROOT)
    names={s.decode('utf8') for s in raw.split(b'\0') if s}
    exclude={'ARTIFACT_INDEX.json','GIT_VERSION_AUDIT.json','PUBLICATION_RECEIPT.json','RESOURCE_ACCOUNTING.json'}
    for p in HERE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name not in exclude:
            names.add(p.relative_to(ROOT).as_posix())
    records={n:dict(sha256=digest(ROOT/n),bytes=(ROOT/n).stat().st_size) for n in sorted(names)}
    write(HERE/'ARTIFACT_INDEX.json',dict(schema='rc3-s1b-science-and-analysis-artifact-index-v1',files=records,
        count=len(records),bytes=sum(v['bytes'] for v in records.values()),excluded_mutable_metadata=sorted(exclude),
        original_S1_A_index='rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/teachers256/POOL_INDEX.json',
        original_S1_A_pool_sha256='98c1e7335fced56589d6d978d1eb3c9a895590c065413c8746d77c03a2044a1a',
        scope='Raw scientific checkpoints/logs/receipts plus fixed diagnosis, plots and independent review. Project GIT_MANIFEST also hashes mutable delivery metadata.'))
    print('Independent claim/figure traces, resource supplement, analysis binding and artifact index saved.')

if __name__=='__main__':main()
