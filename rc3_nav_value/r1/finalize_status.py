"""Project bookkeeping after the saved R1 evidence passes; no research calls."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    a=read(OUT/'analysis.json');r=read(OUT/'resources.json');q=read(OUT/'qa_saved.json')
    assert a['status']=='complete' and q['all_passed'] and r['status']=='completed'
    assert a['research_gate']['status']=='fail' and not any(x['promoted'] for x in read(OUT/'promotion.json').values())
    p=read(ROOT/'PROJECT_VERSION.json');p['version']='2026.11.05-nav-value-r1'
    p['navigation_information_direction']='strict_G1_G4_not_started; separate_nav_value_R0_pass_R1_screen_not_passed'
    p['nav_value_proposal'].update(status='reviewed_design_R0_R1_executed_under_separate_protocols',R1='completed_screen_not_passed_no_R2',new_research_calls_scope='0 belongs to proposal-review task only; R0/R1 resources recorded separately')
    p['nav_value_R0']['R1']='completed_under_separate_R1_protocol_screen_not_passed'
    p['nav_value_R1']=dict(status='completed_configuration_closed',gate='not_passed_no_effective_reference_update',
        protocol='rc3_nav_value/r1/protocol.json',report='rc3_nav_value/REPORT_R1.md',training_seeds=1,
        train_cases=32,development_cases=32,development_used_for_promotion_and_screen=True,
        top_level_calls=r['ledger']['total_started'],top_level_cap=16384,full_H_calls=8806,native_calls=480,
        optimizer_updates_actual=6144,logical_updates_each_including_inherited=2048,
        reference_promotions=dict(base=False,nav=False),D_vs_B_gain_percent=0.,D_vs_B_CI95_percent=[0.,0.],
        main_gate='fail',deployment_gate=dict(cold='fail',hot='fail'),retained_test_queries=0,
        R2='not_started',R3='not_started',old_strict_information_gate='not_executed_not_passed',default_methods_unchanged=True)
    write(ROOT/'PROJECT_VERSION.json',p)
    path=ROOT/'README.md';s=path.read_text(encoding='utf-8-sig')
    s=s.replace('2026.11.05-nav-value-r0','2026.11.05-nav-value-r1')
    s=s.replace('未来候选间导航信息方向仅记录立项门槛，尚未启动搜索、模型或训练。','严格候选间导航信息必要性方向仅记录立项门槛，原G1—G4尚未执行；后续独立R0/R1价值研究不构成该严格门槛的通过。')
    s=s.replace('旧支线保持收尾，R1～R3未启动。','旧支线保持收尾；R1随后按独立任务完成，R2/R3未启动。')
    marker='## 获取完整文件'
    addition='R1 已完成单种子有界筛选并收尾当前配置：实际9,286/16,384次顶层评价、6,144步优化；首轮两表示参考均未晋升，终点A=C、B=D。D对B改善0%，对暖起点改善区间跨零，质量与完整时间均未建立强规则部署优势。没有形成有效参考更新的API证据，也没有反证API一般无效。R2/R3未启动，60保留测试未读，默认Fixed3／Detour不变。见 [R1完整报告](rc3_nav_value/REPORT_R1.md) 与 [质量—时间表](rc3_nav_value/r1/quality_time.csv)。\n\n'
    if addition not in s:s=s.replace(marker,addition+marker)
    path.write_text(s,encoding='utf-8')
    path=ROOT/'rc3_nav_value/README.md';s=path.read_text(encoding='utf-8-sig')
    s=s.replace('本目录按新任务开展R0有界教师检验。它复用冻结的覆盖、导航、完整H与验证器，没有创建或训练价值网络；旧同结构策略梯度支线保持收尾。','本目录按独立任务先完成R0有界教师检验，随后完成R1单种子CNR-VN价值学习筛选。复用冻结覆盖、导航、完整H与验证器；旧同结构策略梯度支线保持收尾。')
    s=s.replace('R1/R2/R3和原60保留测试均未启动/未查询','R1已单独完成并未通过继续门槛；R2/R3和原60保留测试均未启动/未查询')
    addition='\n## R1当前状态\n\n[R1完整报告](REPORT_R1.md)：四组同参数、完整前缀、查询优势回归/排序训练。实际9,286次终局评价、6,144步；两表示首轮晋升均失败，两轮A=C/B=D，D相对B改善0%[0,0]。主门槛与冷/热部署门槛均失败，当前配置收尾，不自动追加训练或进入R2。\n\n`r1/protocol.json`、`paper/EXPERIMENT_PLAN.md`为事前协议；`analysis.json`、`quality_time.csv`、`resources.json`及`qa_saved.json`为完整结果和验收。保存数据离线复算入口：`python -B -m rc3_nav_value.r1.summarize` 与 `python -B -m rc3_nav_value.r1.verify_saved --out rc3_nav_value/r1`；均不新增推理、规划、H或训练。\n'
    if '\n## R1当前状态' not in s:s+=addition
    path.write_text(s,encoding='utf-8')
    path=ROOT/'research_proposals/20261105_nav_value_review/paper/EXPERIMENT_TRACKER.md';s=path.read_text(encoding='utf-8-sig')
    s=s.replace('R1仍未启动；见rc3_nav_value','R1随后单独执行；见rc3_nav_value')
    s=s.replace('NOT STARTED，须R0与新任务','DONE NOT PASSED：9286/16384终局、6144更新、无参考晋升、D=B；本配置收尾')
    path.write_text(s,encoding='utf-8')
    path=OUT/'paper/EXPERIMENT_TRACKER.md';s=path.read_text(encoding='utf-8-sig')
    s=s.replace('执行前核查中','完成，已注册并锁定').replace('尚未生成研究场景、标签或优化器更新','登记在首次研究调用之前，见registration_receipt.json')
    s=s.replace('| 待执行 |','| 完成 |')
    s=s.replace('初始化4096调用；标定无优化器','初始化4096调用；32次backward标定、无优化器')
    s=s.replace('每组512步，状态菜单匹配','每组512步，A=C/B=D；两表示晋升均失败')
    s=s.replace('每组512步，不择优选旧检查点','每组512步，仍g0；A=C/B=D，未形成有效API更新')
    s=s.replace('不自动启动R2/R3','主/部署门槛均失败，当前配置收尾；R2/R3未启动')
    s+='\n总资源：9286/16384终局，6144实际更新，2225.67/7200秒阶段壁钟，0失败，60保留测试0读取。主分析完整；qa_saved.json 1,012,842检查全通过。独立主张判定见 review-traces/experiment-result-to-claim/2026-11-05_run01/。\n'
    path.write_text(s,encoding='utf-8')
    path=OUT/'README.md';s=path.read_text(encoding='utf-8-sig');s+='\n本次运行已完成并收尾：主门槛及两种缓存部署门槛均未通过，无参考晋升。请勿重复执行run；结果见父目录REPORT_R1.md及analysis.json。后验QA通过，不启动R2/R3。\n';path.write_text(s,encoding='utf-8')


if __name__=='__main__':main()
