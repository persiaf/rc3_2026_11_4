"""Describe exact future S2 lineage/commands, without granting authority."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rc3_s1b_results_20261106.audit_s1b import HERE,RUN,SEEDS,read,digest,PH,SH,PR

def main():
    predecessors={'S1-A':digest(RUN/'stage_admission/S1-A/COMPLETE.json')}
    predecessors.update({f'S1-B/{s}':digest(RUN/f'stage_admission/S1-B/{s}/COMPLETE.json') for s in SEEDS})
    auth='rc3_global_local_execution_v3/authorizations/20261106_S1_B/SOURCE_REVISION_APPROVAL.json'
    binding='rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json'
    commands=[]
    for seed in SEEDS:
        commands.append(dict(seed=seed,argv=['.\\rc3_learning\\.venv\\Scripts\\python.exe','-B','-m',
            'rc3_global_local_execution_v3.runner','--stage','S2','--seed',str(seed),'--experiment-id',
            'rc3_v3_global_local_20261106_01','--output',RUN.relative_to(ROOT).as_posix(),'--existing-run',
            '--source-binding',binding,'--protocol-approval',auth,
            '--stage-authorization','<NEW_EXPLICITLY_APPROVED_S2_STAGE_AUTHORIZATION.json>']))
    plan=dict(schema='rc3-s2-exact-future-execution-plan-v1',status='NOT_AUTHORIZED_NOT_EXECUTED',
        technical_admission='GO_SOURCE_LINEAGE_AND_NUMERICAL_STABILITY_ONLY',
        protocol_sha256=PH,source_manifest=binding,source_manifest_sha256=SH,
        protocol_and_source_approval=auth,protocol_and_source_approval_sha256=PR,
        stages=['S2'],optimizer_seeds=list(SEEDS),required_upstream_completion_hashes=predecessors,
        same_stage_seed_predecessors='Earlier S2 seed completion hashes are captured by production chain when known; do not invent future hashes or require a separate per-seed authorization.',
        source_endpoints=[dict(seed=s,path=(RUN/f'seed_{s}/S1/update_0256.pt').relative_to(ROOT).as_posix(),
            checkpoint_sha256=digest(RUN/f'seed_{s}/S1/update_0256.pt'),
            tensor_sha256=read(RUN/f'seed_{s}/S1/update_0256.pt.json')['model_tensor_sha256']) for s in SEEDS],
        per_seed_budget=dict(RL_updates=512,continued_SL_updates=512,batch_instances=4,
            RL_terminal_attempts_max=6144,development_other_terminal_attempts_max=2048,total_terminal_attempts_max=8192,
            cumulative_shared_stage_wall_s=7200,new_teacher_queries=0),
        all_seed_budget=dict(optimizer_updates_max=3072,terminal_attempts_max=24576,stage_wall_s_sum_max=21600),
        fork='Matching SL256 model tensors copied into frozen_SL,RL,continuedSL; RL and continuedSL use separate fresh Adam. Saved S1 Adam is preserved but not inherited into either S2 fork.',
        recipe='Unchanged approved v3 weighted three-start REINFORCE with detached LOO baseline and frozen full-H terminal; no PPO/API substitution, reward/mask/budget changes or rule fallback.',
        commands_after_new_stage_authorization_only=commands,
        resume='Only explicit --resume to the last complete checkpoint boundary with the SAME S2 authorization, Adam/RNG/cursor and cumulative resource clock; dirty tail/failed/complete stages cannot be silently repeated.',
        sealed=['S3 target tasks','original60 retained tasks'],
        execution_guard='This JSON is a plan, not an AUTHORIZED receipt, and is never passed as --stage-authorization. Current S1-B authorization cannot authorize S2.')
    (HERE/'S2_EXECUTION_PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    text='# S2独立授权后的准确执行方案\n\n**技术GO；当前未授权、未执行。** 原批准v3/0f0ffd2源码保持不变。可一次明确授权三个固定seed，随后按11622026→11622027→11622028执行，无需逐seed重复申请。\n\n'
    text+='新S2阶段收据必须绑定现有独立源码批准收据、协议、Registry、H、实验ID和下面四份已存在上游完成哈希；完整参数和精确端点见[S2_EXECUTION_PLAN.json](S2_EXECUTION_PLAN.json)。\n\n'
    text+='| 上游 | COMPLETE SHA256 |\n|---|---|\n'+''.join(f'| {k} | `{v}` |\n' for k,v in predecessors.items())
    text+='\n每seed从自己的SL256分叉RL与continuedSL，fresh Adam；冻结SL及原untrained保留。每seed512RL+512continuedSL、batch4、终局调用最多8192、阶段累计7200秒；三seed最多3072更新、24576终局调用、21600阶段秒。prepare/input/cache/H/development/验证/IO均在阶段账中，不能另开免费预处理。\n\n'
    text+='只在用户新增明确S2授权、真实收据已保存并核验后，将以下占位文件替换为真实授权路径运行；本轮没有执行这些命令。\n\n'
    for c in commands:
        text+='```powershell\n'+' '.join(c['argv'])+'\n```\n\n'
    text+='源码哈希、上游或checkpoint不匹配/非有限/不可行/超限即保存已花资源并停，不换seed、不补训练或重置账。中断仅按同授权安全checkpoint恢复；不选择开发最佳权重，不执行S3或原60。S1训练集Top1约39%不是S2效果保证，独立RL增量仍须正式数据。\n'
    (HERE/'S2_EXECUTION_PLAN.md').write_text(text,encoding='utf8',newline='\n')
    print('Future S2 exact plan saved; no authority created and no S2 operation invoked.')

if __name__=='__main__':main()
