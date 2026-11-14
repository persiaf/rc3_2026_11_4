"""Report actual closed-run resources and stop decision; no model execution."""
import json
from pathlib import Path
from rc3_relation_recovery_20261114.persistence import read

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def write(name, value):
    (HOME / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def main():
    audit = read(HOME / 'CLOSED_RUN_AUDIT.json')
    status = read(HOME / 'CURRENT_STAGE_STATUS.json')
    counts = audit['raw_cumulative_counts']
    success = audit['scientific_stage_outcome'] == 'COMPLETE_RECOVERED'
    outcome = audit['scientific_stage_outcome']
    resource = dict(schema='M2a-recovery-execution-accounting-v1', status=outcome,
        historical_requests=1910, historical_optimizer_attempts=238, historical_wall_s=2221.312,
        new_scientific_teacher_requests=audit['new_requests'], new_scientific_optimizer_attempts=audit['new_optimizer_attempts'],
        new_scientific_updates_completed=audit['new_updates_completed'],
        cumulative_request_attempts=counts['request_attempts'], cumulative_optimizer_attempts=counts['optimizer_attempts'],
        cumulative_optimizer_completed=counts['optimizer_completed'], raw_counts=counts,
        historical_request_failed_events_retained=1, new_failed_events=audit['new_failed_events'],
        technical_recompute_request_increment=6, cumulative_limits=dict(request_attempts=4102, optimizer_attempts=512, wall_s=14400),
        outer_wall=audit['outer_wall'], final_cursor=audit['cursor'],
        complete_request_times='analysis/TEACHER_REQUESTS.csv',
        read_only_admission=read(HOME / 'ADMISSION_CHECK.json'),
        post_run_audit_optimizer_calls=0, post_run_audit_teacher_replays=0,
        independent_engineering_updates_this_turn=0, DRL_trajectories=0, M2b_optimizer_calls=0,
        new_parent_tests=0, MATLAB_experiments=0, automatic_second_recovery_attempts=0,
        GPU_peak_memory='NOT_INSTRUMENTED_FOR_THIS_SL_RUN')
    write('RESOURCE_ACCOUNTING.json', resource)
    wall = audit['outer_wall']['cumulative_wall_s']
    runrel = read(ROOT / 'rc3_relation_recovery_20261114/RECOVERY_PROTOCOL_DRAFT.json')['canonical_recovery_folder']
    report = f'''# M2a SL238受控恢复实际报告

**结果：{outcome}。** 本轮只执行一次获批M2a恢复，M2b和DRL未运行。

| 资源 | 历史 | 本轮新增 | 累计 | 上限 |
|---|---:|---:|---:|---:|
| 教师请求 | 1910 | {audit['new_requests']} | {counts['request_attempts']} | 4102 |
| optimizer尝试 | 238 | {audit['new_optimizer_attempts']} | {counts['optimizer_attempts']} | 512 |
| 完成更新 | 238 | {audit['new_updates_completed']} | {counts['optimizer_completed']} | 512 |
| 墙钟（秒） | 2221.312 | {audit['outer_wall']['current_attempt_wall_s']:.6f} | {wall:.6f} | 14400 |

本轮沿用获批提案规范化SHA256 `40a01bf510b3addf474f428708b6c8ee93cf9f53ef30f1ba444afd8f5b0b380c`，执行源码保持106项绑定。准入独立核对SL238模型/Adam/RNG、原账、2148个原请求/optimizer ID及完整剩余日程；授权在启动前提交并核对远端。

原始第239批6次消费保留。恢复从第239批完整8次计算重新开始，所有新回放使用recovery_v1前缀，不复用旧计算图、不抵扣旧账。每次回放保留教师来源、动作及合法性证书；每次更新保留梯度、模型前后哈希和版本。监控仅读取published不可变快照，原账本未被监控并发打开。

最终游标为{audit['cursor']}。精确日程、请求唯一性、模型链、checkpoint、Adam/RNG、资源和历史原字节检查详见[CLOSED_RUN_AUDIT.json](CLOSED_RUN_AUDIT.json)。原始TECHNICAL_INCOMPLETE及complete/failed重叠事件保留；历史raw request_failed=1不可解释为本轮新增失败。

完整请求耗时在[TEACHER_REQUESTS.csv](analysis/TEACHER_REQUESTS.csv)，实际SL损失和梯度在[TRAINING_UPDATES.csv](analysis/TRAINING_UPDATES.csv)。耗时字段覆盖计算、请求记录和账本快照，后续时间事件及发布尾部由阶段墙钟计入；完整外层监督时间包含冷启动与收尾。GPU峰值未专门记录，不借用M1峰值。

原始运行目录：[recovery_v1_20261114](../{runrel})。原科学SL512 COMPLETE仍不存在；恢复完成凭据采用独立RECOVERY_COMPLETE，不冒充无中断的原初始化收据。

本轮SL初始化结果不证明DRL收益；没有运行LOCAL66/FULL12性能评价、M2b对照、额外种子或新父测试。
'''
    (HOME / 'EXPERIMENT_REPORT.md').write_bytes(report.encode())
    if success:
        checkpoint = audit['checkpoint']
        model = checkpoint['export']
        model_index = f'''# 恢复模型及来源

- [SL512_RECOVERED实际模型](../{model['path']})：文件SHA256 `{model['sha256']}`，模型张量SHA256 `{model['model_sha256']}`。
- [完整恢复收据](../{checkpoint['receipt']['path']})：SHA256 `{checkpoint['receipt']['sha256']}`。
- [最终checkpoint（模型/Adam/RNG）](../{checkpoint['reference']['path']})：SHA256 `{checkpoint['reference']['sha256']}`，游标512。
- [原SL238及恢复日程绑定](../rc3_relation_recovery_20261114/RECOVERY_PROTOCOL_DRAFT.json)。
- [实际累计资源](RESOURCE_ACCOUNTING.json) · [逐请求CSV](analysis/TEACHER_REQUESTS.csv) · [逐更新CSV](analysis/TRAINING_UPDATES.csv)。
- [M2b初始化资格映射](M2B_INITIALIZATION_MAPPING.json) · [M2b准入评审](M2B_ADMISSION_REVIEW.md)。

模型和checkpoint按仓库Git LFS规则交付。SL512的Adam状态仅用于恢复证据；M2b将要求两个分别新建的Adam，不能继承这些动量。
'''
        (HOME / 'MODEL_AND_DATA_INDEX.md').write_bytes(model_index.encode())
    decision = ('M2a恢复完成并通过只读审计。模型来源资格PASS；M2b当前准入REVISE，需要准确恢复收据适配和新的独立阶段授权。'
                if success else '本次出现新的技术中断，保留全部消费并停止；不得自动第二次恢复，也不得启动M2b。')
    (HOME / 'NEXT_STEP_DECISION.md').write_bytes(f'''# 本轮停止决定

**{outcome}；本轮执行结束。**

{decision}

本轮未修改已批准源码、原协议、数据、目标或门槛；未自动重试教师计算或optimizer。原历史TECHNICAL_INCOMPLETE、失败事件和累计消费保留。M2b、DRL2048、匹配SL2048、额外种子、消融、新父测试及MATLAB均未运行。
'''.encode())
    (HOME / 'RESULT_TO_CLAIM.md').write_bytes(f'''# 结果到结论

M2a阶段实际状态为{outcome}。只读审计核对恢复模型/Adam/RNG、精确任务日程、请求消费与检查点；这支持相应初始化来源和执行完成结论。

初始化损失变化、梯度存在及本轮持久化成功不能推导DRL改善。当前没有M2b训练或固定点评价，不支持关系Actor规划收益、优于FrozenSL/Fixed3/Detour16或泛化结论。M2b门槛和停止规则保持原登记。
'''.encode())
    print(json.dumps(dict(status=outcome, cumulative_requests=counts['request_attempts'],
                         cumulative_optimizer=counts['optimizer_attempts'], cumulative_wall_s=wall)))


if __name__ == '__main__':
    main()
