"""Create a source qualification mapping only after audited M2a success."""
import json
from pathlib import Path
from rc3_relation_recovery_20261114.persistence import read, sha, digest

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def write(name, value):
    (HOME / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def main():
    audit = read(HOME / 'CLOSED_RUN_AUDIT.json')
    if audit['status'] != 'PASS_CLOSED_RECOVERY_AUDIT' or audit['scientific_stage_outcome'] != 'COMPLETE_RECOVERED':
        raise RuntimeError('No successful audited recovered model to qualify')
    protocol_path = ROOT / 'rc3_relation_drl_20261113/M2_PROTOCOL.json'
    protocol = read(protocol_path)
    checkpoint = audit['checkpoint']
    mapping = dict(schema='M2b-recovered-initialization-qualification-v1', status='SOURCE_QUALIFIED_NOT_AUTHORIZATION',
        scientific_id=protocol['scientific_id'], qualification_audit=ref(HOME / 'CLOSED_RUN_AUDIT.json'),
        original_M2_protocol=ref(protocol_path), recovered_completion=checkpoint['receipt'],
        recovered_SL512_export=checkpoint['export'], exact_model_sha256=checkpoint['export']['model_sha256'],
        origin_checkpoint=checkpoint['reference'], original_Adam_state_verified_step=512,
        stream_initialization=dict(RelationRL=checkpoint['export'], RelationSL=checkpoint['export'],
            RelationInit=checkpoint['export'], both_streams_identical_model_parameters=True,
            separately_fresh_Adam_required=True, reuse_SL512_Adam_for_M2b=False,
            optimizer_specification=protocol['optimizer'], stream_setup_seed=protocol['sampling_seed']),
        original_schedule=protocol['schedule'], original_evaluation_cases=protocol['evaluation_cases'],
        original_evaluation_grid=protocol['evaluation_grid'], original_training=protocol['training'],
        original_evaluation=protocol['evaluation'], original_gates=protocol['gates'],
        paired_limits=protocol['limits']['paired'], original_problem=protocol['retained'],
        unchanged_training_gate_digest=digest(dict(training=protocol['training'], evaluation=protocol['evaluation'],
            gates=protocol['gates'], limits=protocol['limits']['paired'], optimizer=protocol['optimizer'])),
        current_loader_accepts_recovered_receipt=False,
        loader_incompatibility=dict(original_required_status='COMPLETE', actual_status='COMPLETE_RECOVERED',
            original_required_stage='init', actual_stage='M2a_recovery',
            original_required_cursor='top-level cursor=512', actual_cursor='checkpoint.cursor=512',
            original_required_protocol_sha256='original M2 protocol hash', actual_protocol_reference='recovery proposal digest and upstream binding'),
        required_new_admission=dict(recovered_receipt_adapter_source_binding=True,
            explicit_initialization_provenance_amendment=True, new_M2b_protocol_approval=True,
            separate_M2b_stage_authorization=True, no_relabeling_of_old_COMPLETE=True),
        M2b_implementation_verdict='REVISE_RECOVERED_RECEIPT_ADMISSION_REQUIRED',
        M2b_execution_authorized=False, M2b_optimizer_calls=0, M2b_trajectory_calls=0,
        additional_seeds_or_ablations_authorized=False)
    write('M2B_INITIALIZATION_MAPPING.json', mapping)
    report = f'''# M2b阶段准入评审

**模型来源资格：PASS。当前M2b执行准入：REVISE，尚未授权。**

经[CLOSED_RUN_AUDIT.json](CLOSED_RUN_AUDIT.json)核对，恢复模型准确来自SL238及原239—512日程，累计512次更新、4102次请求。最终模型张量哈希为`{mapping['exact_model_sha256']}`。模型、Adam(step512)、RNG、游标、导出和完整恢复收据相互匹配。原TECHNICAL_INCOMPLETE及6次历史消费保留。

准确初始化来源见[M2B_INITIALIZATION_MAPPING.json](M2B_INITIALIZATION_MAPPING.json)：RelationInit、RelationRL和RelationSL均指向同一SL512_RECOVERED导出；RL和继续SL必须各建一个新Adam，不能继续使用SL512的Adam动量。M2b随机种子、2048+2048更新、原配对日程及0/512/2048评价点保持不变。

当前原m2.py的authorize严格要求上游收据status=COMPLETE、stage=init、顶层cursor=512及原protocol_sha256。真实恢复收据为COMPLETE_RECOVERED/M2a_recovery，并通过恢复协议和checkpoint记录来源，因此不能直接冒充旧收据。此差异使旧入口暂不具备恢复结果准入能力，不能靠复制或改名原COMPLETE绕过。

下一阶段应单独绑定一个明确识别恢复收据的新准入适配版本及初始化来源修订，然后取得M2b协议批准和阶段执行授权。该修订仅处理真实来源映射，不改变训练算法、目标、数据、loss、H或质量门槛。本轮没有实施或运行M2b训练分支。

原科学门槛完整保留在机器映射中：LOCAL66与FULL12分别统计，六方法对照保持；512严重退化停止，2048未满足全部门槛为NO_DEVELOPMENT_SIGNAL；不加种子、不延长训练、不开展消融或新父测试。当前M2a成功不支持DRL收益结论。
'''
    (HOME / 'M2B_ADMISSION_REVIEW.md').write_bytes(report.encode())
    print(json.dumps(dict(source_qualification='PASS', M2b_admission='REVISE', M2b_authorized=False,
                          model_sha256=mapping['exact_model_sha256'])))


if __name__ == '__main__':
    main()
