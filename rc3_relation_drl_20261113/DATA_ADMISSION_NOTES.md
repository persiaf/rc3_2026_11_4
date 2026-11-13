# 既有训练数据与工程案例准入

2026-11-13只读调查。未调用模型前向、optimizer、环境转移、H、规划器或IK，未生成新任务。下列选择依据保存输入和reference元数据，不使用成本排名选题。

## 可复用的恢复接口

- `rc3_d5_revision_20261112.data.Pool(budget)`（第9行）：读取96个完整训练题、256个原局部训练标签和66个已暴露开发定义。新工程运行应传入本轮budget适配器，不实例化旧D5/D6的Budget；后者会写旧资源账。
- `Pool.get(tid)`（第18行）：经`SavedInputs.view(spec)`恢复现有scene/F/A/navigation快照；检查物理父身份、完整题父身份及C/D只读性。没有生成或重算导航后备分支。
- `Pool.teacher(tid)`（第28行）：返回`(problem, actions_local_nodes, teacher_source)`。局部标签来自准确旧256池；完整标签来自`FULL_TEACHER_MANIFEST.json`指定gzip对象。此方法本身不调用H或新教师。
- `rc3_conservative_refinement.saved_inputs.SavedInputs.view`（第188行）：核保存字节、原/当前生产源码身份、历史导航表示及唯一父/view身份，然后调用已有`scoped_view`。该函数虽位于历史S3命名模块，但复用的是通用作用域源码，不调用S3任务提供器或读取其保存路线。
- 旧`References(pool,budget).get(tid)`（runtime.py第130行）只读保存FrozenSL完整成本，验证reference key、字节、问题身份和独立验证结果。新代码可复用读取语义，须将旧reference来源与新Actor源码分开绑定。

新budget最小读取接口为`check()`与`counts`（Counter），供`Pool`记录缓存/输入恢复；不得让这个适配器重置本轮已消费资源。

## 精确来源

|文件|SHA256|
|---|---|
|`rc3_d5_revision_20261112/JOINT_DATA_INDEX.json`|`4d27fbc82006102b3d2e443f1cfc7b196d871e8d1ec291b2aca0dcee045e79a7`|
|`rc3_d5_revision_20261112/REFERENCE_MANIFEST.json`|`9c111e0cc8cbf0d1f1b96b80e99c8796dbe1153a0aca5204940f3733ed6a717c`|
|`rc3_d5_revision_20261112/FULL_TEACHER_MANIFEST.json`|`d97afe9008192e82ea24d212612e450ce6f52030c3d63edf96d0324e37f51773`|
|`rc3_d5_revision_20261112/JOINT_DATA_ADMISSION.json`|`2960a028f82e24b7050a86c917ca6057baea8b710074ec014fb073e6423f8971`|
|`rc3_d5_revision_20261112/REVISION_PROTOCOL.json`|`cbd19bff9d7358174a647f94ca66c8fa720251dd97df9f2617e342b342d29e52`|

JOINT_DATA_INDEX含`full`96条、`local`256条。每题记录spec、physical_key、标签/view身份；完整题另含expected_parent_identity及五个input_files哈希。REFERENCE_MANIFEST的`records[tid]`给出gzip路径、sha256、reference key；解压记录含problem_identity、parent_identity、physical_key、actor_tensor_sha256、H_identity、candidate_count、scope.target_task_ids与metrics。参考标量用`metrics.J`；原始与H后指标均保存。

准确FrozenSL tensor hash为`1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`，H身份为`5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14`。本轮只读352条已允许训练reference元数据，最大候选数732，最小保存参考J为38.8486864596，均非零；这些值不是新模型性能结果。

## 建议固定的6个非空工程任务

|ID|规模|候选/任务|模式|端点|A条件|选择依据|
|---|---|---|---|---|---|---|
|F012|10×10|732/252|mixed|closed|0|96完整题中全局最大候选；并列按ID|
|F039|10×10|676/233|blocked|closed|0|blocked完整题最大候选|
|L030|4×3|90/32|mixed|open|0|该规模open题最大候选|
|L245|6×6|253/87|blocked|closed|1|该规模closed题最大候选|
|L170|8×8|462/163|mixed|closed|0|该规模closed题最大候选|
|L164|8×8|461/158|mixed|open|0|该规模open题最大候选|

六题覆盖全部要求尺度、两个布局模式、开闭端点以及非全允许的A1条件。最多8个工程case的剩余两个位置建议用于：F012同一父/端点的空任务closed输入，以及L030同一父/端点的空任务open输入。它们是显式工程边界派生，不计为新增科学训练数据；生成和实际执行由本轮统一预登记后的工程runner负责。本调查没有生成这两个输入。

F012的problem_identity为`20daa85c2d8d31a68a7194f1ca45afc02f2df5a3a356f946258840c34b254dce`；physical_key为`bcf3ea858034490de2ab7111b1c0eebe58a8541d6cf182aea52c95c9827ad3c5`。其他任务精确身份应直接复制原索引/参考记录进入本轮工程登记，不重新命名父场景。

## 66局部与完整10×10开发组

原66题入口为`REVISION_PROTOCOL.json['evaluation']['development']`，亦即`Pool.cases`；48题属于`trained_parent_new_tasks_exposed_selection`，18题属于`independent_parent_exposed_development`。保留全部ID、spec及physical_key，不按结果重新选题。

新增完整开发组建议按用户本轮方向，事前固定`F000`至`F011`，恰好12个已暴露训练父。只复制已有完整任务定义，既不产生新父，也不算独立泛化。该组与66局部组分别统计，不合并父均值；完整组已有训练标签和FrozenSL参考。此选择不使用保存J或方法优劣。

## 教师动作语义与M1计账

已有JOINT_DATA_ADMISSION为PASS，保存了256局部标签的当前mask/完整回放结果；96完整标签的manifest记录`current_environment_replay_passed=true`。这是历史来源证据，不能替代本轮6题针对新接口的工程核验。

`Pool.teacher`只验证动作索引和来源，**不会自动逐步确认正覆盖**。M1应在完整请求先扣账后，对六题逐一执行原Environment的原mask/step，检查每步严格减少剩余任务、无重复动作、最后全覆盖。可采用`Environment(p, lambda p,r:p.metrics(r))`完成无H教师语义回放，再对同一动作序列调用原`terminal.export`和独立`validation.validate`，核端点、成本与保存教师J；本项不需要重新求教师或H。若重新调用finish，则其H和完整请求必须照实计入本轮上限。

建议将六题教师完整回放各计一个完整工程请求；随后每题四条独立策略轨迹、冻结/恢复复测和两个空边界共用128总上限。最终运行次数由主runner一次固定，不通过重试、改目录或换授权重置。

空closed可得J=0，应跳过无动作策略损失并保留完成记录；空open仍有入口至出口成本。参考尺度采用协议固定下界处理零值，不能除零，也不能为了获得梯度为其伪造动作。模型初始状态、Adam、采样器、游标和资源恢复由新runner统一验证；工程权重不得升级为M2科学初始化。
