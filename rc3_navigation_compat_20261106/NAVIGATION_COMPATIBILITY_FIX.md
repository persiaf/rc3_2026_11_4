# S1-A 教师池导航兼容修复

本轮交付是实际生产代码修复及已有数据重放，不是新增教师生成或科学训练。研究目标、网络、MDP、奖励、优化器数学、训练样本分布、H与预算不变。

## 原始代码根因

原 `baseline.navigation` 返回7个参与导航内容身份的字段。`runtime_guards.bind_navigation` 将所有字段及保存NPZ的来源加入完整parent身份；该身份算法正确且本轮未改。原 `SceneProvider._navigation` 只保存4个数组和两个计数，遗漏 `rejected_edge_count`，导致首次内存导航与后续磁盘导航有不同身份。首96条教师使用完整表示，后160条使用简化表示，因此统一补字段会破坏后160条原身份。

另外，原阶段入口要求S1-A与后续训练使用完全相同源码/批准身份，不能安全消费只修改了加载器的新版本。只修JSON或忽略source检查不能解决正式执行链。

## 生产修复

- 新增 `rc3_global_local_trainer/navigation_compat.py`：未来schema `rc3-navigation-artifact-v2` 在NPZ与receipt完整保存7个字段，交叉核对node/edge/rejected计数、map_key、producer源码、文件SHA及内容身份；拒绝未知字段、非法标量、坏数组、错误来源和覆盖已有文件。
- 旧schema仅枚举已保存坐标上的走廊与连接候选边，以总候选边数减已存有效边数恢复缺失整数。不执行碰撞筛选、Dijkstra、场景生成或规划。
- 对每个旧task构造简化与完整两种绑定，用原F/A和scene计算parent身份，要求恰好一个等于原receipt。96条恢复完整表示，160条保留简化表示；没有匹配或多重匹配直接拒绝。
- `SceneProvider` 的LRU缓存保存每图的候选表示集合，逐task选择，不缓存某个task选择结果。`read_only=True`在任何缺失输入处拒绝生成。
- `LEGACY_TEACHER_REPLAY_MANIFEST.json`固定256条原label、task/nav receipt、scene、F/A、导航NPZ及S1-A完成/索引/批准/源码来源的精确哈希。新permit只能经显式 `legacy_teacher_replay` 接受该精确旧池，不能泛化为旧源码白名单。原路径不搬移，原数据不改写。
- `TeacherPool.validate`及每次`example`均复核池索引原哈希，标签也逐次校验。生产`example → SceneProvider.get`检查标签view和真实J。
- 新入口显式接受 `--source-binding`，验证新的批准及对应阶段授权；旧S1-A完成证据只作为精确的只读前驱。原RUN_MANIFEST保留，新阶段以追加SOURCE_REVISION记录来源，新checkpoint/Adam/RNG及日志绑定新源码。

## 验证与边界

实际256例生产重放、负向测试、同图LRU交错检查及独立审查见 `TEACHER_REPLAY_VERIFICATION.md`。原始S1-A证据保留；不是将旧源码绑定改成新哈希。新版源码冻结在独立 `SOURCE_BINDING_UPDATED.json`，旧435b绑定和v3协议原字节不动。

本轮不初始化模型、不更新optimizer，不读取S3或原60任务。S1-B正式执行必须对应新源码批准和独立阶段授权，不能使用旧S1-A许可。
