# 独立执行前审查

日期：2026-11-10。审查者为当前任务的独立子代理，未参与本轮 runner、storage、analysis、engineering 或 trainer 编写；曾编写 saved_inputs，因此其数据模块检查不冒称独立作者审查。此为静态审查，**没有执行规划、模型前向、optimizer 或实际恢复测试**。仅以 Python AST 解析七个新源码文件，均可解析。

当前决定：**工程验收 REVISE；正式训练 REVISE**。以下问题须在首次真实 Adam 前解决；这份审查不是协议批准或阶段执行授权。用户明确将每组 256 个更新机会称为建议，当前不能将其转写为已批准参数。

## 正确实现的关键路径

- 新 runner 使用新 206 维 `ConservativePPO`，没有调用旧 202 维优化入口。ON/OFF 重新在线采样，实际动作与全轨迹在请求记录中保存；监督对照使用已保存教师标签和独立新 Actor。
- Actor 候选更新前保存 correction 与 Adam，guard 不通过即精确回滚；异常时恢复 mutable Actor/Critic/Adam，已付请求、尝试和失败保留。监督更新也使用相同 guard，但需分别报告接受次数，不能称三组有效更新数天然相等。
- `Ledger` 先扣账再请求或调用 optimizer；请求 ID 不允许重放，恢复拒绝脏 checkpoint 后事务，闭合会话累计墙钟。formal output 固定，不能通过命令行随意换目录重启预算。
- checkpoint 绑定 Actor、Critic、两组 Adam、全局 RNG、使用过的 episode ID、外部请求账、源码/协议及进度。正式恢复程序核对文件 hash 和原 journal 边界。
- 原生 Fixed3/Detour16 走已冻结 endpoint adapter，无第二次 common H。模型 raw/H 路线分别验证。Cold model 请求时间包含真实保存输入、来源检查、模型加载、provider 准备、动态特征、策略、H、导出与验证；日志写入另由 ledger 计时。
- 效应计算先配对同案例，再聚合真实父布局；重复计时、窗口和 checkpoint 不充当独立布局。只有一个训练 seed、3 个已暴露独立开发父布局，不能由此声称新的保留测试泛化。

## 首次真实工程 Adam 前须修复

1. **工程预算可被新输出目录重置。** `engineering.run(plan, output)` 只检查 output 不存在，未要求它等于 plan 中固定输出。相同 plan 可以换目录再次消耗整份预算。需要将精确输出、唯一执行 ID 绑定到工程 plan；修订/失败必须核对累计既有账，不以新目录退款。
2. **工程启动资源没有完整计入。** 当前工程 ledger 在导入、来源核验之后创建，正式 runner 更在来源核验、配置和 `SavedInputs()` 之后创建；这些实际开销在 stage cap 外。应加入父进程 launch-to-ready 或等价实际计时，并累加到原账，不能仅以循环 body 为完整资源。

## 正式训练前须修复

3. **正式入口缺少工程 PASS gate。** `runner.run` 检查审批文件后可直接构造训练器及 warm optimizer；未核对绑定当前源码的真实 zero-equivalence/回滚/恢复/部署工程通过收据。应在 formal protocol 中绑定具体报告及 SHA，启动时核验 PASS、关键检查及对应源码，并保留失败收据。
4. **实际环境未核对。** ledger 直接复制 protocol.environment；没有将实际 Python、Torch、NumPy、设备、确定性设置与冻结协议逐项比较。恢复跨环境时也必须以实际值校验，不能把协议要求写成实测环境。
5. **停止条件与用户原句不一致。** 当前 no_gain 要求 ON/OFF 对 FrozenSL 和 continued_SL 四项比较都不正。如果两组均差于监督起点、却优于更差的 continued_SL，就不会触发“均未改善监督起点”的停止。应登记精确规则，例如两个终点 ON/OFF 对 FrozenSL 均不正即停；continued_SL 比较另报。若选择其他规则，须在新协议里明示差异，不能运行后解释。
6. **runner 级恢复未验收。** 当前真实工程比较的是 trainer 的 payload 和单一病例连续/恢复；没有覆盖新 `Ledger` + runner checkpoint/restore 的闭合会话流程。可先用不调用 optimizer 的 fixture 测试：干净暂停可恢复、dirty 后事务拒绝、账/游标/权重/Adam/RNG 不一致拒绝、重复启动或更改预算拒绝。formal 不能只由单一 trainer 恢复检查推定全过程已验收。

## 非阻塞限制与交付注意

- actor-only 导出确实不含 Critic 或 Adam，但当前测时仍使用完整 rollout，保存/打包训练特征及审计数据。其结果是当前完整实现的实际时间；不应写成已经优化过的最轻生产推理时间。
- 当前原生方法的 raw 语义是其内部搜索完成后的原生路线，原生过程本身含修复/搜索；与模型 H 前 raw 的含义不同，源码已有标记，表中需要保留。
- 常数基线由 warm 轨迹固定组件均值构成，再按初始任务数换算 value；应清楚描述这一基线，避免称其为未经证实完全相同的历史标量常数。
- 完整路线 CSV 应保留 raw_J/raw_L/raw_K/J/L/K、合法性、方法、终点、父布局、窗口、重复、模型启动与完整请求时间；长期优势/校准记录要绑定实际状态及 H 信用语义。
- warm、训练、匹配监督、评价和计时均需要在精确协议中冻结数量。已批准工程 cap 不自动批准 proposed 256 训练机会；真实测时后先形成可审查的最终协议与独立阶段授权。

## 静态审查快照

以下 SHA 是本次审查读取的文件版本，不表示后续修复已审查。任何后续变化需要补充复核记录；不得将本文件自动改写为最终 GO。

| 文件 | SHA256 |
|---|---|
| runner.py | 325269d9c03fca2a33daa1383459d18541a6962a6368df6c233dbf6963cce189 |
| storage.py | 95b6fae7a827a496e9d484fed3f2d5b79f6aff5621b56ce74c3ca9fb8bc52e25 |
| analysis.py | 58299257068c53742f3b894ebce0c2b8c6423f537f8058b3176617f3cfaa5762 |
| engineering.py | 3c3c40fdf2a8f59e0f13c667a58f5a769eb7a5ee0aeb07e615a18d6a7190780d |
| trainer.py | 1ea08cc489862a119ee837500ec172465df956c212765932e09e0227b4992aa2 |
| runtime.py | 1b86fe7566a3c8a780929afe0532ef77687a9d4157a193107aabdd60798fab43 |
| saved_inputs.py | febd3d08e068b1efa6b479595dd0aef0f2236d74484c47549bfe5c73baa7b39d |

原60封存输入、旧 S3 结果或未批准 final 输入均未被本次审查读取；审查未授予其后续访问权限。

## 修复复核 R01（同日追加）

根代理修复后重新静态读取，原始 REVISE 记录保留。此次仍未执行任何测试、规划、模型前向或 optimizer；八个文件均通过 AST 解析。

1. 工程 output 现在必须等于 plan 固定路径。相同父目录下所有 `engineering_*` 既有输出必须有资源收据并全部登记，收据 hash 复核后累加；新工程 cap 加历史已用资源不得超过请求32、Actor16、Critic16、1800秒总上限。目录已有内容或未闭合历史尝试会阻止启动。
2. 工程与正式模块在重依赖导入前保存单调时钟，传给 Ledger `started_at`；pre-registration 实际开销纳入累计会话时间。工程资源报告同时保留 body 和已收费会话耗时，避免把两者混称。
3. 正式入口已要求精确工程资源/检查收据 SHA、PASS、相同源码绑定，以及 zero-equivalence、相同初始化、真实拒绝回滚、连续恢复和 actor-only 验收结果。
4. 正式入口已比对实际 Python、Torch、NumPy、platform、CPU设备、线程与确定性算法状态，协议中的环境不再直接充当实测值。
5. 无收益判断改为 ON/OFF 对 FrozenSL 均无正收益，连续两个登记终点触发停止；继续监督的效果继续独立报告。
6. `test_execution.py` 增加 runner checkpoint/restore 和 Ledger fixture：闭合暂停后核游标恢复、dirty 后事务拒绝、资源累计与封顶、来源/cap变化拒绝、中断时钟不能重启。这里仅审查测试实现；实际执行及结果由后续工程报告给出。

**复核决定：工程源码 GO，可在本轮明确登记的工程 plan、固定输出和累计 cap 内执行已要求的有界验收；正式训练仍为 REVISE。** 工程 GO 不等于验收已通过，更不批准每组256训练机会。正式训练还需要真实工程 PASS、实测资源准入、冻结后的精确协议及用户对具体参数的批准和独立阶段授权。旧实验授权与历史资源不复用。

| R01 文件 | SHA256 |
|---|---|
| runner.py | 2c56be63a7591fbbc426dbbc6ba9a9fc20c86bfac41056b48ff2d8bb5af6e04f |
| storage.py | d15e002a5673b1350734c7023a26ee7b27047ced6a94220484892209fb1eeedd |
| analysis.py | 58299257068c53742f3b894ebce0c2b8c6423f537f8058b3176617f3cfaa5762 |
| engineering.py | 32f8f119c5459a9e0f3c3d443456c13c0819e0cf68a8c256bbd6a955ff9f288b |
| trainer.py | 1ea08cc489862a119ee837500ec172465df956c212765932e09e0227b4992aa2 |
| runtime.py | 1b86fe7566a3c8a780929afe0532ef77687a9d4157a193107aabdd60798fab43 |
| saved_inputs.py | febd3d08e068b1efa6b479595dd0aef0f2236d74484c47549bfe5c73baa7b39d |
| test_execution.py | 957f450b00dc54f1c7328817c6a42c2bc053ea87e67a897839906b33e27626bd |

按当前完整 runner 上限计算，warm48请求、训练3072请求、评价1980请求，共5100；另有匹配监督1536标签回放、768次Actor候选尝试及576次Critic尝试（含64次warm）。这些是待批准的精确候选算术，未写成实际执行结果；提前停止后按真实账报告。
