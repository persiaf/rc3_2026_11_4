# M2 正式实验独立结果与图表审查

- role: independent paper architect/reviewer
- scientific_id: `relation_navcover_m2_v1_20261113`
- approved_baseline: `9c878928e7f619bce25ddd00792757431ca705a1`（本轮任务给定）
- review_date: 2026-11-14 Asia/Shanghai
- claim_supported: **no**
- scientific_status: **TECHNICAL_INCOMPLETE**
- confidence: **high**（针对技术中断和科学证据不足的分类；不是对算法优劣的判断）
- paper_claim_audit: unavailable；下游论文措辞审查为 provisional，本次未读取用户论文。
- routing_action: 本轮停止并封存；技术处置或科学继续执行须另外获得明确批准。

`claim_supported=no` 的对象是“显式导航与覆盖关系的 DRL 已带来完整任务规划收益”这一主科学主张。它表示本轮尚无相应证据，不表示 DRL 机制被证伪，也不是固定完整端点的 `NO_DEVELOPMENT_SIGNAL`。

## what_results_support

1. M2a 的 SL 初始化确实执行了 238 次优化器更新。`STATE.json` 的 cursor 与 commit_sequence 均为 238；事件记录有 238 个 optimizer begin 和 238 个 complete；训练导出包含 InitSL 的第 1–238 次更新，均记录权重哈希变化。最后状态元数据关联模型、Adam 与 RNG 哈希。本审查只读取元数据，没有加载 checkpoint 或执行模型。
2. 这些结果支持“初始化更新链路在故障前执行到第 238 次，并留下相应训练诊断”。首尾批次损失为 4.007646 与 1.860136；裁剪前梯度范数记录范围为 0.747867–12.266729。由于批次不同，这些数值不能证明单调收敛，更不能证明规划成本改善。
3. 直接技术故障已定位到 `common.py:put` 中资源账文件的 `os.replace`：将 `RESOURCE_ACCOUNTING.json.pending` 替换为 `RESOURCE_ACCOUNTING.json` 时发生 `PermissionError [WinError 5]`。调用链来自 `init_0239_teacher_5` 的完成记账，随后异常向上传播。
4. 监督回执记录 `WORKER_FAILED`、退出码 1、整个实验进程树已退出、`timeout_triggered=false`、`job_termination_requested=false`。原账最终 `SUPERVISOR_STOPPED` 是状态标签，不能改述为“监督器因超时主动终止”。未触发墙钟上限。
5. 按事件 ID 静态核对，共 1910 个不同 request begin，1910 个不同 complete ID，1 个 failed ID；交集恰为 `init_0239_teacher_5`。因此原账中的 completed=1910 与 failed=1 是重叠事件计数，不能相加当作 1911 个请求，也不能据此算成功率。
6. 初始化 batch=8，最后已提交 cursor=238 对应 1904 次教师 replay。其后 `init_0239_teacher_0` 至 `_5` 共 6 个请求已经扣账，尚未形成第 239 次优化器更新。原账 backward_calls=1909 与此阶段记录相容。`pending={}` 不能消除这一已付费但未提交的差额。
7. 阶段原账累计墙钟为 2221.312 s；监督收据自身 elapsed 为 2221.2883129 s；worker 最后失败事件 elapsed 为 2194.2945414 s。`Ledger` 与外层监督器分别启动时钟，不能把二者差值直接认定为收尾或等待耗时。
8. 新生成的 `M2A_PARTIAL_AUDIT.json` 报告：98 个源码绑定文件未改变，模型／Adam／RNG 与状态匹配，Adam step 均为 238，checkpoint 请求数为 1904，1910 个已保存请求记录的哈希及其中保存的 teacher_validation 均有效，之后 6 个已付请求不在最后 checkpoint。该收据的 PASS 仅指证据审计，明确 `automatic_restore_admissible=false`、`SL512_available=false`、`M2b_executed=false`。本审查读取并纳入此收据，没有重复加载模型或重新运行 teacher_validation；收据中已保存的有效性不能替代本轮未执行的规划评价。

## what_results_dont_support

- SL512 初始化完成、已取得合格的 RelationInit 导出，或 M2b 的配对 RL／继续 SL 已完成。
- 任一 0／512／2048 端点的质量 gate 通过或失败。三个端点均为 `NOT_RUN`；没有可用于判定 gate 的完整评价网格。
- RelationRL 优于 FrozenSL、RelationInit、RelationSL、Fixed3 或 Detour16；任何 J／L／K、收益、时延、可靠性排名或百分比优势。
- “导航／覆盖关系机制有效”或“机制无效”。配对 RL 尚未开始，本轮中断不构成该机制的正、负科学实验结果。
- 新父场景泛化、封存测试表现、实机机械臂可达／安全认证，或实际关节轨迹规划完成。
- Windows 资源账替换失败的具体锁持有者。监控读取同一账本可列为待查假设；当前无文件句柄取证，不能归因于监控器、杀毒软件或某个进程。
- 将 1909 个无 failed 交集的完成 ID 称为“独立验证成功样本”。它们只是由原事件推导的记账分类，且其中 5 个位于最后提交更新之后。

## missing_evidence

1. M2a 的 512 次初始化更新及对应 `COMPLETE.json`／最终导出；当前仅有 238 次。
2. 以同一个完成初始化导出为起点、各自 fresh Adam 的正式配对 RL 与 SL 完整执行记录。
3. LOCAL66 与 FULL12 分开的完整六方法评价、父等权汇总、raw／H 语义、有效性以及完整请求时间。当前覆盖表的 processed 和 valid 均为 0，不是性能值为 0。
4. Windows 原子替换失败的具体原因与锁持有证据；仅调用栈可以确认失败操作，不能完成责任归因。
5. 若未来考虑恢复，需要明确处理 checkpoint cursor=238 与之后 6 个已扣请求的关系，核对权重／Adam／RNG／游标／请求 ID／资源连续性。不能只依据状态文件存在或 pending 为空就认为可无损继续。
6. 没有论文数字／引用审计文件；本次仅审查本轮证据与图表，不构成整篇论文的提交验收。F1／F4 PNG 已进行独立视觉查看，图表结论见下。

## suggested_claim_revision

建议本轮报告采用：

> 原冻结 M2 协议的初始化阶段执行了 238／512 次 SL 更新，在第 239 次更新的第 6 个教师请求完成记账时发生 Windows 资源账原子替换拒绝访问错误，worker 异常退出。最后已提交状态为 update 238，共有 1910 次请求扣账，其中最后同一请求同时留下 complete 与 failed 事件。M2b 及全部规划评价端点未执行。本轮结论为 TECHNICAL_INCOMPLETE；训练记录仅表明故障前初始化更新已发生，尚不能判断显式关系 DRL 对完整任务规划的收益或机制效果。

报告应并列保留三层原义：科学结论 `TECHNICAL_INCOMPLETE`、账本状态 `SUPERVISOR_STOPPED`、监督事实 `WORKER_FAILED`。原协议字段及原始账本／事件不应为统一措辞而回写。

## next_experiments_needed

本轮不运行任何额外实验，不自动恢复，不进入新 seed、消融或替代训练。

若用户以后批准独立技术处置，可另行登记只针对资源账替换的诊断与有界工程验收，先明确源码绑定、是否允许优化器、用例、次数／墙钟额度和验收标准。任何源码或协议实质变化均需形成新的绑定与变更说明；不能借新目录或新授权重置已经消费的资源。

只有处置方案明确了 6 个已扣但未进入 checkpoint 的请求如何保留与记账，并获得后续阶段执行授权后，才可决定是否继续或新设正式实验。科学证据所需内容仍是已批准协议中的初始化完成、配对训练与完整端点；这份建议本身不授予执行权限，也不建议当前增加种子或消融。

## figure_review

### F1：图面通过，作为初始化训练诊断

两行折线适合展示实际的 InitSL 逐更新损失和裁剪前梯度范数。只绘制实际存在的第 1–238 次记录，不补到 512，不添加未运行 RL／SL 曲线，不给单次运行画跨种子置信带。Loss 轴建议明确为 `Teacher-action mean NLL`，梯度轴为 `Pre-clip global gradient norm`。可以标明截断于 238／512；若加入 norm=1 的裁剪阈值虚线，caption 要说明记录的是裁剪前范数。

已独立查看 `fig1_training.png`：两行面板、图例和字体可读，无图内标题或可见裁切，只存在 InitSL 实际记录，没有外推或伪造其他流。当前较一般的 loss 标签可由以下 caption 补足定义，无须为图面审查增加实验。

可用的自包含 caption：

> **M2a 初始化训练诊断。** 上图为每次更新的教师动作负对数似然，先对每任务动作取均值，再对该更新的 8 个任务取均值；下图为参数梯度的裁剪前全局范数，实际梯度裁剪阈值为 1。曲线包含已提交的第 1–238 次 SL 更新，计划为 512 次。第 239 次更新期间发生资源账文件替换错误，未形成新的优化器更新。配对 RL 与继续 SL 尚未运行；这些曲线不表示规划成本收益或端点 gate 结果。

### F2：跳过正确

没有任何完整端点，因而不生成增益曲线。不要画零增益、连接不存在的 0／512／2048 点或以初始化损失替代规划收益。可在交付索引标为 `NOT_GENERATED — all evaluation endpoints NOT_RUN`。

### F3：性能表跳过，允许覆盖缺口表

没有完整方法×群体网格，不生成成本或时延比较表。可以单独给出“端点、群体、计划数、已执行数、状态”的覆盖表，0 只用于已执行数，J／L／K／时间留空或写 `NOT_RUN`。表 caption 应说明 LOCAL66 与 FULL12 是两个分开统计的已暴露开发群体，当前均未执行评价，不能从空格推断成功或失败。

### F4：修订后的三面板图面通过

已独立查看最终 `fig4_resources.png`。三面板分别展示请求事件、优化器事件和失败事件，故障位置用竖虚线标明，横轴明确为 worker-ledger 时间；未将外层累计墙钟接入该曲线。下列原先的语义问题已在图面修订中解决，其解释仍须在 caption 中保留：

1. 失败独立面板清楚展示 request_failed=1、optimizer_failed=0，解决了在 1910 请求纵轴下不可辨识的问题。最后 complete 与 failed 仍须按非互斥事件解释，不称为互斥的成功和失败。
2. 请求轴已改为 `Request event counts`，图例使用 `Attempt events`／`Complete events`。优化器只有 238 次 attempts／complete，没有第 239 次调用。
3. worker 事件时间轴标出最后失败事件；外层累计墙钟 2221.312 s（约 37.02 min）仅单独报告。两个时钟的起点不同，未经映射不能把累计墙钟当成同一曲线的后续点，不能把约 27 s 差额标为故障收尾时间。
4. 若在配套表中补充 `complete IDs excluding failed IDs=1909`，必须说明是只读派生分类，不是改正后的原账，也不是独立任务验证成功率。

建议 caption（须与最终实现一致）：

> **M2a 资源事件与技术停止。** 曲线根据保存的 worker 事件日志展示累计请求和优化器事件，横轴为 worker 账本自身的累计时间。共 1910 次请求 begin、1910 次 complete 和 1 次 failed；最后一个请求同时记录 complete 与 failed，因此后两类并不互斥。优化器共 238 次 begin 和 complete，无 optimizer failed。最后失败事件位于 worker 时钟 2194.295 s；外层监督器计入阶段账的累计墙钟另记为 2221.312 s，二者计时起点不同。故障发生在资源账文件原子替换；worker 异常退出，未触发超时终止。图中资源消耗不代表规划成功率或算法质量。

F1／F4 已提供 PDF 和 PNG，保留 CSV、逐图脚本及 LaTeX 引用的方案合理。PNG 视觉审查确认字号、图例与停止标记可读、颜色加线型可辨、无图内标题或明显裁切。caption 需落实上述定义与证据边界；无需为完成图表重跑模型、教师、规划或 H。本次未将 PNG 检查冒充 PDF 内部矢量对象审计。

## 直接证据来源

- `rc3_relation_drl_20261113/M2_PROTOCOL.json`
- `rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/{STATE.json,current_slot0.json,RESOURCE_ACCOUNTING.json,EVENTS.jsonl}`
- `rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/controls/attempt_0001/{WORKER_STDOUT.log,SUPERVISOR_RECEIPT.json}`
- `rc3_relation_m2_20261113/analysis/{SUMMARY.json,TRAINING_UPDATES.csv,RESOURCE_CURVE.csv,METHOD_TABLE.csv}`
- `rc3_relation_m2_20261113/FIGURE_PLAN.md`
- `rc3_relation_m2_20261113/M2A_PARTIAL_AUDIT.json`
- `rc3_relation_m2_20261113/figures/{gen_fig1_training.py,gen_fig4_resources.py,paper_plot_style.py}`
- `rc3_relation_m2_20261113/figures/{fig1_training.png,fig4_resources.png}`（视觉查看）
- `rc3_relation_drl_20261113/{runtime.py,common.py,m2.py}` 中损失／梯度记录、事件写入与计时源码。

审查仅进行文本读取、原日志计数和派生 CSV 静态核对；未执行模型／优化器／教师／H／规划／工程探针，未读取封存任务语义或用户论文。唯一新增文件为本审查文本。
