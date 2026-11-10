# 离线辅助分析独立静态审查

- 日期：2026-11-10。
- 审查对象：新增 `postprocess_formal.py`；初审读取版本的 SHA256 为 `f7d91c4e5ada9b8d9dc7bf7302e0e7e33a49528a9dc8d570433e371439ee4528`。
- 对照：冻结 `PROTOCOL_DRAFT_R03.json`、`SOURCE_BINDING_R03.json`、`SCHEDULE_R03.json`，以及 `storage.py`、`runner.py`、`runtime.py`、`trainer.py`、`analysis.py`、保存计划 schema 的源码。
- 审查方法：仅阅读源码及上述冻结元数据。**未执行 helper、测试、规划、模型推理、验证器或 optimizer；未读取活动正式 runs 的方案、日志或结果；未读取原60、S3任务或用户 Word。**
- 初审结论：`REVISE`，以下阻断项修复并重新绑定前，不应执行 helper。本文件不是正式结果核验 PASS，不判定训练效果或正式 run 的成功/失败。

## 最终字节静态复核

复核对象 SHA256：`06e80ff1af49615a60868f14a0bb0495899c7ab82fa431080f337a42c7fec00f`。下列初审问题已在新增helper中修复；**未发现剩余的静态代码阻断项，可进入独立登记及下述两例工程控制自检。** 这不是正式分析执行许可，也不是实际结果核验 PASS。正式helper仍须等待正式run终止闭合，登记独立 `ANALYSIS_PLAN.json`并绑定本精确hash，沿用同一原始账本和剩余18000秒总预算，最多6000份保存request、一次分析、最多300秒；不重放训练或增加请求/Adam。

- 已完成endpoint按冻结案例×方法×repeat核对完整请求集和四份表存在；calibration按ON/OFF逐state×四component核对完整唯一键、cohort、problem identity、路线算式目标与warm常数。完整contrast网格、per-parent、point、PCG64 seed/20000次及CI均与原算法完整重算的dict精确比较。
- 大CSV改为流式写；heavy PPO事件流式读取，Ledger只保留序号计数器。repeat0 eval只保留统计所需小字段，避免双份完整日志及全部几何/轨迹驻留。未执行内存压力测试，因此不声称已实测峰值或能在任意剩余内存下运行。
- 新 `StreamingAnalysisLedger`沿用原events文件、writer锁、canonical链、累计caps/counts/wall；锁内复核预检tail/sequence及关闭状态，继承原emit/check/close，并禁用request/debit。任何先前 `analysis_start` 或 `terminal_analysis_*`会话都在main及锁内阻断再分析，失败后换plan/output也不能重置本次分析额度。
- 保存raw/final identity、任务manifest、本地候选ID→保存全局节点互证；所有带targets记录（包括eval）按保存leg、H信用、精确task_count、scale和固定penalty复算return；训练/warm的保存actions/G/V摘要关联ledger事件，A向量精确长度及float32减法核对。
- 哈希改为1MiB流式块并检查计时；各逐行输出、聚合、完成前后均检查，预留2秒关闭余量。此为协作式截止检查，不是对Windows阻塞I/O/调度停顿的硬实时认证。关闭后的实际elapsed必须如实记账，超时/异常不得补跑或篡改elapsed。
- STOP区分已完成、启动未闭合、未启动endpoint，并列计划/实际/失败/未闭合请求数；设计父场景数与实际闭合评价父场景数分开。失败尾部的计时明确为保存成功请求子集。Native原生raw/H语义保留。10份导出表的hash/bytes写入AUDIT，再由ledger绑定AUDIT。

## 独立工程自检登记材料（尚未执行）

- 新 `postprocess_selftest.py` SHA256：`09f6a875b226998350813c0454a67fadede8acfe9bf63f5cc473f84ecb68763d`。
- `ENGINEERING_SELF_CHECK_PLAN.json` SHA256：`c1bf428fe1cc1da4f2ef6db4670e2a6ec4e4821f2a0ac218e4c69902213c40b6`。
- 固定上限：2个case、30秒、一次独立登记调用；0真实run/原60/S3读取、0规划/模型/H/验证器/optimizer。五条literal合成事件中的 `debit`仅为虚构历史counter，不代表真实请求或Adam调用。
- Case1核对原prefix/seq/tail/caps/counts/carry-wall、禁用request/debit API及session关闭/锁释放；Case2使用实际锁内构造器验证同一合成账本不可再次分析，拒绝后字节及锁状态不变。fixture/失败报告保留，不删除或自动重试。
- 自检脚本采用显式条件检查，不依赖会被Python `-O`禁用的assert；输出明确只代表工程控制检查，不作为规划或DRL收益证据。执行与登记由root后续处理，本审查未执行自检。

## 初审阻断项

1. **P1：置信区间未经核验却作为分析结果转交。** 初版第405–429行仅重算父场景数及 point，随后原样输出全部 contrast 字段，包括 `ci95`、`parents`、`bootstrap`、`bootstrap_seed`。改动这些字段可通过现有审核。建议按冻结的20000次 PCG64、固定seed、同一父场景顺序重算完整 per-parent、point、CI，并核对所有协议字段；或明确这些字段未验证，不以整体 PASS 背书。
2. **P1：派生文件及逐状态数据完整性未核对。** 初版第379–429行采用“文件存在才读取”，缺失整个 endpoint 的 calibration/contrast、删掉部分行或复制部分 calibration 行仍可能生成 `AUDIT.status=PASS`。建议由闭合阶段游标、冻结66案例×6方法×注册repeat构造应有请求集；为 calibration 生成 ON/OFF 的完整 state×4 component 唯一键，核对 case、cohort、problem identity、state范围及每个 contrast 的完整组合。失败尾部只能标为缺失/部分证据，不能静默用成功子集作完整比较。
3. **P1：300秒/剩余18000秒的边界检查不覆盖重操作。** 初版 calibration CSV 内循环、contrast内部、聚合、CSV/JSON写出和最终完成路径没有 `check()`；一次大文件hash/read也可跨越检查。末尾可在超时后仍写 `analysis_complete`。建议迭代检查并为关闭账本保留余量，所有报告写入前后及完成前检查；hash流中检查，不使用一次性无界读取。只能承诺协作式边界，若需要严格硬上限，应使用外部受控截止机制并设计被中断后的原账本恢复收据。
4. **P1：内存峰值没有与本机剩余资源匹配。** 初版同时保留 `chain()`全部 events、`Ledger.events`第二份、所有完整 eval record（包括只用于计时的repeat>0）、全部 route/timing/advantage字典；`Ledger`自身还会一次读入日志并 splitlines。6000份“记录数”并不约束字节数/状态数。当前主机最近仅约0.45GB可用，不能把逻辑上有界等同于可执行。建议流式写CSV，仅保留repeat0的最小索引；复用唯一的Ledger events副本，先静态给出有界内存预估及执行时余量门槛，不在活动训练期间做压力探针。

## 需要收紧的身份与状态说明

- 初版 `metadata_for()`没有完整核对 eval 的 request_id 与保存 `endpoint/repeat/case_id/method/cohort`；训练仅核对final plan identity，raw identity没有核对。应使用冻结schedule核对这些字段，保持真实physical parent及cohort身份。
- `validate_saved_plan()`只核对plan内部global节点和J/L/K；输出的保存 `raw_route/route` 尚未与plan核对。PPO记录已保存 candidate全局映射和global route，可仅凭保存字段互证，无需重新加载输入、规划或运行验证器。Native baseline没有相应映射时，应明确该项未验证。
- `unexecuted_endpoints`初版由“未在完成游标中”直接生成；FAILED run可能已开始某endpoint并保存部分请求，不能称其“未执行”。应区分未启动、启动未闭合和已完成。
- `runner.py`对Native Fixed3/Detour16明确保存 `raw_semantics`：其raw等于Native最终路线，内部已包含自身搜索/H。初版helper统一输出RAW/H而丢弃该说明，可能把Native的 `raw_J`误读为共同H前路线。应保留两侧的raw语义和H处理范围，不把Native的raw/final相同解释为共同H消融收益为零。
- 初版仅核对ledger事件内部的 `A=G−V_old`，尚未与该 `episode_id`对应request保存的returns、V_old及动作相连；“事件内算术核对”不能扩写成“完整return已由路线核验”。可利用保存leg、J及task_count按 `costs.py`公式重算目标并建立轻量摘要互证，全程不需要模型或规划。
- 初版硬编码的3个独立父场景和 `original60_reads=0/S3_reads=0`应标明是冻结设计数及helper本身的新增访问范围；FAILED/未完成评价时不能称已实际验证3个独立父场景，正式run访问计数未另核验时应如实标未知。
- 初版预检关闭状态后再创建Ledger，两个时刻之间存在并发变化窗口。应在取得同一writer锁后核对预检tail，并复用锁内日志快照；避免用旧 `events` 配合新Ledger继续分析。`output.mkdir()`在取得Ledger前可能留下未记账空输出，建议纳入失败记录范围。
- helper自身失败后的 `ANALYSIS_FAILED` 不等于正式训练 `FAILED`。后续重试必须保留原账本、原失败及累计墙钟，不新建正式run、不重放训练、不恢复模型/Adam，也不能把analysis尾部当正式阶段结果。正式闭合状态应来自独立 `stage_result` 或明确的正式失败会话。

## 静态确认的作用域

- 初审导入只有标准库、NumPy及 `storage`，包初始化文件没有执行逻辑；未发现Torch、planner、H、导航求解器或验证器导入/调用。
- `SOURCE_BINDING_R03.json`列出600份source及17项artifact；新增helper不在600份source中。应单独记录其最终精确hash及分析计划，不回写冻结binding或600份执行源码。
- `Ledger(..., resume=True, started_at=ENTRY_START)`在成功建立会话后沿用原caps、历史请求/Adam debit和累计墙钟；没有直接重置预算的代码。此静态性质不代表实际账本已核对。
- 对保存polyline/leg、J/L/K、任务manifest和已有validator证书的算术检查，不等于重新障碍碰撞/F&A验证，更不等于真实IK或关节轨迹认证；helper应持续保留此限制。

## 后续复核

初审发现及历史行号保留供追溯，以“最终字节静态复核”节为当前判断。若helper或自检脚本字节再变动，须重新核对并更新登记hash；任何静态“可登记”判断都不替代正式run闭合、独立分析计划登记或实际结果审计。
