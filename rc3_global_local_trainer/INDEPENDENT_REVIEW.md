# 正式训练器独立审查

审查角色：`/root/trainer_independent_review`。基准：`af8048149f6263e358f635385591950ab9a8f7c3`。固定研究目标为大规模复杂父场景DRL训练，冻结策略完成不同规模局部任务的完整快速规划。

**独立判断：正式训练器实现与本轮工程验收可给技术GO；正式S1/S2仍为REVISE。** v2是活动冻结设计，v3尚未批准，采样种子执行歧义也须先澄清。本轮主代理按单独登记执行的21次真实optimizer仅为工程验收，不是正式S2。独立审查只读源码、保存checkpoint/结果/journal与三个原S0父数组，未重复模型前后向、optimizer、终局H、教师、导航或新任务生成。

## 已审查的核心语义

- `engine.py`对每实例构造独立的primary/forced-low/forced-high环境和编码前缀缓存。主行首动作参与logprob；forced首动作不参与，其后缀仍按独立命名流抽样。每行完成同一目标任务与端点，记录合法集合、动作、logprob、熵、逐步奖励、完整终局成本与H/验证耗时。
- 三个回报构造detached leave-one-out基线；混合损失权重1/.25/.25再除1.5，gamma=1，无隐式熵项或优势标准化。batch4逐实例loss/4反向，最后只clip和step一次。数学接口匹配v2已声明目标；数值批次等价须由保存验收确认。
- S1为完整合法教师标签序列的route-mean masked CE，不调用H或教师。空路线无动作loss；全空批次停机、不换样。单首动作两辅助仍独立后缀，保留权重。错误标签、不可行或非有限值失败停机，无Fixed3回退或补零奖励。
- `rng.py`按协议/阶段/optimizer seed/update/slot/role派生独立SHA命名流，保存PCG64与torch状态；恢复检查key、metadata及派生seed一致。源代码没有逐轨迹重置同一个共享采样流。
- `checkpoint.py`以不覆盖的新文件及SHA侧车保存模型、Adam、NamedStreams、全局RNG、ledger、cursors、weight version、terminal state和runner extra。只在update边界保存；恢复先核字节、binding/config、参数形状/有限值、tensor hash和RNG状态，再恢复运行对象。模型、Adam和命名流的连续/恢复精确一致仍须查保存证据。
- 分叉通过独立tensor storage的模型clone，冻结SL禁梯度，RL和continued-SL各fresh Adam。Trainer验证optimizer参数归属恰好是当前模型，避免跨分支误更新。正式科学比较采用何种初始化/重置配方仍属于待批准协议条款。

## 静态审查发现与处理

| 发现 | 处理 | 解释 |
|---|---|---|
| 教师标签最初用int()转换，可能静默截断float/bool | 作者改为原`problem.integer`严格验证 | 工程输入契约修复，未更改合法标签的算法语义 |
| Provider最初只清空nav LRU，保存的恢复key未使用 | 作者改为核对已存geometry/nav来源后按原顺序重建最多2条绑定，加载与时间单列 | 不生成新导航；缓存恢复不是免费准备 |
| checkpoint恢复损坏RNG后可能已载入live weights | 作者增加tensor/shape/stream预检，terminal可用validate hook | 失败应停止，不把坏输入局部载入另一研究模型 |
| 多个完整plan缓存与逐update流状态可能增加内存/IO | 已提醒作者限制terminal LRU并记录cache/restore开销 | 资源风险不能由原S0模型大小估算替代 |
| 首个教师查询失败在首label写入之前，原partial检查可能漏掉events/H目录 | binding02改为非空pool一律拒绝，并在任何准备或查询前保存STARTED；4个纯文件拒绝用例通过 | 修复不能被用来免费重置512查询/1800秒预算；没有重新查询教师 |

## 保存验收与独立复核

| 证据 | 独立检查结果 |
|---|---|
| batch4整批与逐实例累积 | 作者保存的同采样值/梯度通过事前rtol=1e-5、atol=1e-6；max绝对差7.1525574e-7，max相对差4.3053528e-6。本审查读取证据及实现，不重复前后向。 |
| S1连续3步与1+恢复2步 | 独立直接加载两份保存checkpoint字典：模型、Adam tensors、命名和全局RNG、terminal state、ledger、cursor、weight version、config逐项精确相等；作者另存采样/动作一致性证据。 |
| S2连续4步与2+恢复2步 | 同样直接读取两份checkpoint独立精确比较通过；未将相似loss代替状态恢复。 |
| 权重分叉与失效缓存 | 原SL起点与untrained/frozen-SL保持，分支存储不共享、fresh Adam、RL/continued各更新及旧encoded拒绝证据通过。 |
| 退化/坏输入/准入 | 空批次不step、不换样；单首动作零信号按协议step；坏标签/非有限/坏checkpoint与未经批准CLI拒绝。正式入口负例未新增模型或optimizer调用。 |
| 完整父场景batch4 | 4实例×3角色=12真实完整终局/H，12导出/验证，失败0、缓存命中0，1次真实Adam更新；clip前norm888.47376、后0.99999986。只证明工程通路。 |
| 采样日程与编排 | 512个metadata批次产生1024 full与1024 local，local342/341/341；日程恢复精确。生产S1/RL loop各1次mock-provider更新、日志与checkpoint通过。没有生成新scene或教师。 |
| 累计预算 | 独立对journal逐条charge与汇总对账：21次optimizer完成、89 backward、1770 score、168 mock终局、12真实H；分别不超过32 optimizer、24真实终局和900秒上限。 |

独立在原S0数组上核对12份已保存轨迹的1412个原始动作：各步菜单、正覆盖、动作合法性、完整目标覆盖、raw/final成本、逐步奖励总和=-JH及H不增成本均一致；1129个最终服务停靠点计数与保存指标一致。12对真实终局start/complete与相应问题、原始路线和终局路线逐条匹配。没有重跑H、导出或规划。该检查仍是代理F&A及离散导航范围，不是实际IK/关节轨迹认证。

binding01的9组验收与binding02的教师partial补丁合并覆盖10组。独立核对当前28源码hash；01→02仅改变`runner.py`教师开始/partial保护和`verification.py`新增用例，01原字节已归档。核心engine/rng/checkpoint/data/approval和生产update loop未变，故无需重复已收费的优化器/H验收。原v2、v3提案、registry及活动pointer与基准原字节一致，v3没有被激活。

工程运行收据写前snapshot合计15.7333631秒，journal两次run_end合计15.7388021秒，后者包括最终结果写入等收尾；不把二者视作同一计时点。预算以更完整的journal累计为准。独立自动checkpoint核验0.3440946秒、扩展保存证据核验0.4823479秒；这不包括静态阅读、解释器启动及一次定位计时口径差异的只读探针，不能称为完整审查总耗时。

独立脚本`audit_saved_trainer.py`仅加载保存的checkpoint字典、JSON和原始源码做比较，不构造模型。机器结果及扩展journal/轨迹核对保存在`INDEPENDENT_AUDIT.json`。本报告不把作者测试当作独立重复训练，也不把工程PASS解释为DRL规划增量。

## 正式执行仍须REVISE的边界

1. v3未正式批准，包含新配方、样本分配、预算/判定及退化边界；不能以工程profile通过默认激活。registry中的固定sampling seed与当前按optimizer seed命名数据流之间的执行定义须在批准版中明确。
2. 当前run、共享教师和checkpoint绑定同一execution authorization。未来可以事前授权teachers/S1/S2范围后逐CLI显式执行；不能声称已支持任意新授权文件接续旧run，或由本轮扩大执行权限。
3. 正式768训练task draw可行性、完整教师池、三seed大训练、开发全管线和S3均未运行。生产loop用mock provider验收，不能把它扩述为真实数据准备/教师/阶段全流程已通过。
4. 中断恢复限定完整update边界，脏日志拒绝隐式重放；教师partial保留负结果、不免费重试。真实大训练内存/IO、累计阶段墙钟和任务分布仍有风险，超限按获准协议停止，不能自动减对照、增预算或改奖励。

本轮五项工程阻塞已有实现与有界证据关闭。下一步应正式确认完整协议及执行绑定，再由用户另行授权科学S1/S2；不启动S3，不宣称已获得质量、速度或DRL增量。

最终六项主交付已逐项核读，主要数字、工程/科学边界、活动v2及待批v3、建议预算与REVISE结论一致；机器审查收据保存六文件SHA。Git元数据的当前分支、基准、保留refs/tags及14个新工程pt文件与只读工作区查询一致。最终提交/推送及远端SHA发生在本审查之后，由主代理实际核对，不将尚未发生的发布声称为已验证。
