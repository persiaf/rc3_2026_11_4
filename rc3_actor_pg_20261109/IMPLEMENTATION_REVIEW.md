# 独立部署制品与策略梯度机制小试：实施判断

**GO：在当前用户明确授权的独立新实验额度内执行。** 这不是旧v3/S3恢复，不使用旧剩余额度，不改变旧协议/活动执行指针或S3结论。基准为`925a0a1a8957299e3e849b29ac501a99212aa902`，独立分支`codex/actor-pg-screen-20261109`。

一次性方案与资源核查已完成：[PROTOCOL.json](PROTOCOL.json)、[AUTHORIZATION.json](AUTHORIZATION.json)、[INPUT_REGISTRY.json](INPUT_REGISTRY.json)、[请求顺序](DEPLOYMENT_GRID.json)。用户已明确要求在同一任务执行这些有界实验；生成的源码SHA不是声称用户事先读过的文件，而是执行者按获准任务固化的技术身份。准确绑定为[SOURCE_BINDING.json](SOURCE_BINDING.json)，SHA256 `57b0d0d2236b4b1f942a393ce2fc41e6408f4e7ec899df038bdfc4b19a65eee6`。

## 已核查的实施影响

- 十二个原固定权重只导出参数、原架构/输入配置和来源身份；不量化、不剪枝、不蒸馏。完整训练checkpoint不写。导出逐项检查参数名称、shape、dtype、值和tensor hash。
- D0/D1/D2复用原七方法流水线、合法mask、同臂F&A、完整父导航、端点、共同H、导出和独立验证。仅模型加载器可替换；D2缓存键为seed与制品SHA，只常驻参数，每请求重新编码。各模式独立进程，启动、导航预热和请求完整时间分开记录，全部包含在1800秒阶段账。
- 六例固定为已有DEV_000至DEV_005，包含三尺寸与两端点；不按结果选例。756个请求、方法随机顺序和全部文件摘要已预先保存。精确路线与J/L/K等价是实施正确性条件。
- A组直接继承冻结原Trainer数学；B组是明确的新IIDTrainer，三轨迹独立采样、完整logprob、等权detached LOO。网络、MDP、奖励、H、Adam1e-4、clip1、batch4、父/局部比例不变。
- A/B使用同一份原seed11622026前128×4个训练声明，512个输入均已存在；只读加载原scene/F/A/navigation及兼容receipt，不生成任务或改A。原SceneProvider携带的是旧数据解释身份，不冒充旧S2阶段执行授权；真正新执行范围由本轮AUTHORIZATION约束。
- 两组分别从精确SL256复制参数，各建fresh Adam。每组128更新/1536终局请求/1800秒；固定终点后18 DEV×共同起点/A/B共54请求/600秒。3072+54=3126硬上限。缓存命中也预扣请求，失败不能退款或补抽。
- 每更新保存全部动作、mask、logprob、奖励、优势、梯度及结果参数；每32更新保存完整Adam/RNG/cache/checkpoint。单写入者、哈希链、原子锚有界重试及独立父进程计时收尾保留。固定输出路径及STARTED拒绝隐式重跑/预算清零。

## 工程验收

[8项测试全部PASS](ENGINEERING_VERIFICATION.json)：B似然与独立手算梯度、batch4 joint/accumulate等价、A原强制首动作/单合法首动作、三采样流独立重放、参数schema/value负测、actor来源/架构拒绝、D2仅参数身份缓存、单写入者及日志锚。真实optimizer、GlobalLocalPolicy推理、导航、native、H和教师均0；合成参数的autograd独立记账。

## 已登记的解释边界与风险

部署是单机、一个模型seed、一个旧DEV父布局的六个窗口，模式顺序固定D0→D1→D2。它测实际加载生命周期，不能宣称随机交叉的通用硬件因果性能；启动成本不能隐去。部署等价成立也不能改写旧S3时间或门槛。

B同时改变强制起点和1/.25/.25混合权重，这是用户指定的联合机制对照，不能单独证明只有强制首动作导致某现象。机制仅一个优化seed、128更新、三个DEV父布局；必须同时比较B对A和B对共同SL起点，不以少退化充当改进，不以训练loss或回报当冻结性能。该小试不是S3确认或收敛证明。

研究目标仍是大规模父场景DRL训练、冻结策略快速生成完整约束局部路线。任何不足只限定本配置及有限证据，不以监督或规则替代固定目标。旧60封存，旧S3两个失败槽不补跑。
