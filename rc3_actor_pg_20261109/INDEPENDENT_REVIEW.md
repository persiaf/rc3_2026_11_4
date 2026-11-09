# 独立结果与主张审查

**结论：证据审计 GO；本轮候选晋级 REVISE；总体 `claim_supported=partial`。** 这里的 GO 仅表示已完成的有界执行与保存证据可采，不授权新训练、旧 S3 补跑或保留测试解封。

十二个 actor-only 导出与原始参数逐项完全相等，本机固定 DEV 请求中 D1/D2 记录到加载生命周期加速且规划输出精确等价。B 相对 A 的 raw 成本有正向小试证据，但未传递为预登记的 postH 父布局相对收益；B 对共同 SL256 起点也未通过。不能据此确认 DRL 性能提升、收敛或 S3 成功。

## 审查身份与边界

- 独立委派的 paper architect/reviewer；非执行者自行确认。`pendingExternalReview=false`。
- 基准 `925a0a1a8957299e3e849b29ac501a99212aa902`；源码绑定 `57b0d0d2236b4b1f942a393ce2fc41e6408f4e7ec899df038bdfc4b19a65eee6`。
- 独立读取 202 个绑定源码、1,392 个输入来源、12 个导出、756+54 原始评估、3,072 个训练终局与 256 份更新完整梯度/权重。49,638 个历史证据文件共 9,899,434,805 字节只核原始字节 SHA，未解释旧60或旧S3任务内容。
- 本审查模型 forward、rollout、optimizer、backward、场景/任务生成、导航求解、native、H、教师、旧S3重跑全部为 0。旧实验失败与缺口不回填。
- 没有提供论文或 `paper/PAPER_CLAIM_AUDIT.json`：`paper_claim_audit=unavailable`，写作判断为 provisional；实际结果报告仍可交付。

## 精确制品与部署结果

全部 12 个导出均核对名称、shape、dtype、逐值 `torch.equal`、整体及逐参数 hash；原训练 checkpoint、sidecar、commit 共 36 文件保留。actor 仅含原模型参数、架构与来源，未量化、剪枝或蒸馏。每个actor文件268,504字节；用于计时的seed11622026 RL512完整checkpoint为37,950,130字节，其他角色完整checkpoint大小不同，因此各角色提速幅度应分别报告。

独立重建 PCG64/11642026 的 756 请求顺序，与每个请求的预扣账、完成日志、原始文件 hash 一致。42 个 case×method 组合在 D0/D1/D2、cold/hot、3 重复下 raw/final 路线、J/L/K、来源和 assignments 精确相同；108 对 RL_raw/RL_plus_H 的 raw 路线一致。540 次神经请求均重新编码，432 次部署共同 H 与方法定义相符。

| 生命周期 | 启动至 ready (s) | 请求中模型加载 | 驻留命中 |
|---|---:|---:|---:|
| D0 | 4.541369 | 180 | 0 |
| D1 | 3.799154 | 180 | 0 |
| D2 | 3.753878 | 4 | 180 |

D0 为完整 checkpoint 逐请求加载，D1 为 actor-only 逐请求加载，D2 在启动阶段加载 4 个 actor，键为 `(seed, actor SHA256)`，仅模型参数常驻。D2 的动态编码、历史状态和路线不跨请求缓存。

下表为完整请求 median / p95 秒，每单元 18 请求；包含任务解析与来源校验、导航/view、模型访问、fresh encode、构造、H、导出验证及制品写入/hash。hot 导航预热发生在请求计时外，但计入阶段墙钟。嵌套组件不可相加。

| 方法 | 导航 | D0 median / p95 | D1 median / p95 | D2 median / p95 |
|---|---|---:|---:|---:|
| Fixed3_endpoint_adapter | cold | 0.591347 / 0.715873 | 0.582401 / 0.667707 | 0.568430 / 0.748137 |
| Fixed3_endpoint_adapter | hot | 0.365399 / 0.733211 | 0.349378 / 0.392176 | 0.355382 / 0.410961 |
| Detour16_endpoint_adapter | cold | 0.696029 / 1.206404 | 0.700258 / 0.880634 | 0.693657 / 0.803315 |
| Detour16_endpoint_adapter | hot | 0.487209 / 1.051135 | 0.470864 / 0.609640 | 0.463796 / 0.590939 |
| untrained | cold | 0.709669 / 0.959172 | 0.701212 / 0.779971 | 0.700785 / 0.758348 |
| untrained | hot | 0.516658 / 0.986056 | 0.498335 / 0.593756 | 0.481370 / 0.551299 |
| frozen_supervised | cold | 0.717238 / 0.970277 | 0.709897 / 0.811656 | 0.714271 / 0.776556 |
| frozen_supervised | hot | 0.510091 / 1.045021 | 0.500312 / 0.566131 | 0.462265 / 0.592412 |
| continued_SL | cold | 0.960562 / 1.124412 | 0.711941 / 0.792419 | 0.691463 / 0.745481 |
| continued_SL | hot | 0.709045 / 1.207955 | 0.494972 / 0.552386 | 0.494202 / 0.560995 |
| RL_raw | cold | 1.322044 / 1.616229 | 0.696607 / 0.753018 | 0.689994 / 0.749939 |
| RL_raw | hot | 1.123169 / 1.552742 | 0.487903 / 0.620021 | 0.445240 / 0.528123 |
| RL_plus_H | cold | 1.342769 / 1.588658 | 0.702113 / 0.753957 | 0.697890 / 0.834002 |
| RL_plus_H | hot | 1.104739 / 1.506505 | 0.518722 / 0.549511 | 0.465004 / 0.535066 |

以旧 RL512 的 RL_plus_H 为例，D0→D2 的 cold median 为 1.342769→0.697890 s（1.924×），hot 为 1.104739→0.465004 s（2.376×）；p95 分别为 1.588658→0.834002 s 与 1.506505→0.535066 s。

**解释范围：** 只测一台机器、一个计时 seed、同一父布局的六个旧 DEV 窗口，顺序固定 D0→D1→D2。native 对照也发生时间漂移，尤其 p95，因此不能将所有差异归因于 actor 制品或外推通用硬件因果加速。D2 相对 D1 的增量应从表中独立看待。启动成本不能隐去。启动与首请求 span 之和不含中间 receipt/predebit IO，且首方法不同，不能称为精确进程到首响应时间。

## 匹配策略梯度与谱系

A/B 的 0 号完整 checkpoint 均为同一 SL256 精确参数 `1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`，Adam state 为空、查询游标为0、named RNG为空。两组 128×4 个 specs/view identities 完全匹配，64 full + 64 local 批次；各自实际完成 128 更新与1,536 终局请求。新推理 actor 精确等于各自128终点。

独立逐条核对了 3,072 轨迹的角色、seed 派生、action/mask/logprob、reward=-J、LOO baseline/advantage 与加权目标；A primary 首 logprob 保留，1,024 个 forced 首动作不计 logprob；B 的1,536个 policy 首动作均有完整 logprob，三流 seed 独立、等权。两源均显式 detached LOO。每更新25个参数的完整前后裁剪梯度均有限且与范数、clip1、保存 Adam 增量一致；每32更新 checkpoint 的权重、Adam矩/step、RNG、缓存/请求预算和游标均连续。

两组均128/128更新触发clip1、0次zero_signal；裁剪前范数A为79.778912–2466.050051、B为48.781604–1597.049654。由已保存梯度与freshAdam独立计算的权重增量最大绝对误差约5.96e-8。普遍裁剪是应纳入后续DRL设计解释的事实，但不能据此直接认定性能差异的原因。

审查没有用模型重算 logits/梯度：这里确认的是绑定源码、保存的数学关系、完整张量与更新谱系；不能用这些软件事实证明规划变好。B 同时改变 forcing 与 mixture weights，不能单独归因于首动作。

## 固定终点 DEV 结果

54 请求=18例×SL/A/B；18例属于3个真实父布局，各6个尺寸/端点窗口，只有1个优化seed。主统计先在每父布局内对6例求平均成本，再计算相对收益 `(control-B)/control`，最后对3父布局等权平均。以下95%区间独立以 PCG64/11632026、20,000次父布局 bootstrap 复算；仅为描述性区间。

| 比较 | 指标 | 父布局平均相对收益 | 95%描述区间 | 预登记正向条件 |
|---|---|---:|---:|---|
| B vs A128 | raw J | +7.998928% | [+3.669895%, +12.650376%] | 通过 |
| B vs SL256 | raw J | +1.268564% | [-1.747777%, +6.022956%] | 未通过 |
| B vs A128 | postH J | -0.192737% | [-2.871708%, +3.232858%] | 未通过 |
| B vs SL256 | postH J | -0.056127% | [-1.479473%, +2.365241%] | 未通过 |

**两项 postH 条件均未通过，故候选判定 REVISE。** raw B>A 的结果不能代替最终共同H交付质量，也不能用“比A少退化”证明B优于共同起点。独立补算A对SL的raw父布局平均相对收益为−7.344875%（描述区间[−8.823322%,−5.624070%]），说明只看B对A会高估相对共同起点的进步。3簇bootstrap不能扩大为18独立布局或3优化seed的证据。

同时，18例的绝对成本均值为：

| 模型 | raw J | postH J | postH L | postH K |
|---|---:|---:|---:|---:|
| SL256 | 193.944536547 | 156.683831883 | 128.558831883 | 37.500000 |
| A128 | 208.156717545 | 156.594535809 | 128.552869142 | 37.388889 |
| B128 | 190.393370540 | 156.390867888 | 128.474201222 | 37.222222 |

B 的 postH 绝对均值较 A 降低0.130061%、较 SL 降低0.186978%，所以不能写成“绝对平均J恶化”。绝对均值之比按各父布局基线成本隐式加权，与预登记的等权父布局相对收益不同；二者方向差异属聚合定义差异。不能事后改用绝对均值推翻原筛选。完整父布局、单例及尺寸/端点表见审计JSON及原结果CSV。

## 资源与失败核对

| 阶段 | 完成请求/更新 | 墙钟 / 上限 (s) | 状态 |
|---|---|---:|---|
| DEPLOYMENT | 756 requests / 0 updates | 580.676555 / 1800 | COMPLETE |
| A128 | 1,536 terminals / 128 updates | 662.333310 / 1800 | COMPLETE |
| B128 | 1,536 terminals / 128 updates | 648.278817 / 1800 | COMPLETE |
| DEV54 | 54 requests / 0 updates | 65.913741 / 600 | COMPLETE |

训练+DEV机制额度为3,072+54=3,126，不与756部署额度混用。全部科学完成请求合计3,882；实际新导航435（训练0，使用已保存导航）、共同H3,558、native216；教师0。训练terminal cache两组均0命中、0失败，所有预扣账与完成一致；没有重抽、换seed、补抽或重置旧资源的保存证据。工程验收另账8项PASS、1.510073 s/120 s、真实optimizer/模型推理/导航/native/H/教师0；独立审查未重跑这些测试。

本次独立数值审计脚本墙钟 464.644210 s；补充审计 1.417834 s。从首个审查UTC时钟至出具本报告约 1085.6 s，均为维护/审查账，新增科学调用0。

## 主张与下一步路由

| 分项 | claim_supported | confidence | 判定 |
|---|---|---|---|
| actor精确质量及本机观测速度 | yes（限定上述测试范围） | high | GO |
| B相对A的规划提升 | partial（只支持raw） | medium | REVISE |
| B相对共同SL起点的注册提升 | no | medium | REVISE |
| 广义DRL快速高质量/泛化/真实机器人认证 | no | high（明确未获证实） | REVISE |

最小可用表述：Exact actor-only export and parameter residency reduce observed lifecycle overhead without changing saved DEV plans. The joint all-policy/equal-weight PG variant improves raw planning cost relative to original A on this small DEV screen, but fails the preregistered final postH candidate gate against A and the common SL256 start.

保留精确actor制品、完整A/B原证据与负/混合结果，关闭本次有界小试。下一步先围绕raw与postH不一致、普遍裁剪和混合采样机制修订DRL设计，再准备独立优化seed与真实父布局验证；如果要做forcing因果解释，应另行设计单因素/因子对照。该路由不执行新训练，也不授予旧S3或旧60权限。研究目标继续是大规模复杂父场景DRL训练、冻结后完整局部规划，监督只作初始化/辅助/对照。

真实标定、IK、机械臂扫掠路径与关节轨迹认证仍缺；本轮只核保存的代理same-arm F&A、导航与完整任务覆盖验证，不能提升为真实机械臂认证。

独立可复核产物：[主数值审计](paper/review-traces/experiment-result-to-claim/20261109_run01/independent_numeric_audit.json)、[补充聚合/资源核对](paper/review-traces/experiment-result-to-claim/20261109_run01/additional_saved_data_checks.json)、[完整审查轨迹](paper/review-traces/experiment-result-to-claim/20261109_run01/TRACE_MANIFEST.json)。
