# 冻结参考在线PG对照：实际运行报告

**STOP_CURRENT_CONFIGURATION。** 在固定64和128终点，PG-cond没有改善FrozenSL，也没有优于PG-global；相对匹配继续SL同样未通过事前路线质量门槛。当前配置结束，不延长到256，不增加条件、网络、步数或种子。研究目标仍是大规模复杂父场景DRL训练及冻结Actor的完整局部约束规划。

## 准确协议与实现

本轮从`16f5d05952c03c7ddf20623ddcc59b09c0011040`建立独立分支`codex/rc3-frozen-reference-online-pg-20261111`。首次实际调用前，提交`5194d9519de15b3c8d0803fca4cabe8e5446dc4a`冻结协议、全部科学源码、前128机会六问题/六标签日程、66案例与资源。

准确M2表为`rc3_r04_m2_20261111/attempts/02/FIT_ONLY_COMPONENT_ANCHORS.json`，SHA256=`495790b78e383bf78072d613da2e5b0a100c1aa391b3055eec19121e244b3d6b`。33fit物理父、窗口/可观测剩余比例四档、unknown→fit-global及表内容不变。b=−bL−0.75bK+bH由已归一分量合成，**不再次除n**。表及优势detach，运行中从未更新表；表不进入Actor-only部署包或部署选点。

三组均准确加载原SL256及S1来源，冻结基座；同206维距离ON、64/64 A1合法候选pre-tanh居中残差、零末层和±0.25幅度。同Adam lr3e-5、原guard、梯度裁剪、奖励/H/mask/端点和单候选更新语义。PG-global/PG-cond的唯一主要因素是冻结参考；不据本轮单独归因A1或距离摘要。

A_sample=detach(G_complete−b)，gamma=lambda=1，保留原批次全状态RMS缩放、state_SUM/episode_MEAN及原base-KL项；无TD截断、额外epoch或熵项变化。没有创建/更新神经Critic。每个RL机会从各自当前Actor独立生成3条10×10和4×3/6×6/8×8各1条完整轨迹；按事前common-random-number约定匹配随机流，各自采样调用，不共享更新后离策略轨迹。继续SL用同A1/距离ON、同机会和guard，重放已有6标签，route-mean CE；这是机会/结构控制，不声称相同FLOPs或相同任务标签分布。

## 实际工程与边界

首入口因准确来源目录漏登记A1包而在请求前失败：0请求、0更新，累计7.358秒保留。`1b422402ad05e085ce7c198041e6694eb7a1fd7f`仅补入指定基准的准确A1包源码及新入口身份；原PROTOCOL、SCHEDULE和科学worker/model/update源码原字节未变。没有跳过来源检查或更换任务。

真实工程完成19完整请求、6次Actor尝试，含失败/登记累计41.234826秒，在24/16/300上限内。零残差logits和完整raw/H路线等价、组件表单位/分箱/fallback/detach、两RL真实更新和匹配监督、真实Adam候选后注入拒绝的精确head+Adam回滚、checkpoint模型/Adam/RNG/游标/已用ID/查询账恢复及下一更新一致、Actor-only角色/阶段拒绝和更新后完整路线导出一致均PASS。注入拒绝只证明事务恢复，不冒充自然guard拒绝；工程模型全部弃作科学起点。

科学阶段从三份全新相同头初始化开始。三组各128机会，384次实际Actor更新；1536条在线完整轨迹、768条既有标签重放。64和128各66×6=396评价，共792评价请求；科学完整请求2328/2328成功、失败0，全部raw/H代理约束验证有效。三组各128候选自然guard全接受，回滚0；Critic更新0。

原60、旧S3任务与未授权最终测试未访问；新场景、导航求解、新教师均0。M1 G3FAIL、M2删除CB、旧停止记录与权重、旧v2/v3指针原字节保留。未恢复旧R03/M3或使用CB/CG模型作起点。

## 路线结果与事前判断

主要标准在首次计算前固定：PG-cond须先改善FrozenSL和匹配继续SL，再优于PG-global；两暴露cohort分别要求raw-J、H-J父布局等权节省≥0.5%且配对95%下界>0，各局部尺寸对FrozenSL/继续SL无点退化，全部路线有效。父布局配对20000次，PCG64 seed11652026。8个训练父的新任务selection与3个暴露development父分别报告；单seed、已暴露案例的区间为描述性证据。

64终点，所有PG-cond主比较均为0%节省，区间[0,0]，未通过。128终点，PG-cond对FrozenSL和PG-global仍均为0%，区间[0,0]；对继续SL，selection仍0，development的H-J节省−0.004556013%、raw-J节省+0.000333204%，区间均触0，未通过。128的8×8 H-J相对继续SL还违反无点退化：11父汇总节省−0.002312177%，3个development父为−0.008477983%。不能只因某个弱RL对照或某个raw点差异通过。

66案例的H-J均值，PG-cond/FrozenSL/PG-global两终点均137.889648；128继续SL为137.888000。原生Fixed3为134.764308、Detour16为132.926705。完整逐案例J/L/K、raw/H与分尺寸配对区间见[端点表](ENDPOINT_RESULTS.md)、[完整请求索引](execution_v2/science/REQUEST_INDEX.csv)及[完整对照表](execution_v2/science/QUALITY_CONTRASTS.csv)。

## 在线参考与Actor诊断

两RL组各记录60136在线更新状态，监督25037状态，合计145309。对同一已记录在线状态，G−Bglobal的标准差为0.943238，G−Bcond为0.458068；unknown回退0。条件参考确实改变了中心化回报的描述性分布，不能把它解释为路线收益或独立方差减少定理。

PG-cond的合法候选有效残差范围中位数0.000369、最大0.000892；对应基座top1/top2间隔中位数0.367437。PG-global残差范围中位数0.000173。PG-cond仅在1个已记录在线状态出现base→更新后top1差异，PG-global为0；这不是评价路线收益。全部guard接受，说明本轮没有因拒绝回滚挡住候选；这些诊断仍不足以确认零收益的唯一因果机制。

逐状态G、两种B、两种中心化回报、实际A_sample、有效残差、基座间隔、候选/应用后ratio、KL/TV、guard与回滚均保留。旧collector容器中的V_old=0/null Critic只是未使用占位，不是实际训练参考；实际值由新Trainer和诊断归档绑定。详见[诊断汇总](ONLINE_REFERENCE_SUMMARY.csv)、[在线计数](ONLINE_SUMMARY.json)及[schema说明](SCHEMA_AND_ACCOUNTING_NOTES.md)。

## 完整时间与资源

科学外层完整墙钟2490.281秒，工程累计41.234826秒；新增报告与独立主张审查另计入本任务资源账。科学计账含输入恢复、源验证/缓存、采样、真实更新、监督重放、64/128完整评价、写入、数值审计。标签重放768条/25037状态，实际344.983686秒，未作新教师请求。

128终点完整请求中位数：PG-cond0.510359秒、PG-global0.481923、继续SL0.488458、FrozenSL0.417793、Fixed3 0.363377、Detour16 0.431446。未观察到本配置质量/速度优势。PG-cond两终点132次评价的新增距离摘要/准备/拼接开销合计2.817021秒；完整请求及新增特征、主要I/O分别在[时间表](REQUEST_TIMING_SUMMARY.csv)和[完整计时收据](execution_v2/science/REQUEST_TIMING.csv)中。

计时从输入恢复前到完整归档/索引/完成journal之后；计时收据自身写入在阶段墙钟内。已有runtime内冷模型加载不等于每次新Python进程启动。文件系统缓存未控制；同进程准确身份字节缓存有93729命中，实际2533个完整字节哈希、691495302字节；不是路线缓存。请求动态特征计138110状态，监督状态另计；子阶段包含时间不重复相加作为总墙钟。

科学显式共同H完成2064次；原生请求264次。全阶段recipe792、construct792、two_opt10582、joint_search2856的实际调用及包含/独占时间分别保留（这些总数含Actor共同H及原生内部调用，不全归于native）。原生最终路线属于完整原生搜索/后处理输出，raw列不是新的去H对照；原生全部pre/post-H中间路线没有单独捕获，原协议这项预期覆盖过宽。该限制已如实披露，不能声称独立原生H归因；它不影响Actor/SL/PG-global主比较的零收益判断。共享H与原生内部调用边界见[计账说明](SCHEMA_AND_ACCOUNTING_NOTES.md)，未补跑或覆盖冻结协议。

## 独立复核与结论边界

新输出独立数值路径复核1326384项、2328完整请求、145309在线诊断状态和384角色机会，PASS；包括完整G、冻结表/优势、J=L+.75K、raw/H约束证据、配对区间及门槛。没有重跑历史千万项CSV审计。工程/科学两外层进程树均正常退出、活动进程0。

本轮支持“当前预算和配置下未得到PG-cond路线收益，应结束当前配置”。不支持条件参考一般无效、所有DRL无效、A1或距离摘要独立有效、独立泛化、实机IK认证或部署优势。后续任何新研究另行设计和授权；当前不增加步数/网络/条件，不重开神经Critic离线扩展，不恢复旧M3/R03。独立主张审查与最终路由见[RESULT_TO_CLAIM.md](RESULT_TO_CLAIM.md)。
