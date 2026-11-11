# 独立审查结论

总体 claim_supported=no，confidence=high。本固定QP机制筛选质量FAIL，应关闭此32状态诊断支线。实际保留的最终网络仅13/32（40.625%）严格margin>1e-5，低于事前26/32门槛；原float32 hinge从0.057234799256548285降至0.01057092403061688。终止原因OUTER_LIMIT，32次外层均接受一个实际候选；hinge下降和求解完整不能代替质量门槛。

读取54行子问题记录后，只有每个outer的最终记录用于方向。22个中间切平面轮为INCOMPLETE_NEEDS_CUT_OR_SOLVE，随后追加约束完成；不是32个外层最终子问题不完整，也不能把54轮全部称完整全问题。32个最终记录均为COMPLETE：每轮覆盖3961个非teacher合法alternative（3993个合法候选含32个teacher），方向含全部17473原参数。32份最终证书的最大全问题dual gap为9.095879532900931e-7，最大全约束残差7.578421666598611e-8，均小于1e-6。活动行有效rank32至34只指32至34行活动Jacobian，不是3961行全Jacobian的秩。

对新增压缩证书做了有限stdlib算术重构，没有加载权重或运行模型/求解器：alpha非负、每状态和≤1/32；||d||²与alpha·Gram·alpha/REG²最大绝对差5.329070518200751e-15；full primal=REG/2||d||²+mean(full_slack)重构误差≤6.938893903907228e-18；dual=b_active·alpha-REG/2||d||²误差≤7.589415207398531e-19；gap误差≤6.938893903907228e-18；3961约束相对每状态slack的最大残差重构误差≤2.859583229930518e-18。它们支持记录与该数值容差下的局部soft-slack凸子问题完成一致，不能独立重建未保存的全Jacobian。soft slack本身允许用slack保证可行，所以“可解”不等于所有线性约束无slack分离，更不等于真实非线性网络可达到全32严格排名。

实际响应共112候选：32接受、80拒绝并逐次精确恢复。独立从新ACTUAL_STEP_RESPONSE.csv复算：79拒绝的实际原hinge未下降，13拒绝的protection失败，交集12；所以loss-only67、guard-only1、both12。13个protection失败均记录incremental KL>0.001，概率变化≤0.02、base TV≤0.1，112候选均finite。不能将本run归因为“全被guard挡住”，也不能说guard没有起作用。全部32个scale=1候选中，只有1个实际hinge下降而该次guard失败，其余31个hinge不降；最终接受scale=0.5一次、0.25十四次、0.125十七次。完整线性方向的预测收益在原网络全步上普遍没有兑现，回溯能得到真实下降，却未达到冻结质量阈值。该观察支持本固定局部一阶模型/正则/回溯组合在给定预算的不足，不能唯一归因负梯度内积、guard、表示能力或DRL不可能。也没有观测到在某一轮六档全部拒绝；实际均在前四档内接受。

接受判定确实由experiment.py的原float32 hinge下降>1e-9加当前old_logits到候选的incremental guard以及mask/幅度/finite共同决定。OPTIMIZATION_RESULT.final_states中的guard_reference=base_to_model是另一参考口径；其base到累计final的max_step_KL失败标志不能倒推此前接受的incremental步骤不合格。质量计数仍取最终保留网络，未选择最好中间网络。全局累计base-TV约束另行保留。

solver.py实现固定REG=1e-3的每状态slack均值加L2，dual每状态非负权重和≤1/32，活动精确梯度行配全部合法margin的AD JVP分离；方向只在最终完整条件满足后试用。源码和保存证书对原参数列与合法约束规模一致。模型/实验未引入新状态ID或结构，本审查没有运行新科学调用。

隔离路径PASS仅是另一个小claim。experiment.py先保存JOINT_final和OPTIMIZATION_RESULT，再进入隔离stage，读取共享见证到initial的deepcopy；control.register在首算前已对WITNESS执行SHA完整性字节读取并绑定来源；这不是参数反序列化或优化使用。运行worker的新trace中见证哈希/读取位于isolated22_position_existence_path_AFTER_optimization阶段，experiment.py中torch.load及参数语义访问也只在final保存后发生。因此协议“bytes first read”和DATA receipt的not_read措辞不能字面解释为首算前从未读过任何字节；正确且有源码支持的结论是共享参数未在主优化中作初值、目标或方向，首次参数语义加载在优化后。此澄清保留冻结文件原字节。21个隐藏参数插值位置的末层始终为零，输出与SL等价；第22个位置直接赋值完整共享见证，32/32且原guard通过，优化final模型hash保持不变。由此支持“这些标签相关诊断参数赋值可在所检查22离散位置通过保护，优化未用见证作初值/目标/方向”。它不证明标签无关优化、梯度学习或DRL可到达见证；也没有验证最后一次输出参数跳跃之间的连续轨迹，更不是正式训练路径。

旧hinge18/32 FAIL、CE14/32及共享存在性32/32 PASS全部保留。13/32低于这两个固定历史计数是描述事实；QP/dual/JVP和Adam128的计算机会不同，不支持公平预算优劣或算法普遍排序。无新场景、导航、教师请求、Critic更新、native、H或完整规划请求，不能据此声称完整规划或长期回报改善，更不能声称真实机械臂认证。

资源必须实账：32次QP优化、54切平面轮、9050 dual迭代、1050精确Jacobian行、54次全合法JVP扫描、112实际参数候选/32应用/80恢复、22隔离赋值。ENTRY_CONSUMPTION核心含来源登记扣账62.6488287773127秒；worker内部wall60.957364559173584秒和entry52.48399999999947秒是不同计时scope，不能相加当作总预算。完整审查由父执行器以REVIEW_START到全部artifact完成后的终点累计，最终任务总墙钟由其收口，1800秒上限未因本审查重置；本trace不预报尚未结算的总数。

缺失证据包括真实学习可达性、泛化、完整任务/轨迹和长期回报、等算力对照及唯一因果识别；这些缺口限制论文推论，本次关闭规则不授权为填补缺口而加实验。修订claim应为：“在固定32状态、原共享网络和原保护下，单配置完整线性化soft-slack QP经真实回溯降低hinge，但在32外层后最终13/32，未通过26/32机制筛选；隔离共享赋值仅提供标签相关存在性证据。”

路由为CLOSE_DIAGNOSTIC_BRANCH，不自动补充实验/损失/配置/步数、不恢复旧阶段、不在线DRL、不路线评价；全部诊断权重禁止部署或正式初始化。主研究目标不变，本失败不构成DRL不可能结论。未发现root或本包paper/PAPER_CLAIM_AUDIT.json，下游写作标记provisional — no paper claim audit available；此缺失不影响本次冻结质量判定no。
