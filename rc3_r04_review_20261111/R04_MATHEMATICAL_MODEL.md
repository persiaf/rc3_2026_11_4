# R04数学定义、单位与接口约束

本文件定义待审R04模型，不激活训练。固定目标仍是在10×10等大规模复杂父场景训练DRL，冻结后完成4×3、6×6、8×8等局部窗口的完整可行规划。保持冻结SL、小幅Actor修正和训练期Critic。R03的128停止、256未开始及后续离线诊断的负结论保持原身份。

## 1. 完整状态、观测和固定归一尺度

令完整规划状态为

\[
S_t=(\mathcal P,\mathcal W,\mathcal U_t,c_t,\rho_{0:t},e_{in},e_{out}),
\]

其中父布局和不可变通行表为\(\mathcal P\)，窗口为\(\mathcal W\)，剩余任务集合为\(\mathcal U_t\)，当前位置为\(c_t\)，有序已选停靠前缀为\(\rho_{0:t}\)，端点固定。合法候选集合为\(\mathcal A(S_t)\)。本轮不改变覆盖、导航和代理可达/安全mask。

Critic只接收已有\(x_t=\psi(S_t)\in\mathbb R^{268}\)。旧实现为四个64维向量与12个上下文量，\(x_{256}\)是当前剩余任务比例，\(10x_{265},10x_{266}\)是窗口宽高。268维并未保存完整有序几何前缀；不能据此宣布观测是Markov充分状态。严格的价值定义在\(S_t\)上，网络近似的是基于\(x_t\)的条件期望，可能存在观测混叠。

初始任务数\(n=|\mathcal U_0|\)在整条轨迹固定，\(s_n=\max(1,n)\)。不得替换成变化的\(|\mathcal U_t|\)。对\(n=0\)没有动作状态，Critic接口只允许\(T=0\)空张量；其scale=1只是除零约定，不生成或评价空任务路线。完整无动作终端代价仍由既有接口处理，不调用本轮新H。

## 2. H历史前缀修正和完整回报

给定已保存完整raw路线\(\rho\)，raw代价为

\[
J_{raw}=L_{raw}+\lambda K_{raw},\qquad \lambda=0.75.
\]

冻结后处理H得到\(J_H\)，定义有符号整轨迹会计信用

\[
\Delta_H(\rho)=J_{raw}(\rho)-J_H(\rho).
\]

H可能删除或重排已经构建的前缀，\(\Delta_H\)不是“从当前位置还要走的物理距离”。令动作前raw剩余长度\(L_t\)包含最后退出腿，\(K_t\)为尚余服务次数，则保存目标为

\[
G_t=-\frac{L_t+\lambda K_t-\Delta_H(\rho)}{s_n}.
\]

该信用只作为原终端奖励的一次会计修正，不在每个动作重新调用H；同一个完整轨迹信用出现在各\(G_t\)的目标分解中，不表示奖励被重复发放。未来H与历史前缀的依赖也是268维混叠风险的一部分。当前仅二维代理研究，此H不表示真实双臂IK或实机认证。

单位约定：\(L_t/s_n\)为距离/初始任务，\(K_t/s_n\)为停靠次数/初始任务；\(\lambda\)将停靠次数换算为同一代价单位，\(\Delta_H/s_n\)为有符号代价信用/初始任务，\(G_t\)与\(V_t\)为负代价/初始任务。

## 3. C0与C1：只改输出参数化

C0输出raw单位的\((\widehat L_t,\widehat K_t,\widehat H_t)\)，组合值为

\[
\widehat V_{C0}=-(\widehat L_t+\lambda\widehat K_t-\widehat H_t)/s_n.
\]

C1保持268-128-128-3 MLP结构，直接输出

\[
\widehat\ell_t=\operatorname{softplus}(a_L(x_t)),\quad
\widehat k_t=\operatorname{softplus}(a_K(x_t)),\quad
\widehat h_t=a_H(x_t),
\]

\[
\widehat V_{C1}=-(\widehat\ell_t+\lambda\widehat k_t-\widehat h_t).
\]

目标为\((\ell_t,k_t,h_t)=(L_t,K_t,\Delta_H)/s_n\)，保留分量损失

\[
\mathcal L_{comp,t}=\tfrac12[(\widehat\ell_t-\ell_t)^2+
(\widehat k_t-k_t)^2+(\widehat h_t-h_t)^2].
\]

本候选不自动加入C2组合Value损失。C2曾在固定C1基础上加\(\tfrac12(\widehat V-G)^2\)，组合误差改善不能抵消H分量退化这一历史事实。

新`C1Prediction`显式记录schema、固定初始n和单位；`as_training_dict()`先恢复raw字段\((s_n\widehat\ell,s_n\widehat k,s_n\widehat h)\)，旧`remaining_cost_loss`随后只除一次\(s_n\)，与上式逐状态等价。将normalized字段直接冒充raw字段再交给旧loss会二次除n，接口和人工张量测试专门检查该错误。

逐状态损失公式与状态归约是两个选择。R03 PPO使用trajectory内state **sum**、再按episode数平均，上一轮离线比较使用state **mean**。R04接口返回未归约向量，不静默改变此选择；任何未来训练协议须明确各组相同归约，不能把归约变化混入C1单因素。

相同初始净张量不等于相同初始物理预测：C0与C1输出单位不同。已有离线改善支持C1作为新Critic基础候选，不建立“仅梯度尺度”因果解释，更不建立路线收益。未来比较必须保留初始化与终点结果，明确是否另设物理预测匹配控制。

## 4. 可观测条件基线及分量一致性

候选条件只使用规划时已经知道的量：

\[
z(x_t)=(w,h,q_t),\quad
q_t=\min\{3,\max\{0,\lfloor4(1-\operatorname{clip}(x_{256},0,1))\rfloor\}\}.
\]

不使用\(t/T\)、最终路径长度、最终H、验证标签或验证均值。训练折中的H和完整RTG可以作为**拟合标签**；它们不能成为查询时输入。`ObservableCondition`只接受宽、高和当前剩余任务比例，携带未来量的输入字典会被拒绝。

在预先隔离的fit物理父布局上，先对同父同trajectory的条件内状态求均值，再对trajectory等权、父布局等权，估计三个归一分量均值

\[
b(z)=(b_\ell(z),b_k(z),b_h(z)).
\]

\[
V_{cond}(x)=-(b_\ell(z(x))+\lambda b_k(z(x))-b_h(z(x))).
\]

未知条件回退到同一fit折的全局分量均值。每条轨迹先按自身固定n归一，再估计这些统计量；不能将跨n的raw均值除以当前查询n。artifact绑定fit物理父ID、来源hash、聚合口径、条件schema和内容hash；validation/test/未登记父或重复状态输入硬拒绝。

若评估残差Critic，保持分量解释，定义

\[
(\widehat\ell,\widehat k,\widehat h)=b(z)+r_\phi(x),
\quad
f_\phi(x)=-(r_L(x)+\lambda r_K(x)-r_H(x)),
\]

\[
V(x)=V_{cond}(x)+f_\phi(x).
\]

不是另设一个无法与L/K/H解释相容的独立scalar Value头。接口要求\(r_L\ge-b_\ell,r_K\ge-b_k\)，合成后的L/K非负；违反时硬拒绝，不静默clip。零残差严格回到条件基线。当前仅实现合成/单位接口，不实现或训练残差网络。若未来选择可优化的正值参数化，须把变换和边界行为写入新协议，例如对正的baseline分量使用inverse-softplus offset；baseline为0时的floor及其改变的初始锚点也须预先说明，不能事后修补。

条件表是当前强参照，并不证明残差架构“必要”。已有C1未全面超过它，C2也只在父等权总表超过它。只有在明确固定预算的C1对照下，条件残差在两种口径、主要窗口、分量校准及完整时间上体现增量价值，才有理由增加该复杂度。表估计与查询均属于Critic训练侧，部署Actor不携带Critic、baseline表或Adam。

## 5. Actor合法集合内的相对修正

令冻结SL合法logit为\(z_j^0\)，新网络原始候选分数为\(f_j\)，\(M=|\mathcal A(S)|\ge1\)：

\[
\bar f_M=\frac1M\sum_{k\in\mathcal A(S)}f_k,\quad
u_j=f_j-\bar f_M,\quad
\delta_j=0.25\tanh(u_j),\quad
z_j=z_j^0+\delta_j.
\]

非法动作不进均值，最终logit为−∞。全非法状态拒绝而非除零。单合法动作\(u=0\)，因此修正为0。末层零初始化给出所有\(f_j=0\)、所有合法\(\delta_j=0\)，合法logits/mask与SL严格等价。

对任意共同常数c，\((f_j+c)-\operatorname{mean}_{legal}(f+c)=u_j\)。因此pre-tanh共同偏移和共享输出bias被消去，其共同方向导数为0。这是结构性质，不是训练性能结论。

**但它不保证tanh后的修正均值为0。**例如\(u=(-a,-a,2a)\)均值为0，但对\(a>0\)，\(\tanh(2a)<2\tanh(a)\)，故\(\operatorname{mean}(\delta)<0\)。因此不能声称本式彻底消除了旧诊断的post-tanh共同残差能量或必然改变top1；仍需记录原始分数、合法均值、\(u\)、真实\(\delta\)及其再去均值后的有效部分。再对\(\delta\)减均值可能突破±0.25，属于另一方案，本轮不得偷偷加入。

每个合法动作仍满足\(|\delta_j|\le0.25\)，两动作残差差至多0.5。因此SL top1/top2 gap严格大于0.5时，唯一首选仍不可能被翻转；gap=0.5最多触及并列，须保留tie规则。旧样本29.79%的多合法状态之结构约束不会因居中自动消失。

guard仍独立检查候选更新的每状态KL、概率最大变化和相对基座TV；居中不代替guard，也不证明未见状态、路线成本或安全收益。工程fixture可用手动参数变化制造接受和拒绝，不执行optimizer。更新前ratio使用原采样policy，更新后ratio另算；候选全排序、稳定tie规则和首选变化必须可复核。

## 6. 三部分依赖和可证伪范围

C1先解决输出单位；条件baseline是在固定C1语义下增加的可选锚点；相对Actor改变合法候选之间的表达。它们不是同一个因素，不能由Critic误差下降推导Actor改进，也不能由排序变化推导完整规划收益。

可证伪假设分别为：C1在固定条件下改善留出校准；条件残差较直接C1有两口径稳定增量且开销值得；居中Actor在相同Critic下学到更有用的相对排序，而非仅移除一个无效参数方向。若任一失败，保留原结果并拒绝将它升级为已有效机制。必要工程测试只检验单位、隔离、代数恒等式、幅度、mask、guard与记录接口，不能代替这些科学证据。
