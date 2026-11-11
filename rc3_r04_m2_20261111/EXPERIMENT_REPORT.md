# R04-M2 条件锚点Critic可删除性实验

**决定：停止当前CB候选，不保留这版神经残差进入后续训练；训练期参考优先采用fit-only条件分量表B-cond。** CB有额外点改善，但未同时满足预登记可靠性和各尺寸不退化标准。CG相对自身全局锚点确有Value误差改善，却仍明显弱于B-cond；“学习完全无效”不成立。

## 设计与准确身份

基准 `7aeab00d50793a3bd8f35adf99e74581e29350fb`；独立分支 `codex/rc3-r04-m2-conditional-anchor-20261111`。协议及科学源码在 `26ab03306` 首次提交；默认Python导入Torch失败且消费0更新/0数据，随后在 `a8df31312` 登记M1准确解释器与独立输出目录，原协议及worker/model/analysis/audit源码字节不变。真实数值执行使用该提交。所有失败时间保留，不重置旧账。

[准确协议](M2_PROTOCOL.json)的SHA256为 `ef26eabda1899e7d4d35901f0bce40882722cc8c8a5726c3db8364455af55979`；[实际入口源码绑定](attempts/02/SOURCE_BINDING.json)独立于旧M1绑定；[实际环境](attempts/02/REAL_PREFLIGHT.json)为Python3.10.20、Torch2.12.1+cu126、NumPy2.2.6、单线程确定性CPU。复用已验收Windows Job监督器和真实Torch来源分类器。

116条轨迹、33fit/16holdout物理父布局及共同日程完全复用：64fit＋52holdout轨迹、共7992状态。准确268维已保存数据的LFS字节、数据/原始恢复源码/index/SL256基座身份与各轨迹输入张量哈希均核验后直接加载；数据只加载一次，没有轨迹重放、编码器前向、导航或新标签。[数据复用核验](attempts/02/DATA_REUSE_VERIFICATION.json)保留准确身份链。原M1登记中的“当时尚未保存268维”是历史状态，本轮加载随后已保存的准确artifact，不回写旧记录。

B-cond只用窗口宽高与可观测v[256]剩余任务比例四档，33fit父等权、轨迹等权、条件内状态均值；unknown回退fit-global。仅L/n、K/n、H/n作拟合标签，禁止运行时目标输入；不拟合holdout尺度或偏置。[条件分量表](attempts/02/FIT_ONLY_COMPONENT_ANCHORS.json)与旧fit-only参照的分量统计一致，Value重新由同一分量线性组合，未单独拟合Value。

CG/CB均为268→128→128→3 ReLU网络、51331个可训练参数；相同隐藏/净张量，末层权重与偏置全零。两者仅外部锚点条件不同。L/K使用事前固定的softplus(invsoftplus(max(b,1e-6))+u)，输出下限1e-6；H=bH+uH保持有符号，Value=−(L/n+0.75K/n−H/n)。工程测试覆盖零残差身份、零值下限、极端值/梯度、未知条件回退、fit隔离和运行时输入接口；optimizer0。

Adam lr1e-4、betas(.9,.999)、eps1e-8、weight_decay0、梯度裁剪1；共享原日程前128批，每批8个不同fit父布局。拟合归约为每状态0.5×三归一分量误差平方和、每轨迹state_SUM再episode_MEAN；无C2额外Value项。CG/CB各128步，总256次Critic更新；每次backward前扣账。step0/128的4份实际checkpoint均保存并验证模型/Adam/RNG/游标/条件表/资源快照的精确往返；step0恢复状态直接用于训练，无中间选模。准确旧C1step128只加载固定export，未重训。

## 主比较与继续门槛

[端点结果](ENDPOINT_RESULTS.md)同时列出两聚合的L/K/H及Value MSE、分尺寸和所有主比较；[完整指标](attempts/02/M2_METRICS.csv)含fit/holdout、全部原始和归一分量、bias与父内误差；[配对区间](attempts/02/PAIRED_PARENT_INTERVALS.csv)覆盖Value及各分量、全部尺寸。物理父布局配对bootstrap20000次，PCG64 seed11622026，状态不作为独立样本。

首次计算前固定：CB对B-cond与CG在两聚合下均须Value比≤0.95且配对95%区间上界<1；各归一分量比≤1.10；各尺寸至少3父、Value点比≤1。C1仅为同预算参照。CG相对自身全局锚点接受同样Value诊断，但该诊断不自动晋升CG。

CB/B-cond的Value比为0.832069 [0.767051,0.940156]与0.893331 [0.793903,1.015481]；CB/CG为0.331435与0.297004，区间均低于1。CB的三归一分量总体无门槛级退化；但父等权区间未通过，6×6的CB/B-cond点比为1.037428与1.106671。32项检查29项通过、3项失败，判定STOP_DELETE_CB_RESIDUAL；没有追加步数、条件、网络或种子。

CG/B-global为0.635704与0.680475，区间均低于1，说明全局锚点上存在可学习的离线残差。B-cond/B-global更低，为0.253218与0.226236；CG/B-cond仍为2.510501与3.007812。CB/C1为0.290593与0.264075；CG/C1为0.876771与0.889130。二者优于旧C1，不单独证明新增模块必要。

## 归因与下一步

主要误差收益来自初始条件统计。CB在此之上另有16.79%/10.67%的Value点改善；状态汇总区间支持该离线改善，父布局等权区间仍跨无差异，且6×6未达到不退化标准。因此不能说全部收益只来自统计，也不能说神经残差已值得保留。按已登记的复杂性门槛，删除当前CB残差，冻结B-cond作为后续训练期的分量参考与必要基线；不以它替代父场景DRL或冻结Actor的完整规划目标。

仍缺当前Actor在线分布下的回报/优势校准与方差、TD/信用分配适配、匹配预算下Actor真实更新/KL及拒绝行为、完整合法路线的raw与同H质量收益和完整请求时间。未来需要独立授权和冻结协议的在线对照，才能判断训练期条件参考如何影响策略学习。当前不启动这些工作。

holdout已在M1暴露，本轮只能称开发机制筛选，不是新的独立泛化确认；离线MSE不等于路线收益、速度优势或真实机械臂IK认证。M1step128 G3FAIL、256未运行、全部旧资源、R03/旧M3、原60、旧S3与用户Word均保持原边界。

## 验证与资源

独立数值实现只读本轮55944行导出，独立频数矩阵bootstrap复核5880项、32个门槛和256条固定批次更新，PASS；最大指标/区间绝对差约3.64e-12/2.73e-12。未重跑历史千万项CSV审计。真实优化器更新256，Actor、新场景、导航、教师、native、H请求全部0。详见[新数值复核](attempts/02/INDEPENDENT_NUMERICAL_RECHECK.json)、[阶段计时](TIMING_SUMMARY.json)和[累计资源账](RESOURCE_ACCOUNTING.json)。外层监督器确认两次入口各自进程树退出、活动子进程0。

项目交付包含真实模型、step0/128表、新资源账、独立数值复核及结果到主张审查。旧源码/数据文件没有覆盖；监督器只在原run_claims目录新增本次两入口的登记及终止收据。Git与LFS发布核对见GIT_LFS_VERIFICATION.json，未合并main。
