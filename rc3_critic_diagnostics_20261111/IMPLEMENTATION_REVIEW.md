# 本轮离线实施审查

对象为独立的Actor排序响应与Critic匹配校准，不是恢复R03或开始联合PPO。用户2026-11-11任务已明确授权最多128保存轨迹、C0/C1/C2各256次Critic更新、累计科学墙钟3600秒；Actor优化及新任务/导航/教师/native/H请求均为0。协议批准与执行授权将在精确登记SHA固定后分别保存，旧活动指针和旧停止记录不改。

初始工作区位于准确基准83b0df9，用户已有Word修改及新论文文件/ZIP不纳入。origin fetch成功；远端旧研究分支为83b0df9，main为5d5fe25754ffa032f97937a3fb54f15901c2fb7d；main是本分支祖先，差异0/67。原正式制品LFS fsck PASS。旧v2设计、v3状态及R03最新决定已核阅，本轮不读取旧S3任务或原60。

审查范围：

- 准确SL256及上游S1权重谱系沿用R03哈希；600份旧执行源码、17项冻结锚点和新增代码均在登记中绑定。H身份与奖励/终点/完整标签保持原值，不生成新H标签。
- 仅通过原SavedInputs恢复已保存训练输入；物理身份使用geometry physical_key，不能用包含任务的parent_identity划分。完整父布局先固定fit/holdout，不允许状态随机拆分。
- 两阶段×四尺寸各8个ON/OFF请求对。样本、批次、输出初始化、状态归约、损失系数及终点在第一次新增前向前冻结；不按成本选样，不做中途验证调参。
- Actor只读准确更新前后correction，±0.25原界不改。原完整logits/mask/features未保存，明确记录重建来源与原selected logit/probability核对。ratio的pre与post独立列。
- Critic为独立新实例/Adam；C0保留原损失公式，三组统一trajectory state mean再batch mean，区别于原PPO state sum。相同净张量的归一输出意味着初始物理预测不同，必须列出初始化结果。
- 条件常数仅由fit父估计，条件使用窗口和v[256]可观测剩余任务比例，不使用最终路径长度。分别保留pooled及父等权MSE/偏差分解，不替换R03原停止指标。
- 不请求原生基线或重跑H；原Fixed3/Detour结果保持历史证据。本轮只讨论数值校准及排序机制，不讨论新增路线收益。
- 一次性日志在每次来源读取、前向、反向和optimizer前扣账。失败保留并拒绝重启；独立工具只读取新导出数值，首算前共同绑定。

最终静态独立审查结论和工具身份见[STATIC_REVIEW.md](STATIC_REVIEW.md)。仅当其结论GO且精确注册完成后开始本轮离线计算；任何GO不授予PPO或封存测试权限。
