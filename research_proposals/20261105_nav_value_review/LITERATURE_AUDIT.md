# 文献身份与适配边界核查

日期：2026-11-05。笔记站点仅作发现入口；技术判断回到官方论文、作者机构或作者代码。所有文献核查均为只读，没有下载运行模型或适配外部框架。

| 标记 | 核实身份与原方法 | 可借鉴及不能推导的内容 | 主要来源 |
| --- | --- | --- | --- |
| R1 | Son等，RRNCO，ICLR2026；坐标与真实道路边关系融合 | 可借导航关系表示，不把城市非对称VRP、GPU批处理及增广时延迁移为RC3单请求性能 | [官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9446c291a8744a125a0bda5b18f4d5a1-Abstract-Conference.html) |
| R2 | Drakulić、Michel、Andreoli，GOAL，ICLR2025；初稿2024 | 可借异构mixed-attention；专家模仿不是本项目RL闭环；tail-recursive状态压缩不能直接套到完整前缀H | [官方论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/826aea2253363fe04e8c4991b2a8869e-Abstract-Conference.html) |
| R3 | Yi、Huang、Ma、Cao，RADAR，ICLR2026；SVD入/出代价与非对称距离表示 | 当前对称D无需人为改成非对称；源方法不证明覆盖任务价值学习有效 | [官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d28077e5ff52034cd35b4aa15320caea-Abstract-Conference.html) |
| R4 | Bertsekas、Tsitsiklis、Wu，Rollout Algorithms for Combinatorial Optimization，Journal of Heuristics3:245–262，1997 | 固定一致参考的精确终局评价可支持条件性改进；神经误差保证另需假设 | [作者原文](https://www.mit.edu/~jnt/Papers/J066-97-rollout.pdf) |
| R5 | Pirnay、Grimm，Self-Improvement for Neural Combinatorial Optimization: Sample Without Replacement, but Improvement，TMLR2024 | Gumbeldore使用无放回搜索和最好解监督学习；不能仅复制最好解模仿就称已验证本方案RL闭环 | [原文](https://arxiv.org/abs/2403.15180)、[作者代码](https://github.com/grimmlab/gumbeldore) |
| R6 | Liao、Chen、Wang、Zhang、Wang，BOPO，ICML2025，PMLR267:37456–37475 | 最佳锚定、真实目标差的偏好优化可作适配对照；源实验TSP/JSP/FJSP不是覆盖双臂约束 | [官方论文](https://proceedings.mlr.press/v267/liao25a.html) |
| R7 | Zhou、Lin、Wang、Zhang，Learning to Reduce Search Space for Generalizable Neural Routing Solver，SIGKDD2026；arXiv初稿2025，v3为2026-06-01 | L2R可参考候选缩减，但不能默认保留唯一覆盖动作；首轮只稀疏消息而不裁合法动作 | [作者全文](https://arxiv.org/abs/2503.03137)、[作者代码](https://github.com/CIAM-Group/L2R) |
| R8 | Li等，Deep Reinforcement Learning for Combinatorial Optimization: Covering Salesman Problems；2021v1预印本与2022正式期刊不同版本 | 动态覆盖、REINFORCE及后验局部搜索相关；任务/候选分离、F和绕障D仍需适配 | [预印本v1](https://arxiv.org/html/2102.05875v1)、[共同作者机构记录第96项](https://www.au.tsinghua.edu.cn/info/1077/3265.htm) |
| R9 | Liao等，SuperMimic，arXiv2610.28480，2026-10-20；作者标注CoRL2026 Oral | 可微世界模型及BPTT属于连续控制；训练动作覆盖不能当泛化，异硬件/数据比较不能移植为RC3加速比 | [原文](https://arxiv.org/abs/2610.28480)、[作者项目](https://supermimic.github.io/) |
| R10 | Fang、Weng、Ban，ASAP，ICML2026，PMLR306:29417–29446 | 提议/选择及MAML适配可作后续参考；首版不叠加在线适配或硬裁动作 | [官方论文](https://proceedings.mlr.press/v306/fang26t.html) |
| R11 | Li、Zou、Wu、Wen，LMask，ICLR2026；预印本2025 | 回溯与lazy masking面向TSPTW/TSPDL；当前精确正增益覆盖掩码无同类死端，不据此放宽约束 | [官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ade039c1db0391106a3375bd2feb310a-Abstract-Conference.html)、[作者代码](https://github.com/optsuite/LMask) |

## CSP版本修正

正式期刊：Kaiwen Li、Tao Zhang、Rui Wang、Yuheng Wang、Yi Han、Ling Wang；IEEE Transactions on Cybernetics，2022，52(12):13142–13155，DOI10.1109/TCYB.2021.3103811。DOI中2021不是正式卷期年份。2021年arXiv v1为五作者版本，不将其作者表与2022卷期混合。机构记录为共同作者清华大学主页第96项。

CSP_Attention仓库存在，但README主要继承Kool AM内容和ICLR2019引用。仅凭该README不能认定已核实完整官方CSP实现；计划中保留为待适配代码入口，不称已复现。

## 入口与核查范围

[AI Paper Notes](https://m0rtzz.github.io/paper-notes/)主页的直接抓取失败，已通过可检索的具体论文条目发现入口，再核对上述主源。没有依赖二手解读给出性能或理论保证。网页和论文来源只支持其自身任务与实验条件，本项目的效果均须另行验证。
