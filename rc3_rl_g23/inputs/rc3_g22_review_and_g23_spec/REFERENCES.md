# 方法依据与已核对的原始来源

本次联网读取范围是下面的官方仓库说明、原始论文页面和作者论文。没有重新下载或逐文件审计开源训练代码；此前各版本的源码核对不计为本轮新工作。以下方法只提供设计依据，不证明本项目的改造有效。

1. **Learning to Delegate for Large-scale Vehicle Routing**，Li, Yan, Wu，NeurIPS 2021。
   - 官方仓库：https://github.com/mit-wu-lab/learning-to-delegate
   - 可借鉴：用网络选子问题，再交给确定性子求解器，保留局部修复接口。
   - 边界：仓库说明的学习部分是子问题回归/分类，不能引用它来证明当前 PPO 已有效。

2. **Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt**，Ma, Cao, Chee，NeurIPS 2023。
   - 论文：https://papers.nips.cc/paper_files/paper/2023/hash/9bae70d354793a95fa18751888cea07d-Abstract-Conference.html
   - 官方仓库：https://github.com/yining043/NeuOpt
   - 可借鉴：当前解与历史最好解分开，以及学习多步改进操作的研究结构。
   - 边界：其 TSP/CVRP、k-opt 与不可行区域探索不能直接搬来当作本项目的双臂约束处理或跨规模有效性结果。

3. **Approximate Policy Iteration: A Survey and Some New Methods**，Dimitri P. Bertsekas，Journal of Control Theory and Applications 9(3), 310–335，2011。
   - 作者稿：https://web.mit.edu/dimitrib/www/Bertsekas_ApproximatePolicyIteration_2011.pdf
   - MIT 页面：https://dspace.mit.edu/entities/publication/ee592310-f10c-4f5c-8e8f-6743938265de
   - DOI：https://doi.org/10.1007/s11768-011-1005-3
   - 可借鉴：分开固定策略的价值评价与策略改进。本文也讨论近似评价、探索、噪声和策略振荡，不能把精确策略迭代的改进性质无条件移植给有限数据神经网络。

## 项目原始证据

G2.2 原始包 SHA256：`e99f9f8d1a8d46f1d7bc728391237f1000e652f1bcb1f11f79b57415eeb627db`。

本次项目结论来自 `REPORT_G22.md`、72 条结果、48 条共享动作轨迹及其计划/记账，不来自上述论文的性能数字。下一轮图编码、停止优势、有限预算和实验门槛是针对本项目提出的设计，尚未实验验证。
