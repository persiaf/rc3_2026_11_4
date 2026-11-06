# 全局—局部 DRL 方案文献核实

核实日期：2026-11-06。范围是用户列出的六篇研究与 RL4CO 的存在性、版本及可支持的设计含义；不是整篇论文的逐项引用审计，也不是本项目实验结果。全部研究目标保持为“大规模复杂父场景训练深度强化学习策略，冻结后完成不同规模局部任务的完整可验证规划”。本核实没有场景、导航、规划、H、推理、反向传播、优化更新或保留测试调用，没有安装库。

## 核实结果与规范元数据

| 研究 | 可核对的规范记录 | 版本注意 |
|---|---|---|
| CSP Attention | Kaiwen Li, Tao Zhang, Rui Wang, Yuheng Wang, Yi Han, Ling Wang. *Deep Reinforcement Learning for Combinatorial Optimization: Covering Salesman Problems*. IEEE Transactions on Cybernetics, 52(12):13142–13155, 2022. DOI `10.1109/TCYB.2021.3103811`。[期刊记录](https://pubmed.ncbi.nlm.nih.gov/34437087/)、[作者全文](https://arxiv.org/html/2102.05875v1) | 正式卷期为2022；arXiv v1为2021，作者展示与正式六作者记录有差异，不能混并。 |
| POMO | Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, Seungjai Min. *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning*. NeurIPS 2020。[正式摘要](https://proceedings.neurips.cc/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html)、[论文](https://proceedings.neurips.cc/paper/2020/file/f231f2107df69eab0a3862d50018a9b2-Paper.pdf) | 附件简写省略的“for Reinforcement Learning”属于完整题名。 |
| UAV global/local | Mirco Theile, Harald Bayerlein, Richard Nai, David Gesbert, Marco Caccamo. *UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning*. ICAR 2021, pp.539–546. DOI `10.1109/ICAR53236.2021.9659413`。[作者版本](https://arxiv.org/abs/2010.06917v4)、[机构全文](https://www.eurecom.edu/publication/6377/download/comsys-publi-6377.pdf) | arXiv首发2020；v4修订2021并注明ICAR 2021。 |
| GLOP | Haoran Ye, Jiarui Wang, Helan Liang, Zhiguang Cao, Yong Li, Fanzhang Li. *GLOP: Learning Global Partition and Local Construction for Solving Large-Scale Routing Problems in Real-Time*. AAAI 2024, 38(18):20284–20292. DOI `10.1609/aaai.v38i18.30009`。[会议正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/30009)、[作者版本](https://arxiv.org/abs/2312.08224v2) | arXiv首发2023，正式发表年2024。 |
| UDC | Zhi Zheng, Changliang Zhou, Xialiang Tong, Mingxuan Yuan, Zhenkun Wang. *UDC: A Unified Neural Divide-and-Conquer Framework for Large-Scale Combinatorial Optimization Problems*. NeurIPS 2024. DOI `10.52202/079017-0197`。[会议正式记录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b8e4c8468273ee3bafb288229c0acbc-Abstract-Conference.html)、[作者版本](https://arxiv.org/abs/2407.00312) | arXiv v4修订于2025；第三作者在arXiv显示为Tong Xialiang，本表采用会议记录。 |
| 渐进式训练 | Zhi Cao, Yaoxin Wu, Yaqing Hou, Hongwei Ge. *Enhancing neural combinatorial optimization by progressive training paradigm*. Neurocomputing, 659:131707, **2026-01-01**. DOI `10.1016/j.neucom.2025.131707`。[出版商记录](https://www.sciencedirect.com/science/article/pii/S0925231225023793)、[作者贡献与章节片段](https://www.sciencedirect.com/science/article/abs/pii/S0925231225023793) | 论文确实存在，卷期为2026；DOI含2025不能据此改写卷年。正文直接访问403，核实基于出版商返回的元数据、摘要、介绍及CRediT片段，不称已通读全文。 |

逐篇来源、完整作者、原文定位和访问限制保存在 `literature_sources.json`。其中技术含义均来自出版商、会议、作者 arXiv 或机构全文；不以转载解读作为方法证据。

## 能借鉴的部分与不能推出的结论

**CSP Attention。** 原文采用注意力和动态覆盖嵌入，用完整覆盖回合长度训练REINFORCE，配贪心rollout基线。可参考覆盖状态与终局回报；其城市距离覆盖没有本项目的逐臂F/A、底盘碰撞或IK约束，文中速度和最优差不能转记为本项目性能。[作者正文§IV–V](https://arxiv.org/html/2102.05875v1)

**POMO。** 原文在TSP、CVRP和0-1背包验证多起点与共享基线。§4.2的轨迹概率从第二步开始，共享同实例回报均值；可以作为未来S2的探索对照。覆盖任务不同首动作未必具有循环移位等价性，也不保证产生不同最优解或本项目收益。强制首动作的概率若不是策略输出，不能当作采样log-prob求梯度；全部rollout和推理增广都应计账。[论文§4.1–4.3](https://proceedings.neurips.cc/paper/2020/file/f231f2107df69eab0a3862d50018a9b2-Paper.pdf)

**UAV global/local。** DDQN在覆盖与数据采集任务中结合压缩全局地图和细粒度局部观察。它支持多分辨率输入的设计动机，局部观察并非裁剪任务后可行性不变的定理；UAV移动与着陆动作也不等于双臂服务停靠动作。[作者摘要与版本](https://arxiv.org/abs/2010.06917v4)

**GLOP。** 原文将非自回归全局划分与自回归局部构造结合，包含TSP和Shortest Hamiltonian Path层级，测试TSP、ATSP、CVRP与PCTSP。可参考局部构造及连接端点接口；固定矩形窗口环境不能直接称作已经实现GLOP的学习分区或全局收益。[会议正式摘要](https://ojs.aaai.org/index.php/AAAI/article/view/30009)

**UDC。** 原文以全局GNN划分、定长子路径求解和Divide-Conquer-Reunion训练处理划分与求解耦合。可参考全局与局部目标协调；不能由其组合优化结果推出本项目任意窗口泛化、覆盖或碰撞约束已成立。[NeurIPS正式摘要](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b8e4c8468273ee3bafb288229c0acbc-Abstract-Conference.html)

**渐进式训练。** 出版商摘要支持“小规模SL初始化→更大规模RL微调”，并用分解减轻大规模训练开销；验证对象为LEHD上的TSP/CVRP。摘要的求解规模至10,000节点不应写成全部训练均在10,000节点。它支持监督辅助后继续真正RL的顺序，不证明移动双臂覆盖任务的有效性或具体超参数。[出版商摘要与介绍](https://www.sciencedirect.com/science/article/pii/S0925231225023793)

## RL4CO 的用途

Federico Berto等，*RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark*，作者arXiv v6注明KDD 2025 Oral，相关DOI为`10.1145/3711896.3737433`；预印本首发2023。可参考环境、策略、训练算法和评估配置的模块分离；其库不是移动双臂几何或IK验证器。本轮仅参考组织方式，不安装或替换已验证内核。[论文版本](https://arxiv.org/abs/2306.17100v6)、[官方仓库](https://github.com/ai4co/rl4co)

## 对本轮协议的约束

以下属于本项目应验证的建模要求，不能写成上述文献已替本项目证明的性质：

- 任务窗口应保留全部有效服务候选及必要边界候选；全局障碍、外部允许关系与逐臂有效覆盖要保持一致。候选减少来自合法任务选择，不来自教师终局收益筛选。
- 明确“在大父场景中选取局部任务，导航可经全局通道”和“物理裁剪地图”两种语义。入口/出口、越界导航许可和连接成本必须一致。
- S2的父场景、全任务/局部窗口采样方式需登记；S3训练父地图上的新窗口与未见父地图上的窗口分开报告。前者不是未见地图泛化证据。
- 若取未折扣回报，基本步奖励与终局修正的代数可设计为：基本累计回报`-J_raw`，再加`J_raw-J_H`，总回报即`-J_H`。该等式需在S0逐轨迹验收，H也须保持当前入口/出口语义；它不自动带来优化增量。
- S0的前向、反向和参数不变检查仅验证环境/梯度接口。核心DRL训练、冻结模型性能与相对监督起点的独立增量仍需正式S2/S3证据；原R1/T1负结果不改变研究内容三定位。
