# 一次历史说明与本候选定位

旧S2、GL-v4、GL-v5均已经实施SL初始化后的全Actor训练。“主Actor可训练”本身不是未尝试的新机制，上一版老师材料的相应表述不准确，现予补充纠正。

- S2：[已完成三seed各RL512/继续SL512](../rc3_s2_results_20261107/README.md)，[engine.make_optimizer](../rc3_global_local_trainer/engine.py)使用全model.parameters，旧Trainer明确拒绝部分冻结模型；从S1 SL端点分叉后训练全Actor。这里只核对报告/训练器，不读旧S3评价任务。
- GL-v4：[实际报告](../rc3_gl_v4_20261109/EXPERIMENT_REPORT.md)、[原自然起点对照](../rc3_gl_v4_20261109/START_OBJECTIVE_ABLATION.md)及[run.py](../rc3_gl_v4_20261109/run.py)明确model.train().requires_grad_(True)，A/B/C从共同SL256 fresh Adam各128更新，全Actor配方沿用原engine。
- GL-v5：[实际报告](../rc3_gl_v5_20261109/EXPERIMENT_REPORT.md)、[run.py](../rc3_gl_v5_20261109/run.py)同样使全Actor可训练，G128从同一SL256 fresh Adam，改变冻结贪心参考，而非第一次解冻Actor。

本轮独立候选只改变明确参数范围：冻结global/task/candidate表示与prefix GRU，仅更新原key.weight、dynamic.weight、nav.weight、query.weight、query.bias；保持原评分数学、MDP/H/完整终端回报。需要检验的是该受限范围能否获得完整任务增量，不声称五张量已经有效或优于全Actor。

原53行任务表和固定32封存记录保留原字节；32状态权重禁止用作任何正式初始化。旧Actor、Trainer、协议、结果与停止记录不修改、不重跑，不重做历史全量审计。
