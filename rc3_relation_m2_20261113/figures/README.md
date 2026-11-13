# 实际生成的图表

本轮结论为TECHNICAL_INCOMPLETE。仅生成实际SL初始化和资源记录，不生成缺失的DRL性能比较。

**F1：[训练诊断PDF](fig1_training.pdf) / [PNG](fig1_training.png)。** M2a在独立随机起点上按固定教师日程完成更新1—238时的训练损失与裁剪前梯度范数。损失先对每任务动作负对数似然取均值，再对8任务取均值；梯度记录在阈值1裁剪之前。每点为对应真实更新记录，无平滑；批次含不同任务，曲线下降只表明训练拟合过程，不代表完整规划收益。第239次更新未提交，SL512未完成，M2b未运行。数据：[TRAINING_UPDATES.csv](../analysis/TRAINING_UPDATES.csv)，脚本：[gen_fig1_training.py](gen_fig1_training.py)。

**F4：[资源曲线PDF](fig4_resources.pdf) / [PNG](fig4_resources.png)。** 上、中、下图分别为请求attempt/complete事件、optimizer尝试/完成、请求/optimizer失败事件。请求事件并非互斥：同一`init_0239_teacher_5`先记录complete再记录failed，因此原计数1910 complete与1 failed不能相加为1911请求，也不能算成功率；唯一请求共1910，排除failed重叠后1909。竖虚线标记worker账本中失败事件时刻。横轴仅为worker-ledger计时域；外层累计墙钟2221.312秒是另一个计时原点，单独在资源报告列示，不拼接到此曲线。数据：[RESOURCE_CURVE.csv](../analysis/RESOURCE_CURVE.csv)，脚本：[gen_fig4_resources.py](gen_fig4_resources.py)。

**F2：NOT_GENERATED。** 没有任何完整0/512/2048配对评价端点，不能画零增益占位点。脚本[gen_fig2_gains.py](gen_fig2_gains.py)会明确跳过。

**F3：NOT_GENERATED。** 六方法的LOCAL66/FULL12均未评价。仅提供[覆盖缺口表](../analysis/METHOD_TABLE.csv)；J/L/K/raw/H/时间为缺失，不填0。原始任务评价CSV只有表头，无虚构观测。

PDF为矢量图，PNG为300dpi预览。统一字体/配色/线型，无图内标题，图例未遮挡数据。图表全部来自保存结果，未重新运行模型、H或教师。
