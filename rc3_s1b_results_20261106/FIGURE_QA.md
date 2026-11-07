# 训练图表验收

PASS。执行者及独立审查者均查看两张PNG；PDF保留矢量字体，无栅格截图替代数据图。独立重算全部768原始曲线点/尾随均值与6端点汇总，图与日志/CSV一致。

图1展示实际batch CE（浅色原始、深色最多16点尾随均值）及裁剪前梯度对数轴/阈值1；图2使用固定全部训练教师，左为路线宏CE，右为状态微Top1，0–100%纵轴。三种子颜色和线型区分，图例与标签可读，未选最佳seed或checkpoint。

说明保留训练集/teacher forcing/固定端点/未验证路线和DRL收益的边界。复现脚本、CSV、PNG/PDF和LaTeX include均在figures/。审查原文见INDEPENDENT_REVIEW.md，独立审查不是正式论文数字/引用审计。
