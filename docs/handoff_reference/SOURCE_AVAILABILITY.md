# 来源可用性与缺项清单

| 材料 | 已附 | 状态与获取方式 |
|---|---|---|
| 用户固定需求与实验总结 | 是 | 主需求书与历史表，来源区分明确 |
| 综合残差PPO方案原文 | 是 | docs/proposals/Integrated_RPPO_v1_UNEXECUTED.md；未执行提案 |
| 原MATLAB MobileManipulationPlanning README | 文本摘录 | evidence/MATLAB_PROJECT_README_EXTRACT.md，非整个原工程 |
| 原MATLAB .m、配置和实验数据 | 否 | 未取得真实文件或可靠固定仓库；由项目负责人补交 |
| 原论文uavSim源码 | 两份原样代码＋许可 | 完整源码用fetch_locked_sources.py paper；BSD-3-Clause保留 |
| 当前RC3最新源码 | 未整体复制 | 私有仓库固定7d2cfe75919b25bac59c5274321baca4d8e98f77；需要读权限 |
| 旧RC3 v5源码子集 | 是 | 来自用户上传rc3_v5_delivery.zip；不是最新GL-v5 |
| 原实验checkpoint／SL256精确权重 | 否 | 私有仓库LFS；按WEIGHT_LINEAGE和actor索引选取 |
| 当前科研场景／原60封存测试 | 否 | 本包仅新工程场景；不得把封存数据用于开发 |
| 新接口代码与26项测试 | 是 | 本次编写与执行，不声称历史原源码 |
| MATLAB新导出模板 | 是 | 未MATLAB执行；调用者映射旧字段与索引 |
| 实机IK／关节轨迹／标定 | 否 | 当前代理结论不能替代真实认证 |

**两个同名README的区别：** 项目Files提供的README是MobileManipulationPlanning；运行容器中/mnt/data/README.md为RC3 v4说明。此包只从已核对的Files文本建立MATLAB说明摘录，未将后者冒充前者。

## 教师接手前请项目负责人补齐

MATLAB工程主入口及其调用依赖；config三文件；一个场景与候选/F/A/D可导出结果；MATLAB版本与toolbox清单。当前RC3读权限；三个SL256权重、source sidecar、模型索引；冻结H版本。若计划实机，补机器人模型、安装变换、升降限位、底盘footprint、可达表生成条件及真实验证记录。

## 访问与保密

当前RC3仓库为private；交接包是定向研究材料，不改变原代码授权。不携带token、SSH私钥、个人虚拟环境或其他无关文件。外部原论文代码独立保留BSD许可。新的接口代码是生成的研究辅助代码，无真实机器人安全保证。
