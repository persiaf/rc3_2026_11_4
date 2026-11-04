# 夹爪归属补丁接入复核

本轮唯一功能变更来自上传的 `robot_patch/gripper_chain.patch`：在声明角色之外，沿 URDF 父链核对夹爪所属侧的全部运动关节、排除对侧运动关节、包含升降关节。固定工具关节保留在祖先遍历中，因此不会误拒绝合法固定连接。

代码审查确认：URDF 清单解析已先验证唯一根、无环、单一父关节和完整连通性；进入新分支时夹爪 joint 必须在匹配角色列表，因而属于已解析的非固定关节。缺角色、错误角色和其他配置错误仍保留在 blockers 中，不会因祖先集合计算而被清空。没有发现必须扩大功能范围的问题。

上传完整修订 SHA-256：`653194a8fb79e8445cbe65ed374f4710d8259430b4d7937e5bbac12d55e3afc6`。

最终维护文件 SHA-256：`501a5100061d02ec8c4f1967080d15f7de24ff5ba9debbf2f4f4c16bd4dc512b`。

两者唯一额外差异是顶部帮助说明的模块名从 `rc3_v5.robot_input` 改为 `rc3_v5_1.robot_input`，避免维护入口把用户引回冻结旧版。功能代码与上传补丁逐字一致，差异保存在 `upload_to_maintenance.patch`，复现器通过字节比较核验。

实际运行原 15 项及新增 4 项测试，**19/19 通过**；3 项 CLI 检查通过（正确维护入口帮助、真实资产清单、未填真实模板的预期退出码 2）。测试直接导入 `rc3_v5_1`，没有替换 `sys.modules` 或 monkeypatch 原 `rc3_v5`。

左右夹爪角色/映射/park 同时交换的合成反例：原版 `inputs_complete`，维护版 `blocked_inputs`，同时给出 `grippers.left.chain` 和 `grippers.right.chain`。正确合成输入仍通过。真实空模板与原版结果完全相同：`blocked_inputs`、135 个字段及派生阻断。输入缺失不是物理不可行；IK、碰撞核验标记仍为 false。

复现命令（交付根目录）：

```powershell
python -B -m rc3_v5_1.robot_results.reproduce --output outputs/gripper_patch_recheck
python -B -m unittest rc3_v5_1.test_robot_input -v
```

完整复现器读取冻结 `rc3_v5` 的原测试、资产，以及 `rc3_v5_acceptance_import/rc3_v5_acceptance_evidence/robot_patch/rc3_v5/robot_input.py` 的上传补丁；这些是来源比较与反例前后对照依赖。实际预检 CLI 只需维护文件、指定资产清单和配置文件。

本次没有训练、模型推理、规划或性能重跑，也没有真实标定、IK 或碰撞计算。功能变更不修改规划/缓存/MLP 模块及其原记录。
