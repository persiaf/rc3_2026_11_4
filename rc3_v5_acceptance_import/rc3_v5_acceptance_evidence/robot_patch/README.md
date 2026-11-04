# 夹爪与所属机械臂链一致性补丁

本目录是针对v5输入预检的隔离维护件，原上传源码与运行模板没有改动。没有填写真实参数、做标定、求IK或运行碰撞实验。

## 修复问题

原检查只要求`grippers.left.joint`引用被声明为`left_gripper`的关节。若左右夹爪角色、引用和park字段一同互换，实际URDF上属于右臂的夹爪可以被当作左夹爪，且预检返回`inputs_complete`。

新检查从所选夹爪关节的父link向URDF根遍历，要求：

- 本侧所有运动臂关节均为其祖先。
- 对侧运动臂关节不在该祖先链上。
- 配置指定的升降关节是其祖先。

遍历保留固定关节，所以腕部到夹爪之间的合法固定工具安装连接不会被误拒绝。约束只使用已有URDF树与配置角色，不加入Aloha名称、关节数量、限位或几何数值假设。当前版本本就不支持未明确建模的mimic耦合，本补丁不扩展该范围。

## 已执行验证

**19/19通过＝15项原机器人单测＋4项新增回归，0失败、0异常、0跳过。**

新增回归分别验证左右夹爪互换、声明左夹爪实际挂右臂、夹爪位于本臂运动链上游的拒绝，以及合法固定工具连接的接受。原合成完整配置仍通过；接受结果仍明确`physical_ik_verified=false`和`collision_verified=false`。

真实未填模板的报告与原版逐字段相同，仍为`blocked_inputs`、135个blocker。反例由原版`inputs_complete`变为补丁`blocked_inputs`，记录在`results/counterexample_before_after.json`。

## 文件与使用

- `rc3_v5/robot_input.py`：隔离修订副本。
- `gripper_chain.patch`：相对原v5的统一差异。
- `test_gripper_chain_patch.py`：在进程内加载副本的回归入口，不覆盖原文件。
- `results/SUMMARY.json`、`results/tests.log`：实际执行结果。

```bash
python -B test_gripper_chain_patch.py --source-root /path/to/v5 --output-dir /path/to/new-patch-results
```

需要采用时，将差异审查后应用于单独维护分支或开发副本，记录新的版本与源码哈希。不要修改原v5冻结清单以使旧测试看起来来自新文件。该补丁只修改机器人输入预检，不改变规划、训练或缓存，因此不需要重训模型。

原源码SHA-256：

`c30db09cb23f1e66bb759a90c6dfdccd1786d0f3f5bd5b8dc0174ff4768a400a`

隔离修订SHA-256：

`653194a8fb79e8445cbe65ed374f4710d8259430b4d7937e5bbac12d55e3afc6`

## 范围

这是配置结构一致性修复，不是标定认证或IK闭环。即使输入检查通过，仍必须由下游真实约束验证服务见证和运动可行性。它不影响原v5代理路线、缓存提速或旧MLP收益区间的实验结论。
