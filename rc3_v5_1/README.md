# v5 冻结与 v5.1 夹爪归属预检维护

**规划主线冻结为 v5。** 二维静态批次任务、代理可达性和共同约束范围不变：Fixed3/Detour 作为默认方法，旧 MLP 可选，Structured 保留研究入口。训练、性能及原 v5 文件不修改；本轮不重新训练或测量性能。真实标定和 IK 按用户安排继续暂缓。

本包是配合原 `rc3_v5_delivery.zip` 使用的独立维护件。将新目录放在原 v5 解压根目录下，保留 `rc3_v5/`。**今后的机器人配置预检使用 `rc3_v5_1.robot_input`；原 `rc3_v5.robot_input` 仅保留作冻结记录。** 规划入口仍为 `rc3_v5.plan`。

## 唯一功能修复

原检查只核对夹爪声明角色，交换左右归属及相应配置后，实际连接另一侧机械臂的夹爪可能误通过。维护版沿 URDF 父链检查：夹爪必须在本侧所有运动臂关节和升降机构之后，不能属于另一侧运动链；合法固定工具连接继续允许。

功能逐字采用上传补丁；唯一附加修改是帮助文字中的模块路径。最终源文件 SHA-256：

`501a5100061d02ec8c4f1967080d15f7de24ff5ba9debbf2f4f4c16bd4dc512b`

上传补丁 SHA 为 `653194a8fb79e8445cbe65ed374f4710d8259430b4d7937e5bbac12d55e3afc6`，冻结原文件 SHA 为 `c30db09cb23f1e66bb759a90c6dfdccd1786d0f3f5bd5b8dc0174ff4768a400a`。来源与额外单行差异保存在 `robot_results/`。

## 本轮实际验证

- **19/19 回归通过**：原机器人输入15项＋新链归属4项，直接导入维护模块，没有替换旧模块。不是新增19个独立物理场景。
- **3/3 CLI 检查通过**：新入口帮助、资产清单、真实空模板。
- 合成错误归属由原版 `inputs_complete` 变为维护版 `blocked_inputs`；合法固定工具连接仍通过。
- 真实空模板的完整输出与原版相同，仍有135个未填/派生字段阻断，未进行真实IK或碰撞认证。
- 上传验收包65个清单文件一致；冻结v5的541个payload哈希一致，当前工作区539个对应源文件保持原样，另外2个是原包内生成的说明/继承清单。v4的270个payload及继承manifest继续保留。

已保存的54项测试、252份方案复验、缓存耗时及MLP区间属于上传验收或既有实验，不写成本轮重跑。完整分母与引用范围见 [ACCEPTANCE_SCOPE.md](ACCEPTANCE_SCOPE.md)。本轮测试原始结果见 `robot_results/SUMMARY.json`；迁移后的维护测试与输入检查见交付包旁的 `.smoke.json` 回执。

## 使用与复现

预检模块只使用 Python 标准库。原规划器依赖保持原 v5 配置。

```powershell
python -B -m unittest rc3_v5_1.test_robot_input -v
python -B -m rc3_v5_1.robot_input --help
python -B -m rc3_v5_1.robot_input check --manifest rc3_v5/robot_assets/manifest.json --config rc3_v5/robot_assets/runtime_config.template.json --output outputs/preflight_v5_1.json
```

空模板的最后一条命令应退出2并报告 `blocked_inputs`，是预期输入阻断。资产已齐备，无需重新下载；将来复制模板并填写真实标定值后再检查。预检通过也仅表示输入结构一致，不证明标定准确、可达、碰撞安全或运动可执行。

```powershell
python -B -m rc3_v5_1.robot_results.reproduce --output outputs/gripper_patch_recheck
python -B -m rc3_v5_1.audit_freeze --output outputs/freeze_audit.json
python -B -m rc3_v5.plan --method fixed3 --scale 10x10 --mode blocked --seed 771903 --output outputs/fixed3.json
```

前两条默认从原工作区根运行；完整哈希核对需要原 `rc3_v5_delivery.zip`，可通过 `audit_freeze --base-archive` 指定位置。补丁复现还使用维护包内的上传验收副本进行原/新字节对照。不要将新报告回写到冻结 v5 的目录。

## 论文展示入口

[DISPLAY_INDEX.md](DISPLAY_INDEX.md) 与 `display_manifest.json` 固定已有场景、计划标签、JSON指针和文件哈希，供后续制图复用，避免按学习收益择优选图。本轮仅整理展示数据入口，没有增加新实验或绘制新论文图。v4继续作为主实验；v5及此维护件分别作为工程与输入检查证据。
