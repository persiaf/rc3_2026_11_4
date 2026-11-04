# RC3 G2.1 离线诊断交付

已实际完成5次有限离线训练，共8704次更新、0新环境查询、0 PPO动作。16状态可学习性检查通过；后续所有最终模型未达到预注册机会门槛。主结论与完整范围见 [REPORT_G21.md](REPORT_G21.md)。本包没有新的可推荐部署模型，Fixed3／Detour仍为默认。

本轮新增测试73/73通过，保存的5个checkpoint零更新重放一致；原包2617个本地对应文件保持一致。交付检查日期2026-11-04；训练协议和记录保留原2026-11-03日期。

## 安装位置

这是**配合冻结原G2使用的增量包**。将 `rc3_rl_g21/` 放在原 `rc3_rl_g2/`、`rc3_fast/` 等目录旁，不覆盖任何旧文件。依赖原 `rc3_rl_g2_delivery.zip`，SHA256：`2fe860781980ccf7bffbe2fdc0d7945efc5b8941a64c7fe0d0ac3b5bfddc2908`。不重复打包教师、原BC和旧规划资产；它们必须来自该原包。

验证环境：Python3.10.20、torch2.12.1+cu126、numpy2.2.6、scipy1.15.2、threadpoolctl3.5.0；画图另用matplotlib3.10.9。本轮训练实际在CPU单线程运行。

## 从共享根目录复验

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m unittest discover -s rc3_rl_g21 -t . -p 'test_*.py' -v
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_rl_g21.audit_results --require-all
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_rl_g21.replay_checkpoints --output-dir rc3_rl_g21/outputs/replay_recheck_01
```

前两项是合成／记录检查；第三项加载权重进行零更新推理，均不查询环境。第三项需使用尚未保存重放结果的新输出目录，再次运行可换为 `replay_recheck_02`。审计产物只写G21，原回执保留。交付时为重放入口增加了新输出目录参数，原始重放回执使用的脚本保存于 `inputs/replay_checkpoints_v1.py`，哈希与原回执一致。原G2历史测试没有在本轮重新运行，不混计。

## 训练入口与记录保留

`python -B -m rc3_rl_g21.train_supervised --stage toy` 运行固定16状态检查；`--stage sampling --variant original|root_half` 运行采样对照；`--stage gain --variant control|gain` 运行有条件的收益对照。入口拒绝覆写已有run目录，需在独立复制的工作区进行复现并保留原结果。协议和选择文件不能重写；重新注册新实验应另建实验目录，而非篡改本轮已冻结记录。

若要从头复现本轮，在**原G2及本包的新副本**中将5个已保存run文件夹移动到另一个保留目录，然后依次运行上述命令；先toy，核对门槛，再采样，重新检查root50门槛并核对保存的gain授权条件，最后两组gain。严禁在原交付副本删除结果。训练不是复验的必要步骤，本轮已提供每步日志与独立核对。

`train_ppo_pilot.py`、`evaluate_g21.py` 为后续预备接口。它们有显式门禁，当前无 `outputs/ppo_authorization.json` 和根 `opportunity_gate.json`，因而不启动交互。请勿手工改布尔值绕过当前失败门槛。早停rule_dev网格、学习一步控制、多进程PPO均未端到端运行；详细边界见 `BASELINES.md` 与 `PPO_PREPARED.md`。

## 文件入口

- `outputs/protocol.json`、`selection.json`：训练前固定的协议和16状态。
- `outputs/{toy16,sampling_original,sampling_root_half,gain_control,gain_gain}/`：权重、训练日志、逐状态诊断、完整回执。
- `outputs/decision.json`：门槛结果与不进入PPO的决定。
- `outputs/data_audit.json`、`results_audit.json`、`checkpoint_replay.json`：数据、结果及权重重放核验。
- `outputs/freeze_before.json`、`freeze_after.json`：原包与本地冻结核对。
- `figures/`：可复现PDF/PNG诊断图；不是路径效果图。
- `EXPERIMENT_RESULT_TO_CLAIM.md`、`paper/review-traces/`：独立结果解释及审查追踪。

交付清单和外部打包回执记录全部新文件SHA。补充审查ZIP的库链接未能读取，本包保留用户粘贴正文及其哈希，不冒称已读外部任务文件。

同级 `rc3_rl_g21_delivery_receipt.json` 记录ZIP身份；`rc3_rl_g21_delivery.smoke.json` 记录实际ZIP解压后的73项测试、记录审计、权重重放以及PPO门禁拒绝结果。可以用 `python -B -m rc3_rl_g21.portable_smoke` 复核，它只在G21内部新建隔离检查目录，不训练或查询环境。
