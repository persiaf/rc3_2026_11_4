# RC3 G2.2：冻结策略完整规划反馈

已实际运行12个旧extra开发场景×6方法的72份完整规划，全部通过二维代理验证。两个学习模型保持G2.1原权重，训练更新0、PPO动作0。主要结果和后续判断见 `REPORT_G22.md`，独立结果复算见 `EXPERIMENT_RESULT_TO_CLAIM.md`。

本包是增量证据包。将 `rc3_rl_g22/` 放在原共享根目录，与 `rc3_rl_g2/`、`rc3_rl_g21/` 等并列。需要原G2及G2.1交付的源码、教师和权重：

- `rc3_rl_g2_delivery.zip` SHA256：`2fe860781980ccf7bffbe2fdc0d7945efc5b8941a64c7fe0d0ac3b5bfddc2908`。
- `rc3_rl_g21_delivery.zip` SHA256：`79b0985a1ec459f2a069e6b1c4d4841a6a54d7c609df9f020c25e7414086c50b`。

本轮采用Python3.10.20、torch2.12.1+cu126、numpy2.2.6、scipy1.15.2、threadpoolctl3.5.0，CPU单线程；图表另用matplotlib3.10.9。

## 直接查看

`outputs/evaluation/results.csv` 是全部72行结果。`outputs/plan_index.json` 把场景、方法、指标和具体 `plan.json` 对应起来；每份方案有底盘折线、停靠顺序和左右臂分配。相邻 `evidence.json.gz` 保存逐步动作、实际回报、工作量、原教师匹配范围和模型原始输入。

`outputs/action_feedback.csv` 区分精确教师状态中动作已查询、动作未查询，以及状态未见。原未知Q值保留空值；即时奖励、冻结策略后缀回报与同状态STOP解析记账是不同列。后一项不是额外分叉实验，假定前缀终验不回退；不能把它们混成最优Q或原两步教师标签。

## 零搜索复验

在共享根目录运行：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m unittest rc3_rl_g22.test_feedback -v
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_rl_g22.audit
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_rl_g22.replay
```

第一项是16项纯数据检查；第二项检查保存方案和账本；第三项只对保存的52个模型输入进行零更新前向，不调用环境或提案。后两项只更新G2.2复核回执。原G2的历史71项、G2.1的73项测试不混计为本轮重跑成果。

## 执行范围

`outputs/protocol.json` 在规划前固定了12场景、六方法、两份权重哈希、H=16、共享工作量软预算200万、早停参数和完整5秒软保护限。`outputs/evaluation/attempts.jsonl` 及schedule记录全部72槽位；没有额外试验场景、自动补跑或查询教师树分支。

`evaluate.py` 是已执行入口，会拒绝覆写已有evaluation目录。不要为复验重新执行它；需要完整重现实验时，应在新副本和明确的新执行范围中保存旧证据后运行，并把额外提案重新计数。

G2.1原机会门槛、失败结论及PPO门禁保持原样。本协议只授权冻结评估，没有训练入口。两个模型未被推荐为部署默认；本轮仍保留Fixed3／Detour的定位。

## 证据入口

- `outputs/summary.json`、`report_table.json`：质量、完整时间、实际共享回报与动作反馈摘要。
- `outputs/plan_audit.json`、`inference_replay.json`：72份方案与52次模型输入复验。
- `outputs/freeze_before.json`、`freeze_after.json`：原G2/G2.1及旧版本前后核对。
- `outputs/decision.json`、`final_checks.json`：本轮边界和交付检查。
- `figures/`：PDF/PNG、生成脚本、图注及视觉审查。
- `paper/review-traces/`：独立审查提示、原始回复、重算程序和结论。

主表时间为模型常驻时的完整规划，含新问题准备和最终验证；模型加载及证据序列化单列，单次计时不可推断稳定优势。所有场景仍是旧开发数据，所有验证仍是代理范围。真实标定与IK继续暂缓。

补充库链接未能读取，未声称复用了其未读提案文件。本包完整保存了本轮用户正文和独立登记的实际实施参数。
