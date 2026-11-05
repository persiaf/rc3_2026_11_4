# 覆盖与导航价值规划的独立研究

本目录按新任务开展R0有界教师检验。它复用冻结的覆盖、导航、完整H与验证器，没有创建或训练价值网络；旧同结构策略梯度支线保持收尾。

**R0已完成并通过登记的机会筛选。** 32例对g_B＋H平均配对改善2.2507%，探索性95%区间[1.7990%,2.7341%]；32例均有正收益，最大单例正相对贡献8.9448%，全部方案有效。实际2144次顶层评价≤2304，包括2112次完整H及32次原生Fixed3；2048次候选标签使用包含32次精确缓存命中。

这只支持“完整H后仍有有界教师可利用的动作机会”。额外补全搜索不是神经网络或RL收益；没有证明可学性、D的必要性或部署优势。R1/R2/R3和原60保留测试均未启动/未查询，默认Fixed3／Detour不变。

- [REPORT_R0.md](REPORT_R0.md)：主结果、门槛、资源和解释边界。
- `r0/protocol.json`、`wall_protocol.json`：先冻结源版本、种子、预算与壁钟，再执行。
- `r0/seed_audit*.json`、`seed_audit_inventory.jsonl`、`seed_audit_tool.py`：历史记录查重和严格数字边界。
- `r0/calls.jsonl`、`evaluations.jsonl`、`checkpoints.jsonl`：实际调用账、所有完整H结果、每前缀候选集合与选择。
- `r0/case_results.json`、`per_scene.csv`、`analysis.json`、`resources.json`：32例总表、门槛与资源分解。
- `r0/scenes/`、`baselines/`、`plans/`、`verification.json`：32场景、固定B、96方案和保存重放。
- `r0/registration_attempt1/`、`registration_amendment.json`：首次只读摘要检查解释错误的原记录及校正；校正前研究调用0，不覆盖或删除原记录。
- `paper/review-traces/experiment-result-to-claim/2026-11-05_run01/`：独立结果审查，无额外研究调用。

从仓库根目录可仅复算保存结果：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_nav_value.r0_analyze
```

`r0_run register/calibrate/run/verify`是本轮一次性入口，拒绝隐式重跑；不要删除输出后重复执行。独立验证只重放保存动作、队列和成本，不再补全新路线或调用H。新增阶段需要独立任务和协议，不自动串行开训。
