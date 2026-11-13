# R07动作反事实完整Rollout机会验证

[协议](PROTOCOL.json) · [固定48任务](TASKS.json) · [实施审查](IMPLEMENTATION_REVIEW.md) · [来源](SOURCE_BINDING.json) · [资源](RESOURCE_ACCOUNTING.json)

原SL256冻结，在同一真实可达前缀强制Top1/Top2，仅后续状态不同，后续策略与H完全相同。不训练，不增加教师/场景/导航，不恢复旧D5/D6或固定32等支线。

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r07_20261113.run
```

48个任务预定互异物理父布局；按进度三分箱选择可达状态，缺失不补例。主比较最多288请求，另16同状态重复核对，硬上限320、1800秒。动作价值指这个固定后续策略及固定H下的成本，不是最优Q。潜在改善不当作已训练策略收益。
