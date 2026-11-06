# Fixed3／Detour16开放端点适配核查

日期：2026-11-06。范围是正式执行前的工程准备；没有正式优化器训练、S3任务查询或原60场景读取。

**本模块判断：GO_FOR_BOUNDED_S0_CONFORMANCE。最新22项合成回归全部通过；已有S0保存数据上的真实工程规划检查由主代理统一预算后执行。本报告本身不宣称该后续检查已完成，也不代表正式S1/S2训练已GO。**

## 适配范围

`endpoint_baselines.py`导出：

- `endpoint_fixed3(local_problem, counters=...) -> (plan, info)`
- `endpoint_detour16(local_problem, counters=...) -> (plan, info)`
- `run_endpoint(local_problem, method, ...)`与`make_shim(local_problem)`

复用冻结的`rc3_v4.runtime.run`完整算法流程。Fixed3仍比较配方`(0,1,7)`，Detour仍从同一Fixed3锚点出发、使用原detour_excess决策和最多16次原固定端点DP修复。没有另写启发式伪装成Fixed3，也没有额外调用`rc3_global_local.terminal.finish`或共同H。

shim把原目标任务和可用服务候选重排为旧算法要求的局部连续ID，保留global_id、原全局桌子/障碍与x/y通道范围；F严格等于父图同臂`F&A`切片。临时替换的仅是原runtime导出／认证回调：返回原全局ID计划，调用新S0端点导出和独立端点验证。补丁在串行锁内执行，正常或异常退出都恢复原函数。调用者应保持同一进程内这些全局规划内核无其他并行补丁/规划，并避免并发共享同一计账dict；当前适配器不提供整个应用的跨模块并发调度。

默认`reject_fallback=True`。原runtime任何构造、proposal或最终认证failure/fallback都会保留在异常details中并拒绝作为正常基线结果；不能把锚点fallback静默算成正常Detour。显式软时间限制仍按原runtime语义运行，并在info中保留deadline/overrun；标准质量比较应不给各方法隐式不同的time_limit。

## 为什么端点矩阵可以复用这些内核

局部符号0的出边是`entry→candidate`，入边是`candidate→exit`，D00是`entry→exit`。矩阵整体可不对称，D00在开放空任务时可非零；适配器不会将其强制对称或清零。

服务点之间仍是原父图对称导航距离。2-opt只反转内部服务段，内部反向边长抵消，四条边界差值公式仍正确；joint_search的替换／删除计算明确读取前后有向边。repair_window的DP只在服务候选间转移，入口/出口分别作为固定边界，不会把符号0当作可穿越的免费中间节点。因此无需假定整个增广D是普通单depot对称度量。

适配前检查服务子矩阵对称、F/C/同臂F&A一致，以及每条局部D与父图对应entry/candidate/exit切片完全一致。任务全覆盖、K、全局ID、导航折线、停靠位姿、任务逐臂分配和完整J交付认证均由端点验证器核对。空任务直接输出entry→exit，K=0；入口/出口可与服务位姿重合形成零腿，内部服务ID仍不得重复。

闭环一致性仅声称在**相同局部任务／候选／有效F／导航域**上，适配器与原闭环runtime的路线和J相符，不声称与历史研究场景或报告文件逐字相同。逐臂服务统一使用当前S0有效许可和导出规则；原native的arm距离tie-break不是新的路径质量贡献。

## 首轮合成回归结果与独立预期（历史）

16/16通过，0未预期失败，0错误。用手写坐标、完整图前驱与逐臂许可构造fixture，没有调用场景生成器或导航构建器。开放小图用独立正覆盖枚举核对最优J=12.5；闭环同域用原导出和旧标量验证对照；开放空任务J=11、K=0，闭环空任务J=0。覆盖：

- Fixed3原三配方及closed-native路线/J一致性。
- Detour16开放/闭合修复上限、相对锚点非增。
- 非对称端点矩阵、全局ID/有效F&A映射、完整父几何范围。
- 空任务、端点重合零腿、父输入及原函数对象不变。
- 无额外共同H，非法参数与错误D在搜索前拒绝。
- 注入认证失败和mock fallback会显式失败；回调仍恢复。
- 显式0秒软deadline保留可验证锚点和deadline标记，不伪称完成16次修复。

单测墙钟0.0416156秒；包含Python/依赖导入的工具进程墙钟6.873453秒。后者不是推理延迟或部署速度证据。

## 首轮单独资源账（历史）

下列全部为合成契约验收，不与之后真实S0规划探针混账：

- 适配入口17次，正常返回13次，预期拒绝4次：2次输入拒绝、1次注入认证失败、1次mock runtime fallback。
- 实际冻结native算法执行14次；另2次原闭环native对照。合计16次合成算法执行。
- 源内核合计construct48、recipe48、joint_search48、two_opt91。
- 18次导出尝试全部完成；18次标量认证尝试，15次通过，3次是同一个注入失败测试中三个参考配方的预期失败。
- 完整共同H wrapper调用0；实际场景生成0、导航构建0、S3/原60读取0、神经前向/反向0、优化器0。

适配器自己的algorithm计时不含调用者先前的父图导航、提取或特征准备；正式S3完整请求计时必须纳入这些开销，不能把已知parent/candidate输入当作免费预处理。内部搜索/标量验证与native fallback需逐请求保留。

## 正式执行前仍需完成

1. 主代理按统一预算在保存的S0 closed/open cases执行原任务域一致性检查，核对fallback为0、J/端点/覆盖/独立认证与搜索计数。
2. 若S1教师是“Detour16 endpoint adapter，再共同H”，这是两个明确环节；需要分别记账，不能说本adapter已包含神经方法的共同H。
3. 将本适配器、冻结依赖、所选协议和验证收据哈希绑定到正式执行器；当前没有授权启动该执行器。
4. 端点表/代理认证仍不提供真实IK、机械臂扫掠碰撞和关节轨迹证据，研究DRL核心和物理目标保持不变。

## 机器可读收据

下面JSON由首次16项真实合成单测输出记录；负例为有意注入，不是研究数据或真实S0规划失败。源码哈希见同一收据。

```json
{
  "schema": "rc3-endpoint-adapter-synthetic-unit-v1",
  "client_date": "2026-11-06",
  "author": "/root/endpoint_baseline_adapter",
  "tests": 16,
  "failures": 0,
  "errors": 0,
  "suite_elapsed_s": 0.04161560000011377,
  "exec_process_wall_s": 6.873453,
  "synthetic_accounting": {
    "baseline_attempts": 17,
    "baseline_completed": 13,
    "baseline_failed_expected_negative": 4,
    "mocked_runtime_fallback_cases": 1,
    "actual_adapter_native_runs": 14,
    "legacy_closed_runs": 2,
    "adapter_native_construct": 42,
    "adapter_native_two_opt": 81,
    "adapter_native_joint_search": 42,
    "adapter_native_recipe": 42,
    "legacy_native_construct": 6,
    "legacy_native_two_opt": 10,
    "legacy_native_joint_search": 6,
    "legacy_native_recipe": 6,
    "endpoint_export_attempts": 16,
    "endpoint_export_completed": 16,
    "endpoint_validation_attempts": 16,
    "endpoint_validation_completed": 13,
    "legacy_export_attempts": 2,
    "legacy_export_completed": 2,
    "legacy_validation_attempts": 2,
    "legacy_validation_completed": 2,
    "adapter_wall_s": 0.02543320000040694
  },
  "real_research_calls": {
    "scene_generation": 0,
    "navigation_preparations": 0,
    "S0_saved_case_planning": 0,
    "S3_queries": 0,
    "retained60_reads": 0,
    "model_forwards": 0,
    "backwards": 0,
    "optimizer_updates": 0,
    "common_H_wrapper_calls": 0
  },
  "source_sha256": {
    "endpoint_baselines.py": "94160eb1f99edf5897325d2113fac252164f20a19c4a3b46806e29c5893999c4",
    "test_endpoint_baselines.py": "4ef943af586cba53829beeb31ba2a54ee07f69b4b0fc66f2b1141fbdf7744864"
  },
  "frozen_dependency_sha256": {
    "rc3_v4/runtime.py": "b5c09bf9f1f81f616a30f493f6efb0d6bf919ba40dd45faf2945608310d192ff",
    "rc3_v3/environment.py": "04f2ca6e89837185b731f028b1b36af9bab51b2379f4f01a8450d0e8b0604df0",
    "rc3_improve/environment.py": "ae02801f5339800d43a014add4aa978bfaf3877440df35a211405c6bbac6e04e",
    "rc3_improve/repair.py": "5b3471d4ce7df23cf7381fcf2f1d86ab39eddce2bca7cf23350a5a8c74c1ee36",
    "rc3_fast/solver.py": "7cd3bacaa059f810e0dded038e9118929c035220b1567e2a7a63dea9808c3693",
    "rc3_global_local/problem.py": "64403a306d1158f60c31c5aec51f13e690a671901df619557d6d9c57b30de320",
    "rc3_global_local/terminal.py": "6f4c334176f2d4fcf86e9b02eaaf4ce98c5fa5d8b8bee335e04ed6cea62785af",
    "rc3_global_local/validation.py": "b15b6d0f9e57e64d8e958d08e088299df785342b0b7562fc6336beb48a666777"
  },
  "scope": "Synthetic contract tests only; normal real S0 conformance awaits centrally budgeted parent runner; no automatic training or push."
}
```


## 最终扩展：父全候选闭环对照与只读快照互操作

当前提供 `closed_native_reference(view, method='fixed3'|'detour16', counters=...)`，仅适用于 `entry=exit=0` 的proxy后端。它保留父图全部候选与全部原nav，只把目标task子集重编号，使用父同臂有效F的整列域。执行原runtime、原export_plan和原proxy validator，没有端点回调替换；原函数call/return用临时profile观察计数，结束恢复。对外部F/A后端拒绝以旧proxy validator假作真实认证，对开放端点拒绝当作原闭环方法比较。

该对照比首轮“同压缩域、旧导出”的测试更强：证明没有新增覆盖的父候选被压缩后，相同目标任务上的Fixed3/Detour完整pipeline仍给出相同全局访问序列与J。真实S0的全候选闭环对照仍由主代理54次统一预算runner执行，本代理没有自行运行实际保存S0规划。

`runtime_guards`的只读mapping/list刻意让deepcopy返回self。适配器现用递归plain-copy新建旧算法局部scene，父快照仍不可变；不更改保护层，不解锁F/A/D。两个新增synthetic互操作case分别覆盖开放Detour和closed full-native，并证明父scene/F/D哈希不变。

最新22/22通过、0未预期失败/错误。保留三个必要回归run历史：16项初实现、20项增加父全候选reference、22项修复只读容器互操作。所有run均纯合成，无真实场景、导航、S0保存case规划、共同H wrapper、神经前后向或优化器。负例中的认证失败、参数拒绝和mock fallback是预设检查，不是实际研究数据失败。

三个run累计：61次实际合成native算法执行；construct/recipe/joint_search各183、two_opt375；导出67次尝试和返回；认证67次尝试、58次成功、9次预设无效返回（每run同一负例试三个配方）。单测suite累计0.1403607秒，三个工具进程（含导入）累计12.0682647秒。上述耗时只属工程验证，不作规划效率结论。

下面最终收据保存逐run资源，前两轮不会因扩展被抹去。当前source hash是最终可调用版本；第二轮的中间源未单独落盘，保留其测试与资源原始输出信息，不冒称该中间源hash已被保存。

```json
{
  "schema": "rc3-endpoint-adapter-synthetic-unit-v3",
  "client_date": "2026-11-06",
  "author": "/root/endpoint_baseline_adapter",
  "latest_tests": 22,
  "latest_failures": 0,
  "latest_errors": 0,
  "run_history": [
    {
      "run": 1,
      "tests": 16,
      "failures": 0,
      "errors": 0,
      "suite_elapsed_s": 0.04161560000011377,
      "process_wall_s": 6.873453,
      "purpose": "Initial endpoint adapter synthetic contracts",
      "adapter_source_sha256": "94160eb1f99edf5897325d2113fac252164f20a19c4a3b46806e29c5893999c4",
      "test_source_sha256": "4ef943af586cba53829beeb31ba2a54ee07f69b4b0fc66f2b1141fbdf7744864",
      "counters": {
        "adapter_attempts": 17,
        "adapter_completed": 13,
        "adapter_expected_negative": 4,
        "original_closed_shim_runs": 2,
        "actual_native_runs": 16,
        "construct": 48,
        "recipe": 48,
        "joint_search": 48,
        "two_opt": 91,
        "exports_attempted": 18,
        "exports_returned": 18,
        "validations_attempted": 18,
        "validations_passed": 15,
        "validations_expected_invalid": 3
      }
    },
    {
      "run": 2,
      "tests": 20,
      "failures": 0,
      "errors": 0,
      "suite_elapsed_s": 0.03936790000079782,
      "process_wall_s": 2.3417083,
      "purpose": "Add full-parent native closed comparison",
      "intermediate_source_hash_captured": false,
      "counters": {
        "adapter_attempts": 19,
        "adapter_completed": 15,
        "adapter_expected_negative": 4,
        "full_reference_attempts": 6,
        "full_reference_completed": 3,
        "full_reference_expected_negative": 3,
        "original_closed_shim_runs": 2,
        "actual_native_runs": 21,
        "construct": 63,
        "recipe": 63,
        "joint_search": 63,
        "two_opt": 131,
        "exports_attempted": 23,
        "exports_returned": 23,
        "validations_attempted": 23,
        "validations_passed": 20,
        "validations_expected_invalid": 3
      }
    },
    {
      "run": 3,
      "tests": 22,
      "failures": 0,
      "errors": 0,
      "suite_elapsed_s": 0.05937720000110858,
      "process_wall_s": 2.8531034,
      "purpose": "Immutable-snapshot scene thaw interoperation",
      "counters": {
        "adapter_attempts": 21,
        "adapter_completed": 17,
        "adapter_expected_negative": 4,
        "full_reference_attempts": 7,
        "full_reference_completed": 4,
        "full_reference_expected_negative": 3,
        "original_closed_shim_runs": 2,
        "actual_native_runs": 24,
        "construct": 72,
        "recipe": 72,
        "joint_search": 72,
        "two_opt": 153,
        "exports_attempted": 26,
        "exports_returned": 26,
        "validations_attempted": 26,
        "validations_passed": 23,
        "validations_expected_invalid": 3
      }
    }
  ],
  "all_runs_totals": {
    "actual_native_runs": 61,
    "construct": 183,
    "recipe": 183,
    "joint_search": 183,
    "two_opt": 375,
    "exports_attempted": 67,
    "exports_returned": 67,
    "validations_attempted": 67,
    "validations_passed": 58,
    "validations_expected_invalid": 9,
    "suite_elapsed_s": 0.14036070000202017,
    "process_wall_s": 12.0682647
  },
  "real_research_calls": {
    "scene_generation": 0,
    "navigation_preparations": 0,
    "S0_saved_case_planning": 0,
    "S3_queries": 0,
    "retained60_reads": 0,
    "model_forwards": 0,
    "backwards": 0,
    "optimizer_updates": 0,
    "common_H_wrapper_calls": 0
  },
  "source_sha256": {
    "endpoint_baselines.py": "6ebe1cc90f2372a5f106f8ac6289a582b3d929e9f4d0088d57aae2435209ed1b",
    "test_endpoint_baselines.py": "ad90c31507c662c608bdb2711660065fc0fe9e0d30723d2fa5c62785aae78d6f",
    "runtime_guards.py": "572a4f631438700d86658ab004605551dc525d405a57bef922b6caa928736997"
  },
  "input_or_dependent_frozen_sources_unchanged": true,
  "real_S0_conformance_status": "Await parent centrally budgeted runner; not executed by this reviewer",
  "API": [
    "endpoint_fixed3(problem,counters=...)",
    "endpoint_detour16(problem,counters=...)",
    "closed_native_reference(problem,method,counters=...)"
  ],
  "claim_scope": "Synthetic semantic conformance, source-preserving adaptation, no planning performance or trained DRL claim"
}
```
