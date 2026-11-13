# 下一版代码架构与实施顺序

本目录是R08-A的独立设计/证据/最小接口包。拟建运行目录`rc3_r08_joint_runtime/`**当前不存在**，下表除明确标“已实现”外均为后续受条件控制的开发任务。不会把Protocol或准备报告标成生产训练器已跑通。

|模块/建议文件|主要接口与I/O|复用与必须新增|关键风险/验收|
|---|---|---|---|
|`data.py`|`RegisteredPool.get(problem_id)->ProblemSnapshot`；source/role→原F/Q、端点D、ID、pose|复用SavedInputs确切producer/legacy语义；本轮**已实现**`adapters.SavedEvidence.snapshot`和`ProblemSnapshot`，只读已存NPZ|不依赖旧provider生成fallback；来源/任务/物理身份错误停止；已有数据恢复与新导航求解分开计账|
|`records.py`|`ExistingPlan.from_evidence`→RecordedRoute+原plan/source refs|本轮**已实现**`recorded_route`/`RecordedRoute`；保留raw/H类型、K配置计数|同xy不同配置不能合并；native-final不能当神经no-H|
|`state.py`|`JointState(problem,incumbent,steps_left,requests_left,version)`；`fingerprint`|本轮**已实现**不可变状态和版本指纹；真实状态还须加入attempted菜单签名、完整raw/H锚点|完整方案MDP不能复用旧remaining为空就terminal；状态改变使菜单/GRU缓存失效|
|`menu.py`|`MenuBuilder.build(state)->ActionMenu(actions,mask,menu_sha,construction_stats)`|JointAction和覆盖/端点前检**已实现**；8个drop块×每块2个替代的固定16动作规则待B实现|区别候选生成、覆盖前检与真实下层验证；不会在特征阶段偷偷求16条路线|
|`lower.py`|`LowerOrderingSolver.solve(state,action,SolveLimits)->raw_plan+timing`|本轮只有Protocol/SolveLimits。B新增移除/有界插入/8pass2-opt/normalize，复用原D与验证器；不调用旧DP窗口代替新动作|每非STOP一次完整处理请求先扣账；实际删除/插入/排序及无变化均记录；不能把只算局部边差当完整结果|
|`transition.py`|`JointEnvironment.step(action)->JointState,event`；`terminal()->H_plan`|B实现单步机会处理，C实现最多4步及STOP；完整责任分配沿用原export/validate；终端原H一次|raw中途可增成本，最终返回保护统一；记录保护前候选及H重写；不得把全部失败回退后算学习成功|
|`features.py`|`encode_base(problem)`与`action_features(state,menu)->[M,400]`|C复用准确SL256 t/c/g及prefix GRU；新增当前选中集合池化、完整σ与逐臂/覆盖/预算16标量|base参数全冻结，菜单变化/权重版本变化后重算受影响缓存；特征不含教师目标/候选H成本|
|`actor.py`|`JointMenuActor.logits(state,menu)`、`select(greedy/sampler)`|C新增400→64→1菜单头四张量25729标量；原65088模型全部冻结，只作表示与初解|数值有限、masked概率0、optimizer精确白名单；不是旧A1残差，也不读任何诊断权重|
|`sampling.py`|`collect_episode(actor,problem,stream_rng)->CompleteEpisode`|复用D6 generator序列化/当前Actor hash/精确来源；新增每次联合转移、菜单SHA及终端raw/H|PG各流own-policy；旧轨迹/其他权重不能更新；完整轨迹成本和所有内部调用分账|
|`trainer.py`|`update(batch8)->update_receipt`|仅C实现工程一次更新，D才正式用；复用Adam1e-4、clip1、完整回报、单epoch；无Critic|正式从fresh SL表示+固定新头seed初始化；工程权重禁止正式起点；不偷偷增加PPO epoch|
|`supervision.py`|`structure_target(state,menu,saved_teacher)->action_id`；`CE_batch`|只读352已有教师；合法菜单中，先最小化编辑后集合与教师集合的对称差，再比较**当前顺序删除drop后的保留子序列**在教师共同配置上的逆序数，最后动作ID tie；严格比当前状态更近才选，否则STOP|顺序项是保留顺序的结构proxy，不是假装知道add插入/2-opt后顺序；不给未选菜单跑下层，不新生成教师，不输入教师成本到Actor。仅选中动作真实执行并扣账；SL轨迹独立，匹配任务/机会不是等FLOPs|
|`accounting.py`|`begin_request(purpose,id)`；`record_components`；`close`|复用已验收持久先扣账/失败不退/阶段总时钟/来源分类/私有进程树监督器；新增raw转移和terminal-H用途|parent pipeline与子处理请求分开；禁止全部写engineering；native/H内部搜索、cache命中/feature/I/O都记；不因恢复清零|
|`evaluation.py`|`evaluate_fixed_endpoint(methods,cases,endpoint)`→全请求+父配对|复用D5/D6完整plan证书、父ID聚合、两聚合、raw/H、原生final区分|保存每配置/任务/左右臂与顺序；先验证完整再计质量；等缓存和同机方法顺序用于同期时间，不用旧计时代替|
|`export.py`|`export_actor_only`/`load_deployed`→base+menu head+完整schema/hash|保留原冻结SL256与新头、feature/menu/lower/H身份；无Adam/Critic；冻结贪心|“Actor-only”不等于部署零solver；菜单/下层/H/验证仍是部署依赖，并纳入完整时间|
|`checkpoint.py`|`save_atomic`/`restore_exact`|复用D6双槽+fsync/指针/源身份/Adam/RNG/游标/已付请求；本轮**仅恢复元数据契约**ResumeIdentity已测|C须真正验证权重、Adam、采样、menu次序、下一条轨迹一致；模糊未持久化更新停止，不免费重做|
|`cli.py`|`register/opportunity/acceptance/train/evaluate/resume/close`|B/C分步实现真实入口，D才开正式train；每个入口验证阶段授权与完成收据|当前没有这些可执行CLI，不把改状态为PASS算完成。未来source/content身份不包含可变下游授权SHA|

## 已运行的R08-A命令

仓库根目录，现有真实Python环境：

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r08_joint_decision_design_20261113.analyze_saved
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r08_joint_decision_design_20261113.test_interfaces
```

第一条只读取精确保存成员/数组，产出主分析2974行；再运行README所列`extend_r07`补充既有R07结构后总3550行。第二条12项纯内存测试PASS。分析模块没有Torch/SciPy/历史规划模块导入。原始证据不写，来源索引见ANALYSIS_INPUTS.json。

## 后续CLI合同（开发目标，当前不可运行）

```text
python -m rc3_r08_joint_runtime.cli register --stage B --design <R08_MASTER_PLAN及源码内容hash> --authorization <真实用户收据>
python -m rc3_r08_joint_runtime.cli opportunity --run <同一注册身份>
python -m rc3_r08_joint_runtime.cli acceptance --run <C已授权身份>
python -m rc3_r08_joint_runtime.cli train --run <D已授权身份> --seed 2026111308 --stream RL --target 128
python -m rc3_r08_joint_runtime.cli resume --run <原身份> --checkpoint <原LATEST指针>
```

尖括号为必须由实际收据/注册产物提供的运行身份，不是未决定的算法参数。R08-B实现时必须先让前两条真实跑通；C实现并验收其余驱动。若用户今后一次批准整个B—E条件范围，可直接按批准条件连续推进并登记，不为范围内常规动作重复询问；**本轮没有授予这些阶段执行权**。

## 新增测试与阶段验收

已通过的12测试见INTERFACE_TEST_RESULTS.json，主要检查真实会影响数据含义的契约，不镜像每行实现。未做生产测试：真实联合变更后的mask/完整plan、终端H与STOP一次性、数值梯度和白名单、真实Adam回滚/恢复、冻结导出与部署等价、实际吞吐；这些都是C明确要完成的工作，当前不写PASS。

正式实施顺序：B data/menu/lower的无网络机会；C state/feature/actor/sampling/checkpoint及工程一批；D bounded trainer/supervision/evaluation；E新增父布局授权后的确认/最终与部署优化。每阶段只修改新运行目录，旧Actor、Trainer、H和所有历史证据保持。
