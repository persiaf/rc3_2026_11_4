# G2 公共接口与固定实施选择

G01 原目录只读；所有新增源码、协议、模型和结果位于本目录。SearchConfig 只读复用 G01 同名 dataclass；默认 H=16、epsilon=.02、work_budget=2000000、lambda_work=.01、work_scale=10000、potential_weight=0。

## 环境

`adapter_g2.ResearchAdapterG2`继承旧adapter，from_scene/from_external/payload接口相同。只允许对固定input/validator的完全相同路线复用认证证据；final/load始终真实复验。缓存统计和准备时间可读。原方案代价与最终验证不变。

`env_g2.SearchEnvG2(adapter, config=SearchConfig(), seed=0, record_transitions=True)`保留observe/step/state_dict/load_state_dict/terminate/finalize合同。新增attempt_memory以及observe里的`action_attempts`（与actions顺序一致、各项有attempts/no_change/failed）、`proposal_keys`。菜单diagnostics含action_details(required_tasks/pool_nodes/retained_counts等)。恢复先全部验证，失败环境不变；有限数、bool、integer、路线/证书、终止/返回/累计回报、菜单绑定均验证。

共享proposal_key不包含budget/history，包含problem/full route/operator/window/banned/actual solver caps。已知no-change/failed在菜单截断前过滤补位，所有方法一致。预算/history保留于replay key。不使用proposal缓存。`record_transitions=False`允许训练不长存整条日志，但需可保存完整动态状态、记忆及RNG后续合法采样。

基线`baselines.make_policy(name)`返回.select(observation,rng)、.update(info)、state_dict/load_state_dict；固定名称`detour_memory`、`ils`、`random`、`stop`。无免费全菜单DP。ILS具体参数在开发场景上预先选择后冻结；initial reference Fixed3/Detour单独从旧runtime计完整时间，不收共享菜单费。

## 原始网络观测与模型

`graph_observation.build_policy_observation(env)`返回可pickle的原始NumPy数组字典，至少有精确`actions`、合法`mask`、`features`、`problem_id`和`state_key`。局部二部图包含task/candidate node features、稀疏coverage边及action归属；窗口入口/出口D、required/pool、retained coverage、路线顺序、真实剩余预算、current/best gap和尝试结果显式可见。模型更新重新编码原始输入，不用旧embedding替代新权重。

`actor_critic.ActorCritic(kind='graph'|'mlp',hidden=64)`；`forward(list[obs])->(padded_logits[B,Amax],values[B])`；`evaluate_actions(obslist,actions)->(logp,entropy,values)`；`act(obslist,deterministic=False)->(actions,logp,values)`，输出torch tensors，便于root收集detach/cpu。模型当前device由parameters推断，mask/padding logits=-inf。无dropout/BatchNorm。

`ppo_math.compute_gae(rewards,values,next_values,terminated,gamma=1.,lam=.95)`输入T×B，返回advantages,returns，保持tensor或NumPy接口清楚注明。GAE不能跨terminated，rollout截断本身不终止。Root collector单独处理外部event：恰好一次并入最后真实动作，terminal=True且bootstrap0；零步只记指标。

## 教师

重新采G2教师，不直接复用G01 action_id/Q。depth=2、最多8分支含STOP；初始BC使用64个新teacher场景，继续BC额外32个新teacher场景。由root的protocol.json给种子。各场景内部全树8状态约64queries，失败保留。标签为已查子树最高剩余return，包含最终event；不作为horizon16 critic监督。

每个sample至少为`observation`(build_policy_observation结果)、`evidence_mask`、`q_values`（未知0但mask false）、`ties`（within .001 of best evidenced Q）、`weight`、`problem_id`、`state_signature`、`teacher_depth`。BC归一化只在legality & evidence上；同场景总weight一致、重复route权重均摊。输出samples pickle，另有场景清单、全查询账本、真实menu/proposal工作与耗时。

## 单种子小试

Root实现并行collector、BC/PPO trainer、验证和检查点。一个训练seed，8环境×32step rollout=256真实动作/iteration，共128iteration=32768；每8192/16384/32768步固定验证，不自动扩至65536。gamma1,lambda.95,lr1e-4,clip.2,4epochs,minibatch64,max_grad_norm.5,target_KL.02。初始BC400updates，继续BC额外2048updates（记录真实计算量，不声称与PPO等算力）。正式训练种子池384百桌，validation24(各mode8)，规则开发12(各4)，pipeline6，全部新种子且split互斥。

两个架构均做数值与MLP/graph短管线检查；正式单种子仅graph，BC frozen/continued/PPO共享同一个graph BC初始化。PPO行为来自新环境交互，教师不进入PPO clipped更新。规则参数、奖励系数及模型结构不能看验证后修改。
