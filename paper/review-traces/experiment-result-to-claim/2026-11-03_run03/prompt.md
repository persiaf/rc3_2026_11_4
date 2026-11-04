# 独立审查任务

日期：2026-11-03。技能：experiment-result-to-claim。角色：项目内独立 paper architect/reviewer 子代理；由主代理委派，不参与实验/功能实现，不是外部零上下文审稿。

委派要求：只读核查工作区 `rc3_v5/RESULTS.md`、`README.md`、`EVIDENCE_MAP.md` 的数值与范围是否由原始 JSON 支持。读取 v5 各 summary 与 audit，必要时读取 v4 原始结果。重点核查本机 MLP 区间跨零、两平台不可混拼、缓存 144 决策/36 等价计划与 108 个 profile 计划的不同分母、耗时缩短百分比、机器人配置阻断与真实 IK 未完成、v4 原采用门槛不可追溯改变。

禁止修改实验/源码和重跑性能；允许轻量数据重算。输出 `claim_supported=yes|partial|no`、支持/不支持论点、证据缺口、建议改写、下一步、置信度与路由。无重大问题须直说，不为凑数制造发现。

 intended claims：v5 在保留 Fixed3/Detour 默认和旧门槛前提下修复共同约束；冻结旧 MLP 同软预算结果可作有限描述；Structured 缓存有静态验证等价性及本机耗时改善；真实资产/预检完成但物理闭环尚未完成。

主要证据：两平台 MLP 保存逐次计划 JSON、summary、四份 supplement/protocol audits；cache equivalence 与 benchmark 的 manifest、summary、计划和 head_checks；artifact_reconstruction；external_results/SUMMARY；verification/summary；两份 URDF 与 19 STL、asset_inventory、preflight；v4 原 manifest/summary/逐次计划、小规模 metrics.csv、训练/数据 manifest、40 测试保存日志。

最终审查纳入主代理已明确的范围：Structured 对 Control 跨零只指主要百桌 18 场景固定 8 次；旧 MLP v4 固定 8 次比较属于上传追加描述性审计。技能引用的 shared-references/review-tracing.md 与 assurance-contract.md 在提供的本地技能目录不存在；按主 SKILL.md 的明确字段及用户指定 run03 路径保存完整追踪，不因此停止审查。

本次审查仅对现有保存证据判断，不重新宣称硬件计时、几何/物理或全部历史计划已重新测量。
