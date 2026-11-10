# 交接复现验收

结论：**本轮两个新工程代理方案及现有独立验证均通过；文档/工程交接可审查。** 不能据此宣布DRL性能提升、全历史可复现或真实双臂认证。本轮没有训练、大规模性能实验、S3查询或原60场景语义读取。

## 实际环境与执行身份

2026-11-10北京时间11:11:57开始两个示例。隔离分支`codex/repository-handoff-20261110`，规划源码基准为`0c268426f3760f4d06596a29f65b3d1a5309987f`。Windows11（构建10.0.26300）、AMD64 Family25 Model97、16逻辑CPU；Python3.13.5（Anaconda来源）新建独立venv，NumPy2.1.3、SciPy1.15.3、threadpoolctl3.5.0。无需Torch/GPU/权重。

## README命令验收

| 命令 | 实际结果 | 耗时范围 |
|---|---|---|
| `python -X utf8 -m venv .venv-handoff` | 成功，新环境 | 未单独仪表化，不虚构秒数 |
| 新环境pip安装原requirements-runtime | 三项准确版本安装成功 | 未单独仪表化 |
| 新环境运行`docs/tools/reproduce_baselines.py --output outputs/teacher_handoff` | 四个子命令退出码均0 | 14.797335秒，含子进程启动/规划/导出/两次重验及收据写入 |
| 新环境`python -m pip check` | `No broken requirements found`，退出0 | 未单独仪表化 |

完整argv、stdout、stderr、环境和文件哈希在[VERIFICATION.json](evidence/baselines/VERIFICATION.json)。README中的所有运行命令均已执行。展开命令见[QUICK_START](QUICK_START.md)，没有随后重复运行取最快值。

## 两个完整规划

同一新建`4x3 / regular / seed27182818`代理场景，19任务、96候选，不使用保存数据集。Detour参数预算16，因合法动作耗尽实际执行10次修复调用；不是16次必跑。

| 方法 | J | L（米） | K | CLI进程墙钟 | 规划器内部total_s | 独立结果 |
|---|---:|---:|---:|---:|---:|---|
| Fixed3 | 33.55 | 26.05 | 10 | 10.525282秒 | 0.074726秒 | 全任务覆盖，碰撞0，错误0 |
| Detour16 | 33.55 | 26.05 | 10 | 3.289405秒 | 0.099472秒 | 全任务覆盖，碰撞0，错误0 |

独立验证进程分别0.476408秒和0.470059秒，准确值以JSON为准。进程启动、导入、系统缓存与本机并发会影响单次墙钟；两者成本相同，不能推导质量提升或统计速度比较。该例是代理起点闭环，不是全父地图局部非同端点DRL的性能复现。

结果归档：[Fixed3 JSON](evidence/baselines/fixed3.json)、[服务CSV](evidence/baselines/fixed3.csv)、[独立验证](evidence/baselines/fixed3_validation.json)；[Detour16 JSON](evidence/baselines/detour16.json)、[服务CSV](evidence/baselines/detour16.csv)、[独立验证](evidence/baselines/detour16_validation.json)。归档与原执行输出字节一致，收据路径保留执行时原值。

验证器为未修改的`rc3_baseline.validator`，重新核对覆盖、用臂、代理可达、矩形膨胀碰撞、访问顺序、逐腿/折线长度和J；不信任方案中保存的PASS布尔值。它不是IK验证器，也不认证双臂同步或关节轨迹。

## 依赖、LFS与来源

`git lfs install --local`退出0，1.823811秒；定向pull历史policy.npz退出0，7.043887秒。59,744字节载荷SHA256=`63e200f204b1e53c630f178bf7f60665944222becb60126b11c6461613db234d`，匹配Git指针；没有加载权重。[收据](evidence/LFS_VERIFICATION.json)

旧README原字节完整迁移；交接ZIP与六份文档逐项校验；10项选定冻结身份匹配。[整合收据](evidence/SOURCE_INTEGRATION.json)、[冻结身份](evidence/FROZEN_IDENTITIES.json)。最终文档链接、修改范围和manifest检查结果见[文档检查](evidence/DOCUMENT_CHECKS.json)。

空白检查使用`git -c core.whitespace=cr-at-eol diff --cached --check`，保留历史README、摘录与原manifest的CRLF字节；不为消除默认检查把CR视作尾空格的提示而重写冻结来源。

## 未复现及限制

没有重新克隆整个仓库、恢复/核验全部远端LFS、重训历史模型、重跑S3、补历史失败、运行封存60、执行大规模benchmark或做全仓历史数字独立复算。原交接包26项测试仅保留其Linux报告，本轮未重跑。MATLAB原工程、作者TensorFlow训练、非Windows平台、标定、真实IK/关节轨迹与实机均未复现。

维护新增资源为两次规划、两次独立重验、一个新工程场景；optimizer0、权重加载0、科研H调用0、S3/封存任务语义读取0。规则内部修复调用单列，不冒充正式科研训练评价。[本轮资源账](repository_handoff/RESOURCE_ACCOUNTING.json)
