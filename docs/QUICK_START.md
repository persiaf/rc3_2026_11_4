# 最小复现与版本恢复

## 当前可用版本

本轮整理分支`codex/repository-handoff-20261110`只在本地提交，等待人工检查。父提交为`0c268426f3760f4d06596a29f65b3d1a5309987f`，该RPPO部分证据提交在审计时也未出现在任何origin分支。新老师目前无法从GitHub取得本轮文档或该父提交。

已核对远端RPPO分支为[bb9d1cbb4](https://github.com/persiaf/rc3_2026_11_4/tree/bb9d1cbb4a6ce03f77d43697587ea29332ff21f1)，GL-v5来源为[7d2cfe759](https://github.com/persiaf/rc3_2026_11_4/tree/7d2cfe75919b25bac59c5274321baca4d8e98f77)。两者均不包含本轮整理。

## 本地最小命令

在整理工作树根目录执行[README四条命令](../README.md#最小运行)。已验证Windows、Python3.13.5、NumPy2.1.3、SciPy1.15.3、threadpoolctl3.5.0。新建`.venv-handoff`安装原运行依赖，`pip check`通过；不复制本机旧venv。

编排脚本只生成`4x3 / regular / seed27182818`工程场景，不读取保存数据集。其实际调用等价于：

```powershell
.\.venv-handoff\Scripts\python.exe -B -X utf8 -m rc3_v5.plan --method fixed3 --budget 0 --scale 4x3 --mode regular --seed 27182818 --output outputs/teacher_handoff/fixed3.json
.\.venv-handoff\Scripts\python.exe -B -X utf8 -m rc3_baseline.validator outputs/teacher_handoff/fixed3.json --report outputs/teacher_handoff/fixed3_validation.json
.\.venv-handoff\Scripts\python.exe -B -X utf8 -m rc3_v5.plan --method detour --budget 16 --scale 4x3 --mode regular --seed 27182818 --output outputs/teacher_handoff/detour16.json
.\.venv-handoff\Scripts\python.exe -B -X utf8 -m rc3_baseline.validator outputs/teacher_handoff/detour16.json --report outputs/teacher_handoff/detour16_validation.json
```

这四条已由脚本实际运行，收据保存绝对路径。**二选一**：编排脚本或手动展开命令。不要在同一目录两者都运行，以免手动CLI覆盖输出。脚本拒绝已有输出目录，每个子进程上限180秒，共两次规划和两次独立重验。

JSON包含场景、停靠顺序/位姿、任务用臂分配、逐腿成本和完整折线，CSV是服务表。验证器重新计算代理几何与记账，忽略保存的验证布尔值。

## Git与LFS

本轮实际完成`git fetch origin`、`git ls-remote --heads origin`、`git lfs install --local`和以下定向恢复：

```powershell
git lfs pull --include="rc3_v3/deployment/policy.npz" --exclude=""
```

该权重只作LFS恢复及字节核对，未加载。规则示例不依赖它。SHA256与Git指针一致，[收据](evidence/LFS_VERIFICATION.json)可查；`.gitattributes`原字节未改。

新老师获取已发布源码时，可先设置`$env:GIT_LFS_SKIP_SMUDGE='1'`，再克隆[仓库](https://github.com/persiaf/rc3_2026_11_4)，检出人工选定且已发布的准确SHA，最后删除该环境变量。后续按阶段WEIGHT_LINEAGE和LFS清单定向取文件。不要把main当作最新研究版，也不要以自动源码ZIP代替LFS恢复。

本轮没有全新克隆GitHub仓库、没有下载全部LFS或逐对象验证远端全部历史数据。交接ZIP首次纳入本地整理分支，尚未推送。训练环境须对应历史依赖和来源，不能从最小CPU依赖推断全部DRL可复现。

## 未运行范围

不运行trainer/runner/register/infer、S3、历史全套verify或benchmark；部分历史verify会读冻结数据，不作为新手入口。原60继续封存。非Windows平台、旧MATLAB、上游TensorFlow、历史重训及真实机器人均未在本轮复现。失败须保留原seed和资源账，不能重采样解决。
