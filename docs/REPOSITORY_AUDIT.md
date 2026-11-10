# 仓库真实状态审计

审计日期2026-11-10，初始完整元数据快照为北京时间11:06:55（原始UTC见[JSON](evidence/REPOSITORY_SNAPSHOT.json)）。本轮执行了status、branch、log、remote、fetch、ls-remote、worktree、标签及祖先关系检查，不合并、删除、重置、强推或推送任何分支。

## 工作区与隔离

源工作区当前分支`codex/rc3-integrated-rppo-v1`，快照HEAD为`0c268426f3760f4d06596a29f65b3d1a5309987f`。有四个已跟踪修改：RPPO的`ledger/ACTIVE_CLOCK.pt`、`HEAD.json`、`events.jsonl`，以及`第二章与第三章修订稿_20261029.docx`。初始快照另有1412个未跟踪路径，包含后续checkpoint/frame/request收据、分析脚本、IMPLEMENTATION_REVIEW、教师交接包等；完整路径在JSON中，未把活动产物当作闭合结果。

为避免切换源工作区或提交用户修改，本轮从准确HEAD新建隔离Git工作树和`codex/repository-handoff-20261110`分支。LFS默认跳过smudge，未新建整包ZIP备份。源工作树分支、已有修改和活动日志没有由本轮切换/重置/恢复。只有用户明确指定的交接ZIP从未跟踪输入复制到本整理分支，原件保留；其余未跟踪文件未纳入本轮。

收尾只读复核时源工作区又出现`.gitattributes`及`project_management/integrated_rppo_delivery/checkpoint_publish.py`修改，源HEAD/分支未变；这属于并行工作区变化，未复制到整理工作树。初始四项与收尾六项不是同一时刻的状态。准确收尾记录见[GIT_VERSION_AUDIT](evidence/GIT_VERSION_AUDIT.json)。

## 全部本地分支

以下`相对main新增`由`rev-list --left-right --count main...branch`得出。全部本地分支均包含当前main，main一侧独有提交均为0；这不等于main已经包含研究成果。

| 分支 | 快照SHA | 上游 | 相对main新增 | 不在任一origin分支的提交 |
|---|---|---|---:|---|
| main | 5d5fe2575 | origin/main | 0 | 0 |
| codex/teacher-distill-t0-t2 | 6f595cfa8 | 无 | 2 | 0，已随后续分支上传 |
| codex/global-local-s0 | 057830d58 | 无 | 4 | 0，已随preflight上传 |
| codex/global-local-preflight | 925a0a1a8 | 同名origin | 45 | 0 |
| codex/actor-pg-screen-20261109 | c5f47ec35 | 同名origin | 48 | 0 |
| codex/gl-v4-natural-start-20261109 | ce3de0eab | 同名origin | 48 | 0 |
| codex/gl-v5-frozen-greedy-20261109 | 7d2cfe759 | 同名origin | 51 | 0 |
| codex/integrated-drl-20261109 | 7d2cfe759 | 无 | 51 | 0，与GL-v5同提交 |
| codex/rc3-integrated-rppo-v1 | 0c268426f | 同名origin，领先1 | 57 | 1：0c268426f |
| codex/repository-handoff-20261110 | 创建于0c268426f | 无 | 创建时57；本轮整理另增 | 继承部分证据提交及本轮整理提交 |

T0/T1两个提交`75075a7eb`、`6f595cfa8`没有同名远端分支，但已是`origin/codex/global-local-preflight`及GL-v5远端的祖先。**不能把T0/T1成果称为尚未上传。** S0同理；`integrated-drl`只是GL-v5同SHA的本地别名。

actor-PG与GL-v4为不同研究分支，双向各3个独有提交；actor-PG不是当前RPPO分支的祖先，但其自身已上传。不得假设当前整理分支含所有旁支代码，老师阅读该方向应检出对应分支。本轮没有自动整合旁支。

## GitHub远端实测

origin为[科研仓库](https://github.com/persiaf/rc3_2026_11_4)，fetch完成后`ls-remote --heads`返回：

| 远端分支 | 准确SHA |
|---|---|
| main | `5d5fe25754ffa032f97937a3fb54f15901c2fb7d` |
| codex/global-local-preflight | `925a0a1a8957299e3e849b29ac501a99212aa902` |
| codex/actor-pg-screen-20261109 | `c5f47ec3518bd2b58f26e8efb2d4737676632dba` |
| codex/gl-v4-natural-start-20261109 | `ce3de0eab485056f4b59b96b2a4c620c8ab22775` |
| codex/gl-v5-frozen-greedy-20261109 | `7d2cfe75919b25bac59c5274321baca4d8e98f77` |
| codex/rc3-integrated-rppo-v1 | `bb9d1cbb4a6ce03f77d43697587ea29332ff21f1` |

当前main确实落后于后续研究历史。GitHub API返回default_branch=main、visibility=public；交接包原private说明是历史来源时点，本轮未改权限。`teacher/research-review`只作协作建议，未创建。两个既有snapshot标签保持，未移动。

真正尚未同步的是RPPO部分证据`0c268426f`、源工作区后续未提交成果、本轮整理提交及新纳入的交接ZIP。未经用户进一步审查，不推送这些内容。推送本整理分支将同时使其未发布父提交可被远端访问，人工检查应包含这一点。

## LFS与冻结身份

保留`.gitattributes`原规则。隔离检出中未恢复的大文件是LFS指针，这是有意的最小恢复，不报告为完整离线备份。已实际定向恢复并校验旧`rc3_v3/deployment/policy.npz`，只算维护字节检查，无模型推理。[LFS收据](evidence/LFS_VERIFICATION.json)

对旧v2指针/协议、v3活动状态/协议、导航及scope绑定、冻结terminal、GL-v4/v5报告、LFS配置进行10项选定字节核对，均与基准Git blob一致。[准确SHA256](evidence/FROZEN_IDENTITIES.json)。这不是全仓13GB载荷或远端全部LFS对象审计。最终提交允许修改范围见[变更清单](CHANGESET.md)，所有原算法、权重和结果路径保持。

本轮维护阶段文件在[repository_handoff](repository_handoff/NEXT_STEP_DECISION.md)。没有改变任何研究协议、旧阶段状态或冻结测试边界。
