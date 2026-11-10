# 本轮文档整理变更清单

基准为0c268426f3760f4d06596a29f65b3d1a5309987f；独立分支codex/repository-handoff-20261110。只本地提交，不推送、不合并。

## 修改内容

README首页重写并保留原字节全文；新增八份研究文档、仓库审计、交接整合、实际验收与维护阶段文件；加入Issue/PR模板、两次代理示例的有界编排/文档检查工具及证据。交接原ZIP按原字节首次纳入LFS。更新忽略规则和Git清单。

算法、训练器、模型权重、旧结果、协议/授权及活动v3状态均未修改。源工作区Word、活动账本和其他未跟踪产物不在提交范围。

## 文件清单（47项）

- [.github/ISSUE_TEMPLATE/research_question.yml](../.github/ISSUE_TEMPLATE/research_question.yml)
- [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md)
- [.gitignore](../.gitignore)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [README.md](../README.md)
- [README_HISTORY.md](../README_HISTORY.md)
- [docs/ALGORITHM_DESIGN.md](../docs/ALGORITHM_DESIGN.md)
- [docs/CHANGESET.md](../docs/CHANGESET.md)
- [docs/CODE_ARCHITECTURE.md](../docs/CODE_ARCHITECTURE.md)
- [docs/EXPERIMENT_HISTORY.md](../docs/EXPERIMENT_HISTORY.md)
- [docs/HANDOFF_PACKAGE.md](../docs/HANDOFF_PACKAGE.md)
- [docs/HANDOFF_VERIFICATION.md](../docs/HANDOFF_VERIFICATION.md)
- [docs/QUICK_START.md](../docs/QUICK_START.md)
- [docs/REFERENCES.md](../docs/REFERENCES.md)
- [docs/REPOSITORY_AUDIT.md](../docs/REPOSITORY_AUDIT.md)
- [docs/RESEARCH_OVERVIEW.md](../docs/RESEARCH_OVERVIEW.md)
- [docs/RESEARCH_PROGRESS.md](../docs/RESEARCH_PROGRESS.md)
- [docs/TEACHER_QUESTIONS.md](../docs/TEACHER_QUESTIONS.md)
- [docs/evidence/DOCUMENT_CHECKS.json](../docs/evidence/DOCUMENT_CHECKS.json)
- [docs/evidence/ENVIRONMENT.json](../docs/evidence/ENVIRONMENT.json)
- [docs/evidence/FROZEN_IDENTITIES.json](../docs/evidence/FROZEN_IDENTITIES.json)
- [docs/evidence/GIT_VERSION_AUDIT.json](../docs/evidence/GIT_VERSION_AUDIT.json)
- [docs/evidence/LFS_VERIFICATION.json](../docs/evidence/LFS_VERIFICATION.json)
- [docs/evidence/REPOSITORY_SNAPSHOT.json](../docs/evidence/REPOSITORY_SNAPSHOT.json)
- [docs/evidence/SOURCE_INTEGRATION.json](../docs/evidence/SOURCE_INTEGRATION.json)
- [docs/evidence/baselines/VERIFICATION.json](../docs/evidence/baselines/VERIFICATION.json)
- [docs/evidence/baselines/detour16.csv](../docs/evidence/baselines/detour16.csv)
- [docs/evidence/baselines/detour16.json](../docs/evidence/baselines/detour16.json)
- [docs/evidence/baselines/detour16_validation.json](../docs/evidence/baselines/detour16_validation.json)
- [docs/evidence/baselines/fixed3.csv](../docs/evidence/baselines/fixed3.csv)
- [docs/evidence/baselines/fixed3.json](../docs/evidence/baselines/fixed3.json)
- [docs/evidence/baselines/fixed3_validation.json](../docs/evidence/baselines/fixed3_validation.json)
- [docs/handoff_reference/CODE_MAP.md](../docs/handoff_reference/CODE_MAP.md)
- [docs/handoff_reference/HANDOFF_TEST_REPORT.json](../docs/handoff_reference/HANDOFF_TEST_REPORT.json)
- [docs/handoff_reference/INTERFACE_NOTES.md](../docs/handoff_reference/INTERFACE_NOTES.md)
- [docs/handoff_reference/REQUIREMENTS_TEACHER.md](../docs/handoff_reference/REQUIREMENTS_TEACHER.md)
- [docs/handoff_reference/SOURCE_AVAILABILITY.md](../docs/handoff_reference/SOURCE_AVAILABILITY.md)
- [docs/handoff_reference/SOURCE_LOCKS.json](../docs/handoff_reference/SOURCE_LOCKS.json)
- [docs/repository_handoff/EXPERIMENT_REPORT.md](../docs/repository_handoff/EXPERIMENT_REPORT.md)
- [docs/repository_handoff/IMPLEMENTATION_REVIEW.md](../docs/repository_handoff/IMPLEMENTATION_REVIEW.md)
- [docs/repository_handoff/NEXT_STEP_DECISION.md](../docs/repository_handoff/NEXT_STEP_DECISION.md)
- [docs/repository_handoff/RESOURCE_ACCOUNTING.json](../docs/repository_handoff/RESOURCE_ACCOUNTING.json)
- [docs/repository_handoff/RESULT_TO_CLAIM.md](../docs/repository_handoff/RESULT_TO_CLAIM.md)
- [docs/tools/check_handoff_docs.py](../docs/tools/check_handoff_docs.py)
- [docs/tools/reproduce_baselines.py](../docs/tools/reproduce_baselines.py)
- [project_management/GIT_MANIFEST.json](../project_management/GIT_MANIFEST.json)
- [交接/RC3_Teacher_Handoff_v1.zip](../交接/RC3_Teacher_Handoff_v1.zip)

清单相对本轮准确父提交生成；Git manifest本身按既有规则排除自哈希。新的文档检查证据只核本轮链接与选定冻结身份，不宣称全仓历史审计。
