# 归档、清理与恢复说明

本轮按用户要求备份当前项目、建立 Git 版本管理、上传指定私有仓库，并清理项目内实验残留。清理对象逐项列在 `cleanup_plan.json` 和 `cleanup_receipt.json`，未扫描或删除项目外目录。

## 两类恢复来源

1. **Git＋Git LFS**：保存清理后的完整研究项目，包括源码、模型、训练及测试记录、保存方案、论文资料、机器人资产和原始交付／审查包。克隆后运行 `git lfs pull`，再用 `tools/verify_project_archive.py` 核对文件清单。`PROJECT_VERSION.json` 指明默认方法和冻结边界。
2. **本机清理前完整快照**：`project_backups/2026-11-04_g23-closeout/full_before_cleanup.zip`。这是按内容 SHA256 去重的 ZIP，`SNAPSHOT_MANIFEST.json` 映射原路径到 `objects/SHA256`，保留清理前的缓存、临时副本和本机环境。所有独立内容对象都验证后才允许清理；哈希和数量记录在 `snapshot_receipt.json`。本机快照不上传 Git。

另提供清理后的标准目录 ZIP：同目录的 `rc3_2026_11_4_project.zip`，包括所有已版本管理的实际文件内容与模型，直接解压到新目录即可使用；不包含 Git 内部数据和本机虚拟环境。包内 `ARCHIVE_SNAPSHOT.json` 记录对应提交；包外 `project_archive_receipt.json` 记录包 SHA256 与全部成员校验结果。该 ZIP 由 `tools/build_project_archive.py` 在工作树干净、提交完成后生成，避免生成到一半的记录进入备份。

恢复清理前的全部文件到一个新的空目录：

```powershell
python -B tools/restore_project_backup.py project_backups/2026-11-04_g23-closeout/full_before_cleanup.zip D:/rc3_restored
```

也可以从 ZIP 取出 `RESTORE_PROJECT.py` 后运行。恢复工具拒绝覆盖非空目录，先校验内容对象，再恢复原路径。完整快照是文件内容备份，不是 Windows ACL、进程或系统镜像；虚拟环境迁移到其他机器应按锁定依赖重新安装。

## 清理边界

- 删除迁移检查产生的 `portable_check/`、`portable_test/`、根目录指定 `*_smoke/` 和 `rc3_g23_closeout/work/` 解压副本；保留对应原始 ZIP 与检查收据。
- 删除 `rc3_improve/data_20261102/cache.pkl` 与 `data_v2_20261102/cache.pkl`，保留案例／种子／数据说明、训练记录、权重和结果。原交付脚本已将这两个多 GB 派生缓存排除出交付；需要重训时按对应 README 和 `data.py` 生成新缓存，或从完整快照恢复。
- 删除 `rc3_rl_g2/outputs/prepared_train/`，这是原包明确排除的可重建 `.rc3prep` 缓存；`prepared_cache.py` 会按输入与源码指纹重新生成。保留正式 teacher 数据、PPO 检查点与逐回合日志。
- 删除本项目源码下的 `__pycache__/`；保留 `rc3_learning/.venv/` 及其内部文件，避免破坏已有本机运行环境。

清理程序先解析每个绝对路径并确认严格位于工作区内，拒绝工作区根目录、链接、Git 目录、本机环境及冻结载荷交集；检查备份中的待删文件与当前文件哈希一致，再执行逐路径删除。无通配符跨目录删除。

## 冻结与后续版本

原 v4、v5、v5.1、G0/G1、G2～G2.3、收尾交付包仍保留原哈希。初始提交为当前目录的实际快照，不伪造过去的 Git 提交历史。`snapshot-2026-11-04-g23-closeout` 标签指向完成清理和文件核对后的提交；后续研发应从该点新建分支。

运行以下命令可查看状态与大文件完整性：

```powershell
git status --short
git log --oneline --decorate -5
git lfs fsck
python -B tools/verify_project_archive.py
```

SHA256 清单验证原始内容字节；`.gitattributes` 关闭自动换行转换，LFS 检出后恢复原文件。`project_management/GIT_MANIFEST.json` 不对自身及发布收据做自引用校验；这两类文件由 Git 提交内容和外部交付收据追溯。
