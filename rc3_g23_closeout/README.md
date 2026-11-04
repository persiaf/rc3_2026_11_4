# G2.3 本地收尾与接收复验

日期：2026-11-04。当前状态：**关闭 Fixed3 初解上的局部窗口选择＋STOP 作为第三部分 DRL 核心的当前路线。** Fixed3／Detour 继续作为默认规划方法。新的构造式研究仅保留建模建议，没有启动实现、场景生成或训练。

本目录是独立的补充材料。原 `rc3_rl_g23/`、原交付 ZIP、旧模型及历史结果均保持原样；阅读原报告时，应结合本目录的收尾补充说明。没有通过修改阈值、删去超时结果、选取有利种子或追加训练改变 G2.3 结论。

- `REPORT_CLOSEOUT.md`：五项主比较、质量与时限联合分布、种子差异及更新数口径。
- `PAPER_EXCERPT.md`：可纳入“方法筛选与机制诊断”的文字与表格；未直接修改论文文件。
- `STATUS.json`：已关闭的范围、默认方法与条件性后续研究状态。
- `outputs/closeout_receipt.json`：本地两脚本执行、输入哈希、冻结核对与新增工作范围。
- `outputs/results.json`、`outputs/model_budget.json`：从原 G2.3 ZIP 新解压副本复算的记录；相应 `.log` 保存执行结果。
- `inputs/rc3_g23_closeout_review/`：用户审查补充包的原样副本；`inputs/user_request.txt` 保存本轮要求。
- `closeout.py`：仅用标准库复现接收流程，不导入研究工程、不执行模型 pickle。

## 本次实际执行的工作

两个审查脚本均通过。结果统计与上传补充包一致；结果报告仅对绝对输入路径进行归一化，模型／资源报告逐项相同。上传补充包 11 个清单文件校验通过；G2.3 原交付 282 个清单文件及清单本身共 283 个成员，与当前工作区对应文件全部一致。另核对 8 个历史交付 ZIP 的前后哈希，均未变化。

本次新增训练、环境查询、模型前向、规划、性能测时、几何验证与 IK 检查均为 **0**。本次对保存路径复算 L、K、J 和分配一致性，几何碰撞与可达性仍引用原保存证书；没有重新运行原工程测试套件或历史分支。

## 复现

使用 Python 3.10 或以上，不需安装第三方依赖。将本目录与原 `rc3_g23_closeout_review.zip`、`rc3_rl_g23_delivery.zip` 和历史交付包放在既有工作区，运行：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B .\rc3_g23_closeout\closeout.py
```

也可显式指定原工作区：

```powershell
python -B path\to\rc3_g23_closeout\closeout.py --workspace path\to\workspace
```

脚本只在自身收尾目录内写入。原 G2.3 的新解压副本放在 `work/fresh_source/`，交付补充 ZIP 不重复收录此副本。首次运行会创建它；再次运行只接受内容一致的文件，不替换不同的输入。

如只需复算保存结果，可直接使用输入包 README 中的两个独立标准库命令；这无需恢复旧训练依赖，但不执行本目录额外的历史 ZIP 与工作区冻结核对。
