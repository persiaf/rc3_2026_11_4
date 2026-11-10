# 失败工程版本的事后字节恢复

本目录仅恢复工程 R01/R02 的 `saved_inputs.py` 与 `test_saved_inputs.py`，供审查失败原因。它们是**失败发生后的恢复件，不是事前源码快照**，也不是完整历史工作区备份。恢复依据为已保留的 `SOURCE_BINDING_R01.json`、`SOURCE_BINDING_R02.json`；只将 SHA256 与当时记录完全一致的字节写入 `objects/`。

原工作源码、正在执行的 R03 源码及其绑定文件均未改动。恢复件没有执行、导入或用于训练。**不要用这里的旧版本覆盖当前源码或启动旧版本。** 原33/当前34来源兼容失败仍由原始工程失败账与后续R03修复记录解释，不因本次恢复改写。

| 原路径 | R01/R02共同 SHA256 | 当前恢复状态 |
|---|---|---|
| `rc3_conservative_refinement/saved_inputs.py` | `febd3d08e068b1efa6b479595dd0aef0f2236d74484c47549bfe5c73baa7b39d` | [完全匹配，12927字节](objects/febd3d08e068b1efa6b479595dd0aef0f2236d74484c47549bfe5c73baa7b39d) |
| `rc3_conservative_refinement/test_saved_inputs.py` | `914c6cc58e9eecd291af7156c21da50b18fed72b6314ae59bd37647f5b348575` | [完全匹配，2762字节](objects/914c6cc58e9eecd291af7156c21da50b18fed72b6314ae59bd37647f5b348575) |

两版本的这两个文件分别具有相同 hash，因此共享两个内容对象。按原绑定核验，不依赖推测的运行行为来认定恢复成功。第一次逆向移除新增测试时，模块前的空行数量未还原，测试文件候选未匹配，未保存该候选；原 `RESTORATION.json` 如实记录 MISSING。随后仅纠正已知模块分隔空行，取得原 hash，追加 `RESTORATION_COMPLETION.json`，未覆盖初次记录。

- [R01初次恢复记录](R01/RESTORATION.json)、[R01测试文件恢复完成记录](R01/RESTORATION_COMPLETION.json)
- [R02初次恢复记录](R02/RESTORATION.json)、[R02测试文件恢复完成记录](R02/RESTORATION_COMPLETION.json)
- [初次维护资源记录](MAINTENANCE_RESOURCE.json)

维护预算为30秒CPU。恢复脚本仅读取上述源码及两份绑定，执行内存文本逆变换与 SHA 校验，没有读取任务输入或增加科学操作。Windows `process_time` 对这几个短处理均报告0.0秒，属于计时分辨率限制，不表示物理计算耗时为零。
