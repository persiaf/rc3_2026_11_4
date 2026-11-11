# 新尝试源码分类失败的已定位原因

本次唯一科学尝试因 `Unregistered actual dependency: _ops.py` 在 `check_loaded_modules` 中退出，尚未恢复轨迹、运行模型单元测试或更新Critic。不能在本任务中再启动另一科学尝试。

直接读取本机已安装源码（没有再次import Torch）确认：

- `D:/tools/anocaonda/envs/mobile_dual_arm_planning/Lib/site-packages/torch/_ops.py` 第1447–1456行定义 `_Ops(types.ModuleType)`；第1448行的类属性是 `__file__ = "_ops.py"`，构造器创建名为 `torch.ops` 的虚拟命名空间。
- 同目录 `torch/_classes.py` 第19–25行定义 `_Classes(types.ModuleType)`，第20行同样含虚拟类属性 `__file__ = "_classes.py"`。
- `torch/__init__.py` 第2355–2359行导入这些命名空间对象并以 `torch.ops`、`torch.classes` 键注册到 `sys.modules`。

失败代码对每个 `sys.modules` 项调用 `getattr(module, '__file__', None)`，随后将相对字符串 `"_ops.py"` 按当前工作目录解析成项目文件，并要求其出现在源码哈希目录。这把第三方虚拟命名空间的标签错误识别成真实项目源文件。**原因是来源分类器接口错误，不是缺失RC3依赖、磁盘停顿或Torch模型错误。** 这与原尝试已经确认的主机睡眠是两个不同事件。

新增候选 `provenance_revision.py` 使用模块实例的原始 `__dict__`，避免动态属性访问；真实项目Python源文件必须具有一致的 `ModuleSpec`、`SourceFileLoader`、origin与文件路径。合法来源仍经过准确SHA校验；真实未登记源文件、缺少或不一致的来源规格、字节变更均拒绝，不能用宽泛异常捕获跳过。

`test_provenance_revision.py` 提供9个仅标准库的合成工程用例，涵盖Torch风格的ops/classes虚拟标签、真实登记与未登记源文件、来源规格错误及SHA变化。没有导入Torch、模型、保存轨迹或优化器。测试是否实际通过，以单独登记后的工程执行收据为准。

**失败时绑定的 `provenance.py`、科学尝试源码/授权/账本和输出保持原字节。新候选没有被用于本次科学尝试，也没有授权新拟合。**
