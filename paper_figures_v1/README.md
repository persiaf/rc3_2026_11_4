# 冻结结果的论文图表包

按“场景与输出 → 共同约束必要性 → 百桌同场景对照”的顺序制作，来自已固定的JSON记录。**本轮没有训练、生成新路线、运行模型前向、测量性能或开展真实IK。** 原v4/v5/v5.1均保持不变。

## 直接使用

|内容|图文件|源脚本|
|---|---|---|
|图1：3×3任务、用臂、6站顺序与完整路径|`fig1_scene.pdf/svg/png`|`gen_fig1_scene.py`|
|图2：共同约束合成反例|`fig2_constraints.pdf/svg/png`|`gen_fig2_constraints.py`|
|图3：五方法百桌路线，同坐标|`fig3_routes100.pdf/svg/png`|`gen_fig3_routes100.py`|
|图3补充：固定中心区域放大|`fig3_zoom.pdf/svg/png`|`gen_fig3_zoom.py`|
|表1：五方法L/K/J|`TABLE_1_costs.tex/csv/md`|`gen_table1_costs.py`|

PDF为矢量且嵌入字体，适合排版；SVG保留可编辑文字；PNG为300 dpi，便于Word插图和预览。图片本身不放整图标题，图注见 [CAPTIONS.md](CAPTIONS.md)。[latex_includes.tex](latex_includes.tex) 可复制到已有中文论文工程，需要`graphicx`、`booktabs`及中文支持；未针对用户的具体论文模板编译。长图建议独占一页，保持可读字号；不要强行压缩成单栏小图。

本目录的 `index.html` 可离线查看全部图与成本表。宋体/Times为本次字体；SVG在不同系统查看时需对应字体。PDF已有字体嵌入，不依赖阅读端安装宋体。

## 可重复生成

本包自带必要的冻结源文件副本，因此不依赖原工程或神经网络包。只需Python、NumPy和Matplotlib；本机使用NumPy2.1.3、Matplotlib3.10系列。

```powershell
python -B paper_figures_v1/generate_all.py
```

从其他位置运行时，把命令路径改为本目录实际路径。绘图脚本只读取`sources/`副本，输出本目录图件；再次运行会覆盖图件，不修改原规划工程。若需保留当前成品，先复制整个图表目录。生成中文图时需要宋体及Times New Roman，或在`shared.py`明确配置可覆盖全部中文字符的替代字体；不要忽略缺字警告。

`sources/source_manifest.json`记录原相对路径、SHA与副本位置；`sources/display_manifest.json`保留固定选择规则和JSON指针。`*_metadata.json`记录实际绘制计划、成本与坐标范围。独立审查与冻结文件核对见 `FIGURE_REVIEW.md`、`review_checks.json`、`frozen_release_check.json`。

## 解释边界

- 百桌图固定取原列表第一项、所有方法rep0；Fixed3无修复，其余固定8次，不与软预算记录混用。该例Structured8成本高于Detour8，已如实保留。
- 图2只画保存的成功路径，矩阵仅展示已选的两个候选。其跨桌可达关系为合成逻辑，不代表真实机械臂工作空间。
- 图3放大区是x、y方向各中央20%，面积约为导航包络4%，由地图范围确定。
- J=L+0.75K是等效路程目标；单例和路线图不能证明统计优势、真实任务耗时降低或物理执行成功。完整论证衔接段落在CAPTIONS.md，需与原完整实验表配合使用。
- 此包是论文图件与说明材料，不是新的规划或模型版本；真实标定与IK仍按既定安排暂缓。
