# D5修订图表独立审查

结论：**PASS_WITH_MINOR_PRESENTATION_REVISIONS**。三幅图的数据、比较方向、固定终点和路线选例规则均与保存结果一致；可作本轮单seed开发负结果的图证。下列小修是图注自包含与单位呈现问题，不改变科学结论，不需要新实验。`claim_supported=no`及写作provisional/no paper claim audit保持；本次图表审查由独立代理实际完成，非pending external review。

## 数据与复现核对

- 逐一查看三张PNG。图源8个文件的当前SHA与 `FIGURE_SOURCE_BINDING.json` 一致，Figure Plan SHA也一致。每图有独立生成脚本，读取固定JSON/CSV或保存折线，没有硬编码性能值或执行模型的绘图依赖。`FIGURE_RENDER_RECEIPT.json` 记录真实全进程11.328000秒、0前向/优化/规划；本角色没有重新生成图。
- `gen_endpoints.py` 绘制0/1024/2048/4096，零点共用准确SL256/FrozenSL；自行比较引用summary的raw/H父等权均值，均与先前从594个保存成员重算的数值一致。左右raw与H分开、较低为优、线图适合固定终点轨迹；没有以最佳点替代4096主终点。当前图聚焦PG/SL/FrozenSL，不展示Fixed3与Detour16是图的范围选择，不能替代报告中仍保留的完整原生对照表。
- `gen_parent_gains.py` 的22个主4096父配对HJ值逐一与本角色独立重算一致。先父内6请求均值，再算相对收益，11父等权；正号才改善，灰实线零点、虚线总体均值、点线原1%/0.5%门槛含义一致。SEL8/DEV3都有标签，未构造66任务的伪独立区间。该图能直接说明10/11父对FrozenSL或SL-U退化，符合P1否定。
- 自行从保存的66条PG/FrozenSL HJ对重算分类，容差固定1e-8，得到改善22、持平0、退化44；字典序第一项为 `development_000`、null、`development_001`，与 `ROUTE_EXAMPLE_SELECTION.json` 和实际图相符。空类上下两面板均为NOT_AVAILABLE，没有改阈值或选“接近持平”顶替。两个选中案例来自同一DEV父，是预定取字典序首项的结果，不能当两个独立父或代表性抽样。
- 三个PDF均含字体描述且未发现Image XObject；结合直接使用matplotlib矢量线/散点与文字的源代码，符合矢量PDF交付要求。PNG为300dpi图源的可视预览。

## 视觉质量

统一serif字体、白底、去顶部/右部边框，无装饰性元素、无图内标题；legend未遮住主要数据。端点图用实线圆点/虚线方点，配对图分面并用圆/方点，路线图用实/虚线，因此区别不只依赖蓝/橙颜色；均值虚线、门槛点线和零线也可在灰度下区分。源代码为7.1英寸全宽，基础字号10，主要tick9；路线tick及legend7，按当前figure*全宽呈现仍可读，不宜再缩成单栏。这里是视觉与编码检查，未进行另一次打印或模拟色盲渲染。

路线图轴有m单位，上raw下after-H、入口/出口符号与图注一致。闭端点案例的入口和出口重合属于实际输入，不应为了视觉分开。图只重画导航折线，没有障碍/覆盖/机械臂扫掠证据，因此可以说明保存路线的形状差异，不能用它单独论证碰撞安全、覆盖或具体改进机制。图注已经限制为说明性示例和非关节轨迹认证。

## 两项小修

1. **成本单位和英文聚合说明不充分。** `figures/gen_endpoints.py` 两个成本轴只写cost J，`figures/latex_includes.tex`第一图英文caption也未定义 `J=L+0.75K` 或单位，并省略中文CAPTIONS已写明的“先父内6请求平均、再11父等权”。建议在交付英文caption中补全这三项，明确J为以米计的等效成本；若日后维护轴标签，可使用 `Cost J (m equivalent)`。当前数字和科学比较均正确，不要求为此重跑评价。
2. **路线英文caption仍需独立识别本次案例。** `figures/latex_includes.tex`第三图没有明确4096终点、FrozenSL对D5-PG、1e-8容差、22/0/44实际类别数或两个case ID。中文caption含部分选择规则，但也没有本次选中ID/类别数。建议统一补入，避免读者必须打开外部JSON才能判断例图来源。可以直接写："At update4096, the66 exposed cases contain22 improved,0 tied and44 degraded after-H J comparisons of D5-PG withFrozenSL (absolute tie tolerance1e-8). The lexicographically first cases are development_000 and development_001; the absent tied category remains unavailable." 保留“全部66案例统计决定结论”和“不新增规划/不认证机械臂”的限制。

这两项是呈现修订，不是要求补基线、重训、重规划或新增统计门槛。第二图caption已足够说明父配对口径、两种控制和开发数据暴露；可将原始1%/0.5%具体数值写入英文caption加强自包含，但图上有刻度与门槛线，非阻断项。

## 方法示意与状态消歧

`METHOD_DIAGRAM.md`明确冻结39936标量、训练25152标量、原mask、完整回报、同问题FrozenSL固定参考以及独立PG/SL模型状态；共享结构框的解释不会把两条流写成共享更新权重。模型约束与真实IK/关节认证的界限明确。它没有把未运行的Full-PG或SL-C画成完成的实验。

当前 `STATUS.json` 的 `scientific_result=P1_COMPLETE_NO_DEVELOPMENT_SIGNAL`、`closed=true`、无active_job/phase、`current_error=null`与最终P1结果相符；启动P1的旧next_action和中断error移入 `historical_interruption_fields`，当前next_action改为保存结果审查/发布并明确禁止后续训练/规划/保留评价。P2/P3/P4/P5_confirmation/P5_final均为NOT_RUN，`new_automatic_experiments=false`，旧v1技术停止身份仍保留。这是合适的报告消歧，未把科学否定写成技术未执行。原始 `STATUS_WORKER_FINAL_RAW.json` 保留是必要的证据保全；本角色将两文件的历史字段逐项对照，未更改任何作者状态文件。

原RESULT_TO_CLAIM审查提及的状态歧义已经解决；其他资源细项UNKNOWN、SL-C未匹配、独立层/部署未运行及无paper claim audit的限制继续有效。路由仍为P6_CLOSEOUT，`next_experiments_needed=[]`。
