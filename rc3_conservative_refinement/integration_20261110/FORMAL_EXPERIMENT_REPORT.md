# 保守长期规划DRL：正式保存结果汇总

实验：`conservative_paired_11622026_v1`；训练seed：`11622026`。实际状态：**STOPPED_REGISTERED_RULE**。

最后完整机会：128；完成评价终点：[64, 128]；已开始但未完成终点：无；未开始终点：[256]。

本文件仅汇总已终闭合、保存后审计PASS的结果，不生成路线、拟合模型或补跑阶段。科学结论待新的审查者判断；工程PASS和阶段完成不代表方法有效。

证据：[保存证据审计](runs/conservative_paired_11622026_v1/analysis/AUDIT.json)；[实际停止记录](runs/conservative_paired_11622026_v1/analysis/STOP_SUMMARY.json)；[冻结协议](PROTOCOL_DRAFT_R03.json)；[同一正式账本](runs/conservative_paired_11622026_v1/events.jsonl)。

协议SHA256：`a20eca451e7094bd16390ae5b973c85240a42337e4ef6ae42bd243b9a8cfeb0c`；源码绑定SHA256：`ea605e87a31c1c6fcf8a7090fbc2681202380fe1a6e6ae29de464529c1e701e7`；报告读取的账本末端：`ed4505690528de83d98e6686c89076be9005aff5b97131061f37fdccb00f63dc`。

失败请求：0；未闭合请求：0；已保存完成请求：2376。存在已尝试但未完成整套评价的终点：无；这类数据只保留证据，不补入完整终点表。

## 实际环境与配置

执行绑定环境：`{"deterministic_algorithms": true, "device": "cpu", "numpy": "2.2.6", "platform": "Windows-10-10.0.26300-SP0", "python": "3.10.20", "threads": 1, "torch": "2.12.1+cu126"}`。

设计中的独立暴露开发父布局数：3；实际完整终点评价包含的独立父布局数：3。最后进程stage_result中的自报输入计数：`{"S3_access": 0, "navigation_solves": 0, "original60_access": 0, "source_file_hashes": 7452, "task_generations": 0, "teacher_loads": 384, "view_loads": 1548}`。此字段不是跨暂停/恢复累计计数；SavedInputs在每个新进程重置。失败session没有完整收据时保留UNKNOWN，不推断为零。

准确SL256永久冻结；206维修正末层零初始化；ON/OFF仅新增mean/max距离摘要输入有别。每个机会由各组当前策略独立在线采样，Actor候选1步、Critic1步。注册预热配置为每组Critic最多32步、24条初始策略轨迹，实际执行以本报告资源/最终trainer计数为准。gamma=lambda=1，冻结H、完整覆盖mask和二维代理可达/安全检查。匹配继续监督使用原标签池，匹配机会/Adam/结构/guard预算，输入和FLOPs不相同。

## 完整路线成本

下表仅用每终点首次评价；先对同一真实父布局的各视图求均值，再对父布局等权平均。重复计时不增加质量样本数。raw指标是策略直接路线，H后指标是冻结H处理后的完整路线；原生Fixed3/Detour16的raw字段按接口保存为其原生最终路线，未获得其内部H前方案，不附加共同H。

| 终点 | cohort | 方法 | 父布局/视图 | raw J | raw L | raw K | H后 J | H后 L | H后 K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | independent_parent_exposed_development | Detour16 | 3/18 | 151.0669 | 124.2752 | 35.7222 | 151.0669 | 124.2752 | 35.7222 |
| 64 | independent_parent_exposed_development | Fixed3 | 3/18 | 152.9313 | 126.2646 | 35.5556 | 152.9313 | 126.2646 | 35.5556 |
| 64 | independent_parent_exposed_development | FrozenSL | 3/18 | 193.9445 | 158.1945 | 47.6667 | 156.6838 | 128.5588 | 37.5000 |
| 64 | independent_parent_exposed_development | OFF | 3/18 | 193.9445 | 158.1945 | 47.6667 | 156.6838 | 128.5588 | 37.5000 |
| 64 | independent_parent_exposed_development | ON | 3/18 | 193.9445 | 158.1945 | 47.6667 | 156.6838 | 128.5588 | 37.5000 |
| 64 | independent_parent_exposed_development | continued_SL | 3/18 | 193.9445 | 158.1945 | 47.6667 | 156.6838 | 128.5588 | 37.5000 |
| 64 | trained_parent_new_tasks_exposed_selection | Detour16 | 8/48 | 126.1241 | 100.7179 | 33.8750 | 126.1241 | 100.7179 | 33.8750 |
| 64 | trained_parent_new_tasks_exposed_selection | Fixed3 | 8/48 | 127.9517 | 102.7329 | 33.6250 | 127.9517 | 102.7329 | 33.6250 |
| 64 | trained_parent_new_tasks_exposed_selection | FrozenSL | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |
| 64 | trained_parent_new_tasks_exposed_selection | ON | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |
| 64 | trained_parent_new_tasks_exposed_selection | continued_SL | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |
| 128 | independent_parent_exposed_development | Detour16 | 3/18 | 151.0669 | 124.2752 | 35.7222 | 151.0669 | 124.2752 | 35.7222 |
| 128 | independent_parent_exposed_development | Fixed3 | 3/18 | 152.9313 | 126.2646 | 35.5556 | 152.9313 | 126.2646 | 35.5556 |
| 128 | independent_parent_exposed_development | FrozenSL | 3/18 | 193.9445 | 158.1945 | 47.6667 | 156.6838 | 128.5588 | 37.5000 |
| 128 | independent_parent_exposed_development | OFF | 3/18 | 193.9445 | 158.1945 | 47.6667 | 156.6838 | 128.5588 | 37.5000 |
| 128 | independent_parent_exposed_development | ON | 3/18 | 193.9445 | 158.1945 | 47.6667 | 156.6838 | 128.5588 | 37.5000 |
| 128 | independent_parent_exposed_development | continued_SL | 3/18 | 193.9451 | 158.1951 | 47.6667 | 156.6778 | 128.5528 | 37.5000 |
| 128 | trained_parent_new_tasks_exposed_selection | Detour16 | 8/48 | 126.1241 | 100.7179 | 33.8750 | 126.1241 | 100.7179 | 33.8750 |
| 128 | trained_parent_new_tasks_exposed_selection | Fixed3 | 8/48 | 127.9517 | 102.7329 | 33.6250 | 127.9517 | 102.7329 | 33.6250 |
| 128 | trained_parent_new_tasks_exposed_selection | FrozenSL | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |
| 128 | trained_parent_new_tasks_exposed_selection | ON | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |
| 128 | trained_parent_new_tasks_exposed_selection | continued_SL | 8/48 | 158.5106 | 125.4637 | 44.0625 | 130.8418 | 104.2950 | 35.3958 |

完整raw/H路线、导航节点、成本和原保存验证收据入口：[ALL_RAW_H_ROUTES.csv](runs/conservative_paired_11622026_v1/analysis/ALL_RAW_H_ROUTES.csv)。本报告不重新进行碰撞、双臂F&A或物理IK验证。

## 既有父簇比较

下表转录既有父布局聚类点估计与描述性95%区间，正值表示ON完整成本更低；没有重新bootstrap或据此自动宣布科学成功。这里只展示all窗口的J/raw_J；L/K和各尺寸细分保留在链接中。

| 终点 | cohort | ON对照 | 指标 | 父布局 | 相对节省 | 描述性95%区间 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | independent_parent_exposed_development | OFF | J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | independent_parent_exposed_development | OFF | raw_J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | independent_parent_exposed_development | FrozenSL | J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | independent_parent_exposed_development | FrozenSL | raw_J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | independent_parent_exposed_development | continued_SL | J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | independent_parent_exposed_development | continued_SL | raw_J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | independent_parent_exposed_development | Fixed3 | J | 3 | -2.3002% | [-3.6867%, -0.7225%] |
| 64 | independent_parent_exposed_development | Fixed3 | raw_J | 3 | -26.1507% | [-33.7414%, -21.8696%] |
| 64 | independent_parent_exposed_development | Detour16 | J | 3 | -3.5868% | [-4.7048%, -2.1320%] |
| 64 | independent_parent_exposed_development | Detour16 | raw_J | 3 | -27.7292% | [-35.0545%, -23.5728%] |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | raw_J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | trained_parent_new_tasks_exposed_selection | FrozenSL | J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | trained_parent_new_tasks_exposed_selection | FrozenSL | raw_J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | trained_parent_new_tasks_exposed_selection | continued_SL | J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | trained_parent_new_tasks_exposed_selection | continued_SL | raw_J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 64 | trained_parent_new_tasks_exposed_selection | Fixed3 | J | 8 | -2.2147% | [-3.4740%, -0.7122%] |
| 64 | trained_parent_new_tasks_exposed_selection | Fixed3 | raw_J | 8 | -23.6721% | [-26.3807%, -21.1614%] |
| 64 | trained_parent_new_tasks_exposed_selection | Detour16 | J | 8 | -3.7515% | [-4.8969%, -2.1907%] |
| 64 | trained_parent_new_tasks_exposed_selection | Detour16 | raw_J | 8 | -25.5221% | [-28.0085%, -23.1861%] |
| 128 | independent_parent_exposed_development | OFF | J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | independent_parent_exposed_development | OFF | raw_J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | independent_parent_exposed_development | FrozenSL | J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | independent_parent_exposed_development | FrozenSL | raw_J | 3 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | independent_parent_exposed_development | continued_SL | J | 3 | -0.0046% | [-0.0137%, 0.0000%] |
| 128 | independent_parent_exposed_development | continued_SL | raw_J | 3 | 0.0003% | [0.0000%, 0.0010%] |
| 128 | independent_parent_exposed_development | Fixed3 | J | 3 | -2.3002% | [-3.6867%, -0.7225%] |
| 128 | independent_parent_exposed_development | Fixed3 | raw_J | 3 | -26.1507% | [-33.7414%, -21.8696%] |
| 128 | independent_parent_exposed_development | Detour16 | J | 3 | -3.5868% | [-4.7048%, -2.1320%] |
| 128 | independent_parent_exposed_development | Detour16 | raw_J | 3 | -27.7292% | [-35.0545%, -23.5728%] |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | raw_J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | trained_parent_new_tasks_exposed_selection | FrozenSL | J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | trained_parent_new_tasks_exposed_selection | FrozenSL | raw_J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | trained_parent_new_tasks_exposed_selection | continued_SL | J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | trained_parent_new_tasks_exposed_selection | continued_SL | raw_J | 8 | 0.0000% | [0.0000%, 0.0000%] |
| 128 | trained_parent_new_tasks_exposed_selection | Fixed3 | J | 8 | -2.2147% | [-3.4740%, -0.7122%] |
| 128 | trained_parent_new_tasks_exposed_selection | Fixed3 | raw_J | 8 | -23.6721% | [-26.3807%, -21.1614%] |
| 128 | trained_parent_new_tasks_exposed_selection | Detour16 | J | 8 | -3.7515% | [-4.8969%, -2.1907%] |
| 128 | trained_parent_new_tasks_exposed_selection | Detour16 | raw_J | 8 | -25.5221% | [-28.0085%, -23.1861%] |

[全部父簇比较、各尺寸及CI](runs/conservative_paired_11622026_v1/analysis/QUALITY_CONTRASTS.json)。

## 完整请求与模型加载时间

下表合并同终点/cohort全部尺寸及实际注册重复，给出请求级描述性median/p95（秒）。计时包含保存输入和导航表/路径加载、来源验证、模型加载、provider准备、动态特征、Actor、H、导出和原独立验证；IO另列，Critic诊断不进入部署时间。导航表在本轮没有重新求解。模型逐请求冷加载，文件系统缓存未控制，不称为冷OS进程启动。原生方法无模型加载。

| 终点 | cohort | 方法 | 请求数 | 完整请求 median/p95 | 含落盘 median/p95 | 模型加载 median/p95 | 输入准备 median | 包装及IO median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | independent_parent_exposed_development | Detour16 | 18 | 0.285952/0.456913 | 0.299504/0.474631 | 0.000000/0.000000 | 0.191768 | 0.013665 |
| 64 | independent_parent_exposed_development | Fixed3 | 18 | 0.231428/0.285304 | 0.246228/0.303850 | 0.000000/0.000000 | 0.190247 | 0.013257 |
| 64 | independent_parent_exposed_development | FrozenSL | 18 | 0.282651/0.398766 | 0.295612/0.418867 | 0.006728/0.008242 | 0.191655 | 0.013643 |
| 64 | independent_parent_exposed_development | OFF | 18 | 0.323107/0.619399 | 0.339229/0.648846 | 0.011727/0.012695 | 0.190357 | 0.016905 |
| 64 | independent_parent_exposed_development | ON | 18 | 0.323178/0.629085 | 0.338631/0.655287 | 0.011712/0.012800 | 0.190812 | 0.017397 |
| 64 | independent_parent_exposed_development | continued_SL | 18 | 0.332996/0.623225 | 0.348802/0.647615 | 0.011669/0.013609 | 0.193478 | 0.017862 |
| 64 | trained_parent_new_tasks_exposed_selection | Detour16 | 48 | 0.339526/0.510893 | 0.355608/0.527977 | 0.000000/0.000000 | 0.243113 | 0.014847 |
| 64 | trained_parent_new_tasks_exposed_selection | Fixed3 | 48 | 0.280941/0.335158 | 0.296312/0.350284 | 0.000000/0.000000 | 0.246583 | 0.013811 |
| 64 | trained_parent_new_tasks_exposed_selection | FrozenSL | 48 | 0.330675/0.451109 | 0.349133/0.468902 | 0.006699/0.007428 | 0.243643 | 0.016324 |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | 48 | 0.383970/0.660955 | 0.399841/0.691537 | 0.011952/0.012889 | 0.245208 | 0.016790 |
| 64 | trained_parent_new_tasks_exposed_selection | ON | 48 | 0.380245/0.652114 | 0.399339/0.680249 | 0.011793/0.013102 | 0.242749 | 0.017375 |
| 64 | trained_parent_new_tasks_exposed_selection | continued_SL | 48 | 0.385944/0.655550 | 0.404891/0.683530 | 0.011632/0.013137 | 0.246306 | 0.016294 |
| 128 | independent_parent_exposed_development | Detour16 | 18 | 0.306263/0.487418 | 0.320860/0.510459 | 0.000000/0.000000 | 0.202269 | 0.015492 |
| 128 | independent_parent_exposed_development | Fixed3 | 18 | 0.247168/0.317623 | 0.261077/0.336763 | 0.000000/0.000000 | 0.204915 | 0.014624 |
| 128 | independent_parent_exposed_development | FrozenSL | 18 | 0.300563/0.442535 | 0.316250/0.464564 | 0.007667/0.008472 | 0.202580 | 0.016414 |
| 128 | independent_parent_exposed_development | OFF | 18 | 0.347764/0.669580 | 0.363242/0.700065 | 0.013652/0.015293 | 0.200267 | 0.016690 |
| 128 | independent_parent_exposed_development | ON | 18 | 0.344574/0.684012 | 0.360114/0.711841 | 0.013227/0.014832 | 0.210631 | 0.017269 |
| 128 | independent_parent_exposed_development | continued_SL | 18 | 0.366015/0.681630 | 0.382913/0.708808 | 0.013281/0.014258 | 0.199770 | 0.019651 |
| 128 | trained_parent_new_tasks_exposed_selection | Detour16 | 48 | 0.489449/0.830884 | 0.505672/0.850416 | 0.000000/0.000000 | 0.374726 | 0.016771 |
| 128 | trained_parent_new_tasks_exposed_selection | Fixed3 | 48 | 0.415764/0.548109 | 0.436378/0.571454 | 0.000000/0.000000 | 0.370381 | 0.017330 |
| 128 | trained_parent_new_tasks_exposed_selection | FrozenSL | 48 | 0.493986/0.738620 | 0.513338/0.768168 | 0.010516/0.013624 | 0.370999 | 0.018724 |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | 48 | 0.560676/1.112240 | 0.583327/1.153687 | 0.016853/0.025743 | 0.370275 | 0.020944 |
| 128 | trained_parent_new_tasks_exposed_selection | ON | 48 | 0.575091/1.162927 | 0.598205/1.201343 | 0.017503/0.028420 | 0.389680 | 0.022172 |
| 128 | trained_parent_new_tasks_exposed_selection | continued_SL | 48 | 0.593037/1.138185 | 0.616108/1.176489 | 0.017566/0.026806 | 0.367080 | 0.021284 |

[逐请求全计时](runs/conservative_paired_11622026_v1/analysis/ALL_REQUEST_TIMING.csv)；[按窗口尺寸计时](runs/conservative_paired_11622026_v1/analysis/TIMING_SUMMARY.csv)。

## Critic校准与优势

常数来自各组初始预热目标后固定：value使用归一化RTG的单一均值，L/K/H各有分量均值。H信用为对整条raw路线的有符号调整，不是物理剩余距离。下面先在每个父布局内按状态算误差，再父布局等权汇总；分量保持各自原单位。

| 终点 | cohort | 方法 | 分量 | 父布局 | 父均MSE | 常数父均MSE | 比值 | 父均RMSE | 父均MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | independent_parent_exposed_development | OFF | raw_remaining_K | 3 | 1577.236289 | 1151.495964 | 1.369728 | 38.542855 | 30.011561 |
| 128 | independent_parent_exposed_development | OFF | raw_remaining_L | 3 | 17920.547835 | 15732.212177 | 1.139099 | 131.811327 | 113.351744 |
| 128 | independent_parent_exposed_development | OFF | terminal_H_credit | 3 | 2276.388396 | 16519.176617 | 0.137803 | 45.524560 | 40.291316 |
| 128 | independent_parent_exposed_development | OFF | value | 3 | 2.100228 | 0.943264 | 2.226554 | 1.435345 | 1.211734 |
| 128 | independent_parent_exposed_development | ON | raw_remaining_K | 3 | 1577.236289 | 1151.495964 | 1.369728 | 38.542855 | 30.011561 |
| 128 | independent_parent_exposed_development | ON | raw_remaining_L | 3 | 17920.547835 | 15732.212177 | 1.139099 | 131.811327 | 113.351744 |
| 128 | independent_parent_exposed_development | ON | terminal_H_credit | 3 | 2276.388396 | 16519.176617 | 0.137803 | 45.524560 | 40.291316 |
| 128 | independent_parent_exposed_development | ON | value | 3 | 2.100228 | 0.943264 | 2.226554 | 1.435345 | 1.211734 |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | raw_remaining_K | 8 | 1216.834023 | 1158.166459 | 1.050656 | 33.621226 | 26.456278 |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | raw_remaining_L | 8 | 10267.835364 | 19930.228805 | 0.515189 | 100.303251 | 86.105605 |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | terminal_H_credit | 8 | 1099.483947 | 19386.068718 | 0.056715 | 32.093613 | 28.314113 |
| 128 | trained_parent_new_tasks_exposed_selection | OFF | value | 8 | 1.909265 | 0.865403 | 2.206216 | 1.331325 | 1.128225 |
| 128 | trained_parent_new_tasks_exposed_selection | ON | raw_remaining_K | 8 | 1216.834023 | 1158.166459 | 1.050656 | 33.621226 | 26.456278 |
| 128 | trained_parent_new_tasks_exposed_selection | ON | raw_remaining_L | 8 | 10267.835364 | 19930.228805 | 0.515189 | 100.303251 | 86.105605 |
| 128 | trained_parent_new_tasks_exposed_selection | ON | terminal_H_credit | 8 | 1099.483947 | 19386.068718 | 0.056715 | 32.093613 | 28.314113 |
| 128 | trained_parent_new_tasks_exposed_selection | ON | value | 8 | 1.909265 | 0.865403 | 2.206216 | 1.331325 | 1.128225 |
| 64 | independent_parent_exposed_development | OFF | raw_remaining_K | 3 | 1736.849200 | 1151.495964 | 1.508342 | 40.551056 | 32.365905 |
| 64 | independent_parent_exposed_development | OFF | raw_remaining_L | 3 | 19316.214267 | 15732.212177 | 1.227813 | 136.998150 | 119.394116 |
| 64 | independent_parent_exposed_development | OFF | terminal_H_credit | 3 | 2795.860968 | 16519.176617 | 0.169249 | 50.874851 | 45.946227 |
| 64 | independent_parent_exposed_development | OFF | value | 3 | 2.184843 | 0.943264 | 2.316259 | 1.463434 | 1.238922 |
| 64 | independent_parent_exposed_development | ON | raw_remaining_K | 3 | 1736.849200 | 1151.495964 | 1.508342 | 40.551056 | 32.365905 |
| 64 | independent_parent_exposed_development | ON | raw_remaining_L | 3 | 19316.214267 | 15732.212177 | 1.227813 | 136.998150 | 119.394116 |
| 64 | independent_parent_exposed_development | ON | terminal_H_credit | 3 | 2795.860968 | 16519.176617 | 0.169249 | 50.874851 | 45.946227 |
| 64 | independent_parent_exposed_development | ON | value | 3 | 2.184843 | 0.943264 | 2.316259 | 1.463434 | 1.238922 |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | raw_remaining_K | 8 | 1354.121576 | 1158.166459 | 1.169194 | 35.598221 | 28.750923 |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | raw_remaining_L | 8 | 11316.657419 | 19930.228805 | 0.567814 | 105.401034 | 92.037554 |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | terminal_H_credit | 8 | 1466.115815 | 19386.068718 | 0.075627 | 37.371439 | 33.682459 |
| 64 | trained_parent_new_tasks_exposed_selection | OFF | value | 8 | 2.000933 | 0.865403 | 2.312141 | 1.362116 | 1.159148 |
| 64 | trained_parent_new_tasks_exposed_selection | ON | raw_remaining_K | 8 | 1354.121576 | 1158.166459 | 1.169194 | 35.598221 | 28.750923 |
| 64 | trained_parent_new_tasks_exposed_selection | ON | raw_remaining_L | 8 | 11316.657419 | 19930.228805 | 0.567814 | 105.401034 | 92.037554 |
| 64 | trained_parent_new_tasks_exposed_selection | ON | terminal_H_credit | 8 | 1466.115815 | 19386.068718 | 0.075627 | 37.371439 | 33.682459 |
| 64 | trained_parent_new_tasks_exposed_selection | ON | value | 8 | 2.000933 | 0.865403 | 2.312141 | 1.362116 | 1.159148 |

停止规则使用的value MSE按所有评价状态池化，和上面的父布局等权数值有不同权重。比值分母按原代码下限1e-12；单终点“bad”仅表示该注册数值条件，最终停止原因以保存决策为准。

| 终点 | 方法 | 状态数 | value MSE | 固定warm常数 MSE | MSE比 | >1.25注册标记 |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | OFF | 2973 | 1.821666 | 0.755186 | 2.412208 | True |
| 64 | ON | 2973 | 1.821666 | 0.755186 | 2.412208 | True |
| 128 | OFF | 2973 | 1.745040 | 0.755186 | 2.310742 | True |
| 128 | ON | 2973 | 1.745040 | 0.755186 | 2.310742 | True |

[父布局等权校准](runs/conservative_paired_11622026_v1/analysis/CRITIC_CALIBRATION_PARENT_MACRO.csv)；[逐父布局校准](runs/conservative_paired_11622026_v1/analysis/CRITIC_CALIBRATION_PARENTS.csv)；[预测分位校准](runs/conservative_paired_11622026_v1/analysis/CRITIC_CALIBRATION_BINS.csv)；[实际动作、完整G、V_old与优势](runs/conservative_paired_11622026_v1/analysis/PER_STATE_ADVANTAGES.csv)。

## PPO优势数值诊断

这里只汇总actor_update_state=True的已审计PPO状态，Critic预热行全部排除。窗口尺寸由episode_id的机会/slot映射到冻结日程，不读取保存方案。G是按固定初始任务数归一化、包含冻结H终末信用的完整回报，V是收集时冻结的V_old；A为当次PPO的RMS缩放之前保存的G−V_old。std为状态池化总体标准差，RMS为未中心化均方根。

OFF实际机会1–128：60160个状态、768条在线轨迹；G/V/A均值分别为-0.733570/-0.004984/-0.728586，A RMS为1.187366，正/负比例为24.9202%/75.0798%。
ON实际机会1–128：60160个状态、768条在线轨迹；G/V/A均值分别为-0.733570/-0.004984/-0.728586，A RMS为1.187366，正/负比例为24.9202%/75.0798%。

| 方法 | 注册机会区间（实际） | 窗口 | 状态n | G均值 | V均值 | A均值 | A RMS | A std | A min/max | 正/负比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | 1-64（1–64） | full10x10 | 21714 | -0.6598 | -0.0014 | -0.6584 | 1.1163 | 0.9015 | -3.4821/1.6318 | 26.9319%/73.0681% |
| OFF | 1-64（1–64） | 4x3 | 881 | -1.7332 | -0.0133 | -1.7198 | 2.2702 | 1.4819 | -23.0789/0.9908 | 4.7673%/95.2327% |
| OFF | 1-64（1–64） | 6x6 | 2750 | -0.8974 | -0.0035 | -0.8939 | 1.2506 | 0.8747 | -3.9935/1.1887 | 17.8545%/82.1455% |
| OFF | 1-64（1–64） | 8x8 | 4358 | -0.7866 | -0.0022 | -0.7844 | 1.2031 | 0.9122 | -4.1048/1.4144 | 22.8316%/77.1684% |
| OFF | 65-128（65–128） | full10x10 | 22008 | -0.6655 | -0.0055 | -0.6600 | 1.1168 | 0.9008 | -3.7220/1.4419 | 27.0129%/72.9871% |
| OFF | 65-128（65–128） | 4x3 | 882 | -1.7423 | -0.0434 | -1.6990 | 2.1586 | 1.3316 | -12.0626/0.9871 | 4.6485%/95.3515% |
| OFF | 65-128（65–128） | 6x6 | 2652 | -0.9543 | -0.0152 | -0.9390 | 1.3209 | 0.9290 | -5.5216/1.2201 | 17.9487%/82.0513% |
| OFF | 65-128（65–128） | 8x8 | 4915 | -0.7463 | -0.0082 | -0.7381 | 1.1431 | 0.8728 | -3.5402/1.1737 | 23.4791%/76.5209% |
| ON | 1-64（1–64） | full10x10 | 21714 | -0.6598 | -0.0014 | -0.6584 | 1.1163 | 0.9015 | -3.4821/1.6318 | 26.9319%/73.0681% |
| ON | 1-64（1–64） | 4x3 | 881 | -1.7332 | -0.0133 | -1.7198 | 2.2702 | 1.4819 | -23.0789/0.9908 | 4.7673%/95.2327% |
| ON | 1-64（1–64） | 6x6 | 2750 | -0.8974 | -0.0035 | -0.8939 | 1.2506 | 0.8747 | -3.9935/1.1887 | 17.8545%/82.1455% |
| ON | 1-64（1–64） | 8x8 | 4358 | -0.7866 | -0.0022 | -0.7844 | 1.2031 | 0.9122 | -4.1048/1.4144 | 22.8316%/77.1684% |
| ON | 65-128（65–128） | full10x10 | 22008 | -0.6655 | -0.0055 | -0.6600 | 1.1168 | 0.9008 | -3.7220/1.4419 | 27.0129%/72.9871% |
| ON | 65-128（65–128） | 4x3 | 882 | -1.7423 | -0.0434 | -1.6990 | 2.1586 | 1.3316 | -12.0626/0.9871 | 4.6485%/95.3515% |
| ON | 65-128（65–128） | 6x6 | 2652 | -0.9543 | -0.0152 | -0.9390 | 1.3209 | 0.9290 | -5.5216/1.2201 | 17.9487%/82.0513% |
| ON | 65-128（65–128） | 8x8 | 4915 | -0.7463 | -0.0082 | -0.7381 | 1.1431 | 0.8728 | -3.5402/1.1737 | 23.4791%/76.5209% |

只有实际存在的区间/窗口才列出。状态及同一轨迹内动作相互相关；这些池化数值只诊断回报、Critic尺度和优势信号，不把n解释为独立样本数，不作显著性推断，也不据优势正负或大小宣称路线收益。零优势比例另保留在资源JSON。

## Guard与实际更新

每机会只计一次Actor候选，不能把6条轨迹的guard行数当作6次更新。拒绝和回滚都保留扣账。“最大”列涵盖所有实际候选，包括被拒绝候选；约束只针对这些已收集状态，不保证未知状态或整条贪心路线不变。

| 方法 | 已日志机会 | 接受 | 拒绝 | 精确回滚 | 最大step KL | 最大概率变化 | 最大base TV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ON | 128 | 128 | 0 | 0 | 0.000000000 | 0.000003057 | 0.000034864 |
| OFF | 128 | 128 | 0 | 0 | 0.000000000 | 0.000003064 | 0.000033086 |
| continued_SL | 128 | 128 | 0 | 0 | 0.000000001 | 0.000015468 | 0.000692452 |

FrozenSL、Fixed3与Detour16没有Actor优化器机会。异常机会的尝试/调用仍在资源账与最终trainer计数中，不把缺少成功完成事件的机会补成接受或拒绝。

[逐机会/轨迹guard与梯度日志](runs/conservative_paired_11622026_v1/analysis/GUARD_UPDATES.csv)。

## 实际停止与未执行部分

保存停止原因：`["Two endpoints: neither ON nor OFF improves the frozen supervised start", "Two endpoints: Critic MSE exceeds fixed warm constant by25%"]`。

保存终点决策：`[{"critic_bad": [true, true], "event": "endpoint_decision", "no_gain": true, "opportunity": 64, "previous": "ed8de7faa94fbcde28249298987e0a55be5f3c072fa687c9db5f40b995674a70", "sequence": 5250, "sha256": "2b213833a98ea8a8fa518cecc3d2af5aee349174764c3a360f8a6caebbd4b7bd", "stop_reasons": []}, {"critic_bad": [true, true], "event": "endpoint_decision", "no_gain": true, "opportunity": 128, "previous": "63a09a67fc07677f3327eeaca375c696edf15a612dbff739d30a8de061e45380", "sequence": 10159, "sha256": "52f57b2696c51bc038f4475b944281917fe3d54ba43ba26087b5cd3394303070", "stop_reasons": ["Two endpoints: neither ON nor OFF improves the frozen supervised start", "Two endpoints: Critic MSE exceeds fixed warm constant by25%"]}]`。

完整评价终点：[64, 128]。已开始未完成终点：无；未开始终点：[256]；不补未执行训练、评价或重复。未安排其他训练seed、新独立保留父布局、原60封存任务或旧S3补跑。

## 资源账

下表取同一正式账本的持久debit和所有session_close精确墙钟之和；包含暂停前后训练/评价session与保存后分析session。工程R01/R02/R03另账，不加进本正式训练计数。报告工具自身只做维护汇总，其耗时另记，未改正式账本或重置预算。

| 资源 | 实际累计 | 上限 |
| --- | --- | --- |
| actor_optimizer_attempts | 384 | 768 |
| critic_optimizer_attempts | 320 | 576 |
| requests | 2376 | 5100 |
| teacher_episodes | 768 | 1536 |
| wall_s | 3733.860712 | 18000 |

| session | 范围 | 状态 | 精确秒数 |
| --- | --- | --- | --- |
| 1→5254 | formal_training_evaluation | PAUSED_CLEAN_BOUNDARY | 2414.143536 |
| 5255→10162 | formal_training_evaluation | STOPPED_REGISTERED_RULE | 1298.798396 |
| 10163→10166 | saved_only_analysis | ANALYSIS_COMPLETE | 20.918780 |

各正式session的SavedInputs自报输入计数如下。它们是进程内维护计数，和持久资源预算有不同含义；真实请求、教师重放、Actor/Critic优化器预算始终以上述同一Ledger的累计debit为准。这里只对有完整收据的纯计数求和；任一session缺收据时，跨进程累计保持UNKNOWN，并保留已知session小计。

| session | stage_result收据 | view_loads | teacher_loads | source_file_hashes | 输入计数状态 |
| --- | --- | --- | --- | --- | --- |
| 1→5254 | 5252 | 1596 | 384 | 7690 | 完整session自报 |
| 5255→10162 | 10161 | 1548 | 384 | 7452 | 完整session自报 |

跨session输入计数汇总（自报，不是资源预算）：`{"counts": {"S3_access": 0, "navigation_solves": 0, "original60_access": 0, "source_file_hashes": 15142, "task_generations": 0, "teacher_loads": 768, "view_loads": 3144}, "known_complete_session_subtotals": {"S3_access": 0, "navigation_solves": 0, "original60_access": 0, "source_file_hashes": 15142, "task_generations": 0, "teacher_loads": 768, "view_loads": 3144}, "missing_or_incomplete_session_open_sequences": [], "scope": "Sum of session-local input receipts only; persistent query/teacher/optimizer budgets use ledger debits", "status": "COMPLETE_SELF_REPORTED_SESSION_SUM"}`。

[机器可读正式资源账](FORMAL_RESOURCE_ACCOUNTING.json)；工程分账见[工程累计资源](RESOURCE_ACCOUNTING.json)。

## 结论边界

本配置只有一个训练seed，设计的独立父布局仅3个且都已暴露为开发数据；实际完成评价中的独立父布局数为3。窗口和计时重复不形成新的独立样本。所有有效性证据为二维导航/任务覆盖/双臂代理可达和安全约束，尚未完成真实双臂IK、关节轨迹或实机认证。是否稳定改善SL和继续监督、是否体现完整成本与时间价值，交由新的审查者对这些保存结果判断；本工具不输出成功/失败研究结论。
