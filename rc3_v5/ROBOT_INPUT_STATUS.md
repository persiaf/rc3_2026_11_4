# AlohaMini1 资产与真实配置入口状态（v5）

2026-11-03：**固定版本的几何资产已经获取并校验；真实运行配置仍为 `blocked_inputs`。** 本轮没有制造真实可达矩阵 F，没有把缺参数解释为机械臂物理不可行，也没有运行真实 IK、机械臂碰撞或轨迹实验。旧 v4 文件保持冻结。

## 已取得和核对的事实

主身份固定为 [lemon198/AlohaMini-Simulation 的 URDF](https://github.com/lemon198/AlohaMini-Simulation/blob/83f06e734653f2c267d2c30421267024dd4e1fae/src/Aloha/urdf/Aloha.urdf)，提交 `83f06e734653f2c267d2c30421267024dd4e1fae`。对照为 [liyiteng/AlohaMini 的 AlohaMini1 URDF](https://github.com/liyiteng/AlohaMini/blob/17c6a98d79881a45ab869c1f392ed89c0723a298/AlohaMini1/simulation/src/Aloha/urdf/Aloha.urdf)，提交 `17c6a98d79881a45ab869c1f392ed89c0723a298`。这里没有引用 AlohaMini2 的几何或控制参数。

两份 URDF 已分别保存，去掉 CRLF/LF 换行差异后内容完全相同。两文件的原始字节分别为 26,646 和 25,581 bytes，SHA-256 分别为：

```text
simulation: 0c968794bb1ccd90ea80f1ad6195c9fcfef9cf9df2a3629f809f311f2d26b3ad
reference:  92175fefa161717d4e7f845096fde9513e5ad27e24d06beca5af4799abacae79
```

两份 URDF 的 Git blob SHA-1 与固定提交目录接口返回值一致。所有 **19 个** visual/collision 引用 mesh 已选择性下载，合计 **15,108,946 bytes**；每个文件同时记录 SHA-256、字节长度和 Git blob SHA-1，且两仓库相应 mesh 的 Git blob SHA-1 全部相同。不需要拉取完整仓库。URI `package://Aloha/...` 在本地显式映射到 `robot_assets/simulation/Aloha/`。

URDF 共 19 links、18 joints：2 fixed、3 continuous（车轮）、12 revolute、1 prismatic。现在可以直接从原文确认：`vertical_move` 与左右 `joint1`–`joint6` 共 **13 个有界可动关节**的 `lower`、`upper`、`effort`、`velocity` 均为零。这些是不可用于真实运动规划的导出值。三车轮 continuous joints 未提供 limit 元素。详细轴、origin、树和全部引用在 `robot_assets/asset_inventory.json`。

```text
base_link
  wheel1 -> wheel1                 [continuous]
  wheel2 -> wheel2                 [continuous]
  wheel3 -> wheel3                 [continuous]
  vertical_move -> vertical_link   [prismatic; bounds=0,0]
    left_base -> left_base         [fixed]
      left_joint1 -> left_link1    [revolute; bounds=0,0]
        left_joint2 -> left_link2
          left_joint3 -> left_link3
            left_joint4 -> left_link4
              left_joint5 -> left_link5
                left_joint6 -> left_link6
    right_base -> right_base       [fixed]
      right_joint1 -> right_link1  [revolute; bounds=0,0]
        right_joint2 -> right_link2
          right_joint3 -> right_link3
            right_joint4 -> right_link4
              right_joint5 -> right_link5
                right_joint6 -> right_link6
```

最后两条链未重复标注的 joint2–joint6 也均为 revolute、bounds=0,0。按用户已有机器人定义，每臂 joint1–joint5 属于运动链，joint6 为夹爪；配置模板保留角色待确认，不把 joint6 自动算作第六个姿态自由度。

## 运行预检

在项目根目录执行（Python 标准库，无神经网络或机器人仿真依赖）：

```powershell
python -m rc3_v5.robot_input inventory --manifest rc3_v5/robot_assets/manifest.json --output rc3_v5/robot_assets/asset_inventory.json
python -m rc3_v5.robot_input check --manifest rc3_v5/robot_assets/manifest.json --config rc3_v5/robot_assets/runtime_config.template.json --output rc3_v5/robot_assets/preflight_report.json
python -m unittest rc3_v5.test_robot_input -v
```

第一条应为 `assets_verified`、退出码 0。第二条对未填模板应为 `blocked_inputs`、退出码 2；当前有 135 个细项 blocker，表示未填字段与其派生一致性要求，不表示 135 种物理缺陷。`physical_infeasibility_claimed`、`physical_ik_verified` 和 `collision_verified` 始终为 false。退出码 2 是预期配置阻断，不是 IK 求解失败。

模板需要重新生成时运行：

```powershell
python -m rc3_v5.robot_input template --manifest rc3_v5/robot_assets/manifest.json --output rc3_v5/robot_assets/runtime_config.template.json
```

不要对已填的运行配置执行覆盖式模板生成。复制模板为另一个 JSON 后补齐；原始 URDF 保持不改，校准有效限位以配置覆盖。

自有版本资产可以用 `manifest` 子命令建立清单，`--output` 所在目录就是资产根；`--package Aloha=相对包路径` 提供 URI 映射。`--repository`、完整 40 位 `--commit`、`--repository-path` 必须显式给出。新版本须重建配置身份与来源绑定，不允许沿用旧 SHA。

## 未填配置与证据要求

`runtime_config.template.json` 中所有物理运行值为 null，`source` 为 null；没有从旧数值模型或另一机器人拷贝限位。完整预检需要：

1. 每个非 fixed 关节的角色、有效上下界（continuous 无需有界位置）、正速度、编码器零偏与方向。角度单位 rad，直线位置 m；`zero_offset` 表示编码器到 URDF 的偏置，**不**强制偏置本身位于关节运动区间。
2. 左右 TCP 的父 link 及 xyz/rpy 变换；检查其确实处于相应臂链上且在升降关节下游。
3. 升降 joint、零位参考绝对高度及地面到 base 的标定变换。名义 600 mm 行程不能代替这些值。
4. 每侧夹爪 joint 与至少两个 `[joint_position, opening_m]` 样本；开口非负、单调、非恒定，样本在有效 joint 范围内。
5. 每侧不工作臂/夹爪 park 位姿，以及含全部非 fixed 关节的 transport 位姿。校验关节集合和位置限位；**没有**宣称这些姿态已经通过碰撞检查。
6. 碰撞裕量、显式豁免 link 对，以及自碰撞、环境、闲置臂、接近路径、升降运动检查开关。配置要求明确启用这些下游检查；这不代表检查已经执行。URDF 必须为每个 link 提供 collision geometry。
7. 任务位置/方向容差、接近距离。

每个配置块引用 `sources` 中的一项。来源包含相对配置目录的本地证据文件、该文件 SHA-256、与主 manifest 完全一致的 identity、非空测量/版本说明；预检会重新读取并核对哈希。证据文件可以放在配置目录的子目录内。这能发现版本错配或文件变化，**不能替代对标定真实性、精度和适用性的审核**。关节零限位、缺失配置、错误版本、缺 mesh、hash 变化等均输出明确字段 blocker，不转成全零 F。

若未来配置全部通过，状态是 `inputs_complete`、`ready_for_service_generation=true`，含义仅为“输入足以进入下游服务配置生成”，不是执行认证。仍需真实 IK witness、残差、独立碰撞/接近/升降检查与实验记录，才能接入真实 F；本模块不声称动力学/力矩可行性。

## 来源与文件

| 文件/目录 | 内容 |
|---|---|
| `robot_assets/manifest.json` | 主身份、URDF/mesh 相对路径与哈希、冻结 URL |
| `robot_assets/simulation/Aloha/` | 原始 URDF、19 个 STL、package.xml |
| `robot_assets/official_reference/` | 对照 URDF、package.xml、上游根 LICENSE |
| `robot_assets/source_comparison.json` | 换行差异、mesh 同一性及许可声明观察 |
| `robot_assets/upstream*_listings.json` | GitHub 固定提交目录 API 原始清单 |
| `robot_assets/source_fetch.json`, `supplementary_fetch.json` | GitHub 插件返回的文本与 URL，保留原始换行 |
| `robot_assets/mesh_fetch_receipt.json` | 逐 mesh 下载、字节与 Git blob 校验回执 |
| `robot_assets/download_meshes.py` | 仅重试固定列表 mesh，已经存在且通过 hash 的文件复用 |
| `robot_assets/runtime_config.template.json` | 未填模板 |
| `robot_assets/preflight_report.json` | 对真实资产和未填模板的实际预检报告 |

许可观察：固定仿真仓库根目录没有 LICENSE 文件，两份 `package.xml` 均声明 BSD；对照官方仓库根 [LICENSE](https://github.com/liyiteng/AlohaMini/blob/17c6a98d79881a45ab869c1f392ed89c0723a298/LICENSE) 是 Apache-2.0，已原样保存。这里保留上游不同层级声明，不擅自认定其中一份已经解决另一份的归属差异。后续公开再分发资产时需核对适用授权。

15 项测试均为合成 URDF/合成配置的校验器测试，覆盖完整输入、缺资产、资产篡改、URI 重定向、目录越界、原 URDF 零限位与有效配置覆盖、有效配置零限位、版本/来源 SHA 错配、未填模板、停车限位、TCP 链、碰撞开关，以及 CLI/API 异常输入。它们不构成 AlohaMini1 真实 IK、碰撞、工作空间或规划性能证据。
