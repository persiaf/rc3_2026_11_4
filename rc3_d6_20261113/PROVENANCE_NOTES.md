# D6来源与身份

准确基准为539ead5ff52cb110a91736d6550cb988171d4d9b；本次科学身份在SCIENTIFIC_IDENTITY.json，执行授权在AUTHORIZATION.json，初始化是合法base-only SL256。原D54096权重仅归档，本轮不加载为初始化。

SOURCE_BINDING继承已核验D5 R04B的原源码/数据SHA字典及真实Torch环境，因此部分历史envelope字段仍引用旧数据生产/恢复版本。这些是来源记录，不是沿用旧实验执行许可；D6_artifacts、D6新源码SHA、run_id/scientific_identity以及新AUTHORIZATION精确标识本次执行。原R04/M1分类器的integration字段也只表示复用的来源检查实现，真实角色由本轮phase、stream、请求run_id和Actor头共同绑定。

Actor内部role仍使用原受验网络类的D5-PG/D5-SL结构配置；科学A/B/SL由新export的rc3-d6-actor-v1头中的stream、identity、role、base与tensor_sha256联合约束。B不冒充旧D5模型，工程/诊断权重不能通过本轮加载条件。

首个新计算前，原32768轨迹压缩归档真实SHA与基准Git LFS OID相等，逐条再校验保存raw SHA和真实问题身份。父F/A、任务/候选、导航与窗口端点来自已有来源。旧教师及FrozenSL参考在原目录只读，通过manifest SHA、problem/H/SL/cost键绑定。PG不读取教师动作；新A/B只用当前各自Actor的实际在线采样。

实际请求计时截止于压缩归档完成，后续指针/事件/资源日志尾部在总墙钟；模型网格加载发生在请求循环外，计入作业总账。四类内部搜索inclusive时间存在嵌套，不直接相加。没有保存的其他细分（例如原生DP内层）不填零。旧D5消费另以原账hash保留，本轮7200秒从保守任务起点累计，恢复不清零。
