# 持久化最小修复与验收

**工程GO。** 原IO故障、Windows启动器兼容问题及新科学输入停止条件分别保留，不混为同一故障。

原 `HEAD.pending → HEAD.json` 替换访问拒绝由有界同事件重试处理：最多8次，只重试锚替换，不重复追加事件；检查pending文件身份/内容、旧HEAD和完整链哈希。独立错误收据不递归依赖失效Journal。OS单写入者租约及固定外层激活标记拒绝并发和改目录清账；旧180文件/原HEAD/pending从未改写。

父进程在Journal解析/收尾前保存独立退出钟，失败关闭后仍有FINAL/SEAL或独立故障收据。正常每请求计算方法、日志计量和计时边界保持，新增检查与重试属于计费IO。

17项零科学回归覆盖瞬态/持续替换失败、残留/被篡改pending、断链、孤立结果、父收尾/早期打开失败、重复恢复、预算不退款/不可清零、Windows真实venv子进程租约、缺失计时不填补及合成原统计一致性。最新套件PASS；工程suite累计82.03700110001955秒，科学函数入口计数均0。

首版修复启动遇到真实Windows venv redirector的父PID兼容问题，0个科学请求，完整9.802301499992609秒收费证据保留。source02使用实际持有的监督者OS租约而非直接PPID，独立真实子进程测试通过。source01与源码提交81290f43及原收据保持，source02及原v3/训练模型身份分开绑定。

实际source02续段新增1896预扣/1895成功/1全父连通服务前置检查失败。出现6份瞬态锚重试收据，对应6个不同事件，均成功，未重复追加；新链46023事件，HEAD齐、无新pending，错误与父FINAL正常闭合。后续停止发生在S3_042输入准备，早于模型/native/H，不是日志修复失效。

最终源码绑定：`767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea`。依据：[测试](ENGINEERING_VERIFICATION_02.json)、[源码独立审查](INDEPENDENT_REVIEW_02.md)、[恢复决定](RECOVERY_REVISION_02.json)、[实际报告](EXPERIMENT_REPORT.md)、[资源审计](results/RECOVERY_AUDIT.json)。
