# 分阶段授权与恢复链

协议批准和阶段执行授权是两项独立决定。本任务仅准备文件和工程适配，未取得任何科学执行权限。

## 稳定身份与逐阶段证据

稳定科学身份绑定准确protocol、registry、全部runtime源码、H、环境binding、experiment_id及同一协议批准收据。逐阶段authorization SHA不进入教师池/权重的稳定身份，而是保存到不可覆盖的AUTHORIZATION、START、COMPLETE/FAILED及原始授权来源记录。

S1-A完成后，S1-B授权引用已知的S1-A完成收据hash。三个S1-B种子使用同一明确seed范围的授权、按固定顺序分别启动；同阶段前seed完成hash自动加到运行收据。全部S1-B完成后，可以使用新的S2授权文件：显式引用教师池及三个S1-B完成收据，核同一科学身份、每seed固定256终点和完整源产物。它不要求重跑S1，也不继承S1 Adam动量。

同一阶段/seed恢复必须使用原authorization。更换授权不能重复已完成阶段、重开失败阶段、修改种子、绕过前seed顺序或重置耗时/预算。实验目录唯一固定为本目录`runs/<experiment_id>`，不能换nested子目录开第二条同名链。

## 教师与权重来源

教师池须有完整256个固定spec、native计划、H终局、验证、任务集合、动作及费用记录，pool index逐文件hash与稳定身份一致。不能将不完整index、变动label或无验证计划当可用教师。

S1-B的untrained与固定256终点均须在本实验本seed路径内，有匹配的protocol/source/H/seed/phase/cursor/weightversion、原始pt内容SHA、完整性sidecar与更新关闭记录。S2逐seed从对应S1终点生成冻结SL、RL、继续SL分叉；RL/继续SL fresh Adam。实际tensor、动量、RNG等仍由原训练器的严格loader核验。路径、metadata守卫和旧真实梯度/恢复验收共同构成依据；本轮不生成正式模型。

工程checkpoint、坏恢复夹具、早期selected模型、其他实验或其他seed权重均不允许作正式来源。UNIT_ONLY回归字节文件不是可恢复模型，也不会产生生产ExecutionPermit。

## 中断与资源

S1-A第一次准备/查询前写STARTED；任意partial教师阶段按不完整停止，禁止隐式续查/从零重启，必须另行审计处理已付费资源。S1-B/S2只支持完整更新边界及优雅SIGINT后的同授权恢复；已有脏日志、缺关闭记录或坏checkpoint直接拒绝，不重放已收费调用。

恢复读取已用调用、游标、随机状态、阶段累计clock与checkpoint IO关闭记录，再计本次授权/来源校验及加载开销。下游新授权不重置上游账，每阶段资源分开留存；S2内RL/继续SL/开发共享一个每seed7200秒边界，不能把嵌套耗时相加或把准备当免费成本。末次产物hash之后再核预算并写完成收据；失败保留部分资源与原因，不能写成科学成功。

正式入口不改变原v2指针；需要用户对候选hash明确批准，并对指定stage/seed提供授权。批准后仍应新建活动执行记录引用准确hash，保留本次候选和旧设计原字节，而非通过改写候选状态改变其已批准hash。
