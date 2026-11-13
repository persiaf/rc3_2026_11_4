# 研究内容三 M2正式科学执行

准确协议和源码由用户基于`9c878928e7f619bce25ddd00792757431ca705a1`批准。稳定科学ID：`relation_navcover_m2_v1_20261113`。本目录是新授权、状态和报告入口；原M1及冻结协议保留原字节。

- [用户授权范围](USER_AUTHORIZATION.md)
- [实施审查GO](IMPLEMENTATION_REVIEW.md) · [准入字节核对](ADMISSION_CHECK.json)
- [独立协议批准](authorizations/PROTOCOL_APPROVAL.json) · [M2a阶段授权](authorizations/M2A_AUTHORIZATION.json)
- [当前执行状态](CURRENT_STAGE_STATUS.json)
- [准确冻结协议](../rc3_relation_drl_20261113/M2_PROTOCOL.json)

M2a首先执行SL512。只有完整成功并验证真实COMPLETE后，才按本轮条件授权创建独立M2b收据并执行固定配对日程。任何原协议停止条件触发即停止；不增加种子、消融、训练池、预算或后续阶段。
