# 恢复模型及来源

- [SL512_RECOVERED实际模型](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/recovery_v1_20261114/SL512_RECOVERED.export.pt)：文件SHA256 `fca6974737e97d045977dbb4d8e5bf2a0511c3e38bab21690434497bd13223e9`，模型张量SHA256 `e32121ee54985863aecd2b281534b546411567f79cf76c05c0fd916f87ed2041`。
- [完整恢复收据](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/recovery_v1_20261114/RECOVERY_COMPLETE.json)：SHA256 `3fc63dce0c19d637c5c2a3f325045a62201c9f038ee23d250cd56a619d28be6f`。
- [最终checkpoint（模型/Adam/RNG）](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/recovery_v1_20261114/current_slot0.pt)：SHA256 `7d46be504a74478a06c52666c7390069e4aa4c5696c599cacd888e2ee8a9457e`，游标512。
- [原SL238及恢复日程绑定](../rc3_relation_recovery_20261114/RECOVERY_PROTOCOL_DRAFT.json)。
- [实际累计资源](RESOURCE_ACCOUNTING.json) · [逐请求CSV](analysis/TEACHER_REQUESTS.csv) · [逐更新CSV](analysis/TRAINING_UPDATES.csv)。
- [M2b初始化资格映射](M2B_INITIALIZATION_MAPPING.json) · [M2b准入评审](M2B_ADMISSION_REVIEW.md)。

模型和checkpoint按仓库Git LFS规则交付。SL512的Adam状态仅用于恢复证据；M2b将要求两个分别新建的Adam，不能继承这些动量。
