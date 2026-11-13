# Git交付记录

2026-11-13，独立分支`codex/rc3-ch5-learning-plan-20261113`，来源`a926562ec97b7a4f132987decc2f1b0b2b7b1281`。

开始时已核status/branch/log/remote/fetch；原分支与对应origin一致，main为`5d5fe25754ffa032f97937a3fb54f15901c2fb7d`，main至来源相差133提交，main是来源祖先。新研究分支延续最新第五章收尾与D5/D6/R07/R08历史。

只提交本目录、根README入口和Git清单新增条目。既有Word改动、未跟踪论文、研究情况目录及压缩包不纳入。本轮不新增模型/归档或修改LFS规则，不重复全历史LFS审计。有限冻结源/指针/旧SL权重哈希见SOURCE_BINDING.json。

发布主体提交：`a7343eeac1f5bd5a88c3813fdc94488b340c8873`。

已推送到`origin/codex/rc3-ch5-learning-plan-20261113`，实际`ls-remote`回读为相同SHA；远端main仍为`5d5fe25754ffa032f97937a3fb54f15901c2fb7d`。本记录补记后产生的最终分支SHA，在最终交付中另以实际回读核对。

主体差异16个文档/元数据路径，逐个变更Git blob中LFS指针0，新增LFS对象0；没有历史源码、结果、权重或停止文件进入提交。文档本地链接、JSON、日期/固定入口一致性、10项有限冻结来源哈希和最终暂存差异已检查。独立审读修正了消融对象和分尺度门槛。

第一次推送遇到TLS EOF，后用当前Git的Schannel后端重试成功，未关闭证书校验。本轮新文档统一LF以通过仓库空白检查；历史文件原字节保留。未合并main、未强推或改写历史。
