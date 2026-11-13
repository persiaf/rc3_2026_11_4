# M2交付记录

仓库：<https://github.com/persiaf/rc3_2026_11_4>。独立分支：`codex/rc3-relation-m2-20261113`。

基准`9c878928e7f619bce25ddd00792757431ca705a1`与用户批准一致。开始时相对main为137条独立历史提交，main为`5d5fe25754ffa032f97937a3fb54f15901c2fb7d`；原M1远端与本地一致。启动前fetch及push遇到间歇TLS握手失败，保留证书验证，使用HTTP/1.1重试后成功。

准确协议批准和M2a授权在科学启动前提交为`451f1b288518591727c08a2b32050386d9334031`，并推送核对实际远端同SHA。原科学ID、冻结协议与98项绑定源码/数据/模型均保留原字节。

本轮交付包括该授权、本轮新材料、已停止的唯一init运行原始输出、两个恢复checkpoint、实际图表和派生CSV。用户Word改动及未跟踪的论文/研究资料排除。M2b未创建授权或运行目录，不合并main、不强推、不生成整包备份。

结果payload提交为`b8e208996d5ecf62bb5af83c8e8f480b8fd4f830`，包含2205个变更路径。实际push成功，随后`git ls-remote`核对该独立分支同SHA；main仍为`5d5fe25754ffa032f97937a3fb54f15901c2fb7d`。两个新LFS checkpoint共6,097,750字节，push输出确认`Uploading LFS objects: 100% (2/2)`；没有另行下载远端对象复验。

结果发布曾遇到TLS握手和LFS锁能力查询传输失败；最终使用当前系统代理、Schannel及HTTP/1.1，并仅在该次命令中关闭LFS锁能力查询后成功。这两个文件的lockable属性未设置，LFS对象上传及证书验证保留；未修改全局Git配置。传输错误不证明服务器不支持锁定API。

准确记录见[发布收据](PUBLICATION_RECEIPT.json)。发布收据与最后的停止说明属于后续元数据提交，不回写已停止的科学运行或冻结协议。
