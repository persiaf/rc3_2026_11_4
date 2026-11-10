# Git / Git LFS 传输诊断记录（2026-11-10）

本记录仅说明本轮 GitHub 交付的网络传输问题与命令级兼容处理。它不属于训练、规划或性能实验，不改变研究协议、冻结证据或阶段执行授权。最终发布结果由主执行者另行写入 publication receipt；本文件不证明完整上传或远端同步成功。

## 运行环境与已观察现象

- Git：`2.54.0.windows.1`。
- Git LFS：`3.7.1`，Windows amd64，构建所用 Go：`1.25.1`。
- Git 的 Schannel 路径反复出现 TLS handshake 失败；命令级 OpenSSL 加 HTTP/1.1 的只读远端查询间歇成功，也出现 unexpected EOF。另一次直连查询以 connection reset 结束。
- Git LFS 曾在低 CPU 占用下长期等待。只读进程连接检查观察到其连接本机代理 `127.0.0.1:7897`；该观察仅证明连接路径，不证明代理自身就是故障源。
- 主执行者随后在 `GODEBUG=http2client=0,tlsmlkem=0` 的单次命令环境中完成一个 ON128 对象的上传，该对象 SHA-256 前缀为 `ccee`。单对象通过仅证明该次对象传输可行，不代表所有对象或完整分支已交付。
- 随后的完整 LFS push 返回退出码 `2`，明确失败点为 `POST .../info/lfs/locks/verify` 返回 `EOF`。这是锁元数据查询的连接失败；不能据此断言 GitHub 不支持锁查询。

以上后两项事实由主执行者提供；本记录的作者未自行运行上传、推送或重启现有传输进程。

## 兼容性判断及官方依据

Go 自 1.24 起默认启用 `X25519MLKEM768`。官方文档说明，不能正确处理较大 TLS 记录的服务可能在握手时超时，`GODEBUG=tlsmlkem=0` 可恢复此前默认密钥交换集合。Go 1.24 同时移除了旧的 `tlskyber` 开关，因此当前 Go 1.25.1 不应使用 `tlskyber=0` 代替此设置。`tlssecpmlkem` 对应 Go 1.26 的后续默认变化，并非本次 Go 1.25.1 所需选项。

官方来源：[Go GODEBUG 兼容性说明](https://go.dev/doc/godebug)、[Go 1.24 发布说明](https://go.dev/doc/go1.24)、[Go 1.25 发布说明](https://go.dev/doc/go1.25)。

本机现象与代理链路稳定性或 TLS/HTTP 兼容问题相符，这是工作假设，不是根因确诊。一次同时关闭 Go HTTP/2 客户端与默认 ML-KEM 的单对象成功，不能区分两个选项的作用，也不能排除瞬时网络变化。Schannel、OpenSSL 和 Go 均出现过异常，更不能把所有失败归因于 ML-KEM。

Git 的 `http.sslBackend=openssl` 影响 Git/libcurl 路径；Git LFS 使用 Go TLS，二者的兼容设置需要分别理解。本机 `git lfs help config` 还确认，`lfs.tlstimeout` 限制单次 TLS 握手，`lfs.activitytimeout` 限制下一次 TCP 读写，而对象重试默认八次并采用指数退避，因此整体静默等待可以长于单次超时。

## 本轮处理边界与待核验状态

主执行者采用的是单次命令环境或 `git -c` 参数，未以本次诊断为由修改全局 Git 配置或系统代理：

- Go 兼容环境：`GODEBUG=http2client=0,tlsmlkem=0`。
- Git/libcurl 兼容参数按单次命令指定，例如 `http.sslBackend=openssl` 与 `http.version=HTTP/1.1`。
- 对锁查询 EOF 的后续重试采用命令级 `-c lfs.locksverify=false`，仅跳过本次推送前锁元数据核验。

关闭该次锁查询不会改变 LFS 对象的 SHA-256 OID 或声明大小，不等于关闭 TLS 证书校验，也不等于跳过对象内容完整性要求。本轮 TLS 证书校验与内容哈希校验保持启用。锁元数据查询是否完成，与对象内容是否完整传输，是不同的验证事项；任何内容完整性或远端可用性的最终结论仍须由主执行者以实际结果确认。

截至本记录编写时，采用上述锁查询参数的完整上传重试仍在进行，尚未核验全部 LFS 对象、远端分支 SHA 或 publication receipt。不得将当前状态写成“已全部上传”“已同步 GitHub”或“完整发布成功”。本记录不包含凭据、认证头、完整代理 URL 或签名下载/上传 URL。
