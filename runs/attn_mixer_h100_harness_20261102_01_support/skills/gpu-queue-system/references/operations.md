# 安装、迁移与恢复

## 系统安装

需要同一 Linux 主机及 PID namespace、Python 3.9+（含 `os.pidfd_open` 和
`signal.pidfd_send_signal`）、支持 pidfd 的内核。本机使用 `/usr/bin/python3`，
不依赖当前 Python 虚拟环境。只有安装和服务需要 root；用户提交任务不需要 sudo。

在源码目录执行：

```bash
sudo /usr/bin/python3 scripts/install.py --activate
gpu-queue-system status
```

| 路径 | 用途 |
| --- | --- |
| `/usr/local/share/agent-skills/gpu-queue-system` | root 管理、所有用户可读的 skill 和代码副本 |
| `/etc/codex/skills/gpu-queue-system` | 指向公共代码的系统级 Codex skill |
| `/usr/local/bin/gpu-queue-system` | 所有用户可调用的 CLI |
| `/etc/systemd/system/gpu-queue-system.service` | root 运行的队列服务 |
| `/run/gpu-queue-system/queue.sock` | 所有用户可连接的 Unix socket，权限 0666 |
| `/var/lib/gpu-queue-system` | root 独占的持久状态及固定 inode 的锁，目录权限 0700 |

代码副本不依赖任何用户的 home。服务用 `SO_PEERCRED` 取得连接进程的 PID/UID，
不信任请求者填写的身份；只接收标签和排队操作，不接收或执行用户命令。取得锁后用
`SCM_RIGHTS` 将 lease FD 传给用户的 supervisor，后者以原 UID 启动命令。
FIFO 按服务接受注册的顺序；每个 UID 最多 16 个任务，全机最多 64 个。

Codex 的 ADMIN skill 目录依据官方文档：<https://developers.openai.com/codex/skills/>。
支持该目录的客户端可发现此 skill；已有会话需要重新加载。发现 skill 不意味着
所有已运行会话已经采用排队规则。

每个用户可自行执行以下命令，为自己的 Codex/Claude 添加链接和全局提示：

```bash
/usr/bin/python3 /usr/local/share/agent-skills/gpu-queue-system/scripts/enable-user.py
```

它更新 `~/.codex/AGENTS.md` 和 `~/.claude/CLAUDE.md` 的 `gpu-queue` 标记块，保留
其他内容，并在 `.agents`、`.codex`、`.claude` 的 skills 目录建立链接。Claude 用户
应执行此步骤；系统安装不假设 Claude 会扫描 Codex 的管理员目录。

## 从单用户队列迁移

1. 暂停提交，等旧 `/tmp/gpu-queue-<UID>` 队列和已运行的 GPU 任务全部完成。
   新服务不会接管旧任务，也不与旧队列互锁。
2. 安装系统服务，各用户运行 `enable-user.py`；把项目 `AGENTS.md`、`CLAUDE.md`
   和启动脚本里的旧 wrapper 改成 `gpu-queue-system run -- ...`。
3. 重新加载 agent 会话；确认参与者统一使用系统队列后恢复提交。

已有会话可显式读取下面的规则：

```text
请读取 /usr/local/share/agent-skills/gpu-queue-system/SKILL.md。
以后本机所有 GPU workload 使用 gpu-queue-system run -- 执行。
这是全用户系统队列，请替换旧的单用户 wrapper，不要嵌套或绕过排队。
```

技能不能强制隔离绕过 wrapper 的程序。容器只挂载 socket 不够：服务需要看到真实
主机 PID/session，默认不支持不同 PID namespace 的客户端。

## 检查和恢复

```bash
systemctl status gpu-queue-system.service
journalctl -u gpu-queue-system.service -n 100
gpu-queue-system status --json
sudo systemctl restart gpu-queue-system.service
```

服务重启通过 boot ID、PID 启动时间、UID 和 session 检查存活任务，不会把服务
重启或 agent 退出当作 GPU 空闲。遇到连接失败的等待客户端退出 125；恢复后先查
status，再重提失败任务。运行中的任务保持 lease，不会被重复启动。

取消卡住时检查任务是否处于不可中断的内核状态，不要删锁或状态。服务无法读取
所需 `/proc` 信息时会报错并停止放行。`status` 仅展示入队任务，不能代替
`nvidia-smi` 检查其他 GPU 使用者。

非 systemd 主机可以先安装（不带 `--activate`），再交给现有系统进程管理器以 root
在固定主机 PID namespace 运行：

```bash
/usr/bin/python3 -I /usr/local/share/agent-skills/gpu-queue-system/scripts/gpu_queue.py serve
```

不要把 agent 临时工具 session 当作长期服务管理器。安装器拒绝覆盖内容不同的文件；
升级时先排空队列、停止服务、备份并替换代码，保留状态与锁。系统代码目录必须由
root 控制且不可被其他用户写入。

## CPU 测试与安装预演

```bash
/usr/bin/python3 scripts/install.py --destdir /tmp/gpu-queue-system-stage
/usr/bin/python3 -m unittest discover -s tests -v
```

测试使用隔离 socket/状态和 CPU 子进程，不接触 GPU 或正式队列。覆盖 FIFO、互斥、
超时、取消、子进程存活及 wrapper/supervisor/服务崩溃恢复。root 专用测试可使用
两个临时测试 UID 验证跨用户互斥和取消权限，不创建系统用户。
