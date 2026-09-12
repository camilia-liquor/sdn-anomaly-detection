# 部署样例（deploy/）

> 这里放的是**可以直接抄的部署配置样例**，不是玩具文档。
> 目标是让"这套系统怎么常驻、怎么被监控、怎么安全对外"这三件事有明确答案。
> 完整说明见项目根目录 [`DEPLOYMENT.md`](../DEPLOYMENT.md)。

| 文件 | 用途 | 适用环境 |
|---|---|---|
| `cida-detect.service` | systemd 常驻单元（自动重启、日志进 journald） | Linux 服务器 |
| `cida-detect.env` | 环境变量模板（监听地址、模型口径、策略默认值） | 两者通用 |
| `nginx-cida.conf` | 反向代理 + HTTP Basic 认证 + `/metrics` 只对内网开放 | Linux（对管理网暴露时） |
| `install-windows-service.ps1` | 用 NSSM 把检测服务注册成 Windows 服务 | Windows Server |

---

## 一、Linux：systemd

```bash
# 1) 假设项目已放在 /opt/cida，且依赖已装进 /opt/cida/.venv
sudo cp deploy/cida-detect.service /etc/systemd/system/
sudo cp deploy/cida-detect.env      /etc/default/cida-detect      # 按需修改
sudo systemctl daemon-reload
sudo systemctl enable --now cida-detect

# 2) 观察
systemctl status cida-detect
journalctl -u cida-detect -f          # 实时日志
curl -s http://127.0.0.1:5000/api/health
curl -s http://127.0.0.1:5000/metrics | head
```

> `cida-detect.service` 里已经写好三条运维要点：**工作目录固定**（模型与产物按相对路径解析）、
> **失败自动重启**、**进程崩溃后 5 秒重试**。

## 二、Windows：注册成服务

```powershell
# 需要先下载 nssm.exe 并加入 PATH（或把路径传成 -NssmPath 参数）
.\deploy\install-windows-service.ps1 -ProjectRoot "D:\sdn_demo" -Port 5000
```

脚本会做三件事：创建服务、设置工作目录与日志重定向、启动服务并做一次健康检查。
卸载：`nssm remove cida-detect confirm`。

## 三、对外暴露时的最小加固（nginx）

```bash
sudo cp deploy/nginx-cida.conf /etc/nginx/conf.d/cida.conf
sudo htpasswd -c /etc/nginx/.htpasswd cida-admin     # 设置访问口令
sudo nginx -t && sudo systemctl reload nginx
```

要点（`nginx-cida.conf` 内已写明注释）：

* `/api/*` 走 **Basic 认证**，只对管理网段开放（`allow/deny`）；
* `/metrics` **只允许监控机访问**，不对外；
* `/`（演示大屏）可对内网开放，也可一并加认证；
* 反向代理设 `proxy_read_timeout`，避免长连接被 nginx 默认超时切断。

## 四、上线顺序（务必按这个来）

1. `python tools/check_env.py` → `python tools/acceptance_check.py`（50 项全过）；
2. 用 `deploy/cida-detect.env` 里的 `CIDA_POLICY_ACTION=observe` + `CIDA_DRY_RUN=1` 起服务，**先只看不动**；
3. 接控制器（Ryu）但**不加 `--execute`**，观察一个完整业务周期的告警质量；
4. 用 `python tools/ops_impact.py` 对照实测告警量与人力的预算，确定最终工作点（阈值 + 确认次数）；
5. 改 `CIDA_POLICY_ACTION=block`、`CIDA_DRY_RUN=0`，并按 `DEPLOYMENT.md` 第 5 节逐条核对安全清单；
6. 打开 `/metrics` 接入监控，把告警量、时延分位、`by_class` 计数纳入日常看板。
