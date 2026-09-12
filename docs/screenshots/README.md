# 运行截图（佐证材料候选）

> 这些截图由**真实运行**的演示服务生成（`python app/app.py` → 浏览器打开 `http://127.0.0.1:5000`），
> 不是设计稿或示意图。可以直接放进「佐证材料」或技术报告的成果展示一节。
> ⚠️ 使用前请按 `competition/05_匿名化与合规自查清单.md` 复核画面里没有姓名/单位/本机路径。

| 文件 | 画面内容 | 证明什么 |
|---|---|---|
| `01_dashboard_ddos_alarm.png` | 注入 DDoS 攻击片段后：风险值 **1.000**、判定「DDoS」、曲线区出现红色告警带、告警日志首条为 `DDoS p=1.000 · 窗口真值 DDoS（命中）` | 在线检测链路端到端可用；**告警与真实标签一一对应（命中）**，不是动画 |
| `02_dashboard_bruteforce_vs_normal.png` | 注入 BruteForce 攻击片段后：判定「BruteForce」、告警日志首条 `BruteForce p=1.000 · 窗口真值 BruteForce（命中）` | 8 类攻击都能被正确识别与命名，而不是只报"异常" |

## 建议再补几张（团队自录，5 分钟就能搞定）

| 建议截什么 | 怎么截 | 对应评分点 |
|---|---|---|
| 验收自检结论行 | 终端跑 `python tools/acceptance_check.py`，截最后「通过 52 项 \| 警告 1 项 \| 失败 0 项」 | 实现完成度与可验证效果（同分第一顺位） |
| 策略热更新 | 截两个 `curl`（GET 查询 + POST 更新阈值/白名单）与其 JSON 返回 | 「运行时可控」，区别于只能离线跑的实验代码 |
| 控制器闭环输出 | 终端跑 `python tools/controller_simulator.py --rows 3000 --batch 50 --policy block --no-dry-run`，截吞吐与 p50/p95 那几行 | AI 与开源（Ryu 控制平面）融合 |
| 审计台账 | 打开 `results/audit_alerts.jsonl` 的一行（时间/源 IP/类别/置信度/动作/触发窗口） | 可追溯性 —— 能否进生产的分水岭 |
| 运维人力图 | `results/figures/ops_triage_load.png` | 把误报翻译成人工工时，直接回应行业痛点 |

> 截终端时请先 `cls` 清屏、并把提示符里的用户名去掉（或改用 `D:\sdn_demo` 这样的干净路径）。
