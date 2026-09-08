[English](./README.en.md) · [Website](https://agentfuse-sdk.lei6393.com) · [GitHub](https://github.com/SuperMarioYL/agentfuse-sdk)

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/hero-dark.svg">
  <img src="./assets/presentation/hero-light.svg" width="960" alt="Hero diagram">
</picture>

# agentfuse-sdk

**用任务预算检查下一次调用。**

AgentFuse SDK 为经过包装的 LiteLLM 调用提供任务预算，在委托前预留估计成本，返回用量后结算。

v0.8.0 还修复了未知价格模型在 fallback 模式下不累计 USD 花费的问题，并将 CLI demo 随 wheel 一同安装。fallback 使用保守估价，不能替代服务商账单。

## 为什么需要它

并发调用各自可能看似可负担，合计却超过剩余预算。待结算预留让下一次准入决策看到在途估计。

- **调用前准入** — 超预算预留在委托前失败。
- **计入待结算工作** — 并发估计共用待结算余额。
- **按用量结算** — 预留显式释放或提交。

## 架构

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-dark.svg">
  <img src="./assets/presentation/architecture-light.svg" width="960" alt="Architecture diagram">
</picture>

Fuse 确定活跃任务预算。包装器估计调用并预留成本，准入后才调用 LiteLLM。成功时按报告用量结算，失败时释放预留。预算可约束累计费用、token 与单次费用。

| 组件 | 职责 |
| --- | --- |
| `Fuse task scope` | src/agentfuse/fuse.py |
| `Pre-call estimate` | src/agentfuse/pricing.py |
| `Budget reservation` | src/agentfuse/budget.py |
| `LiteLLM / settlement` | src/agentfuse/wrap.py |

## 安装与快速上手

使用仓库清单指定的运行时版本构建，并在仓库根目录运行示例。

```bash
git clone https://github.com/SuperMarioYL/agentfuse-sdk.git
cd agentfuse-sdk
```

纯预算示例可用 Python 3.11+ 从源码运行，使用明确合成预留；完整 LiteLLM 适配需要 pyproject.toml 中的依赖。

```bash
PYTHONPATH=src LITELLM_LOCAL_MODEL_COST_MAP=True python3 examples/presentation-demo.py
```

## 实际运行示例

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/process-dark.svg">
  <img src="./assets/presentation/process-light.svg" width="960" alt="Process diagram">
</picture>

A second reservation is blocked while the first is pending; settlement records 0.40 spent and 0.60 remaining.

```text
pending before completion: 0.60
second reservation: blocked before delegation
spent: 0.40; pending: 0.00; remaining: 0.60
```

完整命令与输出保存在 [docs/demo-results.json](./docs/demo-results.json). 输入和复现代码均随仓提供。

![已有终端录制](./assets/demo.gif)

保留已有录制供参考；上方文字示例给出当前可复现的操作。

## 用法

CLI 提供以下操作。示例之外的命令需要替换成你的文件路径或标识。

```bash
# Full adapter installation:
python -m pip install -e .
# Offline core example:
PYTHONPATH=src python3 examples/presentation-demo.py
```

## 配置

使用 agentfuse.install() 包装 LiteLLM completion 函数，再在 Fuse(max_spend_usd=...) 中运行；也可直接调用 agentfuse.completion。应设置生成上限并明确选择 on_unpriced；默认 block，而 warn-pass 会让未知模型绕过有效费用约束。

## 集成与职责分工

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-dark.svg">
  <img src="./assets/presentation/integrations-light.svg" width="960" alt="Integrations diagram">
</picture>

以下路径已有源码实现。按任务选择输入，并把生成的结果与项目一起保存。

| 路径 | 已实现职责 |
| --- | --- |
| LiteLLM completion | Wrapped sync and async calls |
| Fuse / decorator | Task-local budget context |
| Streaming usage | Consumption-aware metering |
| Local records | Optional task ledger |

## 限制与后续方向

- 只有经过包装器且处于活跃预算内的调用受管理；直接 SDK、外部进程和无关服务不在预算范围。
- 准入依赖 token 与价格估计；提供方价格不准确或实际用量超预期时，最终账单可能偏离估计。
- 示例使用合成金额验证预留机制，不调用 LiteLLM，也不测量提供方实际消费。

提供方计量与负载估计需要持续验证；调用者自行计算成本时，也可独立使用纯 Budget API。

## 许可与贡献

许可见 [LICENSE](./LICENSE). 反馈问题时请提供最小输入、执行命令和实际输出。
