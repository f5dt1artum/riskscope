# RiskScope

这是一个面向金融风险管理的市场风险、信用风险与流动性风险管理引擎。长期目标是提供头寸与估值曲线、风险因子与敏感度、VaR 与期望损失、压力测试与回溯检验、信用敞口、交易对手净额结算和限额预警，把风险管理沉淀为可复用引擎。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m riskscope.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `RISKSCOPE_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

## 历史模拟 VaR 与期望损失

`POST /market-risk/historical-var` 对组合执行历史模拟：

- 请求体为 JSON 对象，包含严格位于 0 和 1 之间的 `confidence`、非空 `positions` 与至少两个 `observations`；头寸含非空 `id` 与非空 `sensitivities`，观察含非空 `date` 与非空 `factor_returns`，因子名均为非空字符串、值均为有限数字；`currency` 省略时取 `USD`。观察数超过 10000 时返回 413。
- 每个观察下，各头寸对同一因子的敏感度先聚合再乘因子变动，全部因子损益之和的相反数为组合损失。损失升序排列后 VaR 取 `ceil(confidence × 观察数) - 1` 位置；期望损失为最差 `k = max(1, ceil((1 - confidence) × 观察数))` 个观察的平均损失，损失相同时按观察输入顺序选取尾部。
- 成功响应包含 `currency`、`var`、`expected_shortfall`、按输入顺序带 `date` 的 `losses`，以及尾部观察内各因子平均损失 `factor_expected_shortfall_contributions`（其和等于期望损失）。
- 失败时只返回既有 `{"error": {"code", "message"}}` 对象、不返回部分结果：`400 invalid_request`（JSON 无法解析或顶层不是对象）、`422 invalid_input`（confidence 越界、集合数量不合规、类型错误、NaN/Infinity）、`422 duplicate_position`、`422 duplicate_observation`、`422 missing_factor`、`413 request_too_large`。观察中出现但头寸未引用的因子不影响结果。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线刻意不包含风险因子、VaR 计算与压力测试的实现，以便后续任务从已冻结事实出发独立设计并验证这些能力。
