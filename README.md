# RiskScope

这是一个面向金融风险管理的市场风险、信用风险与流动性风险管理引擎。长期目标是提供头寸与估值曲线、风险因子与敏感度、VaR 与期望损失、压力测试与回溯检验、信用敞口、交易对手净额结算和限额预警，把风险管理沉淀为可复用引擎。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m riskscope.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `RISKSCOPE_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

## 历史模拟 VaR 与期望损失

`POST /market-risk/historical-var` 对组合头寸按历史因子变动做历史模拟：

```json
{
  "confidence": 0.95,
  "currency": "USD",
  "positions": [
    {"id": "eq-book", "sensitivities": {"eq": 100.0, "ir": -50.0}}
  ],
  "observations": [
    {"date": "2024-01-01", "factor_returns": {"eq": 0.01, "ir": 0.0}}
  ]
}
```

- `confidence` 严格位于 (0, 1)；`positions` 非空、`observations` 至少两个；头寸 `id`、敏感度、观察 `date`、`factor_returns` 均须类型正确且数值有限。`currency` 省略时为 `USD`。
- 每个观察把各头寸同因子敏感度汇总后乘因子变动，因子损益之和的相反数为组合损失；损失按输入顺序附 `date` 返回。
- VaR 取损失升序中 `ceil(confidence×n)-1` 位置；期望损失为最差 `k=max(1, ceil((1-confidence)×n))` 个观察的平均损失，损失并列时按输入顺序取尾部；`factor_expected_shortfall_contributions` 为尾部观察各因子平均损失，之和等于期望损失。
- 错误均通过 `{"error": {"code", "message"}}` 返回且不含部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（越界、数量或类型错误、NaN/Infinity）、`422 duplicate_position`、`422 duplicate_observation`、`422 missing_factor`、`413 request_too_large`（超过 10000 个观察）。观察中出现未被头寸引用的额外因子不影响结果。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线刻意不包含风险因子、VaR 计算与压力测试的实现，以便后续任务从已冻结事实出发独立设计并验证这些能力。
