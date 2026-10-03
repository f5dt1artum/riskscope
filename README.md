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

## 批量敏感度压力测试

`POST /market-risk/stress-test` 把每个场景作用于整组头寸：

```json
{
  "currency": "USD",
  "positions": [
    {"id": "eq-book", "sensitivities": {"eq": 100.0, "ir": -50.0}}
  ],
  "scenarios": [
    {"id": "crash", "factor_shocks": {"eq": -0.10, "ir": 0.02}}
  ]
}
```

- `currency` 为非空字符串，省略时取 `USD`；`positions`、`scenarios` 均为非空数组。头寸 `id` 与场景 `id` 必须唯一且非空；`sensitivities`、`factor_shocks` 至少包含一个风险因子，数值接受整数和浮点数且必须有限（拒绝布尔值、NaN、Infinity）。额外字段忽略。
- 头寸在某因子上的损失为 `-(敏感度 × 冲击)`，场景 `loss` 为全部头寸全部因子损失之和。场景缺少的已引用因子按零冲击处理；仅出现在场景中的额外因子忽略。
- 响应含 `currency`、按输入顺序排列的 `results` 与 `worst_scenario`。每项结果含场景 `id`、`loss`、按头寸 `id` 汇总的 `position_loss_contributions` 与按因子汇总的 `factor_loss_contributions`（零值项保留，两组归因之和均等于 `loss`）。`worst_scenario` 为最大 `loss` 的场景 `id` 与 `loss`，并列时取输入最早者；全部损失为负也不截断为零。
- 任何中间计算产生非有限数值时整次失败，不返回部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、类型错误、空集合、NaN/Infinity、计算溢出）、`422 duplicate_position`、`422 duplicate_scenario`、`413 request_too_large`（场景超过 1000 个，或场景数 × 头寸数超过 100000；边界值允许处理）。

## VaR 回溯检验

`POST /market-risk/var-backtest` 对逐日 VaR 预测与已实现损益做突破判定并执行 Kupiec 无条件覆盖检验：

```json
{
  "confidence": 0.95,
  "significance": 0.05,
  "currency": "USD",
  "observations": [
    {"date": "2024-01-01", "var": 10.0, "realized_pnl": 5.0},
    {"date": "2024-01-02", "var": 10.0, "realized_pnl": -15.0}
  ]
}
```

- `confidence` 严格位于 (0, 1)；`significance` 省略时为 `0.05`，否则严格位于 (0, 1)；`currency` 省略时为 `USD`，否则须为非空字符串。`observations` 为 2 至 10000 条；额外字段忽略。
- 每条观察的 `date` 唯一非空，`var` 为非负有限数，`realized_pnl` 为有限数；接受整数但拒绝布尔值、NaN、Infinity 与超大整数。
- 令 `loss = -realized_pnl`，仅当 `loss > var` 时 `breach` 为 `true`，相等不突破。响应按输入顺序返回含 `date`、`var`、`realized_pnl`、`loss`、`breach` 的明细，并给出 `observation_count`、`breach_count`、`breach_rate`，回显 `currency`、`confidence`、`significance`。
- 令 n 为观察数、x 为突破数、`p = 1 - confidence`，按规格公式计算 `kupiec.lr_statistic`（`0·ln0` 按 0 计算，舍入导致的微小负值按 0 处理），`kupiec.p_value = erfc(sqrt(lr_statistic/2))`；`kupiec.accepted` 仅在 `p_value >= significance` 时为 `true`。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、类型或范围错误、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_observation`（date 重复）、`413 request_too_large`（超过 10000 条观察；边界值允许处理）。

## 交易对手信用敞口与预期信用损失

`POST /credit-risk/counterparty-exposure` 按结算集净额汇总交易敞口，再按交易对手的 PD/LGD 计算预期信用损失：

```json
{
  "currency": "USD",
  "counterparties": [
    {"id": "cp1", "pd": 0.02, "lgd": 0.6}
  ],
  "netting_sets": [
    {"id": "ns1", "counterparty_id": "cp1", "collateral": 5.0}
  ],
  "trades": [
    {"id": "t1", "netting_set_id": "ns1", "mtm": 10.0, "add_on": 2.0}
  ]
}
```

- `currency` 为非空字符串，省略时取 `USD`。`counterparties`、`netting_sets`、`trades` 均为非空数组。交易对手含非空 `id`、闭区间 `[0, 1]` 内的 `pd` 与 `lgd`；结算集含非空 `id`、引用存在交易对手的 `counterparty_id` 与非负 `collateral`；交易含非空 `id`、引用存在结算集的 `netting_set_id`、有限 `mtm` 与非负有限 `add_on`。数值接受整数但拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 每个结算集：`gross_exposure` 为所属交易 `max(mtm, 0)` 之和，`net_mtm` 为 `mtm` 之和，`potential_future_exposure` 为 `add_on` 之和，`exposure_at_default = max(net_mtm + potential_future_exposure - collateral, 0)`，`expected_loss = exposure_at_default × pd × lgd`。超额抵押只把违约敞口截断为零，不产生负敞口；`net_mtm` 仍按原值报告。
- 响应回显 `currency`，按输入顺序返回 `netting_set_results`（每个结算集含 `id`、五项金额指标与 `trade_count`，与请求 `netting_sets` 同序，交易对手归属可由同序的 `counterparty_id` 得到）；已声明但没有交易的结算集保留全部零值。`counterparty_totals` 按交易对手输入顺序汇总，`portfolio_totals` 为组合总计；组合总计严格等于各结算集明细之和（同一折叠次序，逐位相等），交易对手汇总是同一笔钱的不同分组，已对齐到该总计（float64 能精确表达划分时逐位相等，否则相差至多数个 ulp）。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空数组、类型或范围错误、引用不存在、非有限计算结果）、`422 duplicate_trade`、`422 duplicate_netting_set`、`422 duplicate_counterparty`（对应 id 重复）、`413 request_too_large`（trades 超过 10000 条或 netting_sets 超过 1000 个；边界值允许处理）。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

健康检查、历史模拟 VaR/期望损失、批量敏感度压力测试与 VaR 回溯检验的行为均已冻结，后续能力（如信用敞口等）须从这些已冻结事实出发独立设计并验证。
