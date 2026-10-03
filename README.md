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

## 协方差估计

`POST /market-risk/covariance-estimate` 根据同步风险因子收益率估计样本均值、协方差、波动率与相关系数：

```json
{
  "factors": ["eq", "ir"],
  "observations": [
    {"date": "2024-01-01", "factor_returns": {"eq": 0.01, "ir": 0.02}},
    {"date": "2024-01-02", "factor_returns": {"eq": 0.03, "ir": -0.01}},
    {"date": "2024-01-03", "factor_returns": {"eq": -0.02, "ir": 0.0}}
  ]
}
```

- `factors` 为非空数组，因子名为唯一非空字符串，其顺序决定全部输出向量与矩阵的行列位置；`observations` 至少两条，每条含唯一非空 `date` 与覆盖全部声明因子的 `factor_returns`，未声明因子及其他字段忽略。收益率接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数。
- 响应回显 `factors` 并返回 `observation_count`、`means`、`volatilities`、`covariance_matrix`、`correlation_matrix`。均值为算术平均；协方差为去均值乘积之和除以 n-1；波动率为协方差对角元素的非负平方根；相关系数为协方差除以两侧波动率，零波动因子与其他因子为 0.0、自身为 1.0。矩阵对称，计算不按 `date` 排序；任一计算出现非有限值时整次失败，不返回部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空集合、类型错误、收益率非法、观察不足）、`422 duplicate_factor`、`422 duplicate_observation`、`422 missing_factor`、`413 request_too_large`（因子超过 100 个、观察超过 10000 条，或因子数 × 观察数超过 1000000；边界值允许处理）。

## 参数法 VaR（Delta-Normal）

`POST /market-risk/parametric-var` 以零均值 Delta-Normal 法计算组合 VaR 与预期损失：

```json
{
  "currency": "USD",
  "confidence": 0.99,
  "factors": ["eq", "ir"],
  "positions": [
    {"id": "p1", "sensitivities": {"eq": 100.0, "ir": 50.0}},
    {"id": "p2", "sensitivities": {"eq": -20.0}}
  ],
  "covariance_matrix": [
    [0.04, 0.0],
    [0.0, 0.01]
  ]
}
```

- `currency` 为非空字符串，默认 `USD`；`confidence` 为 (0,1) 内有限数；`factors` 为非空有序数组，因子名唯一非空；`positions` 非空，每项含唯一非空 `id` 与 `sensitivities` 对象，敏感度只能引用已声明因子，缺项按零处理，额外字段忽略。`covariance_matrix` 顺序同 `factors`，须为 n×n 有限数矩阵，并在 1e-12 相对容差内对称且半正定。数值拒绝布尔值、NaN、Infinity 与超大整数。
- 汇总向量 `s` 后计算 `variance = sᵀΣs`、`volatility = sqrt(variance)`、`var = z × volatility`、`expected_shortfall = φ(z) × volatility / (1 - confidence)`，其中 `z`、`φ` 为标准正态分位数与密度。响应含 `currency`、`confidence`、`factors`、按 `factors` 顺序的 `aggregate_sensitivities` 及 `variance`、`volatility`、`var`、`expected_shortfall` 四项指标。波动率为零时 `var` 与 `expected_shortfall` 均为 0.0；同一容差内的微小负 variance 按零处理，其他负值或非有限结果整次失败，不返回部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_covariance`（矩阵不对称或非半正定）、`422 duplicate_factor`、`422 duplicate_position`、`422 unknown_factor`、`422 invalid_input`（其他输入或计算错误）、`413 request_too_large`（因子超过 100 个、头寸超过 10000 个，或因子数 × 头寸数超过 1000000；边界值允许处理）。

## 交易对手信用敞口与预期信用损失

`POST /credit-risk/counterparty-exposure` 按净额结算集汇总交易敞口并计算预期损失：

```json
{
  "currency": "USD",
  "counterparties": [
    {"id": "cp-a", "pd": 0.02, "lgd": 0.5}
  ],
  "netting_sets": [
    {"id": "ns-1", "counterparty_id": "cp-a", "collateral": 10.0}
  ],
  "trades": [
    {"id": "t-1", "netting_set_id": "ns-1", "mtm": 100.0, "add_on": 5.0},
    {"id": "t-2", "netting_set_id": "ns-1", "mtm": -30.0, "add_on": 2.5}
  ]
}
```

- `currency` 为非空字符串，省略时取 `USD`；`counterparties`、`netting_sets`、`trades` 均为非空数组。交易对手 `id` 非空且 `pd`、`lgd` 位于 [0, 1]；结算集 `id` 非空、`counterparty_id` 必须引用已声明交易对手、`collateral` 非负；交易 `id` 非空、`netting_set_id` 必须引用已声明结算集、`mtm` 有限、`add_on` 非负有限。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 每个结算集：`gross_exposure` 为所属交易 `max(mtm, 0)` 之和，`net_mtm` 为 `mtm` 之和，`potential_future_exposure` 为 `add_on` 之和，`exposure_at_default = max(net_mtm + potential_future_exposure - collateral, 0)`（超额抵押不产生负敞口），`expected_loss = exposure_at_default × pd × lgd`（取所属交易对手的 `pd`、`lgd`）。
- 响应回显 `currency`，按输入顺序返回 `netting_sets` 明细（含 `trade_count`，无交易的已声明结算集保留零值），按交易对手输入顺序返回 `counterparties` 汇总金额，并返回 `portfolio_totals`；各层总计等于对应明细之和。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空数组、类型或范围错误、引用不存在、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_counterparty`、`422 duplicate_netting_set`、`422 duplicate_trade`、`413 request_too_large`（交易超过 10000 条或结算集超过 1000 个；边界值允许处理）。

## 流动性缺口与变现折扣

`POST /liquidity-risk/liquidity-gap` 按期限桶衡量资金缺口与变现折扣：

```json
{
  "currency": "USD",
  "buckets": [7, 30, 90],
  "cashflows": [
    {"id": "cf-1", "day": 5, "amount": 100.0},
    {"id": "cf-2", "day": 10, "amount": -250.0}
  ],
  "liquid_assets": [
    {"id": "la-1", "market_value": 200.0, "haircut": 0.1, "available_day": 0}
  ]
}
```

- `currency` 省略时取 `USD`，否则须为非空字符串；`buckets` 为非空、严格递增且不重复的正整数天数数组；`cashflows` 为非空数组，每项含唯一非空 `id`、正整数 `day` 与有限 `amount`（正数为流入、负数为流出）；`liquid_assets` 省略时为空数组，每项含唯一非空 `id`、非负有限 `market_value`、位于 [0, 1] 的 `haircut` 与非负整数 `available_day`。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 现金流归入首个不小于 `day` 的桶，`day` 超过最终期限则整次失败；资产从首个不小于 `available_day` 的桶起可用（超过最终期限则始终不可用），折后金额为 `market_value × (1 - haircut)`。
- 响应回显 `currency`，并按期限顺序返回各桶的 `day`、`net_cashflow`、`cumulative_net_cashflow`、`available_liquidity`（截至该桶可用的折后资产累计）、`surplus`（累计净现金流加 `available_liquidity`）与 `required_funding`（`max(-surplus, 0)`，零值保留）。`earliest_shortfall` 为首个负 `surplus` 桶的 `day` 与 `required_funding`，无缺口时为 `null`。任何中间计算产生非有限数值时整次失败，不返回部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空数组、类型或范围错误、桶序非法、现金流期限超过最终桶、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_cashflow`、`422 duplicate_asset`。

## 跨账簿、多币种的头寸归并与折算

`POST /portfolio-risk/aggregate` 把头寸的市值与敏感度按各自币种汇率折算到报告币，再按币种与账簿归并：

```json
{
  "reporting_currency": "USD",
  "fx_rates": {"EUR": 1.1, "JPY": 0.01},
  "positions": [
    {"id": "p1", "book": "book-a", "currency": "USD", "market_value": 100.0,
     "sensitivities": {"eq": 10.0, "ir": -2.0}},
    {"id": "p2", "book": "book-b", "currency": "EUR", "market_value": 50.0,
     "sensitivities": {"eq": 5.0}},
    {"id": "p3", "book": "book-a", "currency": "JPY", "market_value": -200.0,
     "sensitivities": {"ir": 3.0, "fx": 7.0}}
  ]
}
```

- `reporting_currency` 为非空字符串，`fx_rates` 为对象，`positions` 为非空数组。每个头寸含唯一非空 `id`、非空 `book`、非空 `currency`、有限 `market_value` 与非空 `sensitivities` 对象，其中因子名非空、敏感度有限。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- `fx_rates` 表示一单位源币兑换的报告币金额：被引用的非报告币汇率必须为有限正数，缺失时整次失败；报告币隐含汇率为 1，显式提供时也只能为 1，否则整次失败；未被任何头寸引用的汇率（即使非法）一律忽略。
- 每个头寸的 `market_value` 与各因子敏感度均乘以其币种汇率，负值保留。任何中间或汇总结果非有限时整次失败，不返回部分结果。
- 响应回显 `reporting_currency` 并返回 `position_count`；`currencies` 与 `books` 均按头寸中首次出现的顺序排列。币种明细含 `currency`、`fx_rate`、`position_count`、`converted_market_value`、`converted_sensitivities`；账簿明细含 `book`、`position_count`、`converted_market_value`、`converted_sensitivities`。`portfolio_totals` 给出全组合折算市值与敏感度；各层汇总严格等于其所属明细之和。因子按头寸及各头寸对象内首次出现的顺序输出，未涉及因子补零、汇总为零仍保留。
- 错误均通过 `{"error": {"code", "message"}}` 返回且不含部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空字符串、结构或数值非法、报告币汇率不为 1、非有限计算结果）、`422 duplicate_position`（头寸 id 重复）、`422 missing_fx_rate`（被引用的非报告币缺少汇率）、`413 request_too_large`（头寸超过 10000 条、`fx_rates` 超过 1000 项，或敏感度条目总数超过 1000000；边界值允许处理）。

## 风险限额评估

`POST /risk-management/limit-check` 把调用方算好的指标观测与声明的限额逐项比对，返回各项状态、告警与整体结论；本功能不调用既有计算、不换汇、不保存历史：

```json
{
  "as_of": "2024-06-30",
  "limits": [
    {"id": "lim-var", "scope": {"type": "book", "id": "book-a"},
     "metric": "var", "unit": "USD", "limit": 100.0},
    {"id": "lim-es", "scope": {"type": "book", "id": "book-b"},
     "metric": "expected_shortfall", "unit": "USD", "limit": 50.0,
     "warning_ratio": 0.5}
  ],
  "measurements": [
    {"limit_id": "lim-var", "value": 120.0}
  ]
}
```

- `as_of` 为非空字符串；`limits` 为非空数组，每条限额含唯一非空 `id`、由非空 `type` 与 `id` 组成的 `scope`、非空 `metric` 与 `unit`、严格大于零的有限 `limit`；`warning_ratio` 省略时为 0.8，否则须为大于零且小于 1 的有限数。`measurements` 可省略或为空数组，每条观测以 `limit_id` 引用已声明限额并提供非负有限 `value`，同一限额至多一条。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 响应回显 `as_of`，`limits` 按输入顺序返回原定义及评估值。有观测时 `utilization = value / limit`、`headroom = limit - value`；`value >= limit` 为 `breach`，未突破但 `value >= limit × warning_ratio` 为 `warning`，否则为 `ok`。无观测时 `value`、`utilization`、`headroom` 为 `null`，状态为 `no_data`。
- `alerts` 仅含 `warning` 与 `breach` 的限额评估项且保持限额顺序；`summary` 统计 `ok`、`warning`、`breach`、`no_data` 四种状态的数量；`overall_status` 按 `breach`、`warning`、`incomplete`、`ok` 的优先级确定，其中 `incomplete` 表示没有更高状态但存在 `no_data`。
- 错误均通过 `{"error": {"code", "message"}}` 返回且不含部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、结构或类型错误、范围或数值有限性非法）、`422 duplicate_limit`（限额 id 重复）、`422 duplicate_measurement`（同一限额观测重复）、`422 unknown_limit`（观测引用未声明限额）、`413 request_too_large`（`limits` 或 `measurements` 超过 10000 条；边界值允许处理）。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

健康检查、历史模拟 VaR/期望损失、批量敏感度压力测试、VaR 回溯检验、协方差估计、交易对手信用敞口、流动性缺口分析、跨账簿多币种头寸归并与风险限额评估的行为均已冻结，后续能力须从这些已冻结事实出发独立设计并验证。
