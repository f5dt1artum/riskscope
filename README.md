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

## VaR 回溯检验：独立性与条件覆盖

`POST /market-risk/var-backtest-validation` 沿用 `var-backtest` 的完整输入语义、默认值与突破判定，在 Kupiec 无条件覆盖检验之外，按观察输入顺序对相邻突破状态同时执行 Christoffersen 独立性检验与条件覆盖检验，用于识别突破的时间聚集。请求体与 `var-backtest` 完全相同：

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

- 输入校验、默认值（`significance=0.05`、`currency="USD"`）、逐日明细与突破判定（`-realized_pnl > var`，相等不突破）与 `var-backtest` 完全一致；`observations` 为 2 至 10000 条，`date` 唯一，`var` 非负且所有数值有限。
- 响应保留 `var-backtest` 的全部内容（`observations` 明细顺序不变），并新增四个一阶马尔可夫转移计数 `n00`、`n01`、`n10`、`n11`（相邻两日突破状态从前态到后态的计数）以及 `independence`、`conditional_coverage` 两个检验对象；两对象均含 `lr_statistic`、`p_value`、`accepted`。
- 独立性检验：令 `q=(n01+n11)/(n-1)`、`q0=n01/(n00+n01)`、`q1=n11/(n10+n11)`，`lnL0=(n00+n10)ln(1-q)+(n01+n11)ln(q)`，`lnL1=n00ln(1-q0)+n01ln(q0)+n10ln(1-q1)+n11ln(q1)`，`LRind=2×(lnL1-lnL0)`。计数为零的对数项按零处理；某前态从未出现（`n00+n01==0` 或 `n10+n11==0`）时，对应 `q0` 或 `q1` 返回 `null`，其似然贡献为零。舍入造成的微小负 `LRind` 归零，其他非有限结果按非法输入处理。`independence` 另含 `q`、`q0`、`q1`，其 `p_value = erfc(sqrt(LRind/2))`（自由度为一的卡方生存函数）。
- 条件覆盖检验：`conditional_coverage.lr_statistic = kupiec.lr_statistic + LRind`（自由度为二），`p_value = exp(-lr_statistic/2)`。
- 两项检验分别在 `p_value >= significance` 时令 `accepted=true`（边界值视为接受）。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、类型或范围错误、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_observation`（date 重复）、`413 request_too_large`（超过 10000 条观察；边界值允许处理）。原 `/market-risk/var-backtest` 入口行为不变。

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

## 参数法 VaR：因子与头寸归因

`POST /market-risk/parametric-var-attribution` 沿用 `parametric-var` 的完整输入语义、默认值、因子/头寸顺序、协方差校验（1e-12 相对容差内对称且半正定）与规模上限，在原组合指标之外，把方差、VaR、预期损失按因子和头寸两层归因。请求体与 `parametric-var` 完全相同：

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

- 输入校验、默认值、`covariance_matrix` 形状与半正定校验、各类数值与规模限制、错误码均与 `parametric-var` 一致；整数接受，布尔值、NaN、Infinity 与超大整数拒绝。
- 设聚合敏感度为 `s`、协方差矩阵为 `Σ`、协方差载荷 `c = Σs`、组合方差 `variance = sᵀc`、`z`、`φ` 为标准正态分位数与密度。响应保留 `currency`、`confidence`、`factors`、`aggregate_sensitivities`、`variance`、`volatility`、`var`、`expected_shortfall`，其取值与 `parametric-var` 完全一致。
- `factor_attributions` 按因子输入顺序返回 `factor`、`aggregate_sensitivity`、`covariance_loading`（`c_i`）、`variance_contribution = s_i × c_i`、`component_var = z × variance_contribution / volatility`、`component_expected_shortfall = φ(z) × variance_contribution / ((1 - confidence) × volatility)`。
- `position_attributions` 按头寸输入顺序返回 `id`，以该头寸自身的敏感度向量 `p`（未引用因子补零）计算 `variance_contribution = pᵀc`，并按相同系数计算两项 component 指标。
- 贡献可为负（不截断），零项不省略。两层的三类贡献（方差、component VaR、component 预期损失）之和分别等于对应组合值，绝对误差不超过 `1e-12 × max(1, |对应组合值|)`。组合波动率为零时，`covariance_loading` 仍按矩阵乘法返回，所有贡献均为 `0.0`。归因计算或汇总出现非有限值时整次失败，返回 `422 invalid_input` 且不含部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_covariance`、`422 duplicate_factor`、`422 duplicate_position`、`422 unknown_factor`、`422 invalid_input`、`413 request_too_large`。原 `/market-risk/parametric-var` 入口的状态码、字段与计算结果保持不变。

## 贴现现金流与关键利率敏感度

`POST /market-risk/discounted-cashflow` 依据零息曲线对头寸现金流贴现，并计算利率敏感度：

```json
{
  "currency": "USD",
  "curve_points": [
    {"day": 365, "zero_rate": 0.02},
    {"day": 730, "zero_rate": 0.03}
  ],
  "positions": [
    {"id": "p1", "cashflows": [{"day": 548, "amount": 1000.0}]}
  ]
}
```

- `currency` 省略时取 `USD`，否则须为非空字符串；`curve_points` 至少含两个点，每点含正整数 `day` 与有限 `zero_rate`（允许为负），`day` 不得重复、输入可乱序；`positions` 非空，每个头寸含唯一非空 `id` 与非空 `cashflows`，每笔现金流含正整数 `day` 与有限 `amount`（允许为负）。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 曲线按 `day` 升序整理；现金流恰在节点时取该节点 `zero_rate`，位于相邻节点间时按 `day` 线性插值，超出首尾节点范围则整次失败。以 `t = day / 365` 计算 `present_value = amount × exp(-zero_rate × t)`，按输入顺序累计。每笔现金流对节点的敏感度为 `amount × exp(-zero_rate × t) × t × 0.0001` 乘该节点插值权重，`dv01` 等于本层节点敏感度之和。
- 响应回显 `currency` 与升序 `curve_points`，`positions` 保持输入顺序；头寸与 `portfolio_totals` 均含 `present_value`、`dv01` 及按节点升序、保留零值的 `key_rate_dv01`。负现金流、现值与敏感度不截断；任何中间计算产生非有限数值时整次失败，不返回部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、类型或范围错误、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_curve_point`（曲线 `day` 重复）、`422 duplicate_position`（头寸 `id` 重复）、`422 curve_out_of_range`（现金流越出曲线范围）、`413 request_too_large`（曲线点超过 100 个、头寸超过 10000 个，或现金流总数超过 100000；边界值允许处理）。

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

## 多期预期信用损失

`POST /credit-risk/expected-loss-schedule` 用期限化违约概率与敞口预测计算信用组合的多期预期损失：

```json
{
  "currency": "USD",
  "periods": [1, 2, 3],
  "discount_factors": [1.0, 0.5, 0.25],
  "counterparties": [
    {"id": "cp-a", "cumulative_pd": [0.125, 0.25, 0.5]}
  ],
  "facilities": [
    {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.5, "ead": [100.0, 200.0, 400.0]}
  ]
}
```

- `currency` 省略时取 `USD`，否则须为非空字符串；`periods` 为非空、严格递增且不重复的正整数数组；`discount_factors` 与 `periods` 等长，每项为大于零且不超过 1 的有限数；`counterparties`、`facilities` 均为非空数组。交易对手含唯一非空 `id` 及与 `periods` 等长的 `cumulative_pd`（位于 [0, 1] 且不随期限下降）；授信含唯一非空 `id`、引用已声明交易对手的 `counterparty_id`、位于 [0, 1] 的 `lgd` 及与 `periods` 等长的非负 `ead`。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 首期边际违约概率取首期 `cumulative_pd`，后续取相邻累计值之差；授信各期 `discounted_expected_loss = ead × lgd × 边际违约概率 × discount_factor`。
- 响应回显 `currency` 与 `periods`，按输入顺序返回 `facilities` 的逐期 `contributions` 及 `total_discounted_expected_loss`，按交易对手输入顺序返回 `counterparties` 的逐期汇总及合计，并返回 `portfolio_total`；无授信的交易对手保留全零结果，各层合计与明细一致，零值不省略。任何中间计算产生非有限数值时整次失败，不返回部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空数组、长度不符、期限顺序错误、类型或范围非法、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_counterparty`、`422 duplicate_facility`、`422 unknown_counterparty`、`413 request_too_large`（交易对手超过 1000 个、授信超过 10000 笔，或期限数 × 授信数超过 1000000；边界值允许处理）。

## Basel 内部评级法（F-IRB）资本

`POST /credit-risk/irb-capital` 按 Basel 公司风险暴露的初级内部评级法计算授信资本要求与风险加权资产：

```json
{
  "currency": "USD",
  "counterparties": [
    {"id": "cp-a", "pd": 0.02}
  ],
  "facilities": [
    {"id": "f-1", "counterparty_id": "cp-a", "lgd": 0.45, "ead": 1000000.0, "maturity": 2.5}
  ]
}
```

- `currency` 省略时取 `USD`，否则须为非空字符串；`counterparties`、`facilities` 均为非空数组。交易对手含唯一非空 `id` 与严格位于 (0, 1) 的 `pd`；授信含唯一非空 `id`、引用已声明交易对手的 `counterparty_id`、位于 [0, 1] 的 `lgd`、非负 `ead` 与位于 [1, 5] 的 `maturity`。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 授信采用所属交易对手的 PD。令 `a=(1-exp(-50×PD))/(1-exp(-50))`，`R=0.12×a+0.24×(1-a)`，`b=(0.11852-0.05478×ln(PD))²`，`MA=(1+(maturity-2.5)×b)/(1-1.5×b)`，`K=lgd×[Φ((Φ⁻¹(PD)+√R×Φ⁻¹(0.999))/√(1-R))-PD]×MA`，其中 `Φ`、`Φ⁻¹` 为标准正态分布函数与分位数；`capital_requirement=ead×K`，`risk_weighted_assets=12.5×capital_requirement`。
- 响应回显 `currency`；`facilities` 按输入顺序返回 `id`、`counterparty_id`、`pd`、`R`、`MA`、`K`、`ead`、`capital_requirement`、`risk_weighted_assets`；`counterparties` 按交易对手输入顺序汇总 `ead`、`capital_requirement`、`risk_weighted_assets`，无授信的交易对手保留零值；`portfolio_totals` 为交易对手汇总之和，各金额不舍入。任何中间计算产生非有限数值时整次失败，不返回部分结果。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空数组、结构、类型或范围错误、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_counterparty`、`422 duplicate_facility`、`422 unknown_counterparty`、`413 request_too_large`（交易对手超过 1000 个或授信超过 10000 笔；边界值允许处理）。

## 评级迁徙期限化违约概率与预期损失

`POST /credit-risk/rating-migration-loss` 把单期评级转移矩阵连续应用到请求期限，从评级迁徙推导期限化违约概率并计算预期损失：

```json
{
  "currency": "USD",
  "horizon": 3,
  "ratings": ["AAA", "BBB", "D"],
  "default_rating": "D",
  "transition_matrix": [
    [0.95, 0.04, 0.01],
    [0.10, 0.85, 0.05],
    [0.0, 0.0, 1.0]
  ],
  "counterparties": [
    {"id": "cp-1", "rating": "BBB", "ead": 1000000.0, "lgd": 0.45}
  ]
}
```

- `currency` 省略时取 `USD`，否则须为非空字符串；`horizon` 为 1 到 50 的正整数；`ratings` 为 1 至 100 个有序且唯一的非空评级名；`default_rating` 必须是其中之一；`counterparties` 为 1 至 10000 条，每条含唯一非空 `id`、引用已声明评级且不得为违约评级的 `rating`、非负有限 `ead` 与位于 [0, 1] 的 `lgd`。`transition_matrix` 为 n×n 有限非负数矩阵，每行之和在 1e-12 内等于 1，且违约态只能以概率 1 留在自身（吸收态）。数值接受整数和浮点数，拒绝布尔值、NaN、Infinity 与超大整数；额外字段忽略。
- 期限转移矩阵为单期矩阵连续应用 `horizon` 次的结果，其第 i 行给出初始评级 i 在期末落入各评级的概率，落入违约态的概率即累计 PD。每个交易对手的 `expected_loss = ead × lgd × 累计 PD`，`expected_defaulted_exposure = ead × 累计 PD`。距 0 或 1 不超过 1e-12 的计算概率归位到边界，其余结果不舍入。任何中间计算产生非有限数值时整次失败，不返回部分结果。
- 响应回显 `currency`、`horizon`、`ratings`、`default_rating`，并返回 `horizon_transition_matrix`；`counterparties` 按输入顺序给出 `id`、`rating`、覆盖全部评级的 `terminal_probabilities`、`cumulative_pd`、`ead`、`lgd` 与 `expected_loss`；`rating_aggregates` 按 `ratings` 顺序汇总非违约初始评级的 `counterparty_count`、`ead`、`expected_defaulted_exposure` 与 `expected_loss`（空评级组保留零值，违约评级不出现）；`portfolio_totals` 为评级汇总之和，各层汇总等于对应明细之和。
- 错误均通过 `{"error": {"code", "message"}}` 返回且无部分结果：`400 invalid_request`（JSON 无法解析或顶层非对象）、`422 invalid_input`（字段缺失、空数组、结构、类型或范围错误、未知或已违约评级、矩阵非法、NaN/Infinity、超大整数、非有限计算结果）、`422 duplicate_rating`、`422 duplicate_counterparty`、`413 request_too_large`（评级超过 100 个或交易对手超过 10000 条；边界值允许处理）。其他公开入口行为保持不变。

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

健康检查、历史模拟 VaR/期望损失、批量敏感度压力测试、VaR 回溯检验、协方差估计、交易对手信用敞口、多期预期信用损失、F-IRB 资本要求、评级迁徙期限化违约概率与预期损失、流动性缺口分析、跨账簿多币种头寸归并与风险限额评估的行为均已冻结，后续能力须从这些已冻结事实出发独立设计并验证。
