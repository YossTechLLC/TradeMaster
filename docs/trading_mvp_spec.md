# MacroHFT trading layer MVP: long/short, venue costs with leverage, turnover penalty

Status: specification for the MVP build (2026-09-30). Target venue: **Kraken US perpetual futures** (theoretical: no live
exchange connection in scope). Everything new lives in `MacroHFT/trading/`; upstream files change only where the
agents select an environment, and every such change is marked `# [TradeMaster]`.

## 0. Non-negotiables

1. **The legacy path is unchanged, bit for bit.** With no new flags (`--trade_mode legacy`, the default), MacroHFT must
   produce identical results: `box/bench/golden.py run <name> macro` then `compare p6 <name>` must print `IDENTICAL`.
   Legacy env classes (`MacroHFT/env/*.py`), `tools/demonstration.py` and `model/net.py` are **not edited**.
2. **Leverage policy (user, 2026-09-30):** on perp venues, leverage L must satisfy **3 < L < 10** (strictly above 3x,
   strictly below 10x). It is validated at config time; invalid values raise `ValueError`. Default L = 5. Spot venues
   require L = 1 and long-only.
3. **Modular:** venue data in YAML, one responsibility per module, no MacroHFT imports inside `trading/` except
   `trading/envs.py` (which may import the legacy env classes for the factory). Pure numpy/python, testable alone.
4. Execution assumption (documented, not hidden): market (taker) orders at the previous bar's close, as upstream.

## 1. Package layout

```
MacroHFT/trading/
  __init__.py          exports: load_venue, TradingConfig, add_trading_args, config_from_args, Account, make_env,
                       make_q_table_trading, turnover_penalty
  README.md            what it is, how to run, assumptions, unverified venue figures
  venues/
    kraken_us_perp.yaml    target venue
    coinbase_us_perp.yaml  alternative (figures partly unverified)
    binance_us_spot.yaml   spot, long-only, L = 1 (the current cost setting, as a venue)
  venue.py             FeeModel, FundingModel, Venue, load_venue, list_venues
  config.py            LEVERAGE_POLICY, TradingConfig, add_trading_args, config_from_args
  penalty.py           turnover_penalty
  accounting.py        StepResult, Account
  teacher.py           make_q_table_trading
  envs.py              VenueTestingEnv, VenueTrainingEnv, make_env
  report.py            PnL breakdown CLI
box/bench/tests/
  test_trading_config.py  test_trading_accounting.py  test_trading_teacher.py
  test_trading_env.py     test_trading_report.py
```

## 2. venue.py and the YAML files

```python
@dataclass(frozen=True)
class FeeModel:
    model: str                 # "pct" (fraction of notional) or "per_contract" (USD per contract per side)
    taker: float
    maker: float = 0.0
    min_per_contract: float = 0.0   # pct model only: USD minimum per contract per side
    def cost(self, contracts_traded: float, price: float, contract_size: float) -> float:
        """USD fee for trading |contracts_traded| contracts (taker) at `price`."""
        # per_contract: taker * |n|
        # pct:          max(taker * |n| * contract_size * price, min_per_contract * |n|)
    def rate(self, price: float, contract_size: float) -> float:
        """fee per unit of notional at `price` (used by the teacher): cost(1, price, cs) / (cs * price)"""

@dataclass(frozen=True)
class FundingModel:
    interval_hours: float      # e.g. 8
    rate_per_interval: float   # fraction of notional; > 0 means longs pay shorts
    @property
    def rate_per_hour(self) -> float: ...

@dataclass(frozen=True)
class Venue:
    name: str; kind: str                # "spot" | "perp"
    description: str
    contract_size: float                # BTC per contract (spot: minimum lot)
    fees: FeeModel
    funding: Optional[FundingModel]     # None for spot
    max_leverage: float
    maintenance_margin_rate: float      # fraction of notional; 0 for spot
    liquidation_fee_usd: float
    allows_short: bool
    as_of: str; sources: list; unverified: list   # provenance, carried into reports

def load_venue(name_or_path: str) -> Venue   # "kraken_us_perp" -> venues/kraken_us_perp.yaml; or a path
def list_venues() -> list[str]
```
YAML field names mirror the dataclasses (`fees: {model, taker, maker, min_per_contract}`,
`funding: {interval_hours, rate_per_interval}` or `funding: null`). `load_venue` validates: kind in {spot, perp};
fee model known; all rates >= 0; contract_size > 0; perp has funding; spot has max_leverage 1 and allows_short false.

Venue figures (research of 2026-09-30; put the unverified ones in `unverified:`):

| | kraken_us_perp | coinbase_us_perp | binance_us_spot |
|---|---|---|---|
| kind | perp | perp | spot |
| contract_size | 0.01 BTC | 0.01 BTC | 0.00001 BTC |
| fees | per_contract, taker 0.15, maker 0.15 (maker unverified) | pct, taker 0.0005, maker 0.0005, min_per_contract 0.20 | pct, taker 0.0002, maker 0.0 |
| funding | 8 h, 0.0001 (baseline interest; premium ±0.05% clamp not modelled) | 1 h, 0.00001 (unverified placeholder) | null |
| max_leverage | 10 (venue figure unverified for US) | 10 | 1 |
| maintenance_margin_rate | 0.05 (unverified) | 0.05 (unverified) | 0 |
| liquidation_fee_usd | 10 | 0 (unverified) | 0 |
| allows_short | true | true | false |

Sources to cite in YAML:
- https://support.kraken.com/articles/us-futures-fees
- https://support.kraken.com/articles/contract-specifications
- https://bitnomial.com/exchange/docs/market-operations/settlements/digital-asset-perpetual-pricing/
- https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/guides/futures
- https://www.binance.us/fees

## 3. config.py

```python
LEVERAGE_POLICY = (3.0, 10.0)   # perp leverage must be strictly inside (user policy, 2026-09-30)

@dataclass(frozen=True)
class TradingConfig:
    mode: str = "legacy"            # "legacy" | "long_only" | "long_short"
    venue: Optional[Venue] = None
    leverage: float = 5.0
    capital: float = 10_000.0       # USD, initial equity
    turnover_penalty_bps: float = 0.0   # training-only shaping, bps of traded notional (section 5)
    reward_scale: float = 100.0     # rewards are fractions of capital times this (percent units)
    sizing: str = "equity"          # "equity": floor(L*equity/(price*cs)) contracts per position change;
                                    # "fixed": fixed_units BTC (legacy-equivalence tests)
    fixed_units: float = 0.01
    def validate(self) -> "TradingConfig": ...   # raises ValueError with a clear message
    legacy: bool           (property)  mode == "legacy"
    n_action: int          (property)  2 for legacy/long_only, 3 for long_short
    directions: tuple      (property)  (0.0, 1.0) for legacy/long_only; (-1.0, 0.0, 1.0) for long_short
    flat_action: int       (property)  index of direction 0.0 (0, or 1 for long_short)
    tag: str               (property)  "" for legacy, else f"{mode}-{venue.name}-L{leverage:g}-tp{turnover_penalty_bps:g}"
```
`validate` rules:
- mode is known; non-legacy requires a venue.
- long_short requires `venue.allows_short`.
- perp venue: `LEVERAGE_POLICY[0] < leverage < LEVERAGE_POLICY[1]` and `leverage <= venue.max_leverage`.
- spot venue: leverage == 1 and mode long_only.
- capital > 0; turnover_penalty_bps >= 0; reward_scale > 0; sizing in {equity, fixed}; fixed_units > 0.
- Legacy mode ignores the venue and trading fields entirely.

```python
def add_trading_args(parser):   # adds: --trade_mode {legacy,long_only,long_short} (default legacy), --venue (default
                                # kraken_us_perp), --leverage (5.0), --capital (10000), --turnover_penalty_bps (0),
                                # --reward_scale (100)
def config_from_args(args) -> TradingConfig   # legacy -> TradingConfig() ; else loads the venue, validates
```

## 4. accounting.py (the one place money is computed)

```python
@dataclass
class StepResult:
    direction: float; contracts: float      # signed contracts held during the bar
    traded: float                           # |contracts change| at the bar's start
    pnl: float; fee: float; funding: float  # USD; funding > 0 = paid
    penalty: float                          # USD-equivalent turnover penalty (training shaping, NOT in equity)
    liquidated: bool; bankrupt: bool; equity: float; notional: float

class Account:
    def __init__(self, cfg: TradingConfig, bar_hours: float): ...
    def reset(self, direction: float = 0.0, price: float = None) -> None
        # equity = capital; contracts = size(direction, price) with NO fee (an inherited starting position); logs cleared
    def size(self, direction: float, price: float) -> float
        # equity sizing: direction * floor(L * equity / (price * cs)) (0 if equity <= 0)
        # fixed sizing:  direction * fixed_units / cs
    def step(self, direction: float, price_prev: float, price_now: float,
             bar_low: float = None, bar_high: float = None) -> StepResult
    def log(self) -> dict[str, np.ndarray]    # per-step arrays of every StepResult field
```
`step` order (the trade happens at `price_prev`, the bar runs to `price_now`):
1. **Target.** If `sign(direction) == sign(contracts)` and `contracts != 0`, keep `contracts`: no rebalancing while the
   direction is held. Else `target = size(direction, price_prev)`, and for direction 0 the target is 0.
2. **Fee and penalty.** `traded = |target - contracts|`, `fee = venue.fees.cost(traded, price_prev, cs)`,
   `penalty = turnover_penalty(traded * cs * price_prev, bps)`, then `contracts = target`.
3. **PnL.** `pnl = contracts * cs * (price_now - price_prev)`.
4. **Funding** (perp only). `funding = contracts * cs * price_prev * funding.rate_per_hour * bar_hours`: long pays when
   the rate is > 0, short receives.
5. **Equity.** `equity += pnl - fee - funding`.
6. **Liquidation** (perp, contracts != 0). The adverse price is `bar_low` for longs or `bar_high` for shorts when given,
   else `price_now`. Compute the equity at that adverse price. If it is below
   `maintenance_margin_rate * |contracts| * cs * adverse`, the position is closed at the adverse price:
   - replace this bar's pnl with `contracts * cs * (adverse - price_prev)`;
   - add `fees.cost(|contracts|, adverse, cs) + liquidation_fee_usd` to the fee;
   - recompute equity; set `contracts = 0` and `liquidated = True`.

   This is conservative: exit at the bar's worst price.
7. **Bankruptcy.** `bankrupt = equity <= 0`, after which size() gives 0.

`notional = |contracts| * cs * price_now`.

**Identity the tests must check:** with a spot pct venue, fixed sizing of 0.01 BTC and no penalty, `pnl - fee` per step
equals the legacy MacroHFT env reward `p1*(C1-C0) - f*|p1-p0|*C0` to 1e-9.

## 5. penalty.py

`turnover_penalty(traded_notional_usd, bps) -> float`: `bps * 1e-4 * traded_notional_usd`.

This is a training-only shaping cost, an extra transaction cost that discourages churn (the cost-multiplier approach).
It never touches equity, PnL, or any evaluation metric. It enters (a) the training reward and (b) the teacher's
Q-table, and it is 0 unless `--turnover_penalty_bps` > 0.

## 6. teacher.py

```python
def make_q_table_trading(close: np.ndarray, cfg: TradingConfig, bar_hours: float, gamma: float = 0.99) -> np.ndarray
    # shape (T, n_action, n_action); q[T-1] = 0; vectorised like MacroHFT/tools/demonstration.py
```
The teacher is the dynamic-programming "optimal action values" used as the KL demonstration. It works in the same units
as the venue env reward: fraction of capital × reward_scale. Its exposure is path-independent: e_i = L for equity
sizing, or `fixed_units * C[i] / capital` for fixed sizing.

For each row i < T-1, with `d = cfg.directions`:
- `r_i = C[i+1]/C[i] - 1`
- `cost_i = fees.rate(C[i], cs)`
- `pen = bps*1e-4`
- `fund = funding.rate_per_hour * bar_hours` (0 for spot)

```
reward[i, p, c] = reward_scale * e_i * ( d[c]*r_i - (cost_i + pen) * |d[c] - d[p]| - fund * d[c] )
q[i, p, c]      = reward[i, p, c] + gamma * max_k q[i+1, c, k]
```
**Test:** equals a brute-force search over all action sequences on a short random series (T = 7, 3 actions) for every
(i, p, c).

## 7. envs.py

`VenueTestingEnv(df, tech_indicator_list, tech_indicator_list_trend, cfg, clf_list=None, back_time_length=1,
initial_action=None, bar_hours=None)`.

- **Same public contract as the legacy envs, so the agents need no other change:**
  - low level (`clf_list is None`): `reset() -> (s, s2, info)` and `step(a) -> (s, s2, r, done, info)`;
  - high level: `reset() -> (s, s2, s3, info)` and `step(a) -> (s, s2, s3, r, done, info)`, where s3 is `clf_state.reshape(-1)`;
  - `info = {"previous_action": a}`;
  - state arrays come from pre-extracted numpy exactly as in `MacroHFT/env/low_level_env.py` (same slicing, same dtypes);
  - terminal when `m >= n_rows - 1`, or earlier if bankrupt.
- **Account:** `initial_action` None means `cfg.flat_action`. `bar_hours` None means it is inferred from the median
  `timestamp` step, falling back to 1/60. The bar extremes are `raw_low`/`raw_high` if present, else `low`/`high`,
  else None.
- **step(a)** calls `account.step(directions[a], close[m-1], close[m], low[m], high[m])`.
- **Reward:** `(pnl - fee - funding - penalty) / capital * reward_scale`.
- **At terminal:**
  - `final_balance = equity - capital`, `required_money = capital`, `pured_balance = final_balance`;
  - `trading_log = account.log()` plus `close` and `timestamp` arrays;
  - print `"the portfit margine is ", final_balance / required_money`.
- `get_final_return_rate(slient=False) -> (final_balance/required_money, final_balance, required_money, total_fee)`.

`VenueTrainingEnv(... same ... )` builds `q_table = make_q_table_trading(close, cfg, bar_hours)` and adds `info['q_value']`
in reset (row `m-1`, previous action) and step (row `m-1`, action), exactly like the legacy Training_Env.

```python
def make_env(kind, level, df, tech_indicator_list, tech_indicator_list_trend, cfg, clf_list=None,
             transcation_cost=0.0002, back_time_length=1, max_holding_number=0.01, initial_action=0):
    # kind "train"|"test", level "low"|"high".
    # cfg.legacy -> the legacy MacroHFT classes with EXACTLY the arguments the agents pass today:
    #   low train:  low_level_env.Training_Env(df=, tech_indicator_list=, tech_indicator_list_trend=, transcation_cost=,
    #               back_time_length=, max_holding_number=, initial_action=, alpha=0)
    #   low test:   low_level_env.Testing_Env(same minus alpha)
    #   high train/test: high_level_env.* with clf_list=
    # else -> VenueTrainingEnv / VenueTestingEnv (clf_list only for level "high"; initial_action passed through).
```

**Env tests:**
- legacy-equivalence of rewards (spot, fixed 0.01 BTC, reward_scale 1, capital 1, so the reward is in USD) against
  `MacroHFT.env.low_level_env.Testing_Env`;
- arity and shape parity with the legacy envs;
- a short position gains when the price falls;
- funding signs;
- a forced liquidation scenario;
- `make_env` legacy returns the legacy classes;
- the leverage policy is enforced.

## 8. Agent integration (low_level.py, high_level.py): minimal, marked

- `add_trading_args(parser)`; in `__init__`: `self.trading = config_from_args(args)`, `self.n_action = self.trading.n_action`.
- **Result key.** `self.result_key = args.dataset if self.trading.legacy else f"{args.dataset}@{self.trading.tag}"`. It is
  used for every `./result/...` path (low-level `result_path` and `best_model_dir`; high-level `result_path` and the
  default `subagent_path`). Data paths keep `args.dataset`.
- **Env construction** goes through `make_env` (same arguments as today in legacy mode).
- **Exploration.** `action_choice = list(range(self.n_action))`. This is identical for legacy: `random.choice` draws the
  same index for a list of the same length.
- **Validation and test starts.** Legacy keeps today's values (low level: initial actions 0 and 1; high level: 0). In
  non-legacy modes the high-level val/test starts flat (`self.trading.flat_action`), and the low-level validation
  averages over every initial action `range(self.n_action)`.
- **Extra outputs.** In non-legacy modes, val/test also save the env's `trading_log` as `trading_log*.npz` next to the
  existing .npy files, and `trading_config.yaml` in the run directory (the venue with provenance, and the config).
- `MacroHFT/profiles/plan_runs.sh`: an optional `TRADING_ARGS` env var is appended to the train-low and train-high jobs,
  and a `TRADING_TAG` suffix goes on the job ids.

## 9. report.py

```
python MacroHFT/trading/report.py RUN_DIR [RUN_DIR ...] [--out report.md]
```
Each RUN_DIR is a high-level run directory with `test/trading_log.npz`.

**Per run:**
- net PnL (USD);
- return on capital;
- buy-and-hold return over the same bars;
- max drawdown of equity;
- annualised Sharpe of per-bar equity returns (bars per year from bar_hours);
- number of position changes, and round trips per day;
- exposure: % of bars long / short / flat;
- PnL attributed to long bars vs short bars;
- total fees, total funding (paid − received);
- liquidations.

**Across seeds:** grouped by (dataset, tag): mean ± std of the key metrics.

The output is markdown; legacy runs without a trading log are skipped with a note. Tested on synthetic logs.

## 10. Out of scope for the MVP (documented in README)

- historical funding series;
- maker/limit execution;
- partial fills and order-book slippage;
- dynamic position sizing between signals;
- live Kraken API;
- re-checking venue figures marked unverified.
