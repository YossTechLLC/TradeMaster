# MacroHFT trading layer

`[TradeMaster]` addition on top of the upstream MacroHFT agents: an optional environment with **long / flat / short**
positions, a **per-venue cost model** (fees, funding, leverage, liquidation) and a **turnover penalty**. Target venue:
Kraken US perpetual futures (theoretical: nothing here talks to an exchange). Contract: `docs/trading_mvp_spec.md`.

With no new flags nothing changes: `--trade_mode legacy` (the default) builds the upstream environments, and
`box/bench/golden.py compare` against the shipped goldens prints `IDENTICAL`.

## Layout

| file | job |
|---|---|
| `venues/*.yaml`, `venue.py` | `FeeModel`, `FundingModel`, `Venue`; `load_venue("kraken_us_perp")` |
| `config.py` | `TradingConfig`, the leverage policy, CLI flags |
| `accounting.py` | `Account.step`: the only place money is computed (fee, pnl, funding, liquidation) |
| `penalty.py` | `turnover_penalty` |
| `teacher.py` | `make_q_table_trading`: the DP "optimal action values" used as the KL demonstration |
| `envs.py` | `VenueTestingEnv`, `VenueTrainingEnv`, `make_env` (legacy factory) |
| `report.py` | PnL breakdown CLI over `test/trading_log.npz` |

## Flags (`RL/agent/low_level.py`, `RL/agent/high_level.py`)

| flag | default | meaning |
|---|---|---|
| `--trade_mode` | `legacy` | `legacy` (upstream, 2 actions), `long_only` (2 actions, venue costs), `long_short` (3 actions: short / flat / long) |
| `--venue` | `kraken_us_perp` | venue name (`trading/venues/<name>.yaml`) or a YAML path |
| `--leverage` | 5 (perp), 1 (spot) | see below |
| `--capital` | 10000 | initial equity in USD |
| `--turnover_penalty_bps` | 0 | training-only penalty, bps of traded notional |
| `--reward_scale` | 100 | reward = (pnl - fee - funding - penalty) / capital * scale |

```bash
cd MacroHFT
../.venv-hft/bin/python RL/agent/low_level.py  --dataset D --clf slope --label label_1 --alpha 1 \
    --trade_mode long_short --venue kraken_us_perp --leverage 5 --turnover_penalty_bps 5   # x6 sub-agents
../.venv-hft/bin/python RL/agent/high_level.py --dataset D --trade_mode long_short --venue kraken_us_perp --leverage 5 --turnover_penalty_bps 5
../.venv-hft/bin/python trading/report.py result/high_level/D@long_short-kraken_us_perp-L5-tp5/exp1/seed_12345 --out report.md
```
In `long_only` / `long_short` the venue fee model and equity-based sizing replace `--transcation_cost` and `--max_holding_number`; those two flags are ignored.
Use the same trading flags for the six sub-agents and the hyper-agent (a sub-agent's action space must match).
Job lists: `TRADING_ARGS="--trade_mode long_short ..." TRADING_TAG=ls5 profiles/plan_runs.sh`.

Outputs of a non-legacy run go under `result/*/<dataset>@<mode>-<venue>-L<lev>-tp<bps>/` (the "result key"; data paths keep
the plain dataset name), so they never overwrite legacy results or each other. Extra files next to the upstream ones:
Runs differing only in non-default `--capital` / `--reward_scale` get `-c<capital>` / `-rs<scale>` appended to the key.
`trading_config.yaml` (config, plus the venue with sources and the unverified list) in the run directory,
`trading_log*.npz` (per-step position, pnl, fee, funding, penalty, equity, ... plus `close`, `timestamp`) beside the val/test `.npy` files.

## Leverage policy

On perp venues leverage must satisfy **3 < L < 10**, strictly (user policy, 2026-09-30), and `L <= venue.max_leverage`.
Default 5. Anything else raises `ValueError` when the config is built (e.g. `--leverage 3` is rejected). Spot venues
require L = 1 and `long_only`. Legacy mode ignores all of these fields.

## How it works

- **Positions.** Directions are `(0, 1)` for long_only and `(-1, 0, 1)` for long_short. A position holds `floor(L * equity / (price * contract_size))`
  contracts, fixed when the direction changes and kept while the direction is held (no rebalancing, so no fees while holding).
- **Step** (trade at the previous close, bar runs to this close): fee on the contracts traded; pnl = contracts * cs * (close - prev close);
  funding = notional * rate_per_hour * bar_hours (longs pay when the rate is positive, shorts receive); then a liquidation check at the
  bar's adverse extreme (`raw_low`/`raw_high` if present): if equity falls below `maintenance_margin_rate * notional`, the position is
  closed at that price and the exit fee plus `liquidation_fee_usd` are charged. Equity <= 0 ends the episode.
- **Reward** = (pnl - fee - funding - penalty) / capital * `reward_scale` (percent of capital by default).
- **Turnover penalty**: `bps * 1e-4 * traded_notional`, an extra transaction cost that discourages churn. It shapes training only: it is in the
  reward and in the teacher's Q-table, never in equity, PnL or any reported metric. 0 unless `--turnover_penalty_bps` > 0.
- **Teacher** (`alpha` KL demonstration): `q[i,p,c] = reward[i,p,c] + gamma * max_k q[i+1,c,k]` with the same units, costs, funding and
  penalty as the environment; checked against brute force in `box/bench/tests/test_trading_teacher.py`.
- **Validation/test starts**: the hyper-agent starts flat; sub-agent validation averages over every initial action.

## Venue figures (research of 2026-09-30) - UNVERIFIED items flagged

| | kraken_us_perp | coinbase_us_perp | binance_us_spot |
|---|---|---|---|
| contract | 0.01 BTC | 0.01 BTC | 0.00001 BTC lot |
| fee | 0.15 USD/contract/side | 0.05% (min 0.20 USD/contract) | 0.02% |
| funding | 8 h, 0.01% baseline (premium clamp not modelled) | 1 h, 0.001% **(unverified placeholder)** | none |
| max leverage | 10 **(unverified for US)** | 10 | 1 |
| maintenance margin | 5% **(unverified)** | 5% **(unverified)** | 0 |
| liquidation fee | 10 USD | 0 **(unverified)** | 0 |

Also unverified: Kraken maker fee (assumed = taker; unused, all orders are taker). Each YAML lists its sources and `unverified` items,
and they are copied into `trading_config.yaml`. Re-check them before trusting absolute numbers.

## Assumptions and out of scope (MVP)

Execution is a taker order at the previous bar's close, as upstream. Not modelled: historical funding series (a constant rate is used),
maker/limit execution, partial fills and order-book slippage, dynamic position sizing between signals, a live Kraken API,
and re-checking the unverified venue figures.

## Tests

`test_trading_env.py` reads `MacroHFT/data/BTCGOLD/train/df_0.feather` (git-ignored): run `.venv-hft/bin/python box/bench/golden.py prepare` first.
The turnover penalty is applied only in `VenueTrainingEnv`; validation/test rewards and the log's `penalty` column stay for reference but the reward excludes it.
`for t in box/bench/tests/test_trading_*.py; do .venv-hft/bin/python $t; done` (from the repo root).
