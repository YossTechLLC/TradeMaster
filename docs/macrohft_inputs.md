# MacroHFT inputs: the default set, the proposed set, and the rules an input must follow

Status: **implemented as input profiles (section 7); no training run launched yet.** The 14 + 4 proposal in
section 4 was revised to 13 + 4 (`kline17`) after the build's redundancy check; see 7.2.

## 1. How MacroHFT consumes its inputs (checked in the code)

Every step the agent sees **one bar** (`back_time_length = 1`); there is no sequence model. Any history the agent
should know about has to be encoded in the features themselves (returns over windows, rolling statistics).

There are three input groups, each a list of column names in the data files:

| Group | Where the list comes from | Used by | How it enters the network (`model/net.py`) |
|---|---|---|---|
| **single** (per-bar state) | `data/feature_list/single_features.npy` | the 6 sub-agents and the hyper-agent | sub-agent: `fc1 = Linear(n_single, 64)`, then `LayerNorm`; hyper-agent: `fc1 = Linear(n_single + n_trend, 32)` |
| **trend** (context the sub-agents condition on) | `data/feature_list/trend_features.npy` | the 6 sub-agents and the hyper-agent | sub-agent: `fc2 = Linear(n_trend, 64)`, added to the position embedding, then turned into a **shift and a multiplicative scale** (`x · (1 + scale) + shift`, "adaLN") applied to the single-feature hidden state |
| **context** (market type) | computed by `decompose` from `close`: `slope_360`, `vol_360` | the hyper-agent only | `fc2 = Linear(2, 32)` → shift/scale of the hyper-agent's hidden state; standardised in code since the fixes |

What this means for substituting inputs:

- **The input sizes are not hard-coded.** `n_single = len(single list)` and `n_trend = len(trend list)` (`low_level.py:141-142`, `high_level.py:137-138`). Nothing in the code assumes 36 or 9. A new list works as long as its columns exist in every data file.
- **No input scaling exists for the single and trend groups.** Values go straight into the first linear layer. The only normalisation, a `LayerNorm`, comes *after* that layer, over the mixed hidden vector. So the data files must already be scaled (section 5).
- **The hyper-agent's episodic memory compares states by distance.** It uses the encoded vector `fc1([single, trend])` with a kernel `1/(d² + 0.001)`. Input scale therefore also decides which past states count as "similar".
- **Changing the lists invalidates every trained model.** All six sub-agents and the hyper-agent must be retrained, and the shipped ETHUSDT checkpoints only work with the original 36 + 9.
- **Two small code changes are needed** (section 6):
  - the lists are loaded from one global `data/feature_list/` rather than per dataset;
  - there is no place to store and apply scaling statistics.

## 2. The default inputs (upstream)

Status is measured on the BTCUSDT fixture, where the 5-level book is synthetic.

### 2a. Single, 36 features

| # | Feature | Meaning | Unit / typical scale | Status with Binance klines |
|---|---|---|---|---|
| 1 | `volume` | base volume of the bar | BTC, std ≈ 49, skew 8 | real, but raw and heavy-tailed |
| 2-6 | `bid1_size_n` … `bid5_size_n` | share of bid depth at level k (of 5) | 0-1 | **noise** (no book data) |
| 7-11 | `ask1_size_n` … `ask5_size_n` | share of ask depth at level k | 0-1 | **noise** |
| 12 | `wap_1` | size-weighted mid at level 1 | USD, ≈ 43,000 | price level; ≈ close + noise |
| 13 | `wap_2` | size-weighted mid at level 2 | USD | price level; ≈ close + noise |
| 14 | `wap_balance` | \|wap_1 − wap_2\| | USD | **noise** |
| 15 | `buy_spread` | bid1 − bid5 distance | USD | **constant** |
| 16 | `sell_spread` | ask5 − ask1 distance | USD | **constant** |
| 17 | `buy_volume` | taker-buy volume | BTC | real (ρ 0.93 with volume) |
| 18 | `sell_volume` | taker-sell volume | BTC | real (ρ 0.92 with volume) |
| 19 | `volume_imbalance` | (buy − sell) / volume | −1…1 | real |
| 20 | `price_spread` | relative bid-ask spread | ≈ 1e-6 | **noise** |
| 21 | `sell_vwap` | VWAP of taker sells | USD, ≈ 43,000 | real, but a raw price level |
| 22 | `buy_vwap` | VWAP of taker buys | USD | real, but a raw price level |
| 23-26 | `log_return_{bid1,bid2,ask1,ask2}_price` | 1-bar log return of a book price | ≈ 8e-4 | **duplicates** of the close return (ρ = 1.000) |
| 27 | `log_return_wap_1` | 1-bar log return of wap_1 | ≈ 8e-4 | **duplicate** (ρ = 1.000) |
| 28 | `kmid` | (close − open) / open | ≈ 8e-4 | real, but ρ 0.9998 with the close return |
| 29 | `klen` | (high − low) / open | ≈ 9e-4, skew 8 | real |
| 30 | `kmid2` | (close − open) / (high − low) | −1…1 | real |
| 31 | `kup` | (high − max(open, close)) / open | ≈ 3e-4 | real |
| 32 | `kup2` | upper shadow / (high − low) | 0…1 | real |
| 33 | `klow` | (min(open, close) − low) / open | ≈ 4e-4 | real |
| 34 | `klow2` | lower shadow / (high − low) | 0…1 | real |
| 35 | `ksft` | (2·close − high − low) / open | ≈ 8e-4 | real |
| 36 | `ksft2` | (2·close − high − low) / (high − low) | −1…1 | real |

Totals: **12 noise, 2 constant, 5 duplicates, 2 price levels mixed with noise, 15 real.** Of the 15 real ones, 3 are raw price levels or raw volumes, and `kmid` duplicates the return.

### 2b. Trend, 9 features

Each is `(x_t − x_{t−60}) / 60`, the average per-bar change over 60 bars, in the unit of `x`.

| # | Feature | Status |
|---|---|---|
| 1 | `ask1_price_trend_60` | ≈ price trend (USD per bar) |
| 2 | `bid1_price_trend_60` | duplicate of #1 |
| 3 | `buy_spread_trend_60` | **constant (0)** |
| 4 | `sell_spread_trend_60` | **constant (0)** |
| 5 | `wap_1_trend_60` | duplicate of #1 |
| 6 | `wap_2_trend_60` | duplicate of #1 |
| 7 | `buy_vwap_trend_60` | real (USD per bar) |
| 8 | `sell_vwap_trend_60` | real (USD per bar) |
| 9 | `volume_trend_60` | real (BTC per bar) |

Scale problem: these are in USD per bar (std ≈ 3.9 at 1m, far larger at 1h and at higher BTC prices). They feed the multiplicative `(1 + scale)` term, so large values distort the whole hidden state.

### 2c. Context, 2 features (unchanged)

`slope_360` is the slope of a low-pass-filtered linear fit of close over 360 bars. `vol_360` is the standard deviation of 1-bar returns over 360 bars. The hyper-agent standardises both with train statistics.

## 3. Is substitution easy?

**Yes, structurally.** A new pair of lists plus the matching columns in the data files is all the network needs. The work is in the data, and it has two parts:
1. The data files must carry the new columns, computed causally and scaled (section 5).
2. The six sub-agents and the hyper-agent must be retrained on them.

## 4. Proposed inputs: 14 single + 4 trend = 18

All from Binance **spot** klines, 2017-08 onward. Every kline carries open, high, low, close, volume, quote volume, trade count, and taker-buy base and quote volume. No order book is needed, and nothing is synthetic. W is the trend window in bars; the default is 60, and section 5.7 covers choosing it per timeframe.

### 4a. Single, 14

| # | Feature | Formula | Replaces | Raw scale |
|---|---|---|---|---|
| 1 | `log_return` | ln(close_t / close_{t−1}) | the 5 duplicate returns and `kmid` | ≈ 1e-3 |
| 2 | `log_quote_volume` | ln(quote_volume_t), the USDT traded in the bar | `volume` (unit-free across price regimes, less skewed) | ≈ 15-22 |
| 3 | `volume_imbalance` | (taker_buy − taker_sell) / volume | same (kept) | −1…1 |
| 4 | `log_trade_count` | ln(1 + number of trades) | new; trade count tracks trading costs (Brauneis et al. 2021) | ≈ 5-12 |
| 5 | `buy_vwap_dev` | buy_vwap / close − 1 | `buy_vwap` (price level → relative) | ≈ 1e-4 |
| 6 | `sell_vwap_dev` | sell_vwap / close − 1 | `sell_vwap` | ≈ 1e-4 |
| 7 | `klen` | (high − low) / open | kept | ≈ 1e-3, skewed |
| 8 | `kmid2` | (close − open) / (high − low) | kept | −1…1 |
| 9 | `kup` | (high − max(open, close)) / open | kept | ≈ 3e-4 |
| 10 | `kup2` | (high − max(open, close)) / (high − low) | kept | 0…1 |
| 11 | `klow` | (min(open, close) − low) / open | kept | ≈ 4e-4 |
| 12 | `klow2` | (min(open, close) − low) / (high − low) | kept | 0…1 |
| 13 | `ksft` | (2·close − high − low) / open | kept | ≈ 1e-3 |
| 14 | `ksft2` | (2·close − high − low) / (high − low) | kept | −1…1 |

Dropped:
- the 12 noise and 2 constant features;
- the 5 duplicate returns and `kmid` (ρ 0.9998 with `log_return`);
- `wap_1`/`wap_2` (price levels);
- `buy_volume`/`sell_volume` (fully determined by volume and `volume_imbalance`).

### 4b. Trend, 4

All four are scale-free. The upstream trends were in USD per bar.

| # | Feature | Formula | Replaces |
|---|---|---|---|
| 1 | `ret_W` | ln(close_t / close_{t−W}) | the 4 price-trend duplicates |
| 2 | `rv_W` | std of `log_return` over the last W bars (realised volatility) | new |
| 3 | `volume_surprise_W` | `log_quote_volume`_t − mean of it over the last W bars | `volume_trend_60` |
| 4 | `flow_W` | mean of `volume_imbalance` over the last W bars (persistent buy/sell pressure) | `buy_vwap_trend_60`/`sell_vwap_trend_60` |

### 4c. Context, 2 (unchanged)

`slope_360` and `vol_360`, with the windows rescaled per timeframe (section 5.7).

### 4d. Optional later (ablation group, not in the first run)

These are the literature liquidity estimators, all computable from klines. Add them only if they beat the shuffled-feature baseline (section 5.8):
- EDGE spread, `bidask.edge_rolling`;
- Kyle's λ, rolling regression of `log_return` on signed quote volume;
- Amihud, |return| / quote volume;
- the Roll measure.

See `docs/book_features.md`.

## 5. Rules an input must follow for MacroHFT to make sense of it

### 5.1 File layout and columns
- The data comes as three files, `data/<PAIR>/df_train.feather`, `df_val.feather` and `df_test.feather`, split by time, never shuffled.
- Required columns: `close`, which drives the environment's execution price, the reward and the Q-teacher. Also `timestamp`, for checks, and every feature column named in the two lists, spelled exactly the same.
- One row is one bar. Bars have a **fixed interval and no gaps**. The environment steps row by row and trades at the previous row's close, so a missing hour silently becomes a one-bar jump. Fill short exchange outages by carrying the last close forward with zero volume, or split the data at long ones.

### 5.2 Timing (no look-ahead)
- A feature at row t may use only data up to and including the close of bar t. The agent observes row t and trades at `close_t`.
- Execution at the same close it just observed means zero latency, which is optimistic. It is acceptable at 15m/1h, but it must not be made worse by any feature that peeks forward: no centred windows, no backward-fill, no full-sample normalisation.
- Rolling windows start empty. Drop the warm-up rows, whose length is the largest window, W or 360. Do it *before* the data is split and chunked. `decompose` cuts the sub-agent chunks from the raw splits, so any NaN reaching them breaks training.

### 5.3 Values: finite, stationary, unit-free
- **No NaN or ±inf** anywhere. Guard every division: bars where high = low need a small epsilon (the upstream code uses one), and `volume_imbalance` needs volume > 0.
- **Stationary:** no raw prices, no cumulative totals, and no raw volumes whose level drifts with the BTC price. Use returns, ratios to the close, logs of volume, and differences from a rolling mean. A value should mean the same thing in 2018 and in 2025.
- **Units:** none are required, and they should cancel out. Prefer quote-currency (USDT) volume over base (BTC) volume, and express prices relative to the close.

### 5.4 Scale: standardised, with bounded tails
This rule matters most, because the code does no scaling. With the default inputs the scale ranges from about 1e-4 (`kup`) to about 4e4 (`buy_vwap`), 8 orders of magnitude:
- The first linear layer effectively ignores the tiny features.
- A single large one dominates the hidden vector before `LayerNorm` rescales it.
- Large trend inputs blow up the multiplicative `(1 + scale)` conditioning.
- Scale also decides the neighbours the episodic memory finds.

Required transform, fitted on **train only** and applied unchanged to val and test:
1. Log first for positive, heavy-tailed quantities: volume and trade count are already logged in section 4; `klen` optionally.
2. Standardise each feature: (x − mean_train) / std_train. For features with fat tails, a robust version (median and IQR) works better.
3. Clip to ±5 after standardising, so one extreme bar (a crash or a listing spike) cannot saturate the network.
4. Bounded ratios already in [−1, 1] or [0, 1] (`volume_imbalance`, `kmid2`, `kup2`, `klow2`, `ksft2`) may stay as they are; their spread is already of order 1.
5. A feature whose train std is 0 (a constant) is rejected.
6. Store the statistics next to the data (for example `data/<PAIR>/feature_scaling.yaml`), so val and test, and later live data, are transformed identically.

Target: every input has mean ≈ 0, variance ≈ 1, and |value| ≤ 5.

### 5.5 Precision
The network computes in float32 (about 7 significant digits); the replay buffer holds float64 copies. After standardisation this is irrelevant. Before it, compute every feature in **float64 from the raw prices**. Never difference raw prices stored as float32: a $60,000 price in float32 is only accurate to about $0.004. Feature files may be stored as float32 or float64.

### 5.6 Redundancy and consistency
- No two features with |ρ| > 0.95 on train; keep the simpler one. `kmid` vs `log_return` is the example.
- The two lists (names *and order*) are part of the model. Training, validation, test and any later live use must use identical lists and identical scaling statistics. Changing either means retraining all six sub-agents and the hyper-agent.
- Each list must contain at least one feature: the trend group feeds `fc2` and cannot be empty.

### 5.7 Windows per timeframe
Window lengths count bars, so they must be chosen per timeframe:
- the trend window W;
- the context window, 360 at 1m (6 hours), set with `MACRO_CONTEXT_WINDOW` / `--context_window`;
- the sub-agent chunk, 4320 at 1m (3 days), set with `MACRO_CHUNK_SIZE`.

| | 1m (upstream) | 15m | 1h |
|---|---|---|---|
| trend window W (bars) | 60 (1 h) | 16 (4 h) or 60 (15 h) | 24 (1 day) |
| context window (bars) | 360 (6 h) | 24-96 (6 h-1 day) | 24-48 (1-2 days) |
| sub-agent chunk (bars) | 4320 (3 days) | 288 (3 days) - 672 (1 week) | 168 (1 week) - 336 (2 weeks) |
| train chunks from 8 years at 60% | ~590 | ~580 at 288 | ~250 at 168 |

These are starting points. The chunk sets the market-type labelling, which needs enough chunks per class: at least ~50 train chunks, since the classes are quantiles.

### 5.8 Evidence that a feature carries signal (the admission test)
Before a feature enters training, on the **train split only**:
1. Compute its rank correlation (IC) with forward returns (1, 4 and 16 bars) and with forward realised volatility.
2. Compute the same for a copy of the feature shuffled in time (the noise baseline), repeated about 20 times.
3. Admit the feature only if |IC| clearly exceeds the shuffled distribution, e.g. above its 95th percentile, for at least one target.

This is the test the synthetic book features would have failed.

## 6. Code changes needed to use a new set (done, see section 7)

1. **Per-dataset feature lists.** Load `data/<PAIR>/feature_list/{single,trend}_features.npy` when present, falling back to `data/feature_list/`. This touches `low_level.py`, `high_level.py` and the default arguments in `env/*_env.py`.
2. **A feature builder** (extend `box/bench/build_data.py` or add a new one). It should:
   - download spot klines per timeframe;
   - compute the 18 features in float64;
   - drop warm-up rows and check bars are gap-free;
   - fit the scaling on train only, apply it to all splits, and write the three feather files plus the lists and scaling statistics.
3. **Validation script** for sections 5.3-5.8: finite, stationary (no feature whose train and test means differ by more than ~1 train std), scaled, non-redundant, and the admission test. It fails the build if any rule is broken.
4. **Retrain:** `decompose` with the per-timeframe windows, then `train-low`, then `train-high`.

## 7. Input profiles (implemented)

A profile is a YAML file in `MacroHFT/profiles/` naming the single and trend features, each with its formula
(`fn` in `profiles/features.py`), its arguments (which may differ per timeframe) and its scaling. Per-timeframe
run settings are in `profiles/timeframes.yaml`: chunk, context window, memory size and warm-up.

### 7.1 The profiles (BTCUSDT spot, 15m and 1h; train 2019-2022, val 2023, test 2024)

| Profile | Single | Trend | What it is |
|---|---|---|---|
| `upstream_real` (P0) | 16 | 3 | Upstream defaults minus every book-derived input (14 noise/constant, wap_1/2, 5 book-price returns, 6 book trends); one close return instead of the five copies. The control: it keeps upstream's raw VWAP price levels on purpose, so its build *warns*: drift, redundancy. |
| `kline17` (P1) | 13 | 4 | Section 4, revised (7.2). |
| `xt10` (P2) | 5 | 5 | XT10 split by persistence: τ < 9 bars per-bar; τ ≥ 9 bars and the calendar feature as the trend group. |
| `xt10_all` (P2b) | 10 | 4 | All ten XT10 coordinates per-bar, with the `kline17` trend group. |

**XT10 defaults (as agreed):**
- windows are in bars exactly as specified: σ = EWMA RMS with half-life 89 over 987 prior bars; "1d" = 96 bars at both timeframes;
- the seasonal slot is the hour of the week (168 slots), fitted on train, and averaged over the slots of the next 13 bars;
- guards: `mid_position` = 0.5 when H = L; `efficiency` = 0 when the path did not move; `trade_size_rel` carries forward over bars without trades;
- `range_over_mean_range_1d` is logged before scaling.

**Timeframe settings:**

| | 15m | 1h |
|---|---|---|
| chunk (sub-agents) | 288 bars (3 days) | 168 bars (1 week) |
| context window (hyper-agent) | 96 bars (1 day) | 48 bars (2 days) |
| memory capacity | 288 | 168 |
| train chunks | 487 | 208 |

### 7.2 Why `kline18` became `kline17`
The build's redundancy check (Spearman |ρ| > 0.95 on train) failed the proposal at both timeframes:
- `log_trade_count` ~ `log_quote_volume` (ρ 0.96). It was replaced by `log_avg_trade_size` = ln(quote volume / trades), USDT per trade.
- `buy_vwap_dev` ~ `sell_vwap_dev` (ρ 0.98-0.99), and both ~ −`ksft` (ρ 0.96). Measured against the close, both mostly restate where the close sits in the bar. They were replaced by one `vwap_spread` = (taker-buy VWAP − taker-sell VWAP) / close.

`volume_surprise_W` uses robust scaling (median/IQR): the 269 (15m) / 66 (1h) gap-filled zero-volume bars are ~−17σ outliers that otherwise shrink its spread to 0.7.

### 7.3 Build, check, plan (run from the repo root)

```bash
MacroHFT/run.sh build --all                          # or: build xt10 15m ; datasets -> MacroHFT/data/BTCUSDT_<tf>_<profile>/
.venv-hft/bin/python box/bench/tests/test_profiles.py   # causality of every feature + each dataset plugs into MacroHFT
DATASET=BTCUSDT_15m_xt10 MacroHFT/run.sh admit        # admission test (7.4)
MacroHFT/profiles/plan_runs.sh /tmp/macro_profiles    # staged job lists; nothing runs
```

Each dataset directory holds:
- the three splits: raw `close` (and `raw_*` OHLCV) plus the scaled features;
- `feature_list/`, which MacroHFT now reads per dataset;
- `feature_scaling.yaml` (the train-fitted transforms);
- `build_report.yaml` (gaps, per-split statistics, warnings);
- `run.env` (chunk and context window for `decompose` and `train-high`, which `run.sh` picks up);
- `admission.yaml`.

**Data:**
- Monthly Binance spot kline archives, verified against their published SHA-256 and cached in `MacroHFT/data/raw_binance/`.
- 2025+ archives use microsecond timestamps; the builder, and now `box/bench/build_data.py`, handle both units.
- Missing bars (21 outages in 2019-2024, the longest 40 × 15m) are filled flat at the last close with zero volume, and reported.
- Features are computed over a 1,200-bar warm-up before the train start, which is then dropped.

**Checks in place:**
- `build.py` fails the build on: non-finite values, a constant feature, and, for strict profiles, redundancy, drift in val/test (mean shift > 1 or std ratio outside 1/3-3), > 1% of values clipped at ±5, or < 50 train chunks.
- `test_profiles.py` recomputes every feature on truncated data at 12 cut points and requires identical values: no look-ahead. It also steps MacroHFT's environment and both networks on each dataset: shapes, finiteness and scaling.

### 7.4 Admission results (train split, `admission.yaml` per dataset)

The null for each feature is its IC after 50 random circular shifts of at least one day. This keeps the feature's own persistence; a plain shuffle would let any slow-moving feature pass.

| Dataset | Admitted | Rejected |
|---|---|---|
| 15m `upstream_real` | 17/19 | `buy_vwap`, `sell_vwap` (raw price levels) |
| 1h `upstream_real` | 15/19 | `buy_vwap`, `sell_vwap`, `buy_vwap_trend_60`, `sell_vwap_trend_60` |
| 15m `kline17` | 17/17 | none |
| 1h `kline17` | 16/17 | `flow_W` |
| 15m / 1h `xt10` | 10/10 | none |
| 15m `xt10_all` | 14/14 | none |
| 1h `xt10_all` | 13/14 | `flow_W` |

- **Direction:** every feature's IC with forward returns is small (|IC| ≤ 0.08); the strongest are short-horizon mean reversion in returns and candle position.
- **Volatility:** ICs are large (up to 0.64, `rv_W` / `klen`).
- **What admission means:** it only says a feature is not noise. Whether it helps the agent is decided by the training comparison. With 4 targets × ~15 features at the 95% level, expect about one false pass per profile.

### 7.5 Launching (not done)

`plan_runs.sh` writes three stages. Run them in order on the BOX:
1. 8 `decompose` jobs.
2. 48 sub-agent jobs: 6 per dataset, one seed.
3. 24 hyper-agent jobs: 8 datasets × seeds 12345/23456/34567, each writing its own `result/high_level/<dataset>/seed<s>/`.

`MacroHFT/profiles/launch_runs.sh QUEUE_DIR` runs the three stages in order and stops at a failing stage.
On the BOX the queue is already written to `box/queue/profiles/`. This runs inside our 16 CPU / 24 GB allocation on the shared BOX (`box/README.md`). Estimated wall time is about 1.5-2 h; per-job speed drops while the other agent is busy.
