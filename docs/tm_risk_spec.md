# `tm_risk`: sizing, admission and drawdown control for TradeMaster (MVP specification)

Status: build contract, 2026-09-30. Plan and rationale: /home/perc/.claude/plans/review-the-current-state-mossy-pine.md
(SIMONS layers adapted to one instrument with one net position).

## 0. Rules
- **Package location and dependencies.** `tm_risk/` sits at the repo root. It uses only numpy, pandas and PyYAML, and
  must import under BOTH `.venv/bin/python` (Python 3.9, pandas 2.3) and `.venv-hft/bin/python` (Python 3.10, pandas 1.5).
  - No `X | None`, `match`, `dataclass(slots/kw_only)`, or pandas-2-only APIs.
  - No gym and no torch in `tm_risk/`.
- **tm_risk knows no model.** Models feed it a `SignalStream`. MacroHFT-specific code lives in `MacroHFT/trading/`.
- **Legacy and existing behaviour are unchanged.**
  - `box/bench/golden.py compare p6 <run>` must print IDENTICAL.
  - Every `box/bench/tests/test_*.py` must pass.
  - `MacroHFT/trading/venue.py` becomes a re-export shim of `tm_risk.venue`, and its public names stay identical.
- **Money units.** USD. Equity E. Notional N = |contracts| × contract_size × price.
  - **Exposure** f = N/E.
  - **Margin leverage** L is the venue account setting: collateral = N/L, with 3 < L < 10 on perps.
  - Exposure is capped at `exposure_cap` ≤ L. f and L are different quantities: never multiply by L to get exposure.
- **Causality.** A decision at the close of bar k uses only data from bars ≤ k.

## 1. Files

```
tm_risk/
  __init__.py      exports the public API below
  venue.py         moved verbatim from MacroHFT/trading/venue.py (FeeModel, FundingModel, Venue, load_venue, list_venues,
                   VENUE_DIR); venues/ yaml moved too
  types.py         AccountState, Decision, PolicyConfig (+ load_policy)
  signals.py       SignalStream + loaders
  ledger.py        SignalLedger (raw-signal outcomes at unit notional)
  credibility.py   shrink
  kelly.py         robust_kelly
  sizing.py        size_request (arms: identity, constant, vol_target, kelly)
  stops.py         atr, bar_vol, stop_distance, stressed_loss_per_unit
  episode.py       RiskEpisode
  admission.py     admit
  breach.py        mark_bar (stop / liquidation / funding / pnl), deleverage
  engine.py        replay
  report.py        summarize, attribution, to_markdown
  replay.py        CLI: python -m tm_risk.replay
  policies/        default_35dd.yaml, identity_L5.yaml, constant_1x.yaml, vol_target_25.yaml
MacroHFT/trading/venue.py          -> shim: from tm_risk.venue import *  (+ VENUE_DIR)
MacroHFT/trading/export_signal.py  full untruncated signal export (section 11)
box/bench/tests/test_risk_*.py     standalone scripts (plain asserts; see box/bench/tests/test_trading_accounting.py)
```

## 2. `types.py`

```python
@dataclass
class AccountState:
    capital: float; equity: float; contracts: float = 0.0   # signed
    entry_price: float = 0.0; stop_price: float = 0.0       # 0 = no stop
    peak_all: float = 0.0; peak_ref: float = 0.0            # all-time peak and trailing-window reference peak
    halted: bool = False; bankrupt: bool = False
    open_risk: float = 0.0          # stressed loss (USD) reserved by the open position
    blocked_direction: int = 0      # after a stop-out: re-entry in this direction is blocked until the signal changes

@dataclass
class Decision:
    k: int; timestamp: object; signal: int; action: str   # "enter","exit","flip","hold","abstain","stop","breach","liquidation","none"
    wanted: float = 0.0; preferred: float = 0.0; permitted: float = 0.0   # notional USD
    contracts: float = 0.0; binding: str = ""; vetoes: tuple = (); reason: str = ""
    exposure_req: float = 0.0   # f requested by sizing

@dataclass(frozen=True)
class PolicyConfig:   # every field below comes from the yaml; validate() raises ValueError
    name: str
    sizing: str                       # identity | constant | vol_target | kelly
    margin_leverage: float = 5.0      # perp: 3 < L < 10 (strict), and <= venue.max_leverage; spot: must be 1
    exposure_cap: float = 1.0         # f cap, 0 < exposure_cap <= margin_leverage
    constant_exposure: float = 1.0
    target_vol_annual: float = 0.25
    vol_window_bars: int = 48
    kelly_fraction: float = 0.5
    kelly_quantile: float = 0.10
    kelly_bootstrap: int = 2000
    kelly_block: int = 10
    kelly_min_support: int = 30       # per side; fewer closed outcomes -> abstain
    credibility_n0: float = 150.0
    kelly_stress_loss: float = 0.05   # solvency: f * (min(r) - stress) > -1
    kelly_vol_scale: bool = True      # f *= clip(sigma_ref / sigma_now, 0.5, 2)
    ledger_seed: str = "none"         # none | val (val seeding doubles n0; flagged as selection-biased)
    stop_atr_mult: float = 3.0        # 0 disables stops
    atr_window: int = 14
    slippage_bps: float = 5.0         # execution cost stress on every fill (bps of notional)
    trade_loss_budget: float = 0.02   # fraction of E: stressed loss per new position
    aggregate_loss_budget: float = 0.05
    dd_halt: float = 0.35             # halt new entries when E < (1 - dd_halt) * peak_ref (a TRAILING control, not an all-time DD limit)
    dd_window_days: float = 90.0      # trailing reference-peak window; 0 = all-time
    participation: float = 0.01       # new notional <= participation * bar quote volume; 0 disables
    episode_max_bars: int = 168
    episode_loss_budget: float = 0.05 # realised-loss allowance per risk episode (fraction of E at episode start)
    breach_band: float = 0.25         # deleverage when exposure > exposure_cap * (1 + band)
    liq_distance_factor: float = 0.5  # stop distance must be < factor * liquidation distance (a cap, see 8)
    checks: bool = True               # identity policy sets False: no stops/admission/halts/breach (bridge parity)

def load_policy(name_or_path) -> PolicyConfig   # tm_risk/policies/<name>.yaml or a path
```

Policies:

| policy | sizing | checks | margin L | exposure cap | stop | slippage | budgets / halt |
|---|---|---|---|---|---|---|---|
| `identity_L5` | identity | false | 5 | 5 | 0 | 0 | none |
| `constant_1x` | constant 1.0 | true | 5 | 1.0 | 3 ATR | 5 bps | default (35% halt, 2% / 5%) |
| `vol_target_25` | vol_target 0.25/yr | true | 5 | 1.0 | 3 ATR | 5 bps | default |
| `default_35dd` | kelly | true | 5 | 1.0 | 3 ATR | 5 bps | default |

## 3. `signals.py`

```python
@dataclass
class SignalStream:
    timestamp: np.ndarray; open: np.ndarray; high: np.ndarray; low: np.ndarray; close: np.ndarray
    quote_volume: np.ndarray            # USD per bar; zeros if unknown (participation then disabled with a warning)
    direction: np.ndarray               # int8 in {-1,0,1}: desired position decided at the CLOSE of bar k, held over bar k+1
    covered: np.ndarray                 # bool: the model produced a decision at k (False -> treated as 0, counted)
    bar_hours: float
    source: dict                        # provenance: model, run dir, split, truncated flag, n_covered
def from_frame(df, direction_col="direction", ...) -> SignalStream   # generic: any model exporting timestamp+OHLC(+volume)+direction
def from_signal_npz(path, bars_df) -> SignalStream                   # npz with timestamp, direction (+covered); joined to bars on timestamp
def from_macrohft(run_dir, dataset_dir, split="test") -> SignalStream
    # prefers <run_dir>/<split>/signal_<split>.npz (export_signal), else trading_log(.npz / _val.npz): direction per log row;
    # log row i holds the position over df row stack+i, decided at df row stack-1+i. Join on timestamp; assert
    # log close == df close at those rows; set source["truncated"] if the log ends before the df.
```

Bar columns come from `raw_open/raw_high/raw_low/raw_quote_volume` when present, else `open/high/low/quote_volume`. If
neither is present, open = close shifted by one bar, and high/low = max/min of the open and close.

## 4. `ledger.py`: raw-signal outcomes (independent of the account)
- **Signal episode.** A maximal run of rows with the same non-zero direction. Direction d is decided at the close of bar
  s, so the entry is `close[s]`. The episode ends at the close of bar e, the first row with a different direction, so the
  exit is `close[e]`.
- **Outcome at unit notional.**
  - `r = d*(close[e]/close[s]-1) - fees.rate(close[s]) - fees.rate(close[e]) - d*funding.rate_per_hour*hours`
  - `hours = (e-s)*bar_hours`
  - `sigma_entry = bar_vol at s`
- **Causal visibility.** `SignalLedger.update(k)` appends every episode whose exit bar is ≤ k.
- **Queries.** `outcomes(side) -> (r, hours, sigma_entry)` for side +1 / -1, and `n(side)`.
- **Seeding.** Optional: `ledger.seed(stream_val)` adds every episode of a validation stream up front.
- **Invariant.** Nothing in the account (stops, halts, admission) changes the ledger.

## 5. `credibility.py` and `kelly.py`
- **`shrink(r, n0)`.** With c = n/(n+n0), returns `r - (1-c)*mean(r)`. The mean moves toward zero edge and the residuals
  are kept.
- **`robust_kelly(r, hours, policy, rng) -> KellyResult(f, growth_q, f_raw, abstain, reason)`:**
  1. Abstain if n < `kelly_min_support`.
  2. Apply `r = shrink(r, n0)`.
  3. Grid f in [0, exposure_cap] (e.g. 201 points), restricted to the solvency domain
     `1 + f*(min(r) - kelly_stress_loss) > 0`.
  4. Growth per hour: `g_b(f) = mean(log1p(f*r_b)) / mean(hours_b)`, over B circular-block bootstrap resamples (block
     `kelly_block`, in episode order).
  5. `G(f) = quantile_q(g_b(f))`, `f_raw = argmax G`. Abstain if `max G <= 0`.
  6. `f = kelly_fraction * f_raw`.
  7. Use a fixed rng seed (`np.random.default_rng(12345)`) so results are reproducible.
- **Vol scaling** is applied by `sizing.py`, not here.

## 6. `sizing.py`
`size_request(k, side, state, stream, ledger, policy) -> (f, info)`, the requested exposure for a new position in
direction `side`:
- `identity`: `f = margin_leverage` (bridge: contracts = floor(L*E/(price*cs)), as in `MacroHFT/trading/accounting.py`).
- `constant`: `f = constant_exposure`.
- `vol_target`: `f = target_vol_annual / (bar_vol(k) * sqrt(bars_per_year))`, with bars_per_year = 8760/bar_hours.
- `kelly`: `robust_kelly(ledger.outcomes(side))`. If `kelly_vol_scale`, multiply by
  `clip(median(sigma_entry)/bar_vol(k), 0.5, 2)`.
- Every arm is clipped to `[0, exposure_cap]` except identity.

## 7. `stops.py`
- **`bar_vol(stream, k, window)`.** Std of the log close returns over bars (k-window, k]. Causal.
- **`atr(stream, k, window)`.** Wilder-style or simple mean of the true range over bars ≤ k. Causal.
- **`stop_distance(k) = stop_atr_mult * atr(k) / close[k]`.** A fraction; 0 when stops are disabled.
- **`stressed_loss_per_unit(k, venue, policy)`** = `stop_distance + 2*fees.rate(close[k]) + 2*slippage_bps*1e-4`.
  - With stops disabled, use `kelly_stress_loss` in place of the stop distance.

## 8. `admission.py`
`admit(k, side, f, state, stream, policy, venue, episode) -> Decision`. Used only when `checks` is true; identity
bypasses it.

**Constraints on the new notional.** With `E = state.equity`, `p = close[k]`, `u = stressed_loss_per_unit`:

| name | capacity (USD notional) |
|---|---|
| `request` | f*E |
| `trade_loss` | trade_loss_budget*E / u |
| `aggregate_loss` | max(0, aggregate_loss_budget*E - state.open_risk) / u |
| `drawdown_headroom` | max(0, E - (1-dd_halt)*peak_ref) / u |
| `exposure_cap` | exposure_cap*E |
| `collateral` | max(0, E - est_fee)*margin_leverage |
| `liq_distance` | E / (stop_distance/liq_distance_factor + venue.maintenance_margin_rate) (perp only) |
| `participation` | participation*quote_volume[k] (skipped when 0 or volume unknown) |
| `episode_loss` | max(0, episode.loss_allowance - episode.loss_spent) / u |

**Vetoes (permitted = 0):** `halted` (E < (1-dd_halt)*peak_ref), `bankrupt`, `blocked_direction == side`.

**Result.**
- `wanted = f*E`; `preferred = min(wanted, episode capacities)`; `permitted = min(all)`.
- `binding` = name of the minimum. On ties, the first in table order.
- `contracts = floor(permitted/(cs*p))`. 0 means action "abstain", with the reason.
- The new position's `open_risk = contracts*cs*p*u`.

## 9. `episode.py`
`RiskEpisode`:
- **Start.** Starts at the first bar with a non-zero signal after flat. Flips stay in the same episode. It ends when the
  signal is 0 or after `episode_max_bars`.
- **Allowance.** At its first admission, `loss_allowance = episode_loss_budget * E`.
- **Spending.** `record_close(realised_pnl)` adds `max(0, -pnl)` to `loss_spent`. Profits never reduce it.
- **Reset.** A new episode resets the allowance. A rejected proposal does not end or reset the episode.

## 10. `breach.py` and `engine.py`

`replay(stream, policy, venue, capital=10000.0, val_stream=None) -> ReplayResult(decisions: list[Decision], bars:
dict[str, np.ndarray], summary: dict)`

For k = 0 .. N-1, in this order:
1. **Mark bar k** (k ≥ 1, contracts ≠ 0; the position was opened or held from close[k-1]):
   `mark_bar(state, stream, k, venue, policy)`. Precedence within the bar, with the order unknown and the worst case
   assumed:
   - **(a) Stop** (checks only): long hits if `low[k] <= stop`. Fill at `min(open[k], stop)` for long, `max(open[k], stop)`
     for short (a gap through fills at the open). Pay fees plus slippage, position → 0, `blocked_direction = side`.
   - After a stop or a liquidation (checks only) `blocked_direction = side`.
   - **(b) Liquidation** (perp): the equity at the adverse extreme (low for long, high for short) is below
     `maintenance_margin_rate * notional_at_adverse`. Exit at the adverse extreme with fees plus `liquidation_fee_usd`,
     exactly as in `MacroHFT/trading/accounting.py`. If both would trigger, use the one at the worse price.
   - **(c) Otherwise** `pnl = contracts*cs*(close[k]-close[k-1])`.
   - **Funding**, in every case: `contracts*cs*close[k-1]*rate_per_hour*bar_hours` (long pays).
   - **Close.** Realised pnl on any close goes to `episode.record_close`.
2. **Peaks.** `peak_all = max(peak_all, E)`. `peak_ref` = max E over the trailing `dd_window_days` (all-time if 0).
   `halted = E < (1-dd_halt)*peak_ref` (checks only).
3. **Ledger.** `ledger.update(k)`.
4. **Breach** (checks only, contracts ≠ 0): if `N/E > exposure_cap*(1+breach_band)`, reduce to
   `floor(exposure_cap*E/(cs*close[k]))` at `close[k]`, with fees and slippage. Decision action `"breach"`. If E ≤ 0:
   flat, `bankrupt`.
5. **Signal** `s = direction[k]`. On the last bar, s = 0.
   - **s == 0:** close any position at `close[k]` (fees and slippage), action `"exit"`. The episode ends.
   - **s == sign(contracts):** `"hold"`. No resizing; the breach step handles drift.
   - **Otherwise:** close any opposite position first (`"flip"`), then enter. The entry is sized against the state after
     the close:
     - identity: `contracts = s*floor(L*E/(close[k]*cs))`, no stop;
     - otherwise: `admit(...)`.
     - **Entry fill:** at `close[k]`, fee = `fees.cost(contracts)` plus `slippage_bps*1e-4*notional`.
     - **Stop:** set `stop_price = close[k]*(1 - side*stop_distance)` when stops are enabled.
     - `blocked_direction` is cleared once the signal differs from it.
6. **Records.** Append a `Decision` for every bar with an action other than `"none"`. Also append the per-bar arrays
   (`bars`): equity, contracts, exposure, pnl, fee, funding, halted, peak_ref, dd_all, dd_ref, signal, stop_price.

`summary` = `report.summarize(result, capital, bar_hours)`.

**Bridge invariant** (`identity_L5`, `checks=False`, slippage 0, no stops): for the same direction sequence and prices,
the equity path equals `MacroHFT.trading.accounting.Account.step` to 1e-9. That includes the bankrupt → size 0 behaviour
and liquidation against the extreme.

## 11. `MacroHFT/trading/export_signal.py`

```
.venv-hft/bin/python MacroHFT/trading/export_signal.py --run_dir <result/high_level/<ds>@<tag>/<exp>/seed_<n>>
    --dataset <ds> [--splits val test]
```
- **Build the agent.** Construct the high-level agent programmatically, as `high_level.py` does: import the module,
  build args with `parser.parse_args([...])`, including `--dataset`, `--device cpu`, the run's trading flags from
  `trading_config.yaml`, `--context_window`/`--memory_capacity` from `data/<ds>/run.env`, and `--subagent_path` =
  `result/low_level/<ds>@<tag>/best_model`.
- **Run.** Load `best_model.pkl` and run `act_test` greedily over each split's `whole/<split>.feather` with a venue env
  using `sizing="fixed"`, so it can never go bankrupt or stop early.
- **Output.** Write `<run_dir>/<split>/signal_<split>.npz` with `timestamp`, `direction` (via `cfg.directions`) and
  `df_row`.
- **Parity check.** Wherever the old `test/action.npy` exists, the exported actions must equal it over its length.
  Report the mismatch count; it should be 0, because `act_test` does not depend on equity.

## 12. `report.py` and `replay.py`
- **`summarize(result, capital, bar_hours)`:**
  - net PnL and return; all-time max drawdown and max trailing-reference drawdown; annualised Sharpe of per-bar equity
    returns;
  - buy-and-hold return over the same bars;
  - exposure mean and p95; % of bars long / short / flat;
  - long vs short PnL; fees, funding and slippage;
  - round trips per day; counts of stops, liquidations and breaches; % of bars halted.
- **`attribution(result)`:** how often each constraint binds among entries; wanted vs permitted (mean ratio); abstain
  reasons.
- **`to_markdown(...)`:** per-run and across-seed tables (mean ± std), as in `MacroHFT/trading/report.py`.
- **CLI.** `python -m tm_risk.replay --macrohft RUN_DIR [RUN_DIR ...] --dataset_root MacroHFT/data --policy default_35dd
  [--policy ...] --venue kraken_us_perp --out report.md`.
  - It resolves the dataset from the run dir name (`<ds>@<tag>`).
  - It replays every (run, policy) pair and prints and writes the markdown.

## 13. Tests: `box/bench/tests/test_risk_*.py`
Required checks:
- **Bridge:** identity parity with `Account` to 1e-9, on the BTCGOLD test log and on a synthetic liquidation path.
- **Alignment:** `from_macrohft` join and close equality.
- **Import:** `import tm_risk` under `.venv/bin/python` without gym or torch.
- **Budgets and vetoes:**
  - an oversized request is capped, with the binding constraint named `trade_loss`;
  - profits don't refill the episode loss;
  - the halt blocks entries but exits still execute;
  - the stressed loss of new entries never exceeds drawdown headroom;
  - breach deleverages.
- **Flip ordering:** the new leg is sized after the old one closes.
- **Stops:** a gap through a stop fills at the open.
- **Causality:**
  - perturbing rows > k leaves decisions ≤ k identical;
  - a truncated stream gives the identical prefix.
- **Ledger:** identical under `default_35dd` and an always-halted policy.
- **Kelly:**
  - abstains on zero-mean noise;
  - f ≈ kelly_fraction·μ/σ² on a large Gaussian sample with small support uncertainty;
  - abstains below min support.
- **Policy validation:**
  - L = 3 or 10 is rejected on perps;
  - exposure_cap > L is rejected;
  - spot with L ≠ 1 is rejected.
- **Export parity** with `action.npy`.
