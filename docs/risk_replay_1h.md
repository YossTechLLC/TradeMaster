# 1h MacroHFT signals under the tm_risk arms (test split, Kraken US perp)

Status: 2026-09-30. Implements the spec in `docs/tm_risk_spec.md`. All numbers below come from the commands in the last section.

## Method

The 18 trained 1h high-level runs (datasets `BTCUSDT_1h_xt10` and `BTCUSDT_1h_upstream_real`, penalties tp0 / tp5 / tp10, 3 seeds each) were
re-run greedily over the full validation and test split with `MacroHFT/trading/export_signal.py`. The export uses a fixed-size environment, so it can
never go bankrupt or stop early. It writes the direction decided at each bar close to `<run>/<split>/signal_<split>.npz`. Parity against the old
`test/action.npy` is 0 mismatches on all 18 runs; one old log (upstream_real tp10 seed 12345) ends at bankruptcy after 5166 of 8735 rows and was
compared over those 5166 rows only. The test split (8735 hourly bars, buy&hold +107.3%) was then replayed
bar by bar under four policies on `kraken_us_perp` with 10000 USD capital. A model direction is only a request: the policy decides the size, the
admission layer caps it, and stops, deleveraging and halts act on the open position. Decisions at bar k use bars up to k only.

| policy | sizing | checks | margin L | exposure cap | stop | slippage | budgets / halt |
|---|---|---|---|---|---|---|---|
| `identity_L5` | identity (5x equity, the old behaviour) | off | 5 | 5 | none | 0 | none |
| `constant_1x` | 1.0x equity | on | 5 | 1.0 | 3 ATR | 5 bps | 2% loss per trade, 5% aggregate, 5% per risk episode, 35% halt on a 90-day peak |
| `vol_target_25` | 25% annual vol target | on | 5 | 1.0 | 3 ATR | 5 bps | same |
| `default_35dd` | robust fractional Kelly (half Kelly, 10% bootstrap quantile, 30 episodes per side needed) | on | 5 | 1.0 | 3 ATR | 5 bps | same |

Kelly learns only from the raw model signal's own closed episodes seen so far (no validation seeding), so it needs 30 closed long and 30 closed
short episodes before it can size anything, and it abstains whenever the bootstrap 10% quantile of growth is not positive.
`identity_L5` reproduces the original `test/trading_log` equity to at most 1.2e-10 USD on every run that has one (table at the end), so the
comparison changes only the risk layer, not the accounting. These replays read `signal_test.npz`, so the agreement also validates the
export-to-bar alignment (`from_signal_npz`) end to end.

Columns: return and Sharpe are on per-bar equity; "max DD (all-time)" is against the all-time peak; "max DD (90d ref)" is against the trailing 90-day
peak that the 35% halt uses; "% in market" is bars with an open position; "liq" is liquidations per run. Cells are mean ± std across the 3 seeds (n-1).
Drawdowns are clamped at 100%: identity_L5 can end with negative equity (bankruptcy, as the original env allows), which no real account can lose. Round trips count every close, including the close of a flip whose new leg was refused.
Buy&hold is the same for every row (+107.3%).

## Results per (profile, penalty) and arm

### BTCUSDT_1h_upstream_real / long_short-kraken_us_perp-L5-tp0

| arm | seeds | return | max DD (all-time) | max DD (90d ref) | Sharpe | exposure mean | % in market | round trips/day | fees USD | stops | % halted | liq | buy&hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity_L5 | 3 | -99.2% ± +0.5% | 99.6% ± 0.3% | 96.9% ± 0.8% | -1.21 ± 0.34 | 3.37 ± 0.54 | 75 ± 13 | 2.37 ± 0.23 | 6128 ± 1023 | 0 ± 0 | 0.0 ± 0.0 | 0.3 ± 0.6 | +107.3% ± +0.0% |
| constant_1x | 3 | -52.3% ± +5.8% | 56.6% ± 5.4% | 30.3% ± 3.5% | -3.13 ± 0.58 | 0.37 ± 0.01 | 61 ± 3 | 2.16 ± 0.12 | 1694 ± 177 | 98 ± 6 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| vol_target_25 | 3 | -48.3% ± +3.6% | 52.8% ± 3.6% | 27.9% ± 2.4% | -3.47 ± 0.42 | 0.31 ± 0.01 | 67 ± 3 | 2.51 ± 0.08 | 1589 ± 111 | 109 ± 3 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| default_35dd | 3 | +0.0% ± +0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | 0.00 ± 0.00 | 0 ± 0 | 0 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |

### BTCUSDT_1h_upstream_real / long_short-kraken_us_perp-L5-tp10

| arm | seeds | return | max DD (all-time) | max DD (90d ref) | Sharpe | exposure mean | % in market | round trips/day | fees USD | stops | % halted | liq | buy&hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity_L5 | 3 | -99.8% ± +0.2% | 99.9% ± 0.1% | 99.6% ± 0.3% | -0.86 ± 0.17 | 3.27 ± 0.26 | 56 ± 5 | 0.10 ± 0.01 | 531 ± 39 | 0 ± 0 | 0.0 ± 0.0 | 4.0 ± 1.0 | +107.3% ± +0.0% |
| constant_1x | 3 | -25.2% ± +7.0% | 31.6% ± 4.6% | 18.9% ± 1.6% | -1.70 ± 0.65 | 0.17 ± 0.03 | 27 ± 4 | 0.15 ± 0.03 | 150 ± 29 | 30 ± 4 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| vol_target_25 | 3 | -17.4% ± +5.2% | 22.2% ± 3.8% | 14.6% ± 0.9% | -1.63 ± 0.65 | 0.11 ± 0.01 | 27 ± 4 | 0.15 ± 0.03 | 107 ± 21 | 30 ± 4 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| default_35dd | 3 | +0.0% ± +0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | 0.00 ± 0.00 | 0 ± 0 | 0 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |

### BTCUSDT_1h_upstream_real / long_short-kraken_us_perp-L5-tp5

| arm | seeds | return | max DD (all-time) | max DD (90d ref) | Sharpe | exposure mean | % in market | round trips/day | fees USD | stops | % halted | liq | buy&hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity_L5 | 3 | -99.4% ± +0.3% | 99.8% ± 0.1% | 99.3% ± 0.2% | -1.55 ± 0.35 | 2.04 ± 0.09 | 38 ± 0 | 0.37 ± 0.09 | 2755 ± 526 | 0 ± 0 | 0.0 ± 0.0 | 3.3 ± 0.6 | +107.3% ± +0.0% |
| constant_1x | 3 | -36.4% ± +8.0% | 45.7% ± 5.1% | 25.4% ± 1.3% | -1.98 ± 0.55 | 0.32 ± 0.01 | 45 ± 2 | 1.00 ± 0.17 | 1029 ± 30 | 81 ± 7 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| vol_target_25 | 3 | -29.2% ± +4.0% | 38.2% ± 1.7% | 20.8% ± 1.1% | -2.10 ± 0.32 | 0.22 ± 0.01 | 45 ± 2 | 1.03 ± 0.20 | 787 ± 79 | 81 ± 7 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| default_35dd | 3 | +0.0% ± +0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | 0.00 ± 0.00 | 0 ± 0 | 0 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |

### BTCUSDT_1h_xt10 / long_short-kraken_us_perp-L5-tp0

| arm | seeds | return | max DD (all-time) | max DD (90d ref) | Sharpe | exposure mean | % in market | round trips/day | fees USD | stops | % halted | liq | buy&hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity_L5 | 3 | +328.3% ± +81.2% | 92.4% ± 3.3% | 91.0% ± 1.9% | 1.88 ± 0.07 | 5.03 ± 0.03 | 100 ± 0 | 2.02 ± 0.19 | 70520 ± 51348 | 0 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| constant_1x | 3 | -8.8% ± +5.2% | 36.4% ± 1.8% | 25.3% ± 1.5% | -0.21 ± 0.22 | 0.49 ± 0.02 | 76 ± 1 | 1.72 ± 0.06 | 1873 ± 410 | 122 ± 5 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| vol_target_25 | 3 | -3.5% ± +8.2% | 27.4% ± 7.6% | 20.2% ± 4.5% | -0.07 ± 0.41 | 0.40 ± 0.01 | 83 ± 1 | 1.91 ± 0.14 | 1586 ± 177 | 131 ± 9 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| default_35dd | 3 | -1.6% ± +4.9% | 6.0% ± 5.6% | 5.1% ± 4.1% | -0.15 ± 0.83 | 0.03 ± 0.02 | 9 ± 6 | 0.21 ± 0.13 | 78 ± 52 | 15 ± 11 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |

### BTCUSDT_1h_xt10 / long_short-kraken_us_perp-L5-tp10

| arm | seeds | return | max DD (all-time) | max DD (90d ref) | Sharpe | exposure mean | % in market | round trips/day | fees USD | stops | % halted | liq | buy&hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity_L5 | 3 | +69.9% ± +223.3% | 97.8% ± 2.1% | 95.6% ± 3.8% | 1.16 ± 0.69 | 4.60 ± 0.15 | 93 ± 1 | 0.52 ± 0.02 | 4312 ± 2415 | 0 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| constant_1x | 3 | +15.9% ± +11.2% | 31.9% ± 2.5% | 23.1% ± 3.4% | 0.70 ± 0.39 | 0.41 ± 0.03 | 61 ± 5 | 0.51 ± 0.01 | 527 ± 16 | 55 ± 1 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| vol_target_25 | 3 | +10.1% ± +6.4% | 22.6% ± 2.0% | 17.6% ± 1.6% | 0.67 ± 0.36 | 0.26 ± 0.02 | 61 ± 5 | 0.52 ± 0.02 | 346 ± 11 | 55 ± 1 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| default_35dd | 3 | +0.0% ± +0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | 0.00 ± 0.00 | 0 ± 0 | 0 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |

### BTCUSDT_1h_xt10 / long_short-kraken_us_perp-L5-tp5

| arm | seeds | return | max DD (all-time) | max DD (90d ref) | Sharpe | exposure mean | % in market | round trips/day | fees USD | stops | % halted | liq | buy&hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identity_L5 | 3 | -62.0% ± +46.4% | 99.8% ± 0.2% | 98.8% ± 0.4% | 0.83 ± 1.11 | 3.94 ± 1.01 | 81 ± 22 | 0.72 ± 0.23 | 16020 ± 8714 | 0 ± 0 | 0.0 ± 0.0 | 1.0 ± 0.0 | +107.3% ± +0.0% |
| constant_1x | 3 | +28.4% ± +19.0% | 31.4% ± 6.6% | 25.0% ± 1.4% | 1.02 ± 0.56 | 0.46 ± 0.02 | 64 ± 3 | 0.85 ± 0.04 | 968 ± 131 | 72 ± 1 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| vol_target_25 | 3 | +19.5% ± +9.0% | 22.2% ± 4.9% | 17.5% ± 1.7% | 1.10 ± 0.42 | 0.29 ± 0.02 | 65 ± 3 | 0.87 ± 0.05 | 640 ± 75 | 74 ± 1 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |
| default_35dd | 3 | +0.0% ± +0.0% | 0.0% ± 0.0% | 0.0% ± 0.0% | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | 0.00 ± 0.00 | 0 ± 0 | 0 ± 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | +107.3% ± +0.0% |

### Attribution (all 18 runs pooled, entries and abstentions of sized proposals)

| arm | entries | abstains | of which sizing declined (Kelly / no vol) | Kelly abstention rate | permitted/wanted at proposals | binding at entries |
|---|---|---|---|---|---|---|
| constant_1x | 6969 | 66260 | 0 | 0.0% | 0.08 | trade_loss 4025, episode_loss 1695, request 1249 |
| vol_target_25 | 7629 | 62314 | 24 | 0.0% | 0.13 | request 6455, episode_loss 844, trade_loss 330 |
| default_35dd | 233 | 151158 | 150245 | 99.2% | 0.92 | request 233 |

### Abstain reasons, default_35dd (all runs)

- sizing: no positive robust growth: 103954
- sizing: insufficient support: 46291
- veto: blocked_direction: 513
- below one contract: 400

### identity_L5 vs the original test/trading_log equity

| run | log rows / bars | max abs equity diff USD | old final | replay final at log end |
|---|---|---|---|---|
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp0 seed_12345 | 8735 / 8735 | 4.21e-12 | 107.10 | 107.10 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp0 seed_23456 | 8735 / 8735 | 3.64e-12 | 104.56 | 104.56 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp0 seed_34567 | 8735 / 8735 | 5.68e-13 | 27.37 | 27.37 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp10 seed_12345 | 5166 / 8735 | 0.00e+00 | -7.43 | -7.43 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp10 seed_23456 | 8735 / 8735 | 1.14e-13 | 35.66 | 35.66 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp10 seed_34567 | 8735 / 8735 | 7.28e-12 | 16.85 | 16.85 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp5 seed_12345 | 8735 / 8735 | 3.64e-12 | 46.23 | 46.23 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp5 seed_23456 | 8735 / 8735 | 1.82e-12 | 90.66 | 90.66 |
| BTCUSDT_1h_upstream_real long_short-kraken_us_perp-L5-tp5 seed_34567 | 8735 / 8735 | 9.66e-13 | 39.02 | 39.02 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp0 seed_12345 | 8735 / 8735 | 5.09e-11 | 38128.48 | 38128.48 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp0 seed_23456 | 8735 / 8735 | 1.46e-11 | 52258.52 | 52258.52 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp0 seed_34567 | 8735 / 8735 | 4.37e-11 | 38222.06 | 38222.06 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp10 seed_12345 | 8735 / 8735 | 3.64e-12 | 42618.70 | 42618.70 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp10 seed_23456 | 8735 / 8735 | 3.64e-12 | 6881.56 | 6881.56 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp10 seed_34567 | 8735 / 8735 | 2.91e-11 | 1503.52 | 1503.52 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp5 seed_12345 | 8735 / 8735 | 1.46e-11 | 2305.37 | 2305.37 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp5 seed_23456 | 8735 / 8735 | 1.16e-10 | 9013.57 | 9013.57 |
| BTCUSDT_1h_xt10 long_short-kraken_us_perp-L5-tp5 seed_34567 | 8735 / 8735 | 7.28e-12 | 93.66 | 93.66 |

## Conclusion

The earlier 89-100% drawdowns came from running the model's direction at 5x equity with no stop, no budget and no halt: on this test split the
identity arm's all-time max drawdown is 92-99.9% in every one of the 6 (profile, penalty) groups (group means), 10 of its 18 runs
were liquidated at least once, and it ends the test year down 62-99.8% in 4 of the 6 groups even though two xt10 groups show large positive
means (xt10 tp0 +328% ± 81% with a 92% drawdown; xt10 tp10 +70% ± 223%).
Capping exposure at 1x with a 3 ATR stop and loss budgets cuts the all-time drawdown to 22-57% depending on the model (group means, both arms), and the 90-day-reference
drawdown to 15-30%; vol targeting cuts it a little more than constant 1x. It does not rescue a model that has no edge: the three upstream_real groups lose under every arm (`constant_1x` and `vol_target_25`
still lose 17-52% at 2 round trips a day, with 5 bps slippage plus fees as the friction). The most informative row is xt10 tp0: the same
direction sequence earns +328% ± 81% at 5x uncapped (with a 92% drawdown), but -8.8% / -3.5% once capped at 1x and stopped, because the
1x cap, the per-episode loss budgets and 3 ATR stops (122-131 stop-outs, 1.7-1.9 round trips a day) turn a high-leverage winner into a
small loser. On xt10 tp5 / tp10 the capped arms earn 10-28% against +107% buy&hold with Sharpe 0.7-1.1, so the drawdown fell far more than the return.

The 35% halt never fired (0.0% of bars halted in all 72 replays): the trade-loss and headroom caps keep losses small enough that a 35% fall
against the 90-day peak is not reached, while a slow bleed can still take the all-time drawdown above 35% (constant_1x upstream_real tp0: 57%),
because the reference peak rolls off. Read "35dd" as a limit on drawdown from the trailing 90-day peak, not on the all-time drawdown.

The Kelly arm mostly abstains, and that should be said plainly: 99.2% of its bar-level proposals were declined for lack of support or for no
positive robust growth, it traded in only 3 of 18 runs, and there it held 0.03x mean exposure and returned -1.6% ± 4.9% with a 6% drawdown. Its
0% drawdown elsewhere is not skill at risk control; it is not trading. Since the ledger is causal and account-independent it is doing what it
was designed to do with weak, unstable edge, but as configured (no validation seeding, 30 episodes per side, 10% quantile) it is a filter that
switches this model off rather than a sizing rule, and the honest reading of these 18 runs is that the 1h models carry little edge that survives
costs. Validation seeding (`ledger_seed: val`) would give Kelly earlier support but is selection-biased and was not run.

Caveats: one test period, 3 seeds per group, theoretical fills (fee per contract, 5 bps slippage, stops fill at the open when gapped through, no
intrabar order knowledge beyond the worst-case rule), funding at the baseline interest rate only, and unverified Kraken US parameters flagged in the venue file.

## Commands

```bash
# 1. full signals, 18 runs x (val, test); parity vs action.npy is printed per run (one run per result key in parallel; seeds share a scratch dir)
cd MacroHFT
for key in $(ls result/high_level | grep '^BTCUSDT_1h_.*@long_short'); do ds=${key%@*}
  ( for rd in result/high_level/$key/*/seed_*; do ../.venv-hft/bin/python trading/export_signal.py --run_dir $rd --dataset $ds --splits val test; done ) &
done; wait; cd ..
# 2. replay 4 arms x 18 runs on the test split (kraken_us_perp, capital 10000), identity check against the old logs, tables
.venv-hft/bin/python box/bench/risk_replay_1h.py --json /tmp/items.json > tables.md
# a single run through the CLI
.venv-hft/bin/python -m tm_risk.replay --macrohft MacroHFT/result/high_level/BTCUSDT_1h_xt10@long_short-kraken_us_perp-L5-tp5/seed12345/seed_12345 \
  --dataset_root MacroHFT/data --policy identity_L5 --policy constant_1x --policy vol_target_25 --policy default_35dd --venue kraken_us_perp
# 3. checks
for t in box/bench/tests/test_*.py; do .venv-hft/bin/python $t; done
.venv/bin/python -c "import tm_risk"
.venv-hft/bin/python box/bench/golden.py run rk_int macro && .venv-hft/bin/python box/bench/golden.py compare p6 rk_int   # IDENTICAL
```
