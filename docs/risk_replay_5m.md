# tm_risk replay: BTCUSDT 5m, P2 (xt10) and P0 (upstream_real), 3x margin, Kraken US perp

This is the overnight BOX campaign of 2026-09-30, queue `box/queue/ls5m_L3`. It ran through `MacroHFT/profiles/campaign.sh`:

| | |
|---|---|
| Datasets | `BTCUSDT_5m_xt10`, `BTCUSDT_5m_upstream_real` |
| Mode / venue / margin | long_short, kraken_us_perp, L=3 |
| Turnover penalties | 0, 5 and 10 bps |
| Seeds | 12345, 23456, 34567 |
| Runs | 18 hyper-agents over 36 sub-agents |
| Test period | ~1 year, 105,119 five-minute bars |
| Buy & hold, 1x, over the test period | +111.4% |

The full tables were pulled to `box/queue/ls5m_L3/replay_BTCUSDT_5m_{xt10,upstream_real}.{md,json}`. The local copy is under `box/queue/ls5m_L3_pulled/`.

The four replay arms:

| Arm | Sizing |
|---|---|
| `identity_L3` | the trained env's own accounting, full 3x |
| `constant_1x_L3` | 1x exposure, 3 × ATR stops, 35% drawdown admission |
| `vol_target_25_L3` | 25% vol target, same stops and admission |
| `default_35dd_L3` | robust half-Kelly, same stops and admission |

`identity_L3` reproduces every run's `test/trading_log` equity to ≤ 7.3e-12 USD. All 18 signal exports match the saved `action.npy` exactly (0 mismatches).

## Results (mean ± std over 3 seeds)

| profile | tp | identity 3x: return / max DD | constant 1x: return / max DD / Sharpe | vol 25%: return / max DD / Sharpe | Kelly |
|---|---|---|---|---|---|
| P2 xt10 | 0 | +113% ± 143 / 83% | +0.1% / 23% / 0.02 | +6.3% / 16% / 0.24 | abstains 100% |
| P2 xt10 | 5 | −12.6% / 91% | +51.0% / 17% / 1.45 | +22.1% / 11% / 1.43 | abstains 100% |
| P2 xt10 | 10 | +92.6% ± 5 / 66% | +37.7% / 30% / 0.97 | +12.7% / 16% / 0.70 | abstains 100% |
| P0 upstream_real | 0 | −96.2% / 98% | −74.4% / 76% / −5.4 | −54.6% / 55% / −6.9 | abstains 100% |
| P0 upstream_real | 5 | −47.2% / 84% | −51.4% / 55% / −2.6 | −26.0% / 30% / −2.5 | abstains 100% |
| P0 upstream_real | 10 | +79.6% ± 303 / 93% | −0.4% / 29% / −0.1 | −2.5% / 10% / −0.3 | abstains 100% |

## Findings

1. **P2's gains are long beta in a bull year, not edge.**
   - The identity arm is long 76–85% of the time.
   - Long legs earn +8k to +51k USD; short legs lose 9.5k–18.9k USD in every one of the 9 runs; funding costs 1.4k–4.6k.
   - Even the best P2 group (tp10, +93% at 3x) trails 1x buy & hold (+111%).
2. **P0 loses under every sizing arm.** Robust Kelly correctly finds "no positive robust growth" (82% of its abstentions).
3. **P2's seeds collapse to one policy.**
   - At tp5 all 3 seeds give identical actions; at tp10, 2 of 3 do.
   - The small ± on P2 tp5/tp10 therefore understates uncertainty. Those groups are effectively n = 1–2.
4. **The Kelly arm never trades at 5m.**
   - On P2 the abstention is "insufficient support": the signal changes direction only 0.02–0.17 times a day, so fewer than the required 30 closed episodes per side accumulate in a year.
   - On P0 the abstention reflects the absence of edge.
5. **The drawdown bound works where the sizing layer trades.** The 1x and vol-target arms keep the 90-day drawdown at 35% or below in every group, against 66–98% for the identity arm.
   - The all-time drawdown is larger on P0 tp0/tp5 (55–76%). The 90-day reference lets the halt lift as the peak rolls out of the window, so losses keep compounding.
   - This is why an all-time-halt option is worth adding.
6. **The 1x and vol arms sit out 42–79% of the time, because of stop churn at 5m.** After a 3 × ATR(14) stop, re-entry waits for the signal to change direction. At 5m, ATR(14) is about 70 minutes of range, so stops fire often: 5–376 per run.

## Compared with 1h (docs/risk_replay_1h.md, 5x margin)

- **Identity arm:** at 1h it had a 92–99.9% max drawdown in all 6 groups. At 5m and 3x it is 66–98%, still unacceptable.
- **Kelly arm:** abstained 99.2% at 1h and 100% at 5m.
- **Where the sizing layer's result holds up:** the best case is P2 tp5/tp10 at 1x, +38–51% at 17–30% drawdown. But that result rests on a long-biased signal whose seeds collapsed.

## Campaign incidents

- **Stage 4 (export):** 2 of 18 jobs failed at first. The concurrent seeds of one result key shared the scratch dir `export_tmp`.
  - Fixed in `MacroHFT/trading/export_signal.py`: the scratch dir is now `export_tmp_seed<n>`.
  - The campaign resumed, skipping stages 1–3, and all 18 exports succeeded.
- **No other failures.**
  - Peak RSS per job was 1.46 GB (hyper-agent) under a 2 GB cap.
  - About 6 GB of `epoch_*/val` diagnostics were pruned mid-run to keep disk above 25 GB. Every `best_model.pkl` was kept.
