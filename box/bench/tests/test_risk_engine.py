"""Engine/acceptance tests for tm_risk.replay (halt, headroom, breach, flip order, causality, ledger, episode, bridge); hand-computed numbers."""
import os, sys, warnings
from dataclasses import replace
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
warnings.simplefilter("ignore")
from MacroHFT.trading.accounting import Account
from MacroHFT.trading.config import TradingConfig
from tm_risk import from_frame, load_policy, load_venue, replay
from tm_risk.stops import stressed_loss_per_unit

KRAKEN = load_venue("kraken_us_perp")     # 0.15 USD/contract/side, 0.01 BTC/contract, funding 1e-4 per 8h = 1.25e-5/h
FUND = 1.25e-5
BASE = load_policy("constant_1x")


def frame(close, direction, low=None, high=None, qv=None):
    n = len(close)
    close = np.asarray(close, dtype=float)
    d = dict(timestamp=pd.date_range("2024-01-01", periods=n, freq="h"), open=close, high=close if high is None else high,
             low=close if low is None else low, close=close, direction=direction)
    if qv is not None:
        d["quote_volume"] = qv
    return pd.DataFrame(d)


def load_fast(name):
    """load_policy with a smaller bootstrap so the kelly arm replays in seconds (behaviour identical, noisier quantile)."""
    p = load_policy(name)
    return replace(p, kelly_bootstrap=100) if p.sizing == "kelly" else p


def loose(**kw):
    """constant 1x, no stops, no halt, budgets far above the request: only request/exposure_cap/episode can bind."""
    return replace(BASE, stop_atr_mult=0, dd_halt=1.0, trade_loss_budget=10.0, aggregate_loss_budget=10.0,
                   episode_loss_budget=10.0, **kw)


def dec(res, k, action=None):
    ds = [d for d in res.decisions if d.k == k and action in (None, d.action)]
    assert len(ds) == 1, (k, [(d.action, d.reason) for d in ds])
    return ds[0]


def synth(n=3000, seed=0, vol=0.012):
    """Lognormal walk with wicks and a noisy 3-bar-foresight signal (a fake model with real edge; runs of flips)."""
    rng = np.random.default_rng(seed)
    c = 50000 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 3, n)))
    fut = np.concatenate([c[3:], np.repeat(c[-1], 3)]) - c
    d = np.sign(fut) * np.where(rng.random(n) < 0.3, -1, 1)
    d[rng.random(n) < 0.15] = 0
    df = frame(c, d.astype(int), l, h, qv=np.full(n, 1e9))
    df["open"] = o
    return df


def test_halt_blocks_entries_but_exits_execute():
    pol = replace(BASE, stop_atr_mult=0, dd_halt=0.01)
    close = [50000, 50000, 40000, 40000, 40000, 40000, 40000, 40000]
    # A: long held through the halt, then exit; later entries vetoed
    res = replay(from_frame(frame(close, [0, 1, 1, 1, 0, 1, -1, 0])), pol, KRAKEN)
    e = dec(res, 1)
    assert e.action == "enter" and e.contracts == 3        # headroom: 100/(0.05+2*3e-4+1e-3)=1938 USD -> 3 contracts (loss 300 then halts)
    assert res.bars["halted"][2] == 1 and dec(res, 2).action == "hold" and dec(res, 3).action == "hold"
    x = dec(res, 4)
    assert x.action == "exit" and x.contracts == -3 and res.bars["contracts"][4] == 0          # exit executes while halted
    for k in (5, 6):
        a = dec(res, k)
        assert a.action == "abstain" and a.vetoes == ("halted",) and a.binding == "halted" and a.permitted == 0
        assert res.bars["contracts"][k] == 0
    # B: a flip while halted closes the old leg but opens nothing
    res = replay(from_frame(frame(close, [0, 1, 1, -1, 1, 0, 0, 0])), pol, KRAKEN)
    f = dec(res, 3, "abstain")
    assert f.vetoes == ("halted",) and res.bars["contracts"][3] == 0
    x = dec(res, 3, "exit")                                                                     # the close of the old leg is recorded
    assert x.contracts == -3 and x.signal == -1
    assert res.bars["fee"][3] == 3 * 0.15                                                       # the old leg really closed
    assert dec(res, 4).action == "abstain" and (res.bars["contracts"][3:] == 0).all()


def test_stressed_loss_of_entries_within_drawdown_headroom():
    pol = replace(BASE, stop_atr_mult=3, dd_halt=0.10, trade_loss_budget=10.0, aggregate_loss_budget=10.0,
                  episode_loss_budget=10.0, margin_leverage=5, exposure_cap=5.0, constant_exposure=5.0, participation=0)
    for seed in (1, 2):
        st = from_frame(synth(3000, seed, vol=0.02))
        res = replay(st, pol, KRAKEN)
        ent = [d for d in res.decisions if d.action in ("enter", "flip")]
        assert len(ent) > 50
        n_bind = 0
        for d in ent:
            k = d.k
            u = stressed_loss_per_unit(st, k, KRAKEN, pol)
            room = res.bars["equity_mark"][k] - (1 - pol.dd_halt) * res.bars["peak_ref"][k]
            stressed = abs(d.contracts) * KRAKEN.contract_size * st.close[k] * u
            assert stressed <= room + 1e-6, (k, stressed, room)
            n_bind += d.binding == "drawdown_headroom"
        assert n_bind > 5                                                     # non-vacuous: headroom actually binds


def test_breach_deleverages_on_rally_while_short():
    pol = loose()
    close = [50000, 50000, 52000, 55000, 58000, 60000, 60000]
    res = replay(from_frame(frame(close, [0, -1, -1, -1, -1, -1, 0])), pol, KRAKEN)
    assert dec(res, 1).contracts == -20 and dec(res, 1).action == "enter"            # 10000 / (0.01*50000)
    # E1 = 10000 - 3 - 5; bar2 -400 (+0.125 funding received), bar3 -600 (+0.13), bar4 -600 (+0.1375)
    e4 = 10000 - 3 - 5 - 400 + 0.125 - 600 + 0.13 - 600 + 0.1375
    assert abs(res.bars["equity_mark"][4] - e4) < 1e-9
    for k in (2, 3):
        assert not any(d.action == "breach" and d.k == k for d in res.decisions)      # exposure 1.08 / 1.22 < 1.25
    b = dec(res, 4, "breach")
    assert b.action == "breach" and b.contracts == 6                  # exposure 11600/8392 = 1.38 -> floor(8392.39/580) = 14 left
    assert res.bars["contracts"][4] == -14
    assert abs(res.bars["equity"][4] - (e4 - 6 * 0.15 - 5e-4 * 6 * 0.01 * 58000)) < 1e-9
    assert res.bars["exposure"][4] <= 1.0 and sum(d.action == "breach" for d in res.decisions) == 1
    assert res.summary["breaches"] == 1


def test_refused_flip_records_the_close():
    # long 20 contracts at 50000 (1x of 10000), short signal at k=3 where volume allows 0 contracts: old leg closes, new is refused
    qv = [1e9, 1e9, 1e9, 1.0, 1e9, 1e9]
    res = replay(from_frame(frame([50000] * 6, [0, 1, 1, -1, -1, 0], qv=qv)), loose(), KRAKEN)
    assert dec(res, 1).action == "enter" and dec(res, 1).contracts == 20
    x, a = dec(res, 3, "exit"), dec(res, 3, "abstain")
    assert x.contracts == -20 and a.binding == "participation" and res.bars["contracts"][3] == 0
    assert res.bars["fee"][3] == 20 * 0.15
    assert res.summary["rt_per_day"] > 0 and sum(d.action in ("exit", "flip") and d.k == 3 for d in res.decisions) == 1


def test_halt_is_a_trailing_control_and_all_time_dd_is_reported():
    # slow bleed of 0.5%/bar: the all-time drawdown passes 35% but a 24-bar trailing peak never does
    n = 90
    close = 50000.0 * 0.995 ** np.arange(n)
    pol = replace(BASE, stop_atr_mult=0, dd_halt=0.35, dd_window_days=1.0, trade_loss_budget=10.0, episode_loss_budget=10.0,
                  aggregate_loss_budget=10.0)
    res = replay(from_frame(frame(close, [0] + [1] * (n - 2) + [0])), pol, KRAKEN)
    assert res.summary["max_dd"] > 0.30 and res.summary["max_dd_ref"] < res.summary["max_dd"]    # all-time DD exceeds the trailing one
    assert res.summary["pct_halted"] == 0.0                                                     # documented: halt is trailing, not all-time


def test_liquidation_blocks_same_direction_reentry_with_checks():
    from tm_risk.breach import mark_bar
    from tm_risk.types import AccountState
    st = AccountState(capital=1000.0, equity=1000.0, contracts=20.0, entry_price=50000.0, stop_price=0.0)
    df = frame([50000, 40000], [1, 1], low=[50000, 40000], high=[50000, 50000])
    df.loc[1, "open"] = 50000.0
    stream = from_frame(df)
    m = mark_bar(st, stream, 1, KRAKEN, BASE)
    assert m.event == "liquidation" and st.contracts == 0 and st.blocked_direction == 1
    st2 = AccountState(capital=1000.0, equity=1000.0, contracts=20.0, entry_price=50000.0)
    m2 = mark_bar(st2, stream, 1, KRAKEN, replace(BASE, checks=False, sizing="identity"))
    assert m2.event == "liquidation" and st2.blocked_direction == 0                             # identity path unchanged


def test_flip_sizes_new_leg_after_closing_old():
    pol = replace(loose(), episode_loss_budget=0.05)
    res = replay(from_frame(frame([50000, 50000, 48000, 48000, 48000], [0, 1, 1, -1, 0])), pol, KRAKEN)
    e = dec(res, 1)
    u1 = 0.05 + 2 * 3e-4 + 2 * 5e-4
    assert e.binding == "episode_loss" and e.contracts == 19 and abs(e.permitted - 500 / u1) < 1e-6
    # the closed long lost 0.19*2000 + 2.85 fee + 4.56 slippage = 387.41 -> 112.59 of the 500 allowance is left for the short
    f = dec(res, 3)
    u3 = 0.05 + 2 * 15 / 48000 + 2 * 5e-4
    assert f.action == "flip" and f.binding == "episode_loss" and f.contracts == -4
    assert abs(f.permitted - (500 - 387.41) / u3) < 1e-6
    assert res.bars["contracts"][3] == -4 and dec(res, 4).action == "exit"     # sized before the close it would have been -20


def test_profits_do_not_refill_episode_loss():
    pol = replace(loose(), episode_loss_budget=0.10)        # 1000 allowance fixed at the first admission (E=10000)
    close = [50000, 50000, 47000, 47000, 40000, 40000, 40000]
    res = replay(from_frame(frame(close, [0, 1, 1, -1, -1, 1, 0])), pol, KRAKEN)
    assert dec(res, 1).contracts == 20
    spent = 600 + 3.0 + 5e-4 * 0.2 * 47000                    # realised loss of the long at the flip: 607.7
    u3 = 0.05 + 2 * 15 / 47000 + 2 * 5e-4
    f3 = dec(res, 3)
    assert f3.action == "flip" and f3.binding == "episode_loss" and f3.contracts == -16
    assert abs(f3.permitted - (1000 - spent) / u3) < 1e-6
    # the short then earned +1114.4 (1120 - 2.4 fee - 3.2 slippage) but the allowance stays 1000 - 607.7
    u5 = 0.05 + 2 * 15 / 40000 + 2 * 5e-4
    f5 = dec(res, 5)
    assert f5.action == "flip" and f5.binding == "episode_loss" and f5.contracts == 18
    assert abs(f5.permitted - (1000 - spent) / u5) < 1e-6
    assert f5.wanted > 10000 and f5.permitted < 8000            # a refilled allowance would have allowed ~26 contracts


def test_causality_perturb_future_and_truncation():
    df = synth(1400, seed=3)
    K, N = 700, len(df)
    rng = np.random.default_rng(9)
    df2 = df.copy()
    for c in ("open", "high", "low", "close"):
        df2.loc[K + 1:, c] = df[c].to_numpy()[K + 1:] * (1 + 0.05 * rng.standard_normal(N - K - 1))
    df2["high"] = df2[["open", "high", "low", "close"]].max(axis=1)
    df2["low"] = df2[["open", "high", "low", "close"]].min(axis=1)
    df2.loc[K + 1:, "direction"] = rng.integers(-1, 2, N - K - 1)
    df2.loc[K + 1:, "quote_volume"] = rng.uniform(1e6, 1e9, N - K - 1)
    assert not np.array_equal(df["close"].to_numpy()[K + 1:], df2["close"].to_numpy()[K + 1:])
    keys = ("k", "action", "signal", "contracts", "wanted", "permitted", "binding", "vetoes", "exposure_req")
    row = lambda d: tuple(getattr(d, x) for x in keys)
    for name in ("default_35dd", "vol_target_25", "constant_1x"):
        pol = load_fast(name)
        full, per = replay(from_frame(df), pol, KRAKEN), replay(from_frame(df2), pol, KRAKEN)
        a, b = [row(d) for d in full.decisions if d.k <= K], [row(d) for d in per.decisions if d.k <= K]
        assert a == b and len(a) > 20, name
        for key in ("equity", "contracts", "pnl", "fee", "stop_price", "peak_ref"):
            assert np.array_equal(full.bars[key][:K + 1], per.bars[key][:K + 1]), (name, key)
        # truncated stream: identical up to the (forced-exit) last bar
        cut = replay(from_frame(df).head(K + 1), pol, KRAKEN)
        a, b = [row(d) for d in full.decisions if d.k < K], [row(d) for d in cut.decisions if d.k < K]
        assert a == b, name
        for key in ("equity", "contracts", "pnl", "fee", "stop_price"):
            assert np.array_equal(full.bars[key][:K], cut.bars[key][:K]), (name, key)
        assert cut.bars["contracts"][K] == 0 and cut.info["truncated"]


def test_ledger_independent_of_account_and_hand_check():
    df = synth(1500, seed=4)
    st = from_frame(df)
    pol = load_fast("default_35dd")
    halted = replace(pol, dd_halt=1e-9)
    a, b, c = replay(st, pol, KRAKEN), replay(st, halted, KRAKEN), replay(st, replace(pol, dd_halt=0.01, slippage_bps=100.0), KRAKEN)
    for other in (b, c):
        for side in (1, -1):
            for x, y in zip(a.info["ledger"][side], other.info["ledger"][side]):
                assert np.array_equal(x, y) and len(x) > 30
    assert b.summary["entries"] == 0 and b.summary["abstains"] > 0                 # headroom ~0: nothing is ever admitted
    assert c.summary["entries"] < a.summary["entries"] / 10                 # a tight-halt, costly account really differs
    assert a.summary["entries"] > 20
    # hand check: long 1->3, short 4->5 (exit rows 3 and 5), 1 h bars, fee 15/price, funding 1.25e-5/h
    c = [50000, 50000, 51000, 52000, 51000, 50000]
    res = replay(from_frame(frame(c, [0, 1, 1, 0, -1, 0])), pol, KRAKEN)
    (r_l, h_l, _), (r_s, h_s, _) = res.info["ledger"][1], res.info["ledger"][-1]
    assert h_l[0] == 2 and h_s[0] == 1
    assert abs(r_l[0] - (0.04 - 15 / 50000 - 15 / 52000 - 2 * FUND)) < 1e-12
    assert abs(r_s[0] - ((1 - 50000 / 51000) - 15 / 51000 - 15 / 50000 + FUND)) < 1e-12


def _account_equity(df, lev):
    cfg = TradingConfig(mode="long_short", venue=KRAKEN, leverage=lev, capital=10_000.0).validate()
    acc = Account(cfg, 1.0)
    for i in range(len(df) - 1):
        acc.step(float(df.direction[i]), df.close[i], df.close[i + 1], df.low[i + 1], df.high[i + 1])
    return acc.log()["equity"]


def test_identity_bridge_random_path_with_flips_and_liquidations():
    for seed, lev, vol in ((5, 5.0, 0.01), (6, 8.5, 0.03)):
        df = synth(800, seed, vol=vol)
        res = replay(from_frame(df), load_policy("identity_L5") if lev == 5.0 else
                     replace(load_policy("identity_L5"), margin_leverage=lev, exposure_cap=lev), KRAKEN)
        ref = _account_equity(df, lev)
        assert np.max(np.abs(res.bars["equity_mark"][1:] - ref)) < 1e-9, (seed, np.max(np.abs(res.bars["equity_mark"][1:] - ref)))
        assert res.summary["entries"] > 30
    assert res.summary["liquidations"] > 0                                    # the high-vol path exercises liquidation


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"OK test_risk_engine: {len(tests)} tests")
