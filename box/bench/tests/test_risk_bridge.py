"""Tests for tm_risk bridge parity (engine identity path vs MacroHFT Account), from_macrohft, mark_bar/deleverage and policy checks."""
import os, shutil, subprocess, sys, tempfile, warnings
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
warnings.simplefilter("ignore")
from MacroHFT.trading.accounting import Account
from MacroHFT.trading.config import TradingConfig
from tm_risk import (AccountState, PolicyConfig, deleverage, from_frame, from_macrohft, load_policy, load_venue, mark_bar,
                     replay)

RUN = os.path.join(TM, "MacroHFT/result/high_level/BTCGOLD@long_short-kraken_us_perp-L5-tp5/exp1/seed_12345")
DATA = os.path.join(TM, "MacroHFT/data/BTCGOLD")
KRAKEN = load_venue("kraken_us_perp")


_TMP = tempfile.mkdtemp()


def log_stream(split="test"):
    """from_macrohft over the trading log only (a signal_<split>.npz next to it would take precedence)."""
    d = os.path.join(_TMP, split)
    if not os.path.isdir(d):
        os.makedirs(d)
        shutil.copy(os.path.join(RUN, split, "trading_log.npz"), d)
    return from_macrohft(_TMP, DATA, split)


def frame(close, direction, low=None, high=None, open_=None):
    n = len(close)
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(dict(timestamp=pd.date_range("2024-01-01", periods=n, freq="h"),
                             open=close if open_ is None else open_, high=close if high is None else high,
                             low=close if low is None else low, close=close, direction=direction))


def account_path(df, lev):
    """Equity per step of Account for the same rows (row i: direction[i] held over bar i+1)."""
    cfg = TradingConfig(mode="long_short", venue=KRAKEN, leverage=lev, capital=10_000.0).validate()
    acc = Account(cfg, 1.0)
    for i in range(len(df) - 1):
        acc.step(float(df.direction[i]), df.close[i], df.close[i + 1], df.low[i + 1], df.high[i + 1])
    return acc.log()


def identity(lev):
    return PolicyConfig(name=f"identity_L{lev}", sizing="identity", margin_leverage=lev, exposure_cap=lev,
                        stop_atr_mult=0, slippage_bps=0, checks=False)


def test_bridge_parity_btcgold_log():
    st = log_stream()
    log = np.load(os.path.join(RUN, "test/trading_log.npz"), allow_pickle=True)
    n, r0 = len(log["equity"]), st.source["stack"]
    res = replay(st, load_policy("identity_L5"), KRAKEN)
    got = res.bars["equity_mark"][r0:r0 + n]
    assert np.max(np.abs(got - log["equity"])) < 1e-9, np.max(np.abs(got - log["equity"]))
    assert np.max(np.abs(res.bars["pnl"][r0:r0 + n] - log["pnl"])) < 1e-9
    assert (np.sign(log["direction"]) != 0).sum() > 0 and log["equity"][-1] != 10_000.0   # non-trivial run
    assert res.bars["signal"][-1] == 0 and res.bars["contracts"][-1] == 0                 # last bar exits


def test_bridge_synthetic_liquidation_and_bankruptcy():
    close = [50000, 50000, 47000, 47000, 47000, 20000, 20000, 20000, 20000]
    low = [50000, 50000, 46000, 47000, 47000, 20000, 20000, 20000, 20000]
    high = [50000, 50500, 50000, 47000, 47000, 47000, 20000, 20000, 20000]
    d = [0, 1, 1, 1, 1, 1, 1, 1, 0]
    df = frame(close, d, low, high)
    res = replay(from_frame(df), identity(9.0), KRAKEN)
    ref = account_path(df, 9.0)
    assert np.max(np.abs(res.bars["equity_mark"][1:] - ref["equity"])) < 1e-9
    # hand computed: entry 180 contracts @50000 (fee 27), bar 2 low 46000: funding 1.125, pnl -7200, exit fee 27 + 10
    assert abs(res.bars["equity_mark"][2] - (10000 - 27 - 7200 - 37 - 1.125)) < 1e-9
    assert [x.action for x in res.decisions if x.action == "liquidation"] == ["liquidation"] * 2
    assert res.bars["equity_mark"][5] < 0                       # second crash bankrupts the account
    assert (res.bars["contracts"][5:] == 0).all() and (ref["contracts"][4:] == 0).all()   # bankrupt -> size 0
    assert np.ptp(res.bars["equity_mark"][5:]) == 0


def test_bridge_synthetic_short_liquidation():
    close = [50000, 50000, 52000, 52000, 52000, 51000]
    high = [50000, 50000, 60000, 52000, 52000, 51000]
    df = frame(close, [0, -1, -1, -1, 1, 0], None, high)
    res = replay(from_frame(df), identity(9.0), KRAKEN)
    ref = account_path(df, 9.0)
    assert np.max(np.abs(res.bars["equity_mark"][1:] - ref["equity"])) < 1e-9
    assert ref["liquidated"].sum() == 1 and any(x.action == "liquidation" for x in res.decisions)


def test_from_macrohft_alignment():
    st = log_stream()
    df = pd.read_feather(os.path.join(DATA, "whole/test.feather"))
    log = np.load(os.path.join(RUN, "test/trading_log.npz"), allow_pickle=True)
    r0, n = st.source["stack"], len(log["direction"])
    assert np.array_equal(st.close, df["close"].to_numpy())
    assert np.array_equal(st.close[r0:r0 + n], log["close"]) and st.close[r0 - 1] == float(log["close_start"])
    assert np.array_equal(st.direction[r0 - 1:r0 - 1 + n], log["direction"].astype(np.int8))
    assert st.covered[r0 - 1:r0 - 1 + n].all() and st.source["n_covered"] == n
    assert st.source["truncated"] is False
    # a log that ends early is flagged truncated and the uncovered tail is direction 0
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "test"))
        m = 1000
        np.savez(os.path.join(tmp, "test/trading_log.npz"), **{k: log[k][:m] for k in ("direction", "close", "timestamp")},
                 close_start=log["close_start"])
        t = from_macrohft(tmp, DATA, "test")
    assert t.source["truncated"] is True and t.covered.sum() == m and (t.direction[r0 - 1 + m:] == 0).all()
    assert np.array_equal(t.direction[:r0 - 1 + m], st.direction[:r0 - 1 + m])
    bad = dict(log)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "test"))
        bad["close"] = bad["close"] + 1.0
        np.savez(os.path.join(tmp, "test/trading_log.npz"), **bad)
        try:
            from_macrohft(tmp, DATA, "test")
        except ValueError:
            pass
        else:
            raise AssertionError("close mismatch not detected")


def test_import_py39_no_gym_torch():
    py = os.path.join(TM, ".venv/bin/python")
    code = ("import sys; sys.path.insert(0, %r); import tm_risk; "
            "assert sys.version_info[:2] == (3, 9), sys.version; "
            "assert 'torch' not in sys.modules and 'gym' not in sys.modules; print('ok')" % TM)
    out = subprocess.run([py, "-c", code], cwd=TM, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-600:]


def test_policy_validation():
    spot = load_venue("binance_us_spot")
    base = dict(name="t", sizing="constant")
    PolicyConfig(**base, margin_leverage=5, exposure_cap=1).validate(KRAKEN)
    for L in (3.0, 10.0, 2.0, 12.0):
        try:
            PolicyConfig(**base, margin_leverage=L, exposure_cap=1).validate(KRAKEN)
        except ValueError:
            continue
        raise AssertionError(f"L={L} accepted on perp")
    for kw in (dict(margin_leverage=5, exposure_cap=5.5), dict(margin_leverage=5, exposure_cap=0)):
        try:
            PolicyConfig(**base, **kw).validate(KRAKEN)
        except ValueError:
            continue
        raise AssertionError(f"{kw} accepted")
    try:
        PolicyConfig(**base, margin_leverage=5, exposure_cap=1).validate(spot)
    except ValueError:
        pass
    else:
        raise AssertionError("spot L=5 accepted")
    PolicyConfig(**base, margin_leverage=1, exposure_cap=1).validate(spot)
    try:
        PolicyConfig(name="t", sizing="bogus").validate()
    except ValueError:
        pass
    else:
        raise AssertionError("bad sizing accepted")
    p = load_policy("identity_L5")
    assert p.sizing == "identity" and not p.checks and p.margin_leverage == 5 and p.exposure_cap == 5


def test_mark_bar_stop_gap_fills_at_open():
    pol = PolicyConfig(name="t", sizing="constant", margin_leverage=5, exposure_cap=1)     # slippage 5 bps, checks on
    df = frame([50000, 48050], [1, 0], low=[50000, 47900], high=[50000, 48100], open_=[50000, 48000])
    st = AccountState(capital=1e4, equity=1e4, contracts=100.0, entry_price=50000.0, stop_price=49000.0)
    m = mark_bar(st, from_frame(df), 1, KRAKEN, pol)
    assert m.event == "stop" and m.price == 48000.0 and st.contracts == 0 and st.blocked_direction == 1
    # pnl 1 BTC * -2000; fee 100*0.15; slippage 5bps of 48000 USD; funding 50000*0.0001/8
    assert abs(st.equity - (10000 - 2000 - 15 - 24 - 0.625)) < 1e-9
    # stop above the open on a long is filled at the stop when the bar trades through it
    df2 = frame([50000, 48950], [1, 0], low=[50000, 48900], high=[50000, 50100], open_=[50000, 50000])
    st2 = AccountState(capital=1e4, equity=1e4, contracts=100.0, entry_price=50000.0, stop_price=49000.0)
    assert mark_bar(st2, from_frame(df2), 1, KRAKEN, pol).price == 49000.0


def test_mark_bar_liquidation_beats_stop_and_short_stop():
    pol = PolicyConfig(name="t", sizing="constant", margin_leverage=9, exposure_cap=1)
    df = frame([50000, 47000], [1, 0], low=[50000, 46000], high=[50000, 50000])
    st = AccountState(capital=1e4, equity=1e4 - 27, contracts=180.0, entry_price=50000.0, stop_price=49000.0)
    assert mark_bar(st, from_frame(df), 1, KRAKEN, pol).event == "liquidation"     # 46000 is worse than the stop fill
    df = frame([50000, 51500], [-1, 0], low=[50000, 50000], high=[50000, 52000], open_=[50000, 50200])
    st = AccountState(capital=1e4, equity=1e4, contracts=-100.0, entry_price=50000.0, stop_price=51000.0)
    m = mark_bar(st, from_frame(df), 1, KRAKEN, pol)
    assert m.event == "stop" and m.price == 51000.0 and st.blocked_direction == -1
    assert abs(m.pnl - (-1000.0)) < 1e-9


def test_deleverage():
    pol = PolicyConfig(name="t", sizing="constant", margin_leverage=5, exposure_cap=1)
    s = from_frame(frame([50000, 50000], [1, 0]))
    st = AccountState(capital=1e4, equity=1e4, contracts=100.0, entry_price=50000.0, open_risk=300.0)
    m = deleverage(st, s, 1, KRAKEN, pol)          # exposure 5 > 1 * 1.25 -> 20 contracts; cut 80: fee 12, slippage 20
    assert m.event == "breach" and st.contracts == 20.0 and abs(st.equity - (10000 - 12 - 20)) < 1e-9
    assert abs(st.open_risk - 60.0) < 1e-9
    assert deleverage(st, s, 1, KRAKEN, pol).event == ""                                # within band: nothing
    st.contracts, st.equity = 100.0, -5.0
    m = deleverage(st, s, 1, KRAKEN, pol)
    assert st.contracts == 0 and st.bankrupt


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"OK test_risk_bridge: {len(tests)} tests")
