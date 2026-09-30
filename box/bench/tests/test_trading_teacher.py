"""Tests for MacroHFT.trading.teacher: brute-force equality, shape, terminal row, penalty, trend behaviour."""
import itertools, os, sys
from dataclasses import replace
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)

from MacroHFT.trading.config import TradingConfig
from MacroHFT.trading.teacher import make_q_table_trading
from MacroHFT.trading.venue import load_venue

BAR_H = 1 / 60


def cfg_for(venue, mode, **kw):
    lev = 5.0 if venue.endswith("perp") else 1.0
    return TradingConfig(mode=mode, venue=load_venue(venue), leverage=lev, **kw).validate()


def brute(close, cfg, bar_hours, gamma):
    """Independent scalar reward + exhaustive enumeration of every action sequence."""
    v, d, T, n = cfg.venue, cfg.directions, len(close), cfg.n_action
    fund_h = v.funding.rate_per_hour if v.funding else 0.0

    def reward(i, p, c):
        e = cfg.leverage if cfg.sizing == "equity" else cfg.fixed_units * close[i] / cfg.capital
        r = close[i + 1] / close[i] - 1
        fee = v.fees.cost(1.0, close[i], v.contract_size) / (v.contract_size * close[i])
        cost = fee + cfg.turnover_penalty_bps * 1e-4
        return cfg.reward_scale * e * (d[c] * r - cost * abs(d[c] - d[p]) - fund_h * bar_hours * d[c])

    def best(i, p, c):
        b = -np.inf
        for seq in itertools.product(range(n), repeat=T - 2 - i):     # actions at rows i+1 .. T-2
            tot, prev = reward(i, p, c), c
            for k, a in enumerate(seq):
                tot += gamma ** (k + 1) * reward(i + 1 + k, prev, a)
                prev = a
            b = max(b, tot)
        return b

    q = np.zeros((T, n, n))
    for i in range(T - 1):
        for p in range(n):
            for c in range(n):
                q[i, p, c] = best(i, p, c)
    return q


def series(T=7, seed=0):
    return 100 + np.cumsum(np.random.RandomState(seed).randn(T)) * 0.5 + 20000  # ~20k so per-contract fees are sane


def test_brute_force_long_short():
    cfg = cfg_for("kraken_us_perp", "long_short", turnover_penalty_bps=2.0)
    c = series(7, 1) * (1 + np.linspace(0, 0.02, 7) ** 2 * 5)
    got, ref = make_q_table_trading(c, cfg, BAR_H, 0.99), brute(c, cfg, BAR_H, 0.99)
    assert np.abs(got - ref).max() < 1e-9, np.abs(got - ref).max()


def test_brute_force_long_only_spot():
    cfg = cfg_for("binance_us_spot", "long_only")
    c = series(7, 2)
    got, ref = make_q_table_trading(c, cfg, BAR_H, 0.9), brute(c, cfg, BAR_H, 0.9)
    assert got.shape == (7, 2, 2) and np.abs(got - ref).max() < 1e-9


def test_brute_force_fixed_sizing_coinbase():
    cfg = cfg_for("coinbase_us_perp", "long_short", sizing="fixed", turnover_penalty_bps=1.0)
    c = series(7, 3)
    assert np.abs(make_q_table_trading(c, cfg, 1.0, 0.99) - brute(c, cfg, 1.0, 0.99)).max() < 1e-9


def test_shape_and_terminal():
    for mode, venue, n in (("long_short", "kraken_us_perp", 3), ("long_only", "kraken_us_perp", 2)):
        cfg = cfg_for(venue, mode)
        q = make_q_table_trading(series(50), cfg, BAR_H)
        assert q.shape == (50, n, n) and np.all(q[-1] == 0) and np.isfinite(q).all()


def test_penalty_lowers_switching_values():
    base = cfg_for("kraken_us_perp", "long_short")
    pen = replace(base, turnover_penalty_bps=5.0)
    c = series(40, 4)
    q0, q1 = make_q_table_trading(c, base, BAR_H), make_q_table_trading(c, pen, BAR_H)
    for p in range(3):
        for a in range(3):
            if a != p:
                assert np.all(q1[:-1, p, a] < q0[:-1, p, a])
    assert np.all(q1[:-1, 0, 0] <= q0[:-1, 0, 0] + 1e-12)


def test_trend_direction_with_zero_costs():
    v = load_venue("kraken_us_perp")
    free = replace(v, fees=replace(v.fees, taker=0.0, maker=0.0, min_per_contract=0.0),
                   funding=replace(v.funding, rate_per_interval=0.0))
    cfg = TradingConfig(mode="long_short", venue=free, leverage=5.0).validate()
    up = 20000 * 1.001 ** np.arange(30)
    dn = 20000 * 0.999 ** np.arange(30)
    assert make_q_table_trading(up, cfg, BAR_H)[0, 1].argmax() == 2
    assert make_q_table_trading(dn, cfg, BAR_H)[0, 1].argmax() == 0
    lo = TradingConfig(mode="long_only", venue=free, leverage=5.0).validate()
    assert make_q_table_trading(up, lo, BAR_H)[0, 0].argmax() == 1


if __name__ == "__main__":
    tests = [f for k, f in sorted(globals().items()) if k.startswith("test_") and callable(f)]
    for f in tests:
        f()
    print(f"OK: {len(tests)} teacher tests passed")
