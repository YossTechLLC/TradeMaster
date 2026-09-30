"""Tests for tm_risk ledger, credibility, kelly and sizing with hand-computed numbers."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
from tm_risk.credibility import shrink
from tm_risk.kelly import robust_kelly
from tm_risk.ledger import SignalLedger
from tm_risk.sizing import size_request
from tm_risk.signals import SignalStream
from tm_risk.types import AccountState, PolicyConfig
from tm_risk.venue import load_venue

VENUE = load_venue("kraken_us_perp")   # 0.15 USD/contract/side, cs 0.01, funding 1e-4 per 8h = 1.25e-5/h


def pol(**kw):
    d = dict(name="t", sizing="kelly", margin_leverage=5.0, exposure_cap=1.0, credibility_n0=1e-6,
             kelly_bootstrap=300, kelly_min_support=30)
    d.update(kw)
    return PolicyConfig(**d).validate()


def stream(close, direction, bar_hours=1.0):
    c = np.asarray(close, dtype=float)
    n = len(c)
    return SignalStream(timestamp=np.arange(n), open=c, high=c, low=c, close=c, quote_volume=np.zeros(n),
                        direction=np.asarray(direction, dtype=np.int8), covered=np.ones(n, dtype=bool),
                        bar_hours=bar_hours, source={})


CLOSE = [50000, 50000, 50500, 51000, 55000, 55000, 52250, 50000]
DIR = [0, 1, 1, 1, 0, -1, -1, 0]


def test_ledger_episodes_and_visibility():
    L = SignalLedger(stream(CLOSE, DIR), VENUE, pol())
    for k in range(4):
        L.update(k)
    assert L.n(1) == 0 and L.n(-1) == 0            # long exits at bar 4: invisible before it
    L.update(4)
    assert L.n(1) == 1 and L.n(-1) == 0
    r, h, _ = L.outcomes(1)
    # long: entry close[1]=50000, exit close[4]=55000, 3 h; fee/side = 15/price; funding 1.25e-5/h paid by longs
    assert abs(r[0] - (0.1 - 15 / 50000 - 15 / 55000 - 3 * 1.25e-5)) < 1e-12 and h[0] == 3.0
    L.update(6)
    assert L.n(-1) == 0
    L.update(7)
    r, h, _ = L.outcomes(-1)
    # short: entry 55000 (bar 5), exit 50000 (bar 7), 2 h; shorts receive funding
    assert abs(r[0] - ((1 - 50000 / 55000) - 15 / 55000 - 15 / 50000 + 2 * 1.25e-5)) < 1e-12 and h[0] == 2.0


def test_ledger_independent_of_account():
    s = stream(CLOSE, DIR)
    a, b = SignalLedger(s, VENUE, pol()), SignalLedger(s, VENUE, pol(dd_halt=0.01, exposure_cap=0.5))
    for k in range(len(s)):
        a.update(k), b.update(k)
    for side in (1, -1):
        assert all(np.array_equal(x, y) for x, y in zip(a.outcomes(side), b.outcomes(side)))
    assert "state" not in SignalLedger.__init__.__code__.co_varnames


def test_ledger_seed():
    s = stream(CLOSE, DIR)
    L = SignalLedger(s, VENUE, pol())
    L.seed(s)
    assert L.n(1) == 1 and L.n(-1) == 1


def test_shrink():
    r = np.array([0.01, 0.03, 0.02, 0.06])           # mean 0.03, n=4, n0=4 -> c=0.5
    out = shrink(r, 4.0)
    assert abs(out.mean() - 0.015) < 1e-15
    assert np.allclose(out - out.mean(), r - r.mean())
    assert np.allclose(shrink(r, 0.0), r)


def test_kelly_abstains_on_noise():
    r = np.random.default_rng(7).normal(0, 0.02, 2000)
    res = robust_kelly(r, np.ones(2000), pol(credibility_n0=150.0, kelly_quantile=0.10))
    assert res.abstain and res.f == 0.0


def test_kelly_matches_theory():
    n = 20000
    r = np.random.default_rng(1).normal(0.002, 0.02, n)
    p = pol(margin_leverage=8.0, exposure_cap=8.0, kelly_quantile=0.5, kelly_bootstrap=200, kelly_fraction=0.5)
    res = robust_kelly(r, np.ones(n), p)
    assert not res.abstain
    want = 0.5 * 0.002 / 0.02 ** 2                    # 2.5
    assert abs(res.f - want) / want < 0.2, res.f


def test_kelly_min_support():
    r = np.full(29, 0.01)
    res = robust_kelly(r, np.ones(29), pol())
    assert res.abstain and "support" in res.reason
    assert not robust_kelly(np.full(30, 0.01) + np.linspace(-1e-3, 1e-3, 30), np.ones(30), pol()).abstain


def test_kelly_solvency_domain():
    r = np.concatenate([np.full(59, 0.02), [-0.60]])
    p = pol(margin_leverage=8.0, exposure_cap=8.0, kelly_quantile=0.5, kelly_fraction=1.0)
    res = robust_kelly(r, np.ones(60), p)              # shrink is ~none; min(r)=-0.6, stress 0.05 -> f < 1/0.65
    assert res.f_raw < 1 / 0.65 and res.f_raw >= 0


class Ledger:
    def __init__(self, sig, n=40):
        self.cache = {}
        self.sig = sig

    def n(self, side):
        return 40

    def outcomes(self, side):
        return np.linspace(0.005, 0.015, 40) + 0.0, np.ones(40), np.full(40, self.sig)


def test_vol_target_formula():
    rng = np.random.default_rng(3)
    c = 50000 * np.exp(np.cumsum(rng.normal(0, 0.001, 200)))
    s = stream(c, np.ones(200))
    p = pol(sizing="vol_target", exposure_cap=5.0, target_vol_annual=0.25, vol_window_bars=48)
    f, _ = size_request(150, 1, AccountState(1e4, 1e4), s, None, p)
    v = np.std(np.diff(np.log(c[102:151])), ddof=1)    # 48 returns over bars (102,150]
    assert abs(f - 0.25 / (v * np.sqrt(8760))) < 1e-9
    f2, _ = size_request(150, 1, AccountState(1e4, 1e4), s, None, pol(sizing="vol_target", exposure_cap=5.0, target_vol_annual=1e-4))
    assert f2 == 1e-4 / (v * np.sqrt(8760)) or abs(f2 - 1e-4 / (v * np.sqrt(8760))) < 1e-12


def test_clipping_and_arms():
    s = stream(np.full(100, 50000.0) * (1 + 0.001 * (np.arange(100) % 2)), np.ones(100))
    st = AccountState(1e4, 1e4)
    assert size_request(90, 1, st, s, None, pol(sizing="constant", constant_exposure=3.0, exposure_cap=1.0))[0] == 1.0
    assert size_request(90, 1, st, s, None, pol(sizing="constant", constant_exposure=0.4))[0] == 0.4
    assert size_request(90, 1, st, s, None, pol(sizing="identity", exposure_cap=1.0))[0] == 5.0
    f, _ = size_request(90, 1, st, s, None, pol(sizing="vol_target", exposure_cap=2.0, target_vol_annual=100.0))
    assert f == 2.0
    # kelly vol scaling: reference sigma 10x the current vol -> scale clipped to 2
    from tm_risk.stops import bar_vol
    v = bar_vol(s, 90, 48)
    p = pol(exposure_cap=5.0, kelly_quantile=0.5, kelly_fraction=1.0)
    f_no, _ = size_request(90, 1, st, s, Ledger(10 * v), pol(exposure_cap=5.0, kelly_quantile=0.5, kelly_fraction=1.0, kelly_vol_scale=False))
    f_sc, info = size_request(90, 1, st, s, Ledger(10 * v), p)
    assert info["vol_scale"] == 2.0 and abs(f_sc - min(5.0, 2 * f_no)) < 1e-12
    f_lo, info = size_request(90, 1, st, s, Ledger(v / 10), p)
    assert info["vol_scale"] == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("OK test_risk_kelly")
