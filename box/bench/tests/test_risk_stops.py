"""Tests for tm_risk/stops.py with hand-computed numbers, plus stop fills through tm_risk.breach.mark_bar."""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
from tm_risk.breach import mark_bar
from tm_risk.signals import SignalStream
from tm_risk.stops import atr, bar_vol, stop_distance, stressed_loss_per_unit
from tm_risk.types import AccountState, PolicyConfig
from tm_risk.venue import FeeModel, FundingModel, Venue


def perp(fees=None, cs=0.01):
    return Venue(name="t_perp", kind="perp", description="", contract_size=cs, fees=fees or FeeModel("per_contract", 0.15),
                 funding=FundingModel(8, 0.0), max_leverage=10.0, maintenance_margin_rate=0.05, liquidation_fee_usd=10.0,
                 allows_short=True, as_of="", sources=[], unverified=[])


def stream(o, h, l, c, bar_hours=1.0):
    n = len(c)
    f = lambda x: np.asarray(x, dtype=float)
    return SignalStream(np.arange(n), f(o), f(h), f(l), f(c), np.zeros(n), np.zeros(n, dtype=np.int8),
                        np.ones(n, dtype=bool), bar_hours, {})


def pol(**kw):
    return PolicyConfig(name="t", sizing="constant", **kw)


C = [100, 102, 101, 105, 103]
H = [101, 103, 104, 106, 105]
L = [99, 100, 99, 102, 101]
S = stream(C, H, L, C)
# true range per bar: 2, max(3,3,0)=3, max(5,2,3)=5, max(4,5,1)=5, max(4,0,4)=4


def test_atr_hand():
    assert abs(atr(S, 0, 14) - 2.0) < 1e-12
    assert abs(atr(S, 1, 14) - 2.5) < 1e-12            # partial window at the start: (2+3)/2
    assert abs(atr(S, 4, 3) - 14 / 3) < 1e-12          # bars 2..4: (5+5+4)/3
    assert abs(atr(S, 4, 1) - 4.0) < 1e-12
    assert abs(atr(S, 3, 2) - 5.0) < 1e-12             # first bar of the window uses the previous close


def test_bar_vol_hand():
    a, b = math.log(101 / 102), math.log(105 / 101)     # returns at bars 2, 3; sample std of two values = |a-b|/sqrt2
    assert abs(bar_vol(S, 3, 2) - abs(a - b) / math.sqrt(2)) < 1e-12
    assert bar_vol(S, 0, 5) == 0.0 and bar_vol(S, 1, 5) == 0.0   # < 2 returns
    r = [math.log(C[i] / C[i - 1]) for i in range(1, 5)]
    m = sum(r) / 4
    assert abs(bar_vol(S, 4, 10) - math.sqrt(sum((x - m) ** 2 for x in r) / 3)) < 1e-12


def test_causality():
    rng = np.random.default_rng(1)
    n = 60
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h, l = c * 1.01, c * 0.99
    s1 = stream(c, h, l, c)
    c2, h2, l2 = c.copy(), h.copy(), l.copy()
    k = 30
    c2[k + 1:] *= 3.0
    h2[k + 1:] *= 5.0
    l2[k + 1:] *= 0.2
    s2 = stream(c2, h2, l2, c2)
    p, v = pol(), perp()
    for j in range(k + 1):
        assert atr(s1, j, 14) == atr(s2, j, 14)
        assert bar_vol(s1, j, 20) == bar_vol(s2, j, 20)
        assert stop_distance(s1, j, p) == stop_distance(s2, j, p)
        assert stressed_loss_per_unit(s1, j, v, p) == stressed_loss_per_unit(s2, j, v, p)


def test_stop_distance():
    assert abs(stop_distance(S, 4, pol(stop_atr_mult=3.0, atr_window=3)) - 3 * (14 / 3) / 103) < 1e-12
    assert stop_distance(S, 4, pol(stop_atr_mult=0.0)) == 0.0


def test_stressed_loss_kraken_per_contract():
    v = perp(FeeModel("per_contract", 0.15), cs=0.01)     # fee rate = 0.15 / (0.01 * price)
    p = pol(stop_atr_mult=3.0, atr_window=3, slippage_bps=5.0)
    want = 14 / 103 + 2 * 0.15 / (0.01 * 103) + 2 * 5e-4
    assert abs(stressed_loss_per_unit(S, 4, v, p) - want) < 1e-12
    want3 = 3 * ((3 + 5 + 5) / 3) / 105 + 2 * 0.15 / (0.01 * 105) + 2 * 5e-4
    assert abs(stressed_loss_per_unit(S, 3, v, p) - want3) < 1e-12
    # the fee term depends on price: same stop/slippage, different close -> different rate
    assert abs(2 * 0.15 / (0.01 * 105) - 2 * 0.15 / (0.01 * 103)) > 1e-4


def test_stressed_loss_pct_venue():
    v = perp(FeeModel("pct", 0.0002), cs=0.01)
    p = pol(stop_atr_mult=3.0, atr_window=3, slippage_bps=5.0)
    assert abs(stressed_loss_per_unit(S, 4, v, p) - (14 / 103 + 0.0004 + 0.001)) < 1e-12


def test_stressed_loss_disabled_stop_fallback():
    v = perp(FeeModel("pct", 0.0002))
    p = pol(stop_atr_mult=0.0, kelly_stress_loss=0.05, slippage_bps=5.0)
    assert abs(stressed_loss_per_unit(S, 4, v, p) - (0.05 + 0.0004 + 0.001)) < 1e-12


def _long(stop=95.0):
    return AccountState(capital=10000.0, equity=10000.0, contracts=100.0, entry_price=100.0, stop_price=stop)


def test_stop_gap_fills_at_open():
    # prev close 100, next bar opens at 90 (gap through the 95 stop)
    s = stream([100, 90], [100, 91], [100, 88], [100, 89])
    st, v = _long(), perp()
    m = mark_bar(st, s, 1, v, pol())
    assert m.event == "stop" and m.price == 90.0
    assert abs(m.pnl - 100 * 0.01 * (90 - 100)) < 1e-12                  # -10
    fee, slip = 0.15 * 100, 5e-4 * 100 * 0.01 * 90                       # 15, 0.045
    assert abs(st.equity - (10000 - 10 - fee - slip)) < 1e-9
    assert st.contracts == 0.0 and st.stop_price == 0.0 and st.blocked_direction == 1


def test_stop_intrabar_fills_at_stop():
    s = stream([100, 99], [100, 100], [100, 94], [100, 96])
    st = _long()
    m = mark_bar(st, s, 1, perp(), pol())
    assert m.event == "stop" and m.price == 95.0
    assert abs(m.pnl - 1 * (95 - 100)) < 1e-12                           # -5
    assert st.contracts == 0.0 and st.blocked_direction == 1


def test_stop_not_hit_and_short_side():
    s = stream([100, 99], [100, 100], [100, 95.5], [100, 97])
    st = _long()
    m = mark_bar(st, s, 1, perp(), pol())
    assert m.event == "" and st.contracts == 100.0 and abs(m.pnl - (-3.0)) < 1e-12
    # short with stop at 105, bar opens at 110: fills at 110
    s = stream([100, 110], [100, 112], [100, 108], [100, 111])
    st = AccountState(capital=10000.0, equity=10000.0, contracts=-100.0, entry_price=100.0, stop_price=105.0)
    m = mark_bar(st, s, 1, perp(), pol())
    assert m.event == "stop" and m.price == 110.0 and st.blocked_direction == -1
    assert abs(m.pnl - (-100 * 0.01 * (110 - 100))) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
    print(f"OK test_risk_stops: {len(fns)} tests passed")
