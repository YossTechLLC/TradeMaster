"""Tests for tm_risk/admission.py, episode.py and the policy yamls with hand-computed numbers (stops stubbed: u=0.05, stop distance 3%)."""
import os, sys, types
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
import tm_risk
stub = types.ModuleType("tm_risk.stops")
stub.stressed_loss_per_unit = lambda stream, k, venue, policy: 0.05
stub.stop_distance = lambda stream, k, policy: 0.03
sys.modules["tm_risk.stops"] = stub
tm_risk.stops = stub
from tm_risk.admission import admit
from tm_risk.episode import RiskEpisode
from tm_risk.types import AccountState, load_policy
from tm_risk.venue import FeeModel, FundingModel, Venue

VENUE = Venue(name="t_perp", kind="perp", description="", contract_size=0.01, fees=FeeModel("pct", 0.0005),
              funding=FundingModel(8, 0.0001), max_leverage=10.0, maintenance_margin_rate=0.05, liquidation_fee_usd=10.0,
              allows_short=True, as_of="", sources=[], unverified=[])
BASE = replace(load_policy("default_35dd"), participation=0.01)


def stream(price=50000.0, qv=1e9):
    return SimpleNamespace(close=np.array([price]), quote_volume=np.array([qv]), timestamp=np.array([0]))


def state(E=10000.0, peak=10000.0, **kw):
    return AccountState(capital=E, equity=E, peak_all=peak, peak_ref=peak, **kw)


def run(f=5.0, st=None, s=None, pol=BASE, side=1, ep=None):
    st = st or state()
    ep = ep or RiskEpisode(pol, start_k=0, equity=st.equity)
    return admit(0, side, f, st, s or stream(), pol, VENUE, ep), ep


def test_oversized_request_capped_by_trade_loss():
    # E=10000, u=0.05: trade_loss = 0.02*10000/0.05 = 4000 (others: aggregate 10000, headroom 70000, cap 10000, episode 10000)
    d, _ = run()
    assert d.binding == "trade_loss" and abs(d.permitted - 4000.0) < 1e-9, d
    assert d.wanted == 50000.0 and d.preferred == 10000.0
    assert d.contracts == 8.0 and d.action == "enter"      # 4000 / (0.01*50000) = 8


def test_drawdown_headroom():
    # peak 15000: E - 0.65*peak = 250 -> 250/0.05 = 5000 notional; loosen the trade budget so headroom binds
    pol = replace(BASE, trade_loss_budget=1.0, aggregate_loss_budget=1.0, episode_loss_budget=1.0, exposure_cap=5.0)
    st = state(peak=15000.0)
    d, _ = run(st=st, pol=pol)
    assert d.binding == "drawdown_headroom" and abs(d.permitted - 5000.0) < 1e-9
    assert d.contracts == 10.0
    assert d.permitted * 0.05 <= st.equity - 0.65 * st.peak_ref + 1e-9
    assert d.contracts * 500.0 * 0.05 <= st.equity - 0.65 * st.peak_ref + 1e-9


def test_halted_veto():
    d, _ = run(st=state(peak=20000.0, halted=True))
    assert "halted" in d.vetoes and d.contracts == 0 and d.permitted == 0 and d.action == "abstain" and d.reason
    d, _ = run(st=state(peak=20000.0))                     # halted flag unset but E < 0.65*peak still vetoes
    assert d.vetoes == ("halted",) and d.contracts == 0


def test_blocked_direction_veto():
    d, _ = run(st=state(blocked_direction=1), side=1)
    assert d.vetoes == ("blocked_direction",) and d.contracts == 0
    d, _ = run(st=state(blocked_direction=1), side=-1)
    assert d.vetoes == () and d.contracts == 8.0


def test_aggregate_budget_with_open_risk():
    # (0.05*10000 - 400)/0.05 = 2000 -> 4 contracts
    d, _ = run(st=state(open_risk=400.0))
    assert d.binding == "aggregate_loss" and abs(d.permitted - 2000.0) < 1e-9 and d.contracts == 4.0
    d, _ = run(st=state(open_risk=900.0))                  # over budget: 0 capacity, not negative
    assert d.permitted == 0.0 and d.contracts == 0 and d.action == "abstain"


def test_participation_binding():
    d, _ = run(s=stream(qv=100000.0))                      # 0.01*100000 = 1000 -> 2 contracts
    assert d.binding == "participation" and abs(d.permitted - 1000.0) < 1e-9 and d.contracts == 2.0
    d, _ = run(s=stream(qv=0.0))                           # unknown volume: skipped
    assert d.binding == "trade_loss"


def test_collateral_binding():
    # E=100, p=500, f=5: est_fee=(0.0005+0.0005)*500=0.5; collateral=(100-0.5)*5=497.5 < request 500 = cap 500
    pol = replace(BASE, trade_loss_budget=10.0, aggregate_loss_budget=10.0, episode_loss_budget=10.0, exposure_cap=5.0,
                  dd_halt=1.0)
    d, _ = run(f=5.0, st=state(E=100.0, peak=100.0), s=stream(price=500.0), pol=pol)
    assert d.binding == "collateral" and abs(d.permitted - 497.5) < 1e-9
    assert d.contracts == 99.0                             # floor(497.5 / 5)


def test_tie_order_and_rounding():
    # request 4000 == trade_loss 4000: first in table order is "request"
    d, _ = run(f=0.4)
    assert d.binding == "request" and d.contracts == 8.0
    d, _ = run(f=0.0099)                                   # 99 USD < one contract (500) -> abstain
    assert d.contracts == 0 and d.action == "abstain" and "contract" in d.reason


def test_episode_loss_binding_and_profits_never_refill():
    st = state()
    ep = RiskEpisode(BASE, start_k=0, equity=st.equity)
    d, _ = run(st=st, ep=ep)
    assert ep.loss_allowance == 500.0 and ep.loss_spent == 0.0
    ep.record_close(-300.0)                                # loss 300 -> remaining 200 -> 200/0.05 = 4000 (ties trade_loss)
    ep.record_close(+1000.0)                               # profit does not refill
    assert ep.loss_spent == 300.0
    ep.record_close(-100.0)                                # remaining 100 -> 2000
    d, _ = run(st=st, ep=ep)
    assert d.binding == "episode_loss" and abs(d.permitted - 2000.0) < 1e-9 and d.contracts == 4.0
    assert d.preferred == 2000.0
    ep.record_close(-500.0)                                # exhausted
    d, _ = run(st=st, ep=ep)
    assert d.permitted == 0 and d.contracts == 0


def test_episode_allowance_fixed_at_first_admission_flip_and_rejects():
    st = state()
    ep = RiskEpisode(BASE, start_k=0, equity=st.equity)
    assert ep.loss_allowance is None
    run(st=state(peak=20000.0), ep=ep)                     # rejected (halted): allowance still fixed from E at admission call
    a0, spent0 = ep.loss_allowance, ep.loss_spent
    st2 = state(E=5000.0, peak=5000.0)                     # equity changed later: allowance must not be reset
    d, _ = run(st=st2, ep=ep, side=-1)                     # flip inside the same episode
    assert ep.loss_allowance == a0 and ep.loss_spent == spent0
    ep.record_close(-50.0)
    run(st=state(blocked_direction=1), ep=ep, side=1)      # rejected proposal
    assert ep.loss_spent == 50.0 and ep.loss_allowance == a0


def test_episode_max_bars():
    ep = RiskEpisode(BASE, start_k=10, equity=1.0)
    assert not ep.expired(10 + 167) and ep.expired(10 + 168)


def test_policy_yamls():
    for n, sz in (("default_35dd", "kelly"), ("constant_1x", "constant"), ("vol_target_25", "vol_target")):
        p = load_policy(n)
        p.validate(VENUE)
        assert p.sizing == sz and p.checks and p.margin_leverage == 5 and p.exposure_cap == 1.0
        assert p.stop_atr_mult == 3 and p.slippage_bps == 5 and p.trade_loss_budget == 0.02
        assert p.aggregate_loss_budget == 0.05 and p.dd_halt == 0.35 and p.dd_window_days == 90
        assert p.participation == 0.01 and p.episode_max_bars == 168 and p.episode_loss_budget == 0.05
    assert load_policy("vol_target_25").target_vol_annual == 0.25 and load_policy("constant_1x").constant_exposure == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK test_risk_admission ({len(fns)} tests)")
