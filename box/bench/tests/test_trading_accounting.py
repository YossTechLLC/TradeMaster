"""Tests for MacroHFT/trading/accounting.py and penalty.py with hand-computed numbers."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
from MacroHFT.trading.accounting import Account, StepResult
from MacroHFT.trading.config import TradingConfig
from MacroHFT.trading.penalty import turnover_penalty
from MacroHFT.trading.venue import FeeModel, FundingModel, Venue


def perp(fees=None, funding=FundingModel(8, 0.0001), mm=0.05, liq_fee=10.0, cs=0.01):
    return Venue(name="t_perp", kind="perp", description="", contract_size=cs,
                 fees=fees or FeeModel("per_contract", 0.15), funding=funding, max_leverage=10.0,
                 maintenance_margin_rate=mm, liquidation_fee_usd=liq_fee, allows_short=True,
                 as_of="", sources=[], unverified=[])


def spot(taker=0.0002):
    return Venue(name="t_spot", kind="spot", description="", contract_size=1e-5, fees=FeeModel("pct", taker),
                 funding=None, max_leverage=1.0, maintenance_margin_rate=0.0, liquidation_fee_usd=0.0,
                 allows_short=False, as_of="", sources=[], unverified=[])


def acct(venue=None, mode="long_short", lev=5.0, capital=10_000.0, bps=0.0, sizing="equity", bar_hours=1.0, **kw):
    cfg = TradingConfig(mode=mode, venue=venue or perp(), leverage=lev, capital=capital, turnover_penalty_bps=bps,
                        sizing=sizing, **kw).validate()
    return Account(cfg, bar_hours)


def test_penalty():
    assert abs(turnover_penalty(10_000.0, 5.0) - 5.0) < 1e-12
    assert turnover_penalty(10_000.0, 0.0) == 0.0


def test_long_and_short_pnl():
    # equity 10000, L=5, price 100, cs 0.01 -> floor(5*10000/(100*0.01)) = 50000 contracts (500 BTC)... use price 50000
    # price 50000: floor(5*10000/(50000*0.01)) = 100 contracts = 1 BTC
    a = acct(venue=perp(funding=None))
    r = a.step(1.0, 50000.0, 51000.0)
    assert r.contracts == 100 and r.traded == 100
    assert abs(r.pnl - 1000.0) < 1e-9 and abs(r.fee - 15.0) < 1e-9       # 100 * 0.15
    assert abs(r.equity - (10000 + 1000 - 15)) < 1e-9
    a = acct(venue=perp(funding=None))
    r = a.step(-1.0, 50000.0, 49000.0)
    assert r.contracts == -100 and abs(r.pnl - 1000.0) < 1e-9
    r = a.step(-1.0, 49000.0, 50000.0)   # short held: still -100 contracts, loses 1 BTC * 1000
    assert r.contracts == -100 and r.traded == 0 and r.fee == 0 and abs(r.pnl + 1000.0) < 1e-9


def test_flip_trades_double_and_charges_both_sides():
    a = acct(venue=perp(funding=None))
    a.step(1.0, 50000.0, 50000.0)                # long 100, fee 15, equity 9985
    # flip at 50000: equity 9985 -> floor(5*9985/500) = 99 contracts short; traded |-99 - 100| = 199
    r = a.step(-1.0, 50000.0, 50000.0)
    assert r.contracts == -99 and r.traded == 199
    assert abs(r.fee - 199 * 0.15) < 1e-9
    r = a.step(0.0, 50000.0, 50000.0)            # to flat trades 99
    assert r.contracts == 0 and r.traded == 99


def test_hold_does_not_retrade_when_equity_changes():
    a = acct(venue=perp(funding=None))
    a.step(1.0, 50000.0, 55000.0)                # equity ~ +10%; sizing would now be larger
    assert a.equity > 10500
    r = a.step(1.0, 55000.0, 55000.0)
    assert r.contracts == 100 and r.traded == 0 and r.fee == 0.0


def test_sizing_equity_floor_and_fixed():
    a = acct(venue=perp(funding=None), lev=4.0, capital=10_000.0)
    # 4*10000/(30000*0.01) = 133.33 -> 133
    assert a.size(1.0, 30000.0) == 133 and a.size(-1.0, 30000.0) == -133 and a.size(0.0, 30000.0) == 0
    b = acct(venue=perp(funding=None), sizing="fixed", fixed_units=0.5)   # 0.5 BTC / 0.01 = 50 contracts
    assert abs(b.size(1.0, 12345.0) - 50) < 1e-9 and abs(b.size(-1.0, 1.0) + 50) < 1e-9
    a.equity = 0.0
    assert a.size(1.0, 30000.0) == 0


def test_fees_per_contract_and_pct():
    a = acct(venue=perp(funding=None))
    assert abs(a.step(1.0, 50000.0, 50000.0).fee - 15.0) < 1e-9            # 100 * 0.15
    v = perp(fees=FeeModel("pct", 0.0005, min_per_contract=0.20), funding=None)
    a = acct(venue=v)
    # 100 contracts * 0.01 * 50000 = 50000 notional; 0.05% = 25; min 0.2*100 = 20 -> 25
    assert abs(a.step(1.0, 50000.0, 50000.0).fee - 25.0) < 1e-9
    v = perp(fees=FeeModel("pct", 0.0001, min_per_contract=0.20), funding=None)
    a = acct(venue=v)
    assert abs(a.step(1.0, 50000.0, 50000.0).fee - 20.0) < 1e-9            # 5 < min 20


def test_funding_signs_and_spot_zero():
    v = perp(funding=FundingModel(8, 0.0001), fees=FeeModel("per_contract", 0.0))
    # 1 BTC notional 50000, rate/h 1.25e-5, 2 h -> 50000*1.25e-5*2 = 1.25
    a = acct(venue=v, bar_hours=2.0)
    r = a.step(1.0, 50000.0, 50000.0)
    assert abs(r.funding - 1.25) < 1e-9 and abs(r.equity - (10000 - 1.25)) < 1e-9
    a = acct(venue=v, bar_hours=2.0)
    r = a.step(-1.0, 50000.0, 50000.0)
    assert abs(r.funding + 1.25) < 1e-9 and abs(r.equity - (10000 + 1.25)) < 1e-9
    a = acct(venue=spot(), mode="long_only", lev=1.0, sizing="fixed", fixed_units=0.01, bar_hours=2.0)
    assert a.step(1.0, 50000.0, 50000.0).funding == 0.0


def test_penalty_not_in_equity():
    a0 = acct(venue=perp(funding=None), bps=0.0)
    a1 = acct(venue=perp(funding=None), bps=10.0)
    r0, r1 = a0.step(1.0, 50000.0, 50500.0), a1.step(1.0, 50000.0, 50500.0)
    # traded notional 100*0.01*50000 = 50000; 10 bps = 50
    assert r0.penalty == 0.0 and abs(r1.penalty - 50.0) < 1e-9
    assert r0.equity == r1.equity


def test_liquidation_long():
    # L=5, price 50000, 100 contracts (1 BTC); bar low 41000 (-18%), close 42000. mm 0.05, liq fee 10, funding off.
    v = perp(funding=None)
    a = acct(venue=v, lev=5.0)
    r = a.step(1.0, 50000.0, 42000.0, bar_low=41000.0, bar_high=50000.0)
    # equity at adverse: 10000 - 15 - 9000 = 985 < 0.05*1*41000 = 2050 -> liquidated
    assert r.liquidated and r.contracts == 0
    assert abs(r.pnl + 9000.0) < 1e-9
    assert abs(r.fee - (15.0 + 100 * 0.15 + 10.0)) < 1e-9                 # entry + exit + liquidation fee
    assert abs(r.equity - (10000 - 9000 - 40.0)) < 1e-9 and not r.bankrupt
    assert a.contracts == 0
    # not liquidated when the low stays healthy: -10% low -> 10000-15-5000 = 4985 > 0.05*45000 = 2250
    a = acct(venue=v, lev=5.0)
    r = a.step(1.0, 50000.0, 47000.0, bar_low=45000.0, bar_high=50000.0)
    assert not r.liquidated and r.contracts == 100 and abs(r.pnl + 3000.0) < 1e-9


def test_liquidation_extreme_leverage_bankrupt_and_short():
    # L=9, -12% low: loss 9*12% = 108% of equity -> liquidated and bankrupt
    a = acct(venue=perp(funding=None), lev=9.0)
    r = a.step(1.0, 50000.0, 45000.0, bar_low=44000.0, bar_high=50000.0)
    assert r.contracts == 0 and r.liquidated and r.bankrupt and r.equity < 0
    assert a.size(1.0, 50000.0) == 0                                        # bankruptcy stops sizing
    # short liquidated on the bar high: 180 contracts... L=9, price 50000: floor(9*10000/500)=180 contracts short
    a = acct(venue=perp(funding=None), lev=9.0)
    r = a.step(-1.0, 50000.0, 51000.0, bar_low=50000.0, bar_high=56000.0)
    assert r.liquidated and abs(r.pnl - (-180 * 0.01 * 6000.0)) < 1e-9


def test_no_extremes_uses_close_and_reset():
    a = acct(venue=perp(funding=None), lev=5.0)
    r = a.step(1.0, 50000.0, 41000.0)     # adverse = close: equity 10000-15-9000 = 985 < 2050
    assert r.liquidated
    a.reset(direction=1.0, price=50000.0)   # inherited position, no fee, fresh equity
    assert a.equity == 10000.0 and a.contracts == 100 and a.log()["equity"].size == 0
    r = a.step(1.0, 50000.0, 50000.0)
    assert r.traded == 0 and r.fee == 0.0


def test_log_arrays():
    a = acct(venue=perp(funding=None))
    a.step(1.0, 50000.0, 50100.0); a.step(0.0, 50100.0, 50000.0)
    lg = a.log()
    assert set(lg) == set(StepResult.__dataclass_fields__) and all(x.shape == (2,) for x in lg.values())
    assert lg["contracts"][1] == 0


def test_legacy_identity():
    # legacy MacroHFT env reward, computed from its buy/sell cash flows independently:
    #   value change + cash flows, trades executed at the previous close p0 with fee f
    f, unit = 0.0002, 0.01
    rng = np.random.default_rng(0)
    price = 20000 * np.exp(np.cumsum(rng.normal(0, 0.002, 201)))
    acts = rng.integers(0, 2, 200)
    a = acct(venue=spot(f), mode="long_only", lev=1.0, sizing="fixed", fixed_units=unit)
    cs = a.venue.contract_size
    C0 = 0.0
    for t in range(200):
        p0, p1 = price[t], price[t + 1]
        C1 = unit * acts[t]
        if C0 >= C1:
            legacy = p1 * C1 + (C0 - C1) * p0 * (1 - f) - p0 * C0
        else:
            legacy = p1 * C1 - (C1 - C0) * p0 * (1 + f) - p0 * C0
        r = a.step(float(acts[t]), p0, p1)
        assert abs((r.pnl - r.fee) - legacy) < 1e-9, (t, r.pnl - r.fee, legacy)
        # spec identity: p1*(C1-C0) - f*|p1-p0|*C0 form via pnl/fee decomposition
        assert abs(r.pnl - C1 * (p1 - p0)) < 1e-9 and abs(r.fee - f * p0 * abs(C1 - C0)) < 1e-9
        C0 = C1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"OK: {len(tests)} accounting tests passed")
