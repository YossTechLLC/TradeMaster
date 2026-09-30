"""Tests for MacroHFT/trading venue.py and config.py."""
import argparse, dataclasses, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
from MacroHFT.trading.venue import load_venue, list_venues, FeeModel, FundingModel
from MacroHFT.trading.config import TradingConfig, add_trading_args, config_from_args, LEVERAGE_POLICY


def raises(fn, text=None):
    try:
        fn()
    except ValueError as e:
        assert text is None or text in str(e), str(e)
        return
    raise AssertionError("expected ValueError")


def cfg(venue="kraken_us_perp", **kw):
    kw.setdefault("mode", "long_short")
    return TradingConfig(venue=load_venue(venue), **kw)


def test_venues_load():
    assert set(list_venues()) >= {"kraken_us_perp", "coinbase_us_perp", "binance_us_spot"}
    for n in list_venues():
        v = load_venue(n)
        assert v.name == n and v.as_of == "2026-09-30" and v.sources
    k = load_venue("kraken_us_perp")
    assert k.kind == "perp" and k.contract_size == 0.01 and k.fees.model == "per_contract" and k.fees.taker == 0.15
    assert k.funding.interval_hours == 8 and k.max_leverage == 10 and k.liquidation_fee_usd == 10 and k.allows_short
    assert k.unverified
    s = load_venue("binance_us_spot")
    assert s.kind == "spot" and s.funding is None and not s.allows_short and s.max_leverage == 1
    assert load_venue(os.path.join(TM, "MacroHFT/trading/venues/kraken_us_perp.yaml")) == k


def test_fee_model():
    pc = FeeModel("per_contract", 0.15, 0.15)
    assert abs(pc.cost(-3, 60000, 0.01) - 0.45) < 1e-12
    assert abs(pc.rate(60000, 0.01) - 0.15 / 600) < 1e-15
    pct = FeeModel("pct", 0.0005, 0.0005, 0.20)
    # notional fee 0.0005*600 = 0.30 > 0.20 minimum
    assert abs(pct.cost(2, 60000, 0.01) - 0.60) < 1e-12
    # minimum binds: 0.0005*100 = 0.05 < 0.20
    assert abs(pct.cost(2, 10000, 0.01) - 0.40) < 1e-12
    assert abs(pct.rate(10000, 0.01) - 0.20 / 100) < 1e-15
    assert abs(pct.rate(60000, 0.01) - 0.0005) < 1e-15
    assert abs(FeeModel("pct", 0.0002).cost(1, 100, 0.5) - 0.01) < 1e-15


def test_funding_model():
    assert abs(FundingModel(8, 0.0001).rate_per_hour - 0.0001 / 8) < 1e-18
    assert FundingModel(1, 0.00001).rate_per_hour == 0.00001


def test_venue_validation():
    import tempfile, yaml
    base = yaml.safe_load(open(os.path.join(TM, "MacroHFT/trading/venues/kraken_us_perp.yaml")))
    def load(**kw):
        d = {**base, **kw}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(d, f)
        try:
            return load_venue(f.name)
        finally:
            os.unlink(f.name)
    load()
    raises(lambda: load(kind="future"), "kind")
    raises(lambda: load(funding=None), "funding")
    raises(lambda: load(contract_size=0), "contract_size")
    raises(lambda: load(fees={"model": "bps", "taker": 1}), "fee model")
    raises(lambda: load(fees={"model": "pct", "taker": -1}), "taker")
    raises(lambda: load(kind="spot", funding=None), "spot")
    raises(lambda: load(fees={"model": "pct", "taker": 1, "bogus": 1}), "bogus")
    raises(lambda: load(fees=None), "venue file")


def test_leverage_policy():
    assert LEVERAGE_POLICY == (3.0, 10.0)
    for L in (3.0, 10.0, 12, 1.0, 2.0):
        raises(lambda: cfg(leverage=L).validate(), "leverage")
    for L in (3.01, 5, 9.99):
        cfg(leverage=L).validate()
    cfg(mode="long_only", leverage=4).validate()
    # venue max below policy ceiling
    v = dataclasses.replace(load_venue("kraken_us_perp"), max_leverage=5)
    raises(lambda: TradingConfig(mode="long_short", venue=v, leverage=6).validate(), "max")
    TradingConfig(mode="long_short", venue=v, leverage=5).validate()


def test_spot_rules():
    cfg("binance_us_spot", mode="long_only", leverage=1).validate()
    raises(lambda: cfg("binance_us_spot", mode="long_only", leverage=5).validate(), "spot")
    raises(lambda: cfg("binance_us_spot", mode="long_only", leverage=2).validate(), "spot")
    raises(lambda: cfg("binance_us_spot", mode="long_short", leverage=1).validate(), "shorting")


def test_other_validation():
    raises(lambda: TradingConfig(mode="bogus").validate(), "mode")
    raises(lambda: TradingConfig(mode="long_only").validate(), "venue")
    nan = float('nan')
    for kw in (dict(capital=nan), dict(reward_scale=nan), dict(turnover_penalty_bps=nan), dict(capital=0), dict(turnover_penalty_bps=-1), dict(reward_scale=0),
               dict(sizing="kelly"), dict(fixed_units=0)):
        raises(lambda: cfg(**kw).validate())
    # legacy ignores everything else
    TradingConfig(mode="legacy", leverage=99, capital=-1, sizing="x").validate()


def test_modes():
    lg = TradingConfig()
    assert lg.legacy and lg.n_action == 2 and lg.directions == (0.0, 1.0) and lg.flat_action == 0 and lg.tag == ""
    lo = cfg(mode="long_only")
    assert not lo.legacy and lo.n_action == 2 and lo.directions == (0.0, 1.0) and lo.flat_action == 0
    assert lo.tag == "long_only-kraken_us_perp-L5-tp0"
    ls = cfg(leverage=4.5, turnover_penalty_bps=2.5)
    assert ls.n_action == 3 and ls.directions == (-1.0, 0.0, 1.0) and ls.flat_action == 1
    assert ls.directions[ls.flat_action] == 0.0
    assert ls.tag == "long_short-kraken_us_perp-L4.5-tp2.5"
    assert cfg(capital=1000).tag != cfg().tag != cfg(reward_scale=10).tag != cfg(capital=1000).tag


def test_config_from_args():
    p = add_trading_args(argparse.ArgumentParser())
    a = p.parse_args([])
    assert a.trade_mode == "legacy" and a.venue == "kraken_us_perp" and a.leverage is None
    assert a.capital == 10000 and a.turnover_penalty_bps == 0 and a.reward_scale == 100
    c = config_from_args(a)
    assert c == TradingConfig() and c.venue is None
    a = p.parse_args(["--venue", "no_such_venue"])  # legacy ignores venue
    assert config_from_args(a).legacy
    a = p.parse_args(["--trade_mode", "long_short", "--leverage", "4", "--turnover_penalty_bps", "1.5"])
    c = config_from_args(a)
    assert c.mode == "long_short" and c.venue.name == "kraken_us_perp" and c.leverage == 4
    assert c.turnover_penalty_bps == 1.5 and c.n_action == 3
    a = p.parse_args(["--trade_mode", "long_short", "--leverage", "10"])
    raises(lambda: config_from_args(a), "leverage")
    assert config_from_args(p.parse_args(["--trade_mode", "long_only"])).leverage == 5.0
    assert config_from_args(p.parse_args(["--trade_mode", "long_only", "--venue", "binance_us_spot"])).leverage == 1.0
    a = p.parse_args(["--trade_mode", "long_only", "--venue", "nope"])
    raises(lambda: config_from_args(a), "unknown venue")
    raises(lambda: load_venue("nope"), "kraken_us_perp")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
    print(f"OK test_trading_config: {len(fns)} tests passed")
