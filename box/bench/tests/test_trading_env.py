"""Tests for MacroHFT/trading/envs.py: legacy equivalence, arity parity, shorts, funding, liquidation, factory."""
import dataclasses, io, contextlib, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
os.chdir(os.path.join(TM, "MacroHFT"))   # legacy env modules load ./data/feature_list at import

from MacroHFT.env import low_level_env as LL, high_level_env as HL
from MacroHFT.trading.venue import load_venue
from MacroHFT.trading.config import TradingConfig
from MacroHFT.trading.envs import VenueTestingEnv, VenueTrainingEnv, make_env
from MacroHFT.trading.teacher import make_q_table_trading

SINGLE, TREND, CLF = LL.tech_indicator_list, LL.tech_indicator_list_trend, HL.clf_list
DF = pd.read_feather("data/BTCGOLD/train/df_0.feather")
for _c in CLF:   # chunk files carry no regime columns; the high level only needs them to exist
    if _c not in DF.columns:
        DF[_c] = np.linspace(-1, 1, len(DF))
KRAKEN, SPOT = load_venue("kraken_us_perp"), load_venue("binance_us_spot")


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def synth(close, low=None, high=None):
    """Minimal frame with one feature column per list entry, for synthetic price paths."""
    close = np.asarray(close, float)
    df = pd.DataFrame({"close": close, "low": close if low is None else low, "high": close if high is None else high,
                       "timestamp": pd.date_range("2024-01-01", periods=len(close), freq="min")})
    df["f"] = 0.0
    return df


def run(env, actions):
    env.reset()
    out = []
    for a in actions:
        out.append(env.step(a))
        if out[-1][-2]:
            break
    return out


def test_legacy_equivalence():
    cfg = TradingConfig(mode="long_only", venue=SPOT, leverage=1, capital=1e9, reward_scale=1e9, sizing="fixed",
                        fixed_units=0.01).validate()
    rng = np.random.RandomState(0)
    acts = rng.randint(0, 2, len(DF) - 1)
    legacy = LL.Testing_Env(DF, SINGLE, TREND, 0.0002, 1, 0.01, 0)
    new = VenueTestingEnv(DF, SINGLE, TREND, cfg, initial_action=0)
    s0 = legacy.reset(); s1 = new.reset()
    assert all(np.array_equal(a, b) for a, b in zip(s0[:2], s1[:2])) and len(s1) == 3
    for a in acts:
        o = quiet(legacy.step, a); n = quiet(new.step, a)
        assert abs(o[2] - n[2]) < 1e-9, (o[2], n[2])
        assert np.array_equal(o[0], n[0]) and np.array_equal(o[1], n[1])
        assert o[3] == n[3]
        assert o[0].dtype == n[0].dtype and o[1].dtype == n[1].dtype


def test_arity_parity():
    cfg = TradingConfig(mode="long_short", venue=KRAKEN).validate()
    for level, clf in (("low", None), ("high", CLF)):
        leg = make_env("test", level, DF, SINGLE, TREND, TradingConfig(), clf_list=clf)
        for kind in ("test", "train"):
            new = make_env(kind, level, DF, SINGLE, TREND, cfg, clf_list=clf, initial_action=1)
            r0, r1 = leg.reset(), new.reset()
            assert len(r0) == len(r1) == (3 if clf is None else 4)
            for a, b in zip(r0[:-1], r1[:-1]):
                assert a.shape == b.shape and a.dtype == b.dtype
            s0, s1 = leg.step(1), new.step(2)
            assert len(s0) == len(s1) == (5 if clf is None else 6)
            for a, b in zip(s0[:-3], s1[:-3]):
                assert a.shape == b.shape and a.dtype == b.dtype
            assert isinstance(s1[-3], float) and s1[-2] is False and s1[-1]["previous_action"] == 2
            assert ("q_value" in s1[-1]) == (kind == "train")
            if kind == "train":
                assert s1[-1]["q_value"].shape == (3,) and r1[-1]["q_value"].shape == (3,)


def test_short_gains_on_fall():
    close = 100 * 0.999 ** np.arange(50) * 500
    cfg = TradingConfig(mode="long_short", venue=KRAKEN, leverage=5, capital=10_000, reward_scale=1).validate()
    env = VenueTestingEnv(synth(close), ["f"], ["f"], cfg)
    outs = run(env, [0] * 30)          # action 0 = short
    assert all(o[2] > 0 for o in outs[1:]), [o[2] for o in outs[:3]]
    assert env.account.contracts < 0
    long = run(VenueTestingEnv(synth(close), ["f"], ["f"], cfg), [2] * 30)
    assert long[5][2] < 0


def test_funding_sign():
    close = np.full(20, 50_000.0)
    cfg = TradingConfig(mode="long_short", venue=KRAKEN, leverage=5, capital=10_000, reward_scale=1).validate()
    logs = {}
    for name, a in (("short", 0), ("long", 2), ("flat", 1)):
        env = VenueTestingEnv(synth(close), ["f"], ["f"], cfg, bar_hours=1.0)
        quiet(run, env, [a] * 10)
        logs[name] = env.account.log()
    assert (logs["long"]["funding"] > 0).all() and (logs["short"]["funding"] < 0).all()
    assert (logs["flat"]["funding"] == 0).all()
    assert np.allclose(logs["long"]["funding"], -logs["short"]["funding"])
    # flat price: long/short pnl is only fees and funding
    assert (logs["long"]["pnl"] == 0).all()


def test_forced_liquidation():
    close = np.full(10, 50_000.0)
    low = close.copy(); low[3] = 40_000.0    # crash bar: low far below close
    cfg = TradingConfig(mode="long_short", venue=KRAKEN, leverage=9, capital=10_000, reward_scale=1).validate()
    env = VenueTestingEnv(synth(close, low=low), ["f"], ["f"], cfg)
    env.reset()
    for _ in range(2):
        env.step(2)
    assert env.account.contracts > 0
    env.step(2)                                # step landing on bar index 3
    log = env.account.log()
    assert env.account.contracts == 0 and log["liquidated"][-1]
    assert log["fee"][-1] >= KRAKEN.liquidation_fee_usd
    assert log["equity"][-1] < cfg.capital


def test_training_q_value():
    cfg = TradingConfig(mode="long_short", venue=KRAKEN, turnover_penalty_bps=1.0).validate()
    env = VenueTrainingEnv(DF, SINGLE, TREND, cfg)
    q = make_q_table_trading(DF["close"].to_numpy(), cfg, env.bar_hours)
    assert np.array_equal(env.q_table, q)
    info = env.reset()[-1]
    assert np.array_equal(info["q_value"], q[0][1])
    for a in (0, 2, 1):
        info = env.step(a)[-1]
        assert np.array_equal(info["q_value"], q[env.m - 1][a])


def test_make_env_legacy_classes():
    assert type(make_env("train", "low", DF, SINGLE, TREND, TradingConfig())) is LL.Training_Env
    assert type(make_env("test", "low", DF, SINGLE, TREND, TradingConfig())) is LL.Testing_Env
    assert type(make_env("train", "high", DF, SINGLE, TREND, TradingConfig(), clf_list=CLF)) is HL.Training_Env
    assert type(make_env("test", "high", DF, SINGLE, TREND, TradingConfig(), clf_list=CLF)) is HL.Testing_Env
    cfg = TradingConfig(mode="long_only", venue=KRAKEN).validate()
    assert type(make_env("test", "low", DF, SINGLE, TREND, cfg)) is VenueTestingEnv


def test_bar_hours_and_terminal():
    cfg = TradingConfig(mode="long_short", venue=KRAKEN).validate()
    env = VenueTestingEnv(DF, SINGLE, TREND, cfg)
    assert abs(env.bar_hours - 1 / 60) < 1e-12
    assert env.initial_action == 1
    outs = quiet(run, env, [2] * (len(DF) + 5))
    assert outs[-1][-2] and len(outs) == len(DF) - 1
    log = env.trading_log
    assert len(log["close"]) == len(log["equity"]) == len(log["timestamp"])
    r, fb, req, fee = env.get_final_return_rate()
    assert req == cfg.capital and abs(fb - (log["equity"][-1] - cfg.capital)) < 1e-9 and fee > 0
    assert env.pured_balance == env.final_balance


def test_buy_hold_and_penalty_scope():
    sys.path.insert(0, os.path.join(TM, "MacroHFT", "trading"))
    from MacroHFT.trading import report
    close = [100, 110, 121, 133.1, 146.41] * 1
    cfg = TradingConfig(mode="long_only", venue=KRAKEN, turnover_penalty_bps=50).validate()
    env = VenueTestingEnv(synth(close), ["f"], ["f"], cfg, initial_action=1)
    quiet(run, env, [1] * 10)
    m = report.summarize_log(env.trading_log, cfg.capital, env.bar_hours)
    assert abs(m["buy_hold"] - 0.4641) < 1e-9, m["buy_hold"]
    # penalty: in the log always, in the reward only for the training env
    rewards = {}
    for cls in (VenueTestingEnv, VenueTrainingEnv):
        e = cls(synth([100, 100, 100, 100]), ["f"], ["f"], cfg, initial_action=0)
        e.reset()
        out = e.step(1)
        rewards[cls] = (out[-3], e.account._log[-1].penalty)
    assert rewards[VenueTestingEnv][1] > 0 and rewards[VenueTestingEnv][0] > rewards[VenueTrainingEnv][0]


def test_leverage_policy_enforced():
    for lev in (3, 10, 2, 12):
        try:
            TradingConfig(mode="long_short", venue=KRAKEN, leverage=lev).validate()
        except ValueError:
            continue
        raise AssertionError(lev)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"OK test_trading_env: {len(tests)} tests passed")
