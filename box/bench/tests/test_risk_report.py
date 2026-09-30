"""Tests for tm_risk/report.py (hand-computed metrics on synthetic replays), the replay CLI and export_signal parity."""
import os, subprocess, sys
from types import SimpleNamespace
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
from tm_risk.report import attribution, parse_run_dir, summarize, to_markdown
from tm_risk.types import Decision

RUN = os.path.join(TM, "MacroHFT/result/high_level/BTCGOLD@long_short-kraken_us_perp-L5-tp5/exp1/seed_12345")


def fake(equity, close, contracts, pnl, fee=None, funding=None, slippage=None, halted=None, dd_ref=None, decisions=()):
    n = len(equity)
    z = np.zeros(n)
    bars = dict(equity=np.array(equity, float), close=np.array(close, float), contracts=np.array(contracts, float),
                pnl=np.array(pnl, float), fee=z if fee is None else np.array(fee, float),
                funding=z if funding is None else np.array(funding, float),
                slippage=z if slippage is None else np.array(slippage, float),
                halted=z if halted is None else np.array(halted, float), exposure=np.array([0, 1, 1, 2.0, 0][:n]),
                dd_ref=z if dd_ref is None else np.array(dd_ref, float))
    return SimpleNamespace(bars=bars, decisions=list(decisions))


def five_bar():
    # capital 100; equity 110, 99, 99, 108.9, 108.9; long held over bars 1-2 (opened at close of bar 0), short over bar 3
    acts = [Decision(0, 0, 1, "enter"), Decision(2, 2, 0, "exit"), Decision(2, 2, 0, "stop"), Decision(3, 3, -1, "flip"),
            Decision(4, 4, 0, "breach"), Decision(4, 4, 0, "liquidation")]
    return fake(equity=[100, 110, 99, 99, 108.9], close=[10, 11, 9.9, 9.9, 9.0], contracts=[1, 1, 0, -1, 0],
                pnl=[0, 10.5, -10.5, 0, 9.9], fee=[0.2, 0, 0.3, 0, 0], funding=[0, 0.5, 0, 0, 0.1], slippage=[0, 0, 0.2, 0, 0],
                halted=[0, 0, 1, 1, 0], dd_ref=[0, 0, 0.1, 0.1, 0], decisions=acts)


def test_summarize_basic():
    s = summarize(five_bar(), 100.0, 24.0)          # bar_hours 24 -> 5 days
    assert abs(s["net_pnl"] - 8.9) < 1e-12 and abs(s["return"] - 0.089) < 1e-12
    assert abs(s["max_dd"] - 0.1) < 1e-12           # 110 -> 99
    assert abs(s["max_dd_ref"] - 0.1) < 1e-12
    assert abs(s["buy_hold"] - (9.0 / 10 - 1)) < 1e-12
    # held side per bar = sign(contracts[k-1]): flat, long, long, flat, short
    assert (s["pct_long"], s["pct_short"], s["pct_flat"]) == (40.0, 20.0, 40.0)
    assert abs(s["pnl_long"] - 0.0) < 1e-12         # bars 1,2: 10.5 - 10.5
    assert abs(s["pnl_short"] - 9.9) < 1e-12        # bar 4
    assert abs(s["fees"] - 0.5) < 1e-12 and abs(s["funding"] - 0.6) < 1e-12 and abs(s["slippage"] - 0.2) < 1e-12
    assert abs(s["rt_per_day"] - 4 / 5) < 1e-12     # closes: exit, stop, flip, liquidation (breach is partial)
    assert (s["stops"], s["liquidations"], s["breaches"]) == (1, 1, 1)
    assert s["pct_halted"] == 40.0
    assert abs(s["exposure_mean"] - 0.8) < 1e-12
    assert s["entries"] == 2


def test_sharpe_hand():
    res = fake([101, 100, 101, 100, 101], [1] * 5, [0] * 5, [0] * 5)
    s = summarize(res, 100.0, 1.0)
    r = np.array([0.01, 100 / 101 - 1, 101 / 100 - 1, 100 / 101 - 1, 101 / 100 - 1])
    want = r.mean() / r.std() * np.sqrt(24 * 365.25)
    assert abs(s["sharpe"] - want) < 1e-9
    flat = summarize(fake([100] * 5, [1] * 5, [0] * 5, [0] * 5), 100.0, 1.0)
    assert flat["sharpe"] == 0.0 and flat["max_dd"] == 0.0 and flat["net_pnl"] == 0.0


def test_max_dd_uses_capital_as_first_peak():
    s = summarize(fake([90, 80, 85, 60, 70], [1] * 5, [0] * 5, [0] * 5), 100.0, 1.0)
    assert abs(s["max_dd"] - 0.4) < 1e-12           # 100 -> 60


def test_attribution_counts():
    ds = [Decision(1, 1, 1, "enter", wanted=100, permitted=50, binding="trade_loss"),
          Decision(2, 2, 1, "flip", wanted=200, permitted=200, binding="request"),
          Decision(3, 3, 1, "enter", wanted=100, permitted=25, binding="trade_loss"),
          Decision(4, 4, 1, "abstain", wanted=100, permitted=0, binding="halted", vetoes=("halted",), reason="halted"),
          Decision(5, 5, 1, "abstain", wanted=0, permitted=0, binding="request", reason="kelly abstain"),
          Decision(6, 6, 0, "exit"), Decision(7, 7, 1, "hold")]
    a = attribution(SimpleNamespace(decisions=ds))
    assert a["n_entries"] == 3 and a["n_abstain"] == 2
    assert a["binding"] == {"trade_loss": 2, "request": 1}
    assert a["abstain_reasons"] == {"halted": 1, "kelly abstain": 1}
    assert a["vetoes"] == {"halted": 1}
    # ratios over entries+abstains with wanted>0: 0.5, 1, 0.25, 0 -> 0.4375
    assert abs(a["permitted_over_wanted"] - 0.4375) < 1e-12
    assert attribution(SimpleNamespace(decisions=[]))["permitted_over_wanted"] == 1.0


def test_markdown_across_seeds():
    def item(seed, net):
        r = five_bar()
        s = summarize(r, 100.0, 24.0)
        s["net_pnl"] = net
        return dict(run_dir=f"/x/result/high_level/DS@tag/exp1/seed_{seed}", policy="p", summary=s, attribution=attribution(r))
    md = to_markdown([item(1, 10.0), item(2, 20.0)])
    assert "| DS | tag | p | 2 |" in md and "15.00 ± 7.07" in md      # mean 15, sample std sqrt(50)
    assert md.count("| DS | tag | exp1/seed_") == 4                       # per-run and attribution tables
    assert parse_run_dir("/a/high_level/K@t/exp1/seed_5") == ("K", "t", "exp1/seed_5")


def test_cli_identity_end_to_end():
    out = subprocess.run([sys.executable, "-m", "tm_risk.replay", "--macrohft", RUN, "--dataset_root", "MacroHFT/data",
                          "--policy", "identity_L5"], cwd=TM, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    # legacy trading report for the same run: net -504.68, fees 117.90, max DD 10.17%
    line = [l for l in out.stdout.splitlines() if l.startswith("| BTCGOLD") and "exp1/seed_12345" in l][0]
    assert "| -504.68 | -5.05% | -1.39% | 10.17% |" in line and "| 117.90 |" in line, line


SKIPPED = []


def test_venue_yaml_copies_match():
    a, b = os.path.join(TM, "MacroHFT/trading/venues"), os.path.join(TM, "tm_risk/venues")
    assert sorted(os.listdir(a)) == sorted(os.listdir(b))
    for f in os.listdir(a):
        assert open(os.path.join(a, f)).read() == open(os.path.join(b, f)).read(), f


def test_abstain_reasons_group_without_amounts():
    from tm_risk.report import attribution
    from tm_risk.types import Decision
    ds = [Decision(k=i, timestamp=i, signal=1, action="abstain", wanted=1.0, reason=f"permitted {100 + i * 1.5:.3f} below one contract (binding x)")
          for i in range(3)]
    r = type("R", (), dict(decisions=ds))()
    assert attribution(r)["abstain_reasons"] == {"permitted below one contract (binding x)": 3}


def test_summary_drawdown_clamped_at_one():
    from tm_risk.report import summarize
    n = 3
    b = {k: np.zeros(n) for k in ("equity", "contracts", "exposure", "pnl", "fee", "funding", "halted", "dd_ref", "close")}
    b["equity"] = np.array([500.0, -100.0, -50.0]); b["close"] = np.ones(n)
    assert summarize(type("R", (), dict(bars=b, decisions=[]))(), 1000.0, 1.0)["max_dd"] == 1.0


def test_export_parity_file():
    sig = os.path.join(RUN, "test", "signal_test.npz")
    if not os.path.isfile(sig):
        print("skip: run export_signal first")
        SKIPPED.append("export_parity")
        return
    z = np.load(sig)
    old = np.load(os.path.join(RUN, "test", "action.npy")).reshape(-1)
    assert (z["action"][:len(old)] == old).all() and len(z["action"]) == len(old)
    assert set(np.unique(z["direction"])) <= {-1, 0, 1} and (z["direction"] == z["action"] - 1).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("test_risk_report OK" + (f" (skipped: {', '.join(SKIPPED)})" if SKIPPED else ""))
