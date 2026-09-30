"""[TradeMaster] Tests of MacroHFT/trading/report.py on synthetic trading logs with hand-computable numbers."""
import os, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TM = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TM)
from MacroHFT.trading import report

CAPITAL = 1000.0


def make_log(scale=1.0):
    """8 hourly bars: long on bars 1-2, flat on 3, short on 4-5, flat on 6-7."""
    pnl = np.array([0, 10, -5, 0, 20, -10, 0, 0], dtype=float) * scale
    fee = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    funding = np.array([0, .5, .5, 0, -.25, -.25, 0, 0])
    contracts = np.array([0, 1, 1, 0, -1, -1, 0, 0], dtype=float)
    ts = np.datetime64("2024-01-01T00:00") + np.arange(8) * np.timedelta64(1, "h")
    return dict(direction=np.sign(contracts), contracts=contracts, traded=fee.copy(), pnl=pnl, fee=fee,
                funding=funding, penalty=np.zeros(8), liquidated=np.zeros(8, bool), bankrupt=np.zeros(8, bool),
                equity=CAPITAL + np.cumsum(pnl - fee - funding), notional=np.abs(contracts) * 100,
                close=np.array([100, 101, 102, 103, 104, 105, 106, 110], dtype=float), timestamp=ts)


def test_infer_bar_hours():
    assert abs(report.infer_bar_hours(make_log()["timestamp"]) - 1.0) < 1e-12
    assert report.infer_bar_hours(np.arange(5)) == report.DEFAULT_BAR_HOURS


def test_metrics():
    m = report.summarize_log(make_log(), CAPITAL, 1.0)
    assert abs(m["net_pnl"] - 10.5) < 1e-9 and abs(m["return"] - 0.0105) < 1e-12
    assert abs(m["buy_hold"] - 0.10) < 1e-12
    assert abs(m["max_dd"] - 10.75 / 1021.25) < 1e-12
    eq = np.array([1000, 1000, 1008.5, 1003, 1002, 1021.25, 1011.5, 1010.5, 1010.5])
    r = np.diff(eq) / eq[:-1]
    want = r.mean() / r.std() * np.sqrt(24 * 365.25)
    assert abs(m["sharpe"] - want) < 1e-9 * abs(want), (m["sharpe"], want)
    assert m["changes"] == 4 and abs(m["rt_per_day"] - 2 / (8 / 24)) < 1e-12
    assert (m["pct_long"], m["pct_short"], m["pct_flat"]) == (25.0, 25.0, 50.0)
    assert m["pnl_long"] == 5.0 and m["pnl_short"] == 10.0
    assert m["fees"] == 4.0 and abs(m["funding"] - 0.5) < 1e-12 and m["liquidations"] == 0


def test_flat_and_liquidation():
    log = make_log()
    log["equity"] = np.full(8, CAPITAL); log["pnl"] = np.zeros(8); log["contracts"] = np.zeros(8)
    m = report.summarize_log(log, CAPITAL, 1.0)
    assert m["sharpe"] == 0.0 and m["max_dd"] == 0.0 and m["pct_flat"] == 100.0
    log = make_log()   # a liquidated long bar ends flat but its loss belongs to the long side
    log["contracts"][2] = 0; log["liquidated"][2] = True; log["direction"][2] = 1.0
    m = report.summarize_log(log, CAPITAL, 1.0)
    assert m["liquidations"] == 1 and m["pnl_long"] == 5.0
    assert m["pct_long"] == 25.0 and m["pct_flat"] == 50.0   # liquidated bar counts as long, as the PnL does
    log["close_start"] = 50.0   # buy&hold starts at the pre-first-bar close when logged
    assert abs(report.summarize_log(log, CAPITAL, 1.0)["buy_hold"] - 1.2) < 1e-12


def test_parse_run_dir():
    assert report.parse_run_dir("/x/result/high_level/BTC@long_short-k-L5-tp0/exp1/seed_2") == \
        ("BTC", "long_short-k-L5-tp0", "seed_2")
    assert report.parse_run_dir("result/high_level/BTC/exp1/seed_1") == ("BTC", "legacy", "seed_1")


def write_run(root, key, seed, log, yaml=True):
    d = os.path.join(root, "result", "high_level", key, "exp", f"seed_{seed}")
    os.makedirs(os.path.join(d, "test"))
    if log is not None:
        np.savez(os.path.join(d, "test", "trading_log.npz"), **log)
    if yaml:
        open(os.path.join(d, "trading_config.yaml"), "w").write(f"config:\n  capital: {CAPITAL}\n")
    return d


def test_grouping_and_markdown():
    with tempfile.TemporaryDirectory() as root:
        key = "BTC@long_short-kraken_us_perp-L5-tp0"
        d1 = write_run(root, key, 1, make_log(1.0))
        d2 = write_run(root, key, 2, make_log(3.0))
        legacy = write_run(root, "BTC", 1, None, yaml=False)
        assert report.load_run(legacy) is None and abs(report.load_run(d1)["net_pnl"] - 10.5) < 1e-9
        runs = [(d, report.load_run(d)) for d in (d1, d2)]
        g = report.group_runs(runs)
        assert list(g) == [("BTC", "long_short-kraken_us_perp-L5-tp0")]
        mean, std, n = g[list(g)[0]]["net_pnl"]     # nets: 10.5 and 40.5
        assert n == 2 and abs(mean - 25.5) < 1e-9 and abs(std - 30 / np.sqrt(2)) < 1e-9
        out = os.path.join(root, "r.md")
        report.main([d1, d2, legacy, "--out", out])
        md = open(out).read()
        assert "| BTC | long_short-kraken_us_perp-L5-tp0 | seed_1 | 10.50 |" in md
        assert "| BTC | long_short-kraken_us_perp-L5-tp0 | 2 | 25.50 ± 21.21 |" in md
        assert "Skipped" in md and f"- {legacy}" in md
        assert "seed_1 | 10.50" in md and md.count("seed_2") == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"OK: {len(tests)} report tests passed")
