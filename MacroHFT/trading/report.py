"""[TradeMaster] PnL breakdown CLI: per-run metrics from test/trading_log.npz and mean/std across seeds."""
import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # run as a script
from MacroHFT.trading.bars import DEFAULT_BAR_HOURS, infer_bar_hours  # noqa: E402

DEFAULT_CAPITAL = 10_000.0
HOURS_PER_YEAR = 24 * 365.25
# (key, header, format) of the per-run table; the grouped table uses the same keys
COLUMNS = [("net_pnl", "net PnL USD", "{:.2f}"), ("return", "return", "{:.2%}"), ("buy_hold", "buy&hold", "{:.2%}"),
           ("max_dd", "max DD", "{:.2%}"), ("sharpe", "Sharpe", "{:.2f}"), ("changes", "pos. changes", "{:.1f}"),
           ("rt_per_day", "round trips/day", "{:.2f}"), ("pct_long", "% long", "{:.1f}"),
           ("pct_short", "% short", "{:.1f}"), ("pct_flat", "% flat", "{:.1f}"), ("pnl_long", "PnL long", "{:.2f}"),
           ("pnl_short", "PnL short", "{:.2f}"), ("fees", "fees", "{:.2f}"), ("funding", "funding net", "{:.2f}"),
           ("liquidations", "liquidations", "{:.1f}")]


def summarize_log(log, capital, bar_hours):
    """Metrics of one run from a trading_log dict (Account.log() arrays plus close and timestamp)."""
    equity = np.asarray(log["equity"], dtype=float)
    n = len(equity)
    curve = np.concatenate([[capital], equity])
    net = float(equity[-1] - capital) if n else 0.0
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.max((peak - curve) / peak)) if peak.max() > 0 else 0.0
    rets = np.diff(curve) / np.where(curve[:-1] != 0, curve[:-1], np.nan)
    rets = rets[np.isfinite(rets)]
    sd = rets.std() if len(rets) else 0.0
    sharpe = float(rets.mean() / sd * np.sqrt(HOURS_PER_YEAR / bar_hours)) if sd > 0 else 0.0
    close = np.asarray(log["close"], dtype=float)
    start = float(log["close_start"]) if "close_start" in log else float(close[0]) if len(close) else 0.0
    buy_hold = float(close[-1] / start - 1) if len(close) and start > 0 else 0.0
    side = np.sign(np.asarray(log["contracts"], dtype=float))
    liq = np.asarray(log["liquidated"], dtype=bool)
    # a liquidated bar was held in the direction requested, although the position ends the bar at 0
    bar_side = np.where(liq, np.sign(np.asarray(log["direction"], dtype=float)), side)
    prev = np.concatenate([[0.0], side[:-1]])
    closes = int(np.sum((prev != 0) & (side != prev)))     # position closed or flipped
    days = n * bar_hours / 24
    pnl = np.asarray(log["pnl"], dtype=float)
    pct = lambda m: float(100 * np.mean(m)) if n else 0.0
    return dict(net_pnl=net, **{"return": net / capital}, buy_hold=buy_hold, max_dd=max_dd, sharpe=sharpe,
                changes=int(np.sum((np.asarray(log["traded"]) > 0) | liq)), rt_per_day=closes / days if days > 0 else 0.0,
                pct_long=pct(bar_side > 0), pct_short=pct(bar_side < 0), pct_flat=pct(bar_side == 0),
                pnl_long=float(pnl[bar_side > 0].sum()), pnl_short=float(pnl[bar_side < 0].sum()),
                fees=float(np.sum(log["fee"])), funding=float(np.sum(log["funding"])),
                liquidations=int(liq.sum()), bars=n, bar_hours=bar_hours, capital=capital)


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def read_capital(run_dir, default=DEFAULT_CAPITAL):
    path = os.path.join(run_dir, "trading_config.yaml")
    if not os.path.isfile(path):
        return default
    with open(path) as f:
        cap = _find_key(yaml.safe_load(f), "capital")
    return float(cap) if cap else default


def parse_run_dir(run_dir):
    """(dataset, tag, seed label) from .../high_level/<dataset>@<tag>/<exp>/seed_<n>; tag is 'legacy' if absent."""
    parts = os.path.normpath(run_dir).split(os.sep)
    key = None
    if "high_level" in parts[:-1]:
        key = parts[len(parts) - 1 - parts[::-1].index("high_level") + 1]
    seeds = [i for i, p in enumerate(parts) if p.startswith("seed_")]
    if key is None and seeds and seeds[-1] >= 2:
        key = parts[seeds[-1] - 2]
    if key is None:
        return "?", "?", parts[-1]
    dataset, _, tag = key.partition("@")
    return dataset, tag or "legacy", parts[seeds[-1]] if seeds else parts[-1]


def load_run(run_dir, bar_hours=None):
    """Metrics dict for one run dir, or None when it has no trading log (legacy run)."""
    path = os.path.join(run_dir, "test", "trading_log.npz")
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as z:
        log = {k: z[k] for k in z.files}
    bh = bar_hours or (infer_bar_hours(log["timestamp"]) if "timestamp" in log else DEFAULT_BAR_HOURS)
    return summarize_log(log, read_capital(run_dir), bh)


def group_runs(runs):
    """runs: list of (run_dir, metrics). -> OrderedDict[(dataset, tag)] -> {key: (mean, std, n)}."""
    groups = OrderedDict()
    for run_dir, m in runs:
        dataset, tag, _ = parse_run_dir(run_dir)
        groups.setdefault((dataset, tag), []).append(m)
    out = OrderedDict()
    for k, ms in sorted(groups.items()):
        out[k] = {c: (float(np.mean([m[c] for m in ms])), float(np.std([m[c] for m in ms], ddof=1)) if len(ms) > 1
                      else 0.0, len(ms)) for c, _, _ in COLUMNS}
    return out


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in rows]


def to_markdown(runs, skipped):
    lines = ["# Trading report", "", "## Runs", ""]
    heads = ["dataset", "tag", "run"] + [h for _, h, _ in COLUMNS]
    rows = []
    for run_dir, m in runs:
        d, t, s = parse_run_dir(run_dir)
        rows.append([d, t, s] + [f.format(m[k]) for k, _, f in COLUMNS])
    lines += _table(heads, rows) if rows else ["(none)"]
    lines += ["", "## Across seeds (mean ± std)", ""]
    grows = []
    for (d, t), stats in group_runs(runs).items():
        grows.append([d, t, str(next(iter(stats.values()))[2])] +
                     [f"{f.format(stats[k][0])} ± {f.format(stats[k][1])}" for k, _, f in COLUMNS])
    lines += _table(["dataset", "tag", "seeds"] + [h for _, h, _ in COLUMNS], grows) if grows else ["(none)"]
    if skipped:
        lines += ["", "## Skipped (no test/trading_log.npz, legacy run)", ""] + [f"- {p}" for p in skipped]
    lines += ["", "Notes: PnL long/short is gross price PnL by held side; net PnL = PnL - fees - funding. "
              "Sharpe uses per-bar equity returns annualised from the bar spacing; std across seeds uses n-1.", ""]
    return "\n".join(lines)


def build_report(run_dirs, bar_hours=None):
    runs, skipped = [], []
    for d in run_dirs:
        m = load_run(d, bar_hours)
        (skipped.append(d) if m is None else runs.append((d, m)))
    return to_markdown(runs, skipped)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PnL breakdown of MacroHFT trading runs")
    ap.add_argument("run_dirs", nargs="+", metavar="RUN_DIR", help="high-level run dir containing test/trading_log.npz")
    ap.add_argument("--out", default=None, help="write markdown here instead of printing")
    ap.add_argument("--bar_hours", type=float, default=None, help="override the bar spacing inferred from timestamps")
    args = ap.parse_args(argv)
    text = build_report(args.run_dirs, args.bar_hours)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print("wrote", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
