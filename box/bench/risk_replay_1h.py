"""[TradeMaster] Replay all 1h MacroHFT runs under the tm_risk arms (parallel), check identity_L5 against the old logs, print markdown tables."""
import argparse, glob, json, os, sys
from collections import Counter, OrderedDict
from multiprocessing import Pool

import numpy as np

TM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TM)
from tm_risk.engine import replay                                              # noqa: E402
from tm_risk.report import attribution, parse_run_dir                          # noqa: E402
from tm_risk.signals import from_macrohft                                      # noqa: E402
from tm_risk.types import load_policy                                          # noqa: E402
from tm_risk.venue import load_venue                                           # noqa: E402

ARMS = ["identity_L5", "constant_1x", "vol_target_25", "default_35dd"]
DATA = os.path.join(TM, "MacroHFT/data")


def job(args):
    run_dir, arm, venue_name, capital = args
    ds = parse_run_dir(run_dir)[0]
    stream = from_macrohft(run_dir, os.path.join(DATA, ds), "test")
    venue = load_venue(venue_name)
    res = replay(stream, load_policy(arm), venue, capital=capital)
    out = dict(run_dir=run_dir, arm=arm, summary=res.summary, attribution=attribution(res),
               n_uncovered=res.info["n_uncovered"], truncated=res.info["truncated"])
    old = os.path.join(run_dir, "test", "trading_log.npz")
    if arm == "identity_L5" and os.path.isfile(old):                     # sanity: replay equity == the original env log
        import pandas as pd
        log = np.load(old, allow_pickle=True)
        df = pd.read_feather(os.path.join(DATA, ds, "whole", "test.feather"))
        r0 = int(pd.Index(pd.to_datetime(df["timestamp"])).get_indexer(pd.to_datetime(pd.Series(log["timestamp"][:1])))[0])
        n = len(log["equity"])
        out["old_log"] = dict(n=n, of=len(df) - r0, max_abs_diff=float(np.max(np.abs(res.bars["equity_mark"][r0:r0 + n] - log["equity"]))),
                              old_final=float(log["equity"][-1]), new_final_at_n=float(res.bars["equity_mark"][r0 + n - 1]))
    return out


def fmt(x, f):
    return f.format(x)


COLS = [("return", "return", "{:+.1%}"), ("max_dd", "max DD (all-time)", "{:.1%}"), ("max_dd_ref", "max DD (90d ref)", "{:.1%}"), ("sharpe", "Sharpe", "{:.2f}"),
        ("exposure_mean", "exposure mean", "{:.2f}"), ("in_market", "% in market", "{:.0f}"),
        ("rt_per_day", "round trips/day", "{:.2f}"), ("fees", "fees USD", "{:.0f}"), ("stops", "stops", "{:.0f}"),
        ("pct_halted", "% halted", "{:.1f}"), ("liquidations", "liq", "{:.1f}"), ("buy_hold", "buy&hold", "{:+.1%}")]


def tables(items):
    lines = []
    groups = OrderedDict()
    for it in items:
        ds, tag, _ = parse_run_dir(it["run_dir"])
        it["summary"]["in_market"] = 100 - it["summary"]["pct_flat"]
        groups.setdefault((ds, tag), OrderedDict()).setdefault(it["arm"], []).append(it)
    for (ds, tag), arms in sorted(groups.items()):
        lines += ["", f"### {ds} / {tag}", "", "| arm | seeds | " + " | ".join(h for _, h, _ in COLS) + " |",
                  "|---|---|" + "|".join("---" for _ in COLS) + "|"]
        for arm in ARMS:
            its = arms.get(arm, [])
            if not its:
                continue
            cells = []
            for k, _, f in COLS:
                v = np.array([i["summary"][k] for i in its], dtype=float)
                cells.append(f"{f.format(v.mean())} ± {f.format(v.std(ddof=1) if len(v) > 1 else 0.0)}")
            lines.append(f"| {arm} | {len(its)} | " + " | ".join(cells) + " |")
    lines += ["", "### Attribution (all 18 runs pooled, entries and abstentions of sized proposals)", "",
              "| arm | entries | abstains | of which sizing declined (Kelly / no vol) | Kelly abstention rate | permitted/wanted at proposals | binding at entries |",
              "|---|---|---|---|---|---|---|"]
    for arm in ARMS[1:]:
        its = [i for i in items if i["arm"] == arm]
        ent = sum(i["attribution"]["n_entries"] for i in its)
        ab = sum(i["attribution"]["n_abstain"] for i in its)
        bind = Counter()
        for i in its:
            bind.update(i["attribution"]["binding"])
        ab_s = sum(v for i in its for k, v in i["attribution"]["abstain_reasons"].items() if k.startswith("sizing"))
        pw = np.mean([i["attribution"]["permitted_over_wanted"] for i in its])
        lines.append(f"| {arm} | {ent} | {ab} | {ab_s} | {ab_s / max(1, ent + ab):.1%} | {pw:.2f} | "
                     + ", ".join(f"{k} {v}" for k, v in bind.most_common()) + " |")
    lines += ["", "### Abstain reasons, default_35dd (all runs)", ""]
    reasons = Counter()
    for i in (i for i in items if i["arm"] == "default_35dd"):
        for k, v in i["attribution"]["abstain_reasons"].items():
            reasons["below one contract" if k.startswith("permitted") else k] += v
    lines += [f"- {k}: {v}" for k, v in reasons.most_common()]
    lines += ["", "### identity_L5 vs the original test/trading_log equity", "",
              "| run | log rows / bars | max abs equity diff USD | old final | replay final at log end |", "|---|---|---|---|---|"]
    for i in sorted((i for i in items if "old_log" in i), key=lambda i: i["run_dir"]):
        o = i["old_log"]
        ds, tag, s = parse_run_dir(i["run_dir"])
        lines.append(f"| {ds} {tag} {s.split('/')[-1]} | {o['n']} / {o['n'] + 0 if o['of'] == o['n'] else o['of']} | {o['max_abs_diff']:.2e} | {o['old_final']:.2f} | {o['new_final_at_n']:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", default="kraken_us_perp")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--json", default=None, help="write raw items here / read them with --load")
    ap.add_argument("--load", default=None)
    a = ap.parse_args()
    if a.load:
        items = json.load(open(a.load))
    else:
        runs = sorted(glob.glob(os.path.join(TM, "MacroHFT/result/high_level/BTCUSDT_1h_*@long_short-*/*/seed_*")))
        jobs = [(r, arm, a.venue, a.capital) for arm in reversed(ARMS) for r in runs]     # slow kelly jobs first
        with Pool(a.procs) as p:
            items = p.map(job, jobs, chunksize=1)
        if a.json:
            json.dump(items, open(a.json, "w"), default=float)
    print(tables(items))


if __name__ == "__main__":
    main()
