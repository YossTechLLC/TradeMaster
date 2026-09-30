"""[TradeMaster] Replay metrics: summarize (PnL, drawdown, exposure, costs), attribution (binding constraints), markdown tables."""
import os
import re
from collections import Counter, OrderedDict

import numpy as np

HOURS_PER_YEAR = 24 * 365.25
_NUM = re.compile(r"permitted [-+0-9.eE]+")            # strip the USD amount so abstain reasons group
CLOSE_ACTIONS = ("exit", "flip", "stop", "liquidation")
# (key, header, format) of the per-run table; the across-seed table uses the same keys
COLUMNS = [("net_pnl", "net PnL USD", "{:.2f}"), ("return", "return", "{:.2%}"), ("buy_hold", "buy&hold", "{:.2%}"),
           ("max_dd", "max DD", "{:.2%}"), ("max_dd_ref", "max DD (ref)", "{:.2%}"), ("sharpe", "Sharpe", "{:.2f}"),
           ("exposure_mean", "exposure mean", "{:.2f}"), ("exposure_p95", "exposure p95", "{:.2f}"),
           ("pct_long", "% long", "{:.1f}"), ("pct_short", "% short", "{:.1f}"), ("pct_flat", "% flat", "{:.1f}"),
           ("pnl_long", "PnL long", "{:.2f}"), ("pnl_short", "PnL short", "{:.2f}"), ("fees", "fees", "{:.2f}"),
           ("funding", "funding net", "{:.2f}"), ("slippage", "slippage", "{:.2f}"),
           ("rt_per_day", "round trips/day", "{:.2f}"), ("stops", "stops", "{:.1f}"),
           ("liquidations", "liquidations", "{:.1f}"), ("breaches", "breaches", "{:.1f}"),
           ("pct_halted", "% halted", "{:.1f}")]


def summarize(result, capital, bar_hours) -> dict:
    """Metrics of one replay. `result` needs .bars (dict of per-bar arrays) and .decisions (list of Decision)."""
    b = result.bars
    equity = np.asarray(b["equity"], dtype=float)
    n = len(equity)
    if n == 0:
        return dict(net_pnl=0.0, final_equity=float(capital), bars=0, **{c: 0.0 for c, _, _ in COLUMNS if c != "net_pnl"})
    curve = np.concatenate([[capital], equity])
    net = float(equity[-1] - capital)
    peak = np.maximum.accumulate(curve)
    max_dd = float(min(1.0, np.max(1 - curve / peak))) if peak.max() > 0 else 0.0   # negative equity (identity bankruptcy) is clamped: a drawdown cannot exceed 100%
    max_dd_ref = float(np.max(b["dd_ref"])) if "dd_ref" in b else max_dd
    rets = np.diff(curve) / np.where(curve[:-1] != 0, curve[:-1], np.nan)
    rets = rets[np.isfinite(rets)]
    sd = rets.std() if len(rets) else 0.0
    sharpe = float(rets.mean() / sd * np.sqrt(HOURS_PER_YEAR / bar_hours)) if sd > 0 else 0.0
    close = np.asarray(b["close"], dtype=float)
    buy_hold = float(close[-1] / close[0] - 1) if close[0] > 0 else 0.0
    # the position that earned bar k's pnl is the one held at the end of bar k-1
    held = np.sign(np.concatenate([[0.0], np.asarray(b["contracts"], dtype=float)[:-1]]))
    pnl = np.asarray(b["pnl"], dtype=float)
    acts = Counter(d.action for d in result.decisions)
    pct = lambda m: float(100 * np.mean(m))
    days = n * bar_hours / 24
    expo = np.asarray(b["exposure"], dtype=float)
    return dict(net_pnl=net, final_equity=float(equity[-1]), **{"return": net / capital}, buy_hold=buy_hold,
                max_dd=max_dd, max_dd_ref=max_dd_ref, sharpe=sharpe,
                exposure_mean=float(expo.mean()), exposure_p95=float(np.percentile(expo, 95)),
                pct_long=pct(held > 0), pct_short=pct(held < 0), pct_flat=pct(held == 0),
                pnl_long=float(pnl[held > 0].sum()), pnl_short=float(pnl[held < 0].sum()),
                fees=float(np.sum(b["fee"])), funding=float(np.sum(b["funding"])),
                slippage=float(np.sum(b["slippage"])) if "slippage" in b else 0.0,
                rt_per_day=sum(acts[a] for a in CLOSE_ACTIONS) / days if days > 0 else 0.0,
                entries=int(acts["enter"] + acts["flip"]), abstains=int(acts["abstain"]), stops=int(acts["stop"]),
                liquidations=int(acts["liquidation"]), breaches=int(acts["breach"]),
                pct_halted=pct(np.asarray(b["halted"], dtype=float) > 0), bars=n, bar_hours=float(bar_hours),
                capital=float(capital))


def attribution(result) -> dict:
    """Which constraint bound at entries, wanted vs permitted, and why proposals were rejected."""
    entries = [d for d in result.decisions if d.action in ("enter", "flip")]
    aband = [d for d in result.decisions if d.action == "abstain"]
    label = lambda d: d.binding or "none"
    sized = [d for d in entries + aband if d.wanted > 0]
    ratios = [min(d.permitted / d.wanted, 1.0) for d in sized]
    return dict(n_entries=len(entries), n_abstain=len(aband),
                binding=dict(Counter(label(d) for d in entries).most_common()),
                binding_abstain=dict(Counter(label(d) for d in aband).most_common()),
                permitted_over_wanted=float(np.mean(ratios)) if ratios else 1.0,
                abstain_reasons=dict(Counter(_NUM.sub("permitted", d.reason or "unspecified") for d in aband).most_common()),
                vetoes=dict(Counter(v for d in aband for v in d.vetoes).most_common()))


def parse_run_dir(run_dir):
    """(dataset, tag, seed label) from .../high_level/<dataset>@<tag>/<exp>/seed_<n>; '?' when not recognisable."""
    parts = os.path.normpath(str(run_dir)).split(os.sep)
    for i, p in enumerate(parts[:-1]):
        if p == "high_level" and i + 1 < len(parts):
            ds, _, tag = parts[i + 1].partition("@")
            return ds, tag or "legacy", "/".join(parts[i + 2:]) or parts[-1]
    return "?", "?", parts[-1]


def _table(header, rows):
    return (["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] +
            ["| " + " | ".join(r) + " |" for r in rows])


def _fmt_counts(d):
    return ", ".join(f"{k} {v}" for k, v in d.items()) or "-"


def group_runs(items):
    """items: list of dicts (run_dir, policy, summary, ...). -> OrderedDict[(dataset, tag, policy)] -> {key: (mean, std, n)}."""
    groups = OrderedDict()
    for it in items:
        d, t, _ = parse_run_dir(it["run_dir"])
        groups.setdefault((d, t, it["policy"]), []).append(it["summary"])
    out = OrderedDict()
    for k, ms in sorted(groups.items()):
        out[k] = {c: (float(np.mean([m[c] for m in ms])),
                      float(np.std([m[c] for m in ms], ddof=1)) if len(ms) > 1 else 0.0, len(ms)) for c, _, _ in COLUMNS}
    return out


def to_markdown(items, title="Risk replay report", notes=()) -> str:
    """items: list of dicts with run_dir, policy (name), summary, attribution (optional), info (optional)."""
    lines = [f"# {title}", "", "## Runs", ""]
    rows = []
    for it in items:
        d, t, s = parse_run_dir(it["run_dir"])
        rows.append([d, t, s, it["policy"]] + [f.format(it["summary"][k]) for k, _, f in COLUMNS])
    lines += _table(["dataset", "tag", "run", "policy"] + [h for _, h, _ in COLUMNS], rows) if rows else ["(none)"]
    lines += ["", "## Across seeds (mean ± std)", ""]
    grows = [[d, t, p, str(next(iter(st.values()))[2])] + [f"{f.format(st[k][0])} ± {f.format(st[k][1])}" for k, _, f in COLUMNS]
             for (d, t, p), st in group_runs(items).items()]
    lines += _table(["dataset", "tag", "policy", "seeds"] + [h for _, h, _ in COLUMNS], grows) if grows else ["(none)"]
    arows = []
    for it in items:
        a = it.get("attribution")
        if a:
            d, t, s = parse_run_dir(it["run_dir"])
            arows.append([d, t, s, it["policy"], str(a["n_entries"]), str(a["n_abstain"]), f"{a['permitted_over_wanted']:.2f}",
                          _fmt_counts(a["binding"]), _fmt_counts(a["abstain_reasons"])])
    if arows:
        lines += ["", "## Attribution (entries)", ""]
        lines += _table(["dataset", "tag", "run", "policy", "entries", "abstains", "permitted/wanted", "binding at entries",
                         "abstain reasons"], arows)
    warn = [f"- {it['run_dir']} / {it['policy']}: {it['info']}" for it in items if it.get("info")]
    if warn:
        lines += ["", "## Stream warnings", ""] + warn
    lines += ["", "Notes: PnL long/short is gross price PnL by held side; net PnL = PnL - fees - funding - slippage. "
              "Sharpe uses per-bar equity returns annualised from the bar spacing; std across seeds uses n-1. "
              "max DD is all-time peak; max DD (ref) is against the trailing reference peak. "
              "Buy&hold is close-to-close over the replayed bars."] + list(notes) + [""]
    return "\n".join(lines)
