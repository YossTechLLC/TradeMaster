"""[TradeMaster] CLI: replay MacroHFT runs under tm_risk policies and write a markdown report (python -m tm_risk.replay)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # run as a script
from tm_risk.engine import replay  # noqa: E402
from tm_risk.report import attribution, parse_run_dir, to_markdown  # noqa: E402
from tm_risk.signals import from_macrohft  # noqa: E402
from tm_risk.types import load_policy  # noqa: E402
from tm_risk.venue import load_venue  # noqa: E402


def run_pair(run_dir, dataset_dir, policy, venue, split="test", capital=10000.0, val_stream=None, stream=None):
    stream = stream if stream is not None else from_macrohft(run_dir, dataset_dir, split)
    if policy.ledger_seed == "val" and val_stream is None:
        val_stream = from_macrohft(run_dir, dataset_dir, "val")
    res = replay(stream, policy, venue, capital=capital, val_stream=val_stream)
    info = []
    if stream.source.get("truncated"):
        info.append("signal log ends before the bars (truncated)")
    if res.info.get("n_uncovered"):
        info.append(f"{res.info['n_uncovered']} bars without a model decision treated as flat")
    return dict(run_dir=run_dir, policy=policy.name, summary=res.summary, attribution=attribution(res), info="; ".join(info))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tm_risk.replay", description="replay MacroHFT runs under risk policies")
    ap.add_argument("--macrohft", nargs="+", required=True, metavar="RUN_DIR", help="high-level run dir(s) (<ds>@<tag>/<exp>/seed_<n>)")
    ap.add_argument("--dataset_root", default="MacroHFT/data", help="dir holding <dataset>/whole/<split>.feather")
    ap.add_argument("--policy", action="append", default=None, help="policy name or yaml path (repeatable; default default_35dd)")
    ap.add_argument("--venue", default="kraken_us_perp")
    ap.add_argument("--split", default="test")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--out", default=None, help="also write the markdown here")
    a = ap.parse_args(argv)
    venue = load_venue(a.venue)
    policies = [load_policy(p) for p in (a.policy or ["default_35dd"])]
    items = []
    for run_dir in a.macrohft:
        ds = parse_run_dir(run_dir)[0]
        if ds == "?":
            raise SystemExit(f"cannot resolve the dataset from {run_dir} (expected .../high_level/<ds>@<tag>/...)")
        for pol in policies:
            items.append(run_pair(run_dir, os.path.join(a.dataset_root, ds), pol, venue, a.split, a.capital))
    text = to_markdown(items, notes=[f"Venue {venue.name}, split {a.split}, capital {a.capital:g} USD."])
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
