"""[TradeMaster] Build a MacroHFT dataset for one input profile and timeframe from Binance spot klines.

    python profiles/build.py --profile xt10 --timeframe 15m            # run from MacroHFT/ (run.sh does)
    python profiles/build.py --all                                     # every profile x timeframe

Writes data/<SYMBOL>_<TIMEFRAME>_<PROFILE>/:
    df_train.feather, df_val.feather, df_test.feather   raw close (+ raw OHLCV as raw_*) and the scaled features
    feature_list/single_features.npy, trend_features.npy the profile's two input lists (read by the agents)
    feature_scaling.yaml                                 train-fitted transform of every feature
    build_report.yaml                                    gaps, per-split feature statistics, checks, warnings
    run.env                                              per-timeframe settings read by run.sh (decompose / train-high)

Rules enforced (docs/macrohft_inputs.md section 5): fixed-interval gap-free bars (gaps filled flat with zero
volume, reported); causal features (tests/test_profiles.py); warm-up history computed and dropped before the
split; every value finite; transforms fitted on train only; |value| <= 5 after scaling; no constant feature.
Redundancy (|rho| > 0.95 on train) and drift (val/test mean or std far from train) are errors for strict
profiles and warnings otherwise.
"""
import argparse, hashlib, io, os, sys, urllib.request, zipfile, datetime as dt
import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MACRO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import FEATURES, TRAIN_FITTED  # noqa: E402

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
RAW = os.path.join(MACRO, "data", "raw_binance")
CLIP = 5.0


# ----------------------------------------------------------------------------- data
def fetch_month(symbol, interval, month):
    """Monthly spot kline zip from data.binance.vision, verified against its published SHA-256."""
    name = f"{symbol}-{interval}-{month:%Y-%m}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{name}"
    path = os.path.join(RAW, symbol, interval, name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob = urllib.request.urlopen(url, timeout=60).read()
        want = urllib.request.urlopen(url + ".CHECKSUM", timeout=60).read().split()[0].decode()
        if hashlib.sha256(blob).hexdigest() != want:
            raise IOError(f"checksum mismatch for {url}")
        with open(path, "wb") as f:
            f.write(blob)
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
    if not str(df.iloc[0, 0]).isdigit():  # a few archives carry a header row
        df = df.iloc[1:].astype({c: float for c in COLS})
    return df


def load_klines(symbol, interval, start, end):
    months = pd.period_range(start, end, freq="M").to_timestamp()
    k = pd.concat([fetch_month(symbol, interval, m) for m in months], ignore_index=True)
    t = k["open_time"].astype(np.int64)
    # Binance spot archives switched from milliseconds to microseconds on 2025-01-01
    k["timestamp"] = pd.to_datetime(np.where(t > 10**14, t // 1000, t), unit="ms")
    k = k.drop(columns=["open_time", "close_time", "ignore"]).drop_duplicates("timestamp").sort_values("timestamp")
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        k[c] = k[c].astype(np.float64)
    return k[(k.timestamp >= start) & (k.timestamp <= end)].reset_index(drop=True)


def fill_gaps(k, freq):
    """Fixed-interval bars: missing bars (exchange outages) become flat bars at the last close, zero volume."""
    full = pd.date_range(k.timestamp.iloc[0], k.timestamp.iloc[-1], freq=freq)
    k = k.set_index("timestamp").reindex(full)
    missing = k["close"].isna().to_numpy()
    k["close"] = k["close"].ffill()
    for c in ("open", "high", "low"):
        k[c] = k[c].fillna(k["close"])
    for c in ("volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        k[c] = k[c].fillna(0.0)
    runs, longest, cur = 0, 0, 0
    for m in missing:
        cur = cur + 1 if m else 0
        runs += m and cur == 1
        longest = max(longest, cur)
    k = k.rename_axis("timestamp").reset_index()
    return k, {"bars": int(len(k)), "filled_bars": int(missing.sum()), "gaps": int(runs), "longest_gap_bars": int(longest)}


# ----------------------------------------------------------------------------- features and scaling
def resolve(args, timeframe):
    return {a: (v[timeframe] if isinstance(v, dict) and timeframe in v else v) for a, v in (args or {}).items()}


def compute(k, spec, timeframe, train_mask):
    args = resolve(spec.get("args"), timeframe)
    if spec["fn"] in TRAIN_FITTED:
        args["train_mask"] = train_mask
    return FEATURES[spec["fn"]](k, **args).astype(np.float64).rename(spec["id"])


def fit_scale(x_train, how):
    steps = how.split("+")
    stats = {"transform": how}
    for s in steps:
        if s == "log1p":
            if (x_train <= -1).any():
                raise ValueError("log1p on values <= -1")
            x_train = np.log1p(x_train)
        elif s == "log":
            # floored log: zeros (flat bars) map to the 1st percentile of the positive train values
            pos = x_train[x_train > 0]
            if len(pos) < 0.9 * len(x_train):
                raise ValueError("log on a feature that is mostly <= 0")
            stats["log_floor"] = float(np.percentile(pos, 1))
            x_train = np.log(np.maximum(x_train, stats["log_floor"]))
        elif s == "z":
            stats["mean"], stats["std"] = float(x_train.mean()), float(x_train.std())
        elif s == "robust":
            q1, med, q3 = np.percentile(x_train, [25, 50, 75])
            stats["mean"], stats["std"] = float(med), float((q3 - q1) / 1.349)
        elif s != "none":
            raise ValueError(f"unknown scale step {s}")
    return stats


def apply_scale(x, stats):
    x = np.asarray(x, np.float64)
    for s in stats["transform"].split("+"):
        if s == "log1p":
            x = np.log1p(np.maximum(x, -1 + 1e-12))
        elif s == "log":
            x = np.log(np.maximum(x, stats["log_floor"]))
        elif s in ("z", "robust"):
            x = np.clip((x - stats["mean"]) / stats["std"], -CLIP, CLIP)
    return x


# ----------------------------------------------------------------------------- build
def build(profile_name, timeframe, symbol, split, out_root, force=False):
    prof = yaml.safe_load(open(os.path.join(HERE, f"{profile_name}.yaml")))
    tf = yaml.safe_load(open(os.path.join(HERE, "timeframes.yaml")))[timeframe]
    freq = {"15m": "15min", "1h": "1h", "5m": "5min", "30m": "30min", "1m": "1min"}[tf["interval"]]
    name = f"{symbol}_{timeframe}_{profile_name}"
    out = os.path.join(out_root, name)
    if os.path.exists(out) and not force:
        print(f"{out} exists (use --force to rebuild)"); return out
    (tr0, tr1), (va0, va1), (te0, te1) = [tuple(pd.Timestamp(x) for x in s.split(":")) for s in split]
    tr1, va1, te1 = (x + pd.Timedelta(days=1) - pd.Timedelta(freq) for x in (tr1, va1, te1))  # inclusive end days
    warm_start = tr0 - tf["warmup_bars"] * pd.Timedelta(freq)

    k = load_klines(symbol, tf["interval"], warm_start.replace(day=1), te1)
    k = k[k.timestamp >= warm_start].reset_index(drop=True)
    k, gaps = fill_gaps(k, freq)
    ts = k.timestamp
    split_of = np.select([(ts >= tr0) & (ts <= tr1), (ts >= va0) & (ts <= va1), (ts >= te0) & (ts <= te1)],
                         ["train", "val", "test"], "warmup")
    train_mask = split_of == "train"

    report = {"dataset": name, "profile": profile_name, "timeframe": timeframe, "symbol": symbol,
              "split": {"train": [str(tr0), str(tr1)], "val": [str(va0), str(va1)], "test": [str(te0), str(te1)]},
              "gaps": gaps, "errors": [], "warnings": [], "features": {}}
    feats, scaling = {}, {}
    for group in ("single", "trend"):
        for spec in prof[group]:
            x = compute(k, spec, timeframe, train_mask)
            keep = split_of != "warmup"
            bad = int((~np.isfinite(x[keep])).sum())
            if bad:
                report["errors"].append(f"{spec['id']}: {bad} non-finite values after the warm-up "
                                        f"(increase warmup_bars or guard the formula)")
                continue
            xt = x[train_mask].to_numpy()
            if np.std(xt) == 0:
                report["errors"].append(f"{spec['id']}: constant on train"); continue
            try:
                stats = fit_scale(xt, spec.get("scale", "z"))
            except ValueError as e:
                report["errors"].append(f"{spec['id']}: {e}"); continue
            scaling[spec["id"]] = {"group": group, "fn": spec["fn"], "args": resolve(spec.get("args"), timeframe), **stats}
            feats[spec["id"]] = apply_scale(x.to_numpy(), stats)

    frame = pd.DataFrame({"timestamp": ts, "close": k["close"],
                          **{f"raw_{c}": k[c] for c in ("open", "high", "low", "volume", "quote_volume", "trades")},
                          **feats})
    frame = frame[split_of != "warmup"].reset_index(drop=True)
    parts = {s: frame[split_of[split_of != "warmup"] == s].reset_index(drop=True) for s in ("train", "val", "test")}

    # ------------------------------------------------------------------ checks
    ids = list(feats)
    strict = prof.get("strict", True)
    issue = (lambda m: report["errors"].append(m)) if strict else (lambda m: report["warnings"].append(m))
    for f in ids:
        st = {}
        for s, p in parts.items():
            v = p[f].to_numpy()
            st[s] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                     "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
                     "clipped_frac": round(float((np.abs(v) >= CLIP).mean()), 5) if scaling[f].get("std") else 0.0}
            if not np.isfinite(v).all():
                report["errors"].append(f"{f}: non-finite values in {s}")
        report["features"][f] = {"group": scaling[f]["group"], **st}
        tr = st["train"]
        for s in ("val", "test"):
            ratio = st[s]["std"] / tr["std"] if tr["std"] else np.inf
            if abs(st[s]["mean"] - tr["mean"]) > max(1.0, 1.0 * tr["std"]) or not (1 / 3 <= ratio <= 3):
                issue(f"{f}: drift in {s} (mean {st[s]['mean']} vs train {tr['mean']}, std ratio {ratio:.2f})")
            if st[s]["clipped_frac"] > 0.01:
                issue(f"{f}: {st[s]['clipped_frac']:.1%} of {s} values clipped at +-{CLIP}")
    rho = parts["train"][ids].rank().corr().to_numpy()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if abs(rho[i, j]) > 0.95:
                issue(f"redundant: {ids[i]} ~ {ids[j]} (spearman {rho[i, j]:+.3f} on train)")
    chunks = len(parts["train"]) // tf["chunk"]
    if chunks < 50:
        issue(f"only {chunks} train chunks of {tf['chunk']} bars; market-type quantiles need >= 50")
    report["train_chunks"] = chunks
    report["rows"] = {s: len(p) for s, p in parts.items()}

    # ------------------------------------------------------------------ write
    os.makedirs(os.path.join(out, "feature_list"), exist_ok=True)
    for s, p in parts.items():
        p.to_feather(os.path.join(out, f"df_{s}.feather"))
    single = [s["id"] for s in prof["single"] if s["id"] in feats]
    trend = [s["id"] for s in prof["trend"] if s["id"] in feats]
    np.save(os.path.join(out, "feature_list", "single_features.npy"), np.array(single, dtype=object))
    np.save(os.path.join(out, "feature_list", "trend_features.npy"), np.array(trend, dtype=object))
    yaml.safe_dump(scaling, open(os.path.join(out, "feature_scaling.yaml"), "w"), sort_keys=False)
    yaml.safe_dump(report, open(os.path.join(out, "build_report.yaml"), "w"), sort_keys=False)
    with open(os.path.join(out, "run.env"), "w") as f:
        f.write(f"# written by profiles/build.py for {name}\n"
                f"MACRO_CHUNK_SIZE={tf['chunk']}\nMACRO_CONTEXT_WINDOW={tf['context_window']}\n"
                f"MACRO_HIGH_ARGS=\"--context_window {tf['context_window']} --memory_capacity {tf['memory_capacity']}\"\n")
    print(f"{name}: rows {report['rows']}, {len(single)} single + {len(trend)} trend features, "
          f"{chunks} train chunks, gaps filled {gaps['filled_bars']} bars ({gaps['gaps']} gaps, longest {gaps['longest_gap_bars']})")
    for w in report["warnings"]:
        print("  warning:", w)
    for e in report["errors"]:
        print("  ERROR:", e)
    if report["errors"]:
        raise SystemExit(f"{name}: {len(report['errors'])} error(s); see {out}/build_report.yaml")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="profile name (profiles/<name>.yaml)")
    ap.add_argument("--timeframe", help="key in profiles/timeframes.yaml (15m, 1h)")
    ap.add_argument("--all", action="store_true", help="every profile x timeframe")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--train", default="2019-01-01:2022-12-31")
    ap.add_argument("--val", default="2023-01-01:2023-12-31")
    ap.add_argument("--test", default="2024-01-01:2024-12-31")
    ap.add_argument("--out", default=os.path.join(MACRO, "data"))
    ap.add_argument("--force", action="store_true", help="rebuild an existing dataset")
    a = ap.parse_args()
    split = (a.train, a.val, a.test)
    if a.all:
        profiles = sorted(f[:-5] for f in os.listdir(HERE) if f.endswith(".yaml") and f != "timeframes.yaml")
        tfs = list(yaml.safe_load(open(os.path.join(HERE, "timeframes.yaml"))))
        failed = []
        for p in profiles:
            for t in tfs:
                try:
                    build(p, t, a.symbol, split, a.out, a.force)
                except SystemExit as e:
                    failed.append(str(e))
        if failed:
            raise SystemExit("\n".join(failed))
    else:
        if not (a.profile and a.timeframe):
            ap.error("--profile and --timeframe, or --all")
        build(a.profile, a.timeframe, a.symbol, split, a.out, a.force)


if __name__ == "__main__":
    main()
