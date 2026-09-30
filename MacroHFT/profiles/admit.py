"""[TradeMaster] Admission test for a profile dataset (docs/macrohft_inputs.md 5.8), on the TRAIN split only.

    python profiles/admit.py --dataset BTCUSDT_15m_xt10 [--shifts 50]

For every feature: Spearman IC with forward log returns over 1, 4 and 16 bars and with forward realised
volatility (std of the next 16 one-bar returns), all measured from the bar's close, where the agent trades.
Null distribution: the same IC after circularly shifting the feature by a random offset of at least 1 day
(keeps the feature's own autocorrelation, unlike a plain shuffle, so slow features cannot pass on persistence
alone). A feature is admitted if |IC| exceeds the 95th percentile of the null |IC| for at least one target.
Writes data/<dataset>/admission.yaml and prints a table. Informational: it does not change the dataset.
"""
import argparse, os
import numpy as np
import pandas as pd
import yaml

HORIZONS = (1, 4, 16)


def rank(x):
    return pd.Series(x).rank().to_numpy()


def ic(a_rank, b_rank):
    ok = ~(np.isnan(a_rank) | np.isnan(b_rank))
    return float(np.corrcoef(a_rank[ok], b_rank[ok])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--shifts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = os.path.join("./data", a.dataset)
    df = pd.read_feather(os.path.join(root, "df_train.feather"))
    feats = (np.load(os.path.join(root, "feature_list", "single_features.npy"), allow_pickle=True).tolist()
             + np.load(os.path.join(root, "feature_list", "trend_features.npy"), allow_pickle=True).tolist())
    lc = np.log(df["close"].to_numpy(np.float64))
    r = np.diff(lc, prepend=np.nan)
    targets = {f"ret_{h}": np.concatenate([lc[h:] - lc[:-h], np.full(h, np.nan)]) for h in HORIZONS}
    fwd_r = pd.Series(r).shift(-1)
    targets["vol_16"] = fwd_r[::-1].rolling(16).std()[::-1].to_numpy()  # std of r_{t+1..t+16}
    targets["vol_16"][-16:] = np.nan
    t_rank = {k: rank(v) for k, v in targets.items()}

    step = pd.Timestamp(df.timestamp.iloc[1]) - pd.Timestamp(df.timestamp.iloc[0])
    min_shift = max(1, int(pd.Timedelta(days=1) / step))
    rng = np.random.default_rng(a.seed)
    shifts = rng.integers(min_shift, len(df) - min_shift, size=a.shifts)

    out, rows = {}, []
    for f in feats:
        fr = rank(df[f].to_numpy())
        res = {"admitted": False}
        for k, tr in t_rank.items():
            real = ic(fr, tr)
            null = np.array([abs(ic(np.roll(fr, s), tr)) for s in shifts])
            p95 = float(np.percentile(null, 95))
            res[k] = {"ic": round(real, 4), "null_p95": round(p95, 4), "pass": bool(abs(real) > p95)}
            res["admitted"] |= abs(real) > p95
        out[f] = res
        rows.append([f] + [f"{res[k]['ic']:+.3f}{'*' if res[k]['pass'] else ' '}" for k in t_rank] + ["yes" if res["admitted"] else "NO"])
    yaml.safe_dump({"dataset": a.dataset, "split": "train", "null": f"{a.shifts} circular shifts >= 1 day",
                    "features": out}, open(os.path.join(root, "admission.yaml"), "w"), sort_keys=False)
    w = max(len(f) for f in feats)
    print(f"{a.dataset} (train, {len(df)} bars)  IC vs forward target; * = beats the 95% shifted-null")
    print(f"{'feature':<{w}}  " + "  ".join(f"{k:>8}" for k in t_rank) + "  admitted")
    for row in rows:
        print(f"{row[0]:<{w}}  " + "  ".join(f"{c:>8}" for c in row[1:-1]) + f"  {row[-1]}")
    n = sum(v["admitted"] for v in out.values())
    print(f"admitted {n}/{len(feats)}  ->  {root}/admission.yaml")


if __name__ == "__main__":
    main()
