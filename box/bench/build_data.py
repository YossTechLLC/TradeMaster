"""Build benchmark datasets from real Binance BTCUSDT spot klines (data.binance.vision).

Prices, candles, volume and taker-buy flow are real. Binance's free archive has no
order-book history, so the 5-level book (prices/sizes) is synthesised around the real
price; timing depends only on shapes/lengths, not on the values.

usage: build_data.py macro <interval> <YYYY-MM-DD start> <n_train_days> <n_val_days> <n_test_days> [dataset=BTCUSDT]
       build_data.py earn  <YYYY-MM-DD start> <n_days>
       build_data.py count          # bar counts per timeframe (for ETA tables)
"""
import io, os, sys, zipfile, urllib.request, datetime as dt
import numpy as np, pandas as pd

TM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
RAW = os.environ.get("TM_BENCH_RAW", os.path.join(TM, "box", "bench", "raw"))
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
rng = np.random.default_rng(12345)


def fetch(interval, day):
    """Daily file for 1s, monthly for everything else (cached)."""
    if interval == "1s":
        name = f"BTCUSDT-1s-{day:%Y-%m-%d}.zip"; url = f"spot/daily/klines/BTCUSDT/1s/{name}"
    else:
        name = f"BTCUSDT-{interval}-{day:%Y-%m}.zip"; url = f"spot/monthly/klines/BTCUSDT/{interval}/{name}"
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        urllib.request.urlretrieve(f"https://data.binance.vision/data/{url}", path)
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)


def klines(interval, start, days):
    start = dt.date.fromisoformat(start); end = start + dt.timedelta(days=days)
    if interval == "1s":
        parts = [fetch("1s", start + dt.timedelta(d)) for d in range(days)]
    else:
        months = sorted({(start + dt.timedelta(d)).replace(day=1) for d in range(days)})
        parts = [fetch(interval, m) for m in months]
    df = pd.concat(parts, ignore_index=True)
    t = df.open_time.astype("int64")  # Binance spot archives switched from ms to microseconds on 2025-01-01
    df["timestamp"] = pd.to_datetime(np.where(t > 10**14, t // 1000, t), unit="ms")
    df = df[(df.timestamp >= pd.Timestamp(start)) & (df.timestamp < pd.Timestamp(end))]
    return df.reset_index(drop=True)


def features(k):
    o, h, l, c, v = (k[x].astype(float) for x in ("open", "high", "low", "close", "volume"))
    tb = k.taker_buy_base.astype(float)
    df = pd.DataFrame({"timestamp": k.timestamp, "open": o, "high": h, "low": l, "close": c, "volume": v})
    n = len(df); eps = 1e-12
    # --- synthetic 5-level book around the real close
    half = np.maximum(0.005, c * 0.5e-5 * rng.lognormal(0, 0.3, n))
    for i in range(1, 6):
        df[f"bid{i}_price"] = (c - half - (i - 1) * 0.01).round(2)
        df[f"ask{i}_price"] = (c + half + (i - 1) * 0.01).round(2)
        df[f"bid{i}_size"] = rng.lognormal(-1.0, 1.0, n) * (1 + i / 2)
        df[f"ask{i}_size"] = rng.lognormal(-1.0, 1.0, n) * (1 + i / 2)
    tot_b = sum(df[f"bid{i}_size"] for i in range(1, 6)); tot_a = sum(df[f"ask{i}_size"] for i in range(1, 6))
    for i in range(1, 6):
        df[f"bid{i}_size_n"] = df[f"bid{i}_size"] / tot_b
        df[f"ask{i}_size_n"] = df[f"ask{i}_size"] / tot_a
    b1, a1, bs1, as1 = df.bid1_price, df.ask1_price, df.bid1_size, df.ask1_size
    b2, a2, bs2, as2 = df.bid2_price, df.ask2_price, df.bid2_size, df.ask2_size
    df["wap_1"] = (b1 * as1 + a1 * bs1) / (bs1 + as1)
    df["wap_2"] = (b2 * as2 + a2 * bs2) / (bs2 + as2)
    df["wap_balance"] = (df.wap_1 - df.wap_2).abs()
    df["buy_spread"] = (b1 - df.bid5_price).abs(); df["sell_spread"] = (df.ask5_price - a1).abs()
    df["price_spread"] = 2 * (a1 - b1) / (a1 + b1)
    # --- real trade flow
    df["buy_volume"] = tb; df["sell_volume"] = v - tb
    df["volume_imbalance"] = (tb - (v - tb)) / (v + eps)
    df["buy_vwap"] = k.taker_buy_quote.astype(float) / (tb + eps)
    df["sell_vwap"] = (k.quote_volume.astype(float) - k.taker_buy_quote.astype(float)) / (v - tb + eps)
    for x in ("bid1_price", "bid2_price", "ask1_price", "ask2_price", "wap_1"):
        df[f"log_return_{x}"] = np.log(df[x]).diff().fillna(0)
    # --- real candle-shape (Alpha158 k*) features
    rngh = (h - l) + eps
    df["kmid"] = (c - o) / o; df["klen"] = (h - l) / o; df["kmid2"] = (c - o) / rngh
    df["kup"] = (h - np.maximum(o, c)) / o; df["kup2"] = (h - np.maximum(o, c)) / rngh
    df["klow"] = (np.minimum(o, c) - l) / o; df["klow2"] = (np.minimum(o, c) - l) / rngh
    df["ksft"] = (2 * c - h - l) / o; df["ksft2"] = (2 * c - h - l) / rngh
    # --- MacroHFT context (trend) features: 60-bar slope proxies
    for x in ("ask1_price", "bid1_price", "buy_spread", "sell_spread", "wap_1", "wap_2", "buy_vwap", "sell_vwap", "volume"):
        df[f"{x}_trend_60"] = (df[x] - df[x].shift(60)) / 60
    # --- extra rolling features so EarnHFT has 54 second-level / 19 minute-level inputs
    for w in (5, 10, 30, 60):
        df[f"roc_{w}"] = c.pct_change(w); df[f"ma_ratio_{w}"] = c / c.rolling(w).mean() - 1
        df[f"std_{w}"] = c.pct_change().rolling(w).std(); df[f"vwap_dev_{w}"] = df.wap_1 / df.wap_1.rolling(w).mean() - 1
    df["imb_ma_10"] = df.volume_imbalance.rolling(10).mean(); df["imb_ma_60"] = df.volume_imbalance.rolling(60).mean()
    for w in (1, 5, 15, 30, 60):
        df[f"min_roc_{w}"] = c.pct_change(60 * w)
    for w in (5, 15, 30, 60):
        df[f"min_std_{w}"] = c.pct_change(60).rolling(60 * w).std()
        df[f"min_ma_ratio_{w}"] = c / c.rolling(60 * w).mean() - 1
        df[f"min_vol_{w}"] = v.rolling(60 * w).sum()
    for w in (5, 15):
        df[f"min_range_{w}"] = h.rolling(60 * w).max() / l.rolling(60 * w).min() - 1
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df["tic"] = "BTCUSDT"; df["symbol"] = "BTCUSDT"
    return df


MACRO_SINGLE = list(np.load(os.path.join(TM, "MacroHFT/data/feature_list/single_features.npy"), allow_pickle=True))
MACRO_TREND = list(np.load(os.path.join(TM, "MacroHFT/data/feature_list/trend_features.npy"), allow_pickle=True))
EARN_SECOND = MACRO_SINGLE + [f"{p}_{w}" for p in ("roc", "ma_ratio", "std", "vwap_dev") for w in (5, 10, 30, 60)] + ["imb_ma_10", "imb_ma_60"]
EARN_MINUTE = [f"min_roc_{w}" for w in (1, 5, 15, 30, 60)] + [f"min_{p}_{w}" for p in ("std", "ma_ratio", "vol") for w in (5, 15, 30, 60)] + ["min_range_5", "min_range_15"]
assert len(EARN_SECOND) == 54 and len(EARN_MINUTE) == 19


def build_macro(interval, start, ntr, nva, nte, name="BTCUSDT"):
    d = features(klines(interval, start, ntr + nva + nte))
    per_day = len(d) / (ntr + nva + nte)
    a, b = int(ntr * per_day), int((ntr + nva) * per_day)
    out = os.path.join(TM, "MacroHFT/data", name); os.makedirs(out, exist_ok=True)
    keep = list(dict.fromkeys(["timestamp", "open", "high", "low", "close", "volume"] + MACRO_SINGLE + MACRO_TREND))
    for name, part in (("df_train", d[:a]), ("df_val", d[a:b]), ("df_test", d[b:])):
        part[keep].reset_index(drop=True).to_feather(os.path.join(out, f"{name}.feather"))
        print(f"{name}: {len(part)} bars ({interval})")


def build_earn(start, ndays):
    d = features(klines("1s", start, ndays))
    alg = os.path.join(TM, "EarnHFT/EarnHFT_Algorithm")
    os.makedirs(os.path.join(alg, "data/BTCUSDT"), exist_ok=True); os.makedirs(os.path.join(alg, "data/feature"), exist_ok=True)
    # per-pair lists (what run.sh selects via EARNHFT_FEATURE_DIR) + the shared copy the upstream baselines read
    for fdir in ("data/feature/BTCUSDT", "data/feature"):
        os.makedirs(os.path.join(alg, fdir), exist_ok=True)
        np.save(os.path.join(alg, fdir, "second_feature.npy"), np.array(EARN_SECOND))
        np.save(os.path.join(alg, fdir, "minitue_feature.npy"), np.array(EARN_MINUTE))
    d.to_feather(os.path.join(alg, "data/BTCUSDT/df.feather"))
    print(f"EarnHFT df.feather: {len(d)} one-second rows, {d.shape[1]} columns")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "macro":
        build_macro(sys.argv[2], sys.argv[3], *map(int, sys.argv[4:7]), *sys.argv[7:8])
    elif cmd == "earn":
        build_earn(sys.argv[2], int(sys.argv[3]))
