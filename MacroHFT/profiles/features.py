"""[TradeMaster] Feature library for MacroHFT input profiles (see docs/macrohft_inputs.md).

Every feature is a function of the raw kline frame (columns: timestamp, open, high, low, close, volume,
quote_volume, trades, taker_buy_base, taker_buy_quote) and returns a float64 Series aligned with it.

Rules every function follows (checked by box/bench/tests/test_profiles.py):
- causal: row t uses only bars <= t (values never change when later bars are appended);
- computed in float64 from raw prices; divisions guarded so no NaN/inf appears after the warm-up.
Train-fitted features (the seasonal profile) receive the train mask and fit on those rows only.
"""
import numpy as np
import pandas as pd

EPS = 1e-12


# ----------------------------------------------------------------------------- helpers
def _log_close(k):
    return np.log(k["close"].astype(np.float64))


def log_ret(k):
    """r_t = ln C_t - ln C_{t-1}"""
    return _log_close(k).diff()


def _safe_div(a, b, fill):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    out = np.full_like(a, fill, dtype=np.float64)
    ok = np.abs(b) > EPS
    out[ok] = a[ok] / b[ok]
    return out


def ewma_sigma(k, half_life=89, window=987):
    """Prior-bar EWMA RMS of r: weights 0.5^(lag/half_life) over lags 1..window (r_t itself excluded)."""
    r2 = log_ret(k).to_numpy() ** 2
    w = 0.5 ** (np.arange(1, window + 1) / half_life)
    w = w / w.sum()
    out = np.full(len(r2), np.nan)
    # sigma_t^2 = sum_{j=1..window} w_j r_{t-j}^2 ; valid once r_{t-window} exists (t - window >= 1)
    r2f = np.nan_to_num(r2)
    conv = np.convolve(r2f, w, mode="full")[: len(r2)]  # conv[t] = sum_j w_{j} r2[t-(j-1)] ... shift by one below
    out[1:] = conv[:-1]
    out[: window + 1] = np.nan
    return pd.Series(np.sqrt(out), index=k.index)


# ----------------------------------------------------------------------------- upstream (P0) features
def volume(k):
    return k["volume"].astype(np.float64)


def buy_volume(k):
    return k["taker_buy_base"].astype(np.float64)


def sell_volume(k):
    return (k["volume"] - k["taker_buy_base"]).astype(np.float64)


def volume_imbalance(k):
    tb, v = k["taker_buy_base"].to_numpy(np.float64), k["volume"].to_numpy(np.float64)
    return pd.Series(_safe_div(tb - (v - tb), v, 0.0), index=k.index)


def buy_vwap(k, relative=False):
    """VWAP of taker buys (quote/base); carries the close on bars without taker buys."""
    p = _safe_div(k["taker_buy_quote"], k["taker_buy_base"], np.nan)
    p = np.where(np.isnan(p), k["close"].to_numpy(np.float64), p)
    return pd.Series(p / k["close"].to_numpy(np.float64) - 1 if relative else p, index=k.index)


def sell_vwap(k, relative=False):
    p = _safe_div(k["quote_volume"] - k["taker_buy_quote"], k["volume"] - k["taker_buy_base"], np.nan)
    p = np.where(np.isnan(p), k["close"].to_numpy(np.float64), p)
    return pd.Series(p / k["close"].to_numpy(np.float64) - 1 if relative else p, index=k.index)


def candle(k, kind):
    """Alpha158 k-features, as in the upstream data (MacroHFT feature list #28-36)."""
    o, h, l, c = (k[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    rng = h - l
    top, bot = np.maximum(o, c), np.minimum(o, c)
    f = {
        "kmid": (c - o) / o, "klen": rng / o, "kup": (h - top) / o, "klow": (bot - l) / o,
        "ksft": (2 * c - h - l) / o,
        "kmid2": _safe_div(c - o, rng, 0.0), "kup2": _safe_div(h - top, rng, 0.0),
        "klow2": _safe_div(bot - l, rng, 0.0), "ksft2": _safe_div(2 * c - h - l, rng, 0.0),
    }[kind]
    return pd.Series(f, index=k.index)


def trend_diff(k, base, w, **base_args):
    """Upstream '<x>_trend_<w>': (x_t - x_{t-w}) / w, the mean per-bar change over w bars."""
    x = FEATURES[base](k, **base_args)
    return (x - x.shift(w)) / w


# ----------------------------------------------------------------------------- kline18 (P1) features
def log_quote_volume(k):
    return np.log1p(k["quote_volume"].astype(np.float64))


def log_trade_count(k):
    return np.log1p(k["trades"].astype(np.float64))


def log_avg_trade_size(k):
    """ln(quote volume / trades): USDT per trade; bars without trades carry the last value."""
    ats = pd.Series(_safe_div(k["quote_volume"], k["trades"], np.nan), index=k.index)
    return np.log(ats.where(ats > 0)).ffill()


def vwap_spread(k):
    """(taker-buy VWAP - taker-sell VWAP) / close: how much higher aggressive buys paid than aggressive sells."""
    return buy_vwap(k, relative=True) - sell_vwap(k, relative=True)


def window_return(k, w):
    lc = _log_close(k)
    return lc - lc.shift(w)


def realized_vol(k, w):
    return log_ret(k).rolling(w).std()


def volume_surprise(k, w):
    """log quote volume minus its mean over the prior w bars."""
    lqv = log_quote_volume(k)
    return lqv - lqv.shift(1).rolling(w).mean()


def flow(k, w):
    """mean volume imbalance over the last w bars (incl. t)."""
    return volume_imbalance(k).rolling(w).mean()


# ----------------------------------------------------------------------------- XT10 (P2) features
def ret_sigma(k, a, b, half_life=89, window=987):
    """ln(C_{t-a} / C_{t-b}) / (sigma_t * sqrt(b - a))"""
    lc = _log_close(k)
    return (lc.shift(a) - lc.shift(b)) / (ewma_sigma(k, half_life, window) * np.sqrt(b - a))


def range_over_mean_range(k, w):
    """(H - L) / mean(H - L) over the last w bars (incl. t); 1 when the day had no range."""
    rng = (k["high"] - k["low"]).astype(np.float64)
    m = rng.rolling(w).mean()
    return pd.Series(np.where(m.isna(), np.nan, _safe_div(rng, m, 1.0)), index=k.index)


def rv_over_sigma(k, w, half_life=89, window=987):
    """1/2 ln( mean(r^2 over the last w bars, incl. t) / sigma_t^2 ); floored at (1e-3 sigma)^2."""
    s2 = ewma_sigma(k, half_life, window) ** 2
    m = (log_ret(k) ** 2).rolling(w).mean()
    return 0.5 * np.log(np.maximum(m, 1e-6 * s2) / s2)


def efficiency(k, w):
    """|ln C_t - ln C_{t-w}| / sum of |r| over the last w bars; 0 when the path did not move."""
    lc = _log_close(k)
    num = (lc - lc.shift(w)).abs()
    den = log_ret(k).abs().rolling(w).sum()
    return pd.Series(np.where(den.isna(), np.nan, _safe_div(num, den, 0.0)), index=k.index)


def trade_size_rel(k, w):
    """ln(V_t/N_t) minus its mean over the prior w bars; bars without trades carry the last value."""
    ts = pd.Series(_safe_div(k["volume"], k["trades"], np.nan), index=k.index)
    lts = np.log(ts.where(ts > 0)).ffill()
    return lts - lts.shift(1).rolling(w).mean()


def mid_position(k):
    """((O + C)/2 - L) / (H - L); 0.5 when H == L."""
    o, h, l, c = (k[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
    return pd.Series(_safe_div((o + c) / 2 - l, h - l, 0.5), index=k.index)


def seasonal_activity_ahead(k, ahead, train_mask):
    """Mean of the train-fitted hour-of-week log-volume profile over the slots of the next `ahead` bars.
    Uses only the calendar of future bars (known in advance), never their data."""
    ts = pd.DatetimeIndex(k["timestamp"])
    how = (ts.dayofweek * 24 + ts.hour).to_numpy()
    lv = np.log1p(k["volume"].to_numpy(np.float64))
    prof = pd.Series(lv[train_mask]).groupby(how[train_mask]).mean().reindex(range(168))
    if prof.isna().any():
        raise ValueError("train split does not cover every hour of the week")
    step = ts[1] - ts[0]
    out = np.zeros(len(k))
    for j in range(1, ahead + 1):
        fut = ts + j * step
        out += prof.to_numpy()[(fut.dayofweek * 24 + fut.hour).to_numpy()]
    return pd.Series(out / ahead, index=k.index)


FEATURES = {f.__name__: f for f in (
    volume, buy_volume, sell_volume, volume_imbalance, buy_vwap, sell_vwap, candle, trend_diff, log_ret,
    log_quote_volume, log_trade_count, log_avg_trade_size, vwap_spread, window_return, realized_vol, volume_surprise, flow,
    ret_sigma, range_over_mean_range, rv_over_sigma, efficiency, trade_size_rel, mid_position,
    seasonal_activity_ahead,
)}
TRAIN_FITTED = {"seasonal_activity_ahead"}
