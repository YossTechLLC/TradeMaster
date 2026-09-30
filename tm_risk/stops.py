"""[TradeMaster] Causal volatility (bar_vol, ATR), ATR stop distance and stressed loss per unit notional."""
import numpy as np


def bar_vol(stream, k, window) -> float:
    """Sample std (ddof=1) of log close returns over bars (k-window, k]; uses only bars <= k. 0.0 with < 2 returns."""
    lo = max(1, k - int(window) + 1)
    if k < lo:
        return 0.0
    c = np.asarray(stream.close[lo - 1:k + 1], dtype=float)
    r = np.diff(np.log(c))
    return float(np.std(r, ddof=1)) if len(r) >= 2 else 0.0


def atr(stream, k, window) -> float:
    """Simple mean of the true range over the last `window` bars <= k (fewer at the start of the series), in price units.
    TR_j = max(h-l, |h-c[j-1]|, |l-c[j-1]|); TR_0 = h-l."""
    lo = max(0, k - int(window) + 1)
    h = np.asarray(stream.high[lo:k + 1], dtype=float)
    l = np.asarray(stream.low[lo:k + 1], dtype=float)
    tr = h - l
    if lo >= 1:
        pc = np.asarray(stream.close[lo - 1:k], dtype=float)
        tr = np.maximum(tr, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    elif k >= 1:
        pc = np.asarray(stream.close[0:k], dtype=float)
        tr[1:] = np.maximum(tr[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    return float(tr.mean())


def stop_distance(stream, k, policy) -> float:
    """stop_atr_mult * ATR(k) / close[k] as a fraction of price; 0 when stops are disabled."""
    if policy.stop_atr_mult <= 0:
        return 0.0
    return float(policy.stop_atr_mult * atr(stream, k, policy.atr_window) / float(stream.close[k]))


def stressed_loss_per_unit(stream, k, venue, policy) -> float:
    """Stressed loss per unit notional: stop distance (kelly_stress_loss if stops are disabled) + 2 fee rates + 2 slippage."""
    p = float(stream.close[k])
    d = stop_distance(stream, k, policy) if policy.stop_atr_mult > 0 else policy.kelly_stress_loss
    return d + 2 * venue.fees.rate(p, venue.contract_size) + 2 * policy.slippage_bps * 1e-4
