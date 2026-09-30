"""[TradeMaster] size_request: requested exposure f for a new position, per sizing arm (identity, constant, vol_target, kelly)."""
import numpy as np

from tm_risk.kelly import robust_kelly


def size_request(k, side, state, stream, ledger, policy):
    """Returns (f, info). f = requested notional/equity for a new position in direction `side`, clipped to
    [0, exposure_cap] for every arm except identity."""
    from tm_risk.stops import bar_vol
    arm, info = policy.sizing, {"arm": policy.sizing}
    if arm == "identity":
        return float(policy.margin_leverage), info
    if arm == "constant":
        f = policy.constant_exposure
    elif arm == "vol_target":
        v = bar_vol(stream, k, policy.vol_window_bars)
        info["bar_vol"] = float(v)
        ok = np.isfinite(v) and v > 0
        f = policy.target_vol_annual / (v * np.sqrt(8760.0 / stream.bar_hours)) if ok else 0.0
        if not ok:
            info["reason"] = "vol unavailable"
    elif arm == "kelly":
        sd = 1 if side > 0 else -1
        key = ("kelly", sd, ledger.n(sd))               # outcomes of a side change only when that side gains an episode
        res = ledger.cache.get(key)
        if res is None:
            for old in [c for c in ledger.cache if c[:2] == key[:2]]:
                del ledger.cache[old]
            r, h, sg = ledger.outcomes(side)
            res = ledger.cache[key] = (robust_kelly(r, h, policy), sg)
        kr, sg = res
        f = kr.f
        info.update(f_raw=kr.f_raw, growth_q=kr.growth_q, abstain=kr.abstain, reason=kr.reason)
        if not kr.abstain and policy.kelly_vol_scale:
            v, ref = bar_vol(stream, k, policy.vol_window_bars), np.nanmedian(sg) if len(sg) else np.nan
            if np.isfinite(v) and v > 0 and np.isfinite(ref) and ref > 0:
                info["vol_scale"] = float(np.clip(ref / v, 0.5, 2.0))
                f *= info["vol_scale"]
    else:
        raise ValueError(f"unknown sizing arm {arm!r}")
    return float(np.clip(f, 0.0, policy.exposure_cap)), info
