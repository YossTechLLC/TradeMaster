"""[TradeMaster] Robust fractional Kelly: bootstrap-quantile growth-optimal exposure with a solvency domain."""
from dataclasses import dataclass

import numpy as np

from tm_risk.credibility import shrink

GRID = 201


@dataclass
class KellyResult:
    f: float
    growth_q: float
    f_raw: float
    abstain: bool
    reason: str = ""


def _abstain(reason, f_raw=0.0, g=0.0):
    return KellyResult(0.0, g, f_raw, True, reason)


def robust_kelly(r, hours, policy, rng=None) -> KellyResult:
    """Spec section 5. r = per-episode returns at unit notional, hours = episode durations."""
    r, hours = np.asarray(r, dtype=float), np.asarray(hours, dtype=float)
    n = len(r)
    if n < policy.kelly_min_support:
        return _abstain(f"support {n} < {policy.kelly_min_support}")
    rng = rng if rng is not None else np.random.default_rng(12345)
    r = shrink(r, policy.credibility_n0)
    grid = np.linspace(0.0, policy.exposure_cap, GRID)
    grid = grid[1.0 + grid * (r.min() - policy.kelly_stress_loss) > 0]      # solvency domain (f=0 always inside)
    if len(grid) < 2:
        return _abstain("solvency domain empty")
    # circular block bootstrap: resample ceil(n/block) block starts; block sums are precomputed per f
    b = max(1, min(int(policy.kelly_block), n))
    nb = -(-n // b)
    starts = rng.integers(0, n, size=(int(policy.kelly_bootstrap), nb))
    ext = np.concatenate([hours, hours[:b - 1]])
    cs = np.concatenate([[0.0], np.cumsum(ext)])
    hsum = (cs[b:b + n] - cs[:n])[starts].sum(axis=1)                        # (B,) total hours per resample
    rext = np.concatenate([r, r[:b - 1]])
    G = np.empty(len(grid))
    for i, f in enumerate(grid):
        cs = np.concatenate([[0.0], np.cumsum(np.log1p(f * rext))])
        G[i] = np.quantile((cs[b:b + n] - cs[:n])[starts].sum(axis=1) / hsum, policy.kelly_quantile)
    j = int(np.argmax(G))
    if G[j] <= 0:
        return _abstain("no positive robust growth", float(grid[j]), float(G[j]))
    return KellyResult(float(policy.kelly_fraction * grid[j]), float(G[j]), float(grid[j]), False)
