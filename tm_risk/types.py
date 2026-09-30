"""[TradeMaster] Core records of tm_risk: AccountState, Decision, PolicyConfig (yaml-backed, validated)."""
import os
from dataclasses import dataclass, fields

import yaml

POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")
SIZING_ARMS = ("identity", "constant", "vol_target", "kelly")
LEVERAGE_POLICY = (3.0, 10.0)   # perp margin leverage must be strictly inside (user policy, 2026-09-30)


@dataclass
class AccountState:
    capital: float
    equity: float
    contracts: float = 0.0                  # signed
    entry_price: float = 0.0
    stop_price: float = 0.0                 # 0 = no stop
    peak_all: float = 0.0                   # all-time equity peak
    peak_ref: float = 0.0                   # trailing-window reference peak
    halted: bool = False
    bankrupt: bool = False
    open_risk: float = 0.0                  # stressed loss (USD) reserved by the open position
    blocked_direction: int = 0              # after a stop-out: re-entry in this direction is blocked until the signal changes


@dataclass
class Decision:
    k: int
    timestamp: object
    signal: int
    action: str         # enter | exit | flip | hold | abstain | stop | breach | liquidation | none
    wanted: float = 0.0             # notional USD
    preferred: float = 0.0
    permitted: float = 0.0
    contracts: float = 0.0
    binding: str = ""
    vetoes: tuple = ()
    reason: str = ""
    exposure_req: float = 0.0       # f requested by sizing


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    sizing: str                             # identity | constant | vol_target | kelly
    margin_leverage: float = 5.0            # perp: 3 < L < 10 and <= venue.max_leverage; spot: must be 1
    exposure_cap: float = 1.0               # 0 < exposure_cap <= margin_leverage
    constant_exposure: float = 1.0
    target_vol_annual: float = 0.25
    vol_window_bars: int = 48
    kelly_fraction: float = 0.5
    kelly_quantile: float = 0.10
    kelly_bootstrap: int = 2000
    kelly_block: int = 10
    kelly_min_support: int = 30             # per side; fewer closed outcomes -> abstain
    credibility_n0: float = 150.0
    kelly_stress_loss: float = 0.05         # solvency: f * (min(r) - stress) > -1
    kelly_vol_scale: bool = True            # f *= clip(sigma_ref / sigma_now, 0.5, 2)
    ledger_seed: str = "none"               # none | val (val seeding doubles n0; selection-biased)
    stop_atr_mult: float = 3.0              # 0 disables stops
    atr_window: int = 14
    slippage_bps: float = 5.0               # execution cost stress on every fill (bps of notional)
    trade_loss_budget: float = 0.02         # fraction of E: stressed loss per new position
    aggregate_loss_budget: float = 0.05
    dd_halt: float = 0.35                   # halt new entries when E < (1 - dd_halt) * peak_ref
    dd_window_days: float = 90.0            # trailing reference-peak window; 0 = all-time
    participation: float = 0.01             # new notional <= participation * bar quote volume; 0 disables
    episode_max_bars: int = 168
    episode_loss_budget: float = 0.05       # realised-loss allowance per risk episode (fraction of E at episode start)
    breach_band: float = 0.25               # deleverage when exposure > exposure_cap * (1 + band)
    liq_distance_factor: float = 0.5        # stop distance must be < factor * liquidation distance (a cap)
    checks: bool = True                     # False (identity): no stops/admission/halts/breach

    def validate(self, venue=None) -> "PolicyConfig":
        """Raise ValueError on an invalid policy; with `venue`, also check its kind and max leverage (default: perp)."""
        err = lambda m: ValueError(f"policy '{self.name}': {m}")
        if self.sizing not in SIZING_ARMS:
            raise err(f"sizing must be one of {SIZING_ARMS}, got {self.sizing!r}")
        if self.ledger_seed not in ("none", "val"):
            raise err(f"ledger_seed must be 'none' or 'val', got {self.ledger_seed!r}")
        spot = venue is not None and venue.kind == "spot"
        L, lo, hi = self.margin_leverage, LEVERAGE_POLICY[0], LEVERAGE_POLICY[1]
        if spot:
            if L != 1:
                raise err(f"spot venue requires margin_leverage 1, got {L:g}")
        else:
            if not lo < L < hi:
                raise err(f"margin_leverage {L:g} violates policy {lo:g} < L < {hi:g}")
            if venue is not None and L > venue.max_leverage:
                raise err(f"margin_leverage {L:g} exceeds venue {venue.name!r} max {venue.max_leverage:g}")
        if not 0 < self.exposure_cap <= L:
            raise err(f"exposure_cap {self.exposure_cap:g} must satisfy 0 < cap <= margin_leverage {L:g}")
        pos = ("constant_exposure", "target_vol_annual", "vol_window_bars", "kelly_bootstrap", "kelly_block",
               "atr_window", "episode_max_bars", "credibility_n0")
        for f in pos:
            if not getattr(self, f) > 0:
                raise err(f"{f} must be > 0, got {getattr(self, f)}")
        nonneg = ("stop_atr_mult", "slippage_bps", "trade_loss_budget", "aggregate_loss_budget", "dd_window_days",
                  "participation", "episode_loss_budget", "breach_band", "kelly_stress_loss", "kelly_min_support")
        for f in nonneg:
            if getattr(self, f) < 0:
                raise err(f"{f} must be >= 0, got {getattr(self, f)}")
        if not 0 < self.dd_halt <= 1:
            raise err(f"dd_halt must be in (0, 1], got {self.dd_halt}")
        if not 0 < self.kelly_fraction <= 1:
            raise err(f"kelly_fraction must be in (0, 1], got {self.kelly_fraction}")
        if not 0 < self.kelly_quantile < 1:
            raise err(f"kelly_quantile must be in (0, 1), got {self.kelly_quantile}")
        if not 0 < self.liq_distance_factor <= 1:
            raise err(f"liq_distance_factor must be in (0, 1], got {self.liq_distance_factor}")
        return self


def load_policy(name_or_path) -> PolicyConfig:
    """`"default_35dd"` -> policies/default_35dd.yaml, or a path to a YAML file."""
    s = str(name_or_path)
    path = s if s.endswith((".yaml", ".yml")) or os.sep in s else os.path.join(POLICY_DIR, s + ".yaml")
    if not os.path.isfile(path):
        avail = sorted(f[:-5] for f in os.listdir(POLICY_DIR) if f.endswith(".yaml"))
        raise ValueError(f"unknown policy {s!r}; available: {avail}")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    known = {f.name for f in fields(PolicyConfig)}
    extra = sorted(set(d) - known)
    if extra:
        raise ValueError(f"policy file {path}: unknown fields {extra}")
    d.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    try:
        return PolicyConfig(**d).validate()
    except TypeError as e:
        raise ValueError(f"policy file {path}: {e}") from e
