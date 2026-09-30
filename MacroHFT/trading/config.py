"""[TradeMaster] TradingConfig: mode, venue, leverage policy and CLI flags for the MacroHFT trading layer."""
from dataclasses import dataclass
from typing import Optional

from MacroHFT.trading.venue import Venue, load_venue

LEVERAGE_POLICY = (3.0, 10.0)   # perp leverage must be strictly inside (user policy, 2026-09-30)
MODES = ("legacy", "long_only", "long_short")


@dataclass(frozen=True)
class TradingConfig:
    mode: str = "legacy"
    venue: Optional[Venue] = None
    leverage: float = 5.0
    capital: float = 10_000.0           # USD, initial equity
    turnover_penalty_bps: float = 0.0   # training-only shaping, bps of traded notional
    reward_scale: float = 100.0
    sizing: str = "equity"              # "equity" or "fixed"
    fixed_units: float = 0.01

    def validate(self) -> "TradingConfig":
        if self.mode not in MODES:
            raise ValueError(f"unknown trade mode {self.mode!r}, expected one of {MODES}")
        if self.legacy:
            return self
        v = self.venue
        if v is None:
            raise ValueError(f"trade mode {self.mode!r} requires a venue")
        if self.mode == "long_short" and not v.allows_short:
            raise ValueError(f"venue {v.name!r} does not allow shorting; long_short is not available")
        lo, hi = LEVERAGE_POLICY
        if v.kind == "perp":
            if not lo < self.leverage < hi:
                raise ValueError(f"leverage {self.leverage:g} violates policy {lo:g} < L < {hi:g}")
            if self.leverage > v.max_leverage:
                raise ValueError(f"leverage {self.leverage:g} exceeds venue {v.name!r} max {v.max_leverage:g}")
        else:
            if self.leverage != 1:
                raise ValueError(f"spot venue {v.name!r} requires leverage 1, got {self.leverage:g}")
            if self.mode != "long_only":
                raise ValueError(f"spot venue {v.name!r} supports long_only only, got {self.mode!r}")
        if not self.capital > 0:
            raise ValueError(f"capital must be > 0, got {self.capital}")
        if not self.turnover_penalty_bps >= 0:
            raise ValueError(f"turnover_penalty_bps must be >= 0, got {self.turnover_penalty_bps}")
        if not self.reward_scale > 0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")
        if self.sizing not in ("equity", "fixed"):
            raise ValueError(f"sizing must be 'equity' or 'fixed', got {self.sizing!r}")
        if not self.fixed_units > 0:
            raise ValueError(f"fixed_units must be > 0, got {self.fixed_units}")
        return self

    @property
    def legacy(self) -> bool:
        return self.mode == "legacy"

    @property
    def n_action(self) -> int:
        return 3 if self.mode == "long_short" else 2

    @property
    def directions(self) -> tuple:
        return (-1.0, 0.0, 1.0) if self.mode == "long_short" else (0.0, 1.0)

    @property
    def flat_action(self) -> int:
        return 1 if self.mode == "long_short" else 0

    @property
    def tag(self) -> str:
        if self.legacy:
            return ""
        tag = f"{self.mode}-{self.venue.name}-L{self.leverage:g}-tp{self.turnover_penalty_bps:g}"
        if self.capital != 10_000.0:
            tag += f"-c{self.capital:g}"
        if self.reward_scale != 100.0:
            tag += f"-rs{self.reward_scale:g}"
        return tag


def add_trading_args(parser):
    parser.add_argument("--trade_mode", choices=list(MODES), default="legacy",
                        help="legacy: unchanged long-only env; long_only / long_short: venue-aware env")
    parser.add_argument("--venue", type=str, default="kraken_us_perp", help="venue name or YAML path")
    parser.add_argument("--leverage", type=float, default=None,
                        help="perp leverage, strictly inside (3, 10); default 5 on perps, 1 on spot")
    parser.add_argument("--capital", type=float, default=10000.0, help="initial equity in USD")
    parser.add_argument("--turnover_penalty_bps", type=float, default=0.0,
                        help="training-only penalty, bps of traded notional")
    parser.add_argument("--reward_scale", type=float, default=100.0)
    return parser


def config_from_args(args) -> TradingConfig:
    if args.trade_mode == "legacy":
        return TradingConfig()
    venue = load_venue(args.venue)
    leverage = args.leverage if args.leverage is not None else (5.0 if venue.kind == "perp" else 1.0)
    return TradingConfig(mode=args.trade_mode, venue=venue, leverage=leverage,
                         capital=args.capital, turnover_penalty_bps=args.turnover_penalty_bps,
                         reward_scale=args.reward_scale).validate()
