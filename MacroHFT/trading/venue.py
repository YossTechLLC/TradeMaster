"""[TradeMaster] Venue model: fees, funding, margin and shorting rules, loaded from venues/*.yaml."""
import os
from dataclasses import dataclass
from typing import Optional

import yaml

VENUE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venues")
FEE_MODELS = ("pct", "per_contract")


@dataclass(frozen=True)
class FeeModel:
    model: str                      # "pct" (fraction of notional) or "per_contract" (USD per contract per side)
    taker: float
    maker: float = 0.0
    min_per_contract: float = 0.0   # pct model only: USD minimum per contract per side

    def cost(self, contracts_traded: float, price: float, contract_size: float) -> float:
        """USD taker fee for trading |contracts_traded| contracts at `price`."""
        n = abs(contracts_traded)
        if self.model == "per_contract":
            return self.taker * n
        return max(self.taker * n * contract_size * price, self.min_per_contract * n)

    def rate(self, price: float, contract_size: float) -> float:
        """Fee per unit of notional at `price` (used by the teacher)."""
        return self.cost(1.0, price, contract_size) / (contract_size * price)


@dataclass(frozen=True)
class FundingModel:
    interval_hours: float
    rate_per_interval: float        # fraction of notional; > 0 means longs pay shorts

    @property
    def rate_per_hour(self) -> float:
        return self.rate_per_interval / self.interval_hours


@dataclass(frozen=True)
class Venue:
    name: str
    kind: str                       # "spot" | "perp"
    description: str
    contract_size: float            # BTC per contract (spot: minimum lot)
    fees: FeeModel
    funding: Optional[FundingModel]
    max_leverage: float
    maintenance_margin_rate: float  # fraction of notional
    liquidation_fee_usd: float
    allows_short: bool
    as_of: str
    sources: list
    unverified: list


def list_venues():
    return sorted(f[:-5] for f in os.listdir(VENUE_DIR) if f.endswith(".yaml"))


def _validate(v: Venue) -> Venue:
    err = lambda m: ValueError(f"venue '{v.name}': {m}")
    if v.kind not in ("spot", "perp"):
        raise err(f"kind must be 'spot' or 'perp', got {v.kind!r}")
    if v.fees.model not in FEE_MODELS:
        raise err(f"unknown fee model {v.fees.model!r}, expected one of {FEE_MODELS}")
    rates = dict(taker=v.fees.taker, maker=v.fees.maker, min_per_contract=v.fees.min_per_contract,
                 maintenance_margin_rate=v.maintenance_margin_rate, liquidation_fee_usd=v.liquidation_fee_usd)
    if v.funding is not None:
        rates["funding.interval_hours"] = v.funding.interval_hours
    for k, x in rates.items():
        if x < 0:
            raise err(f"{k} must be >= 0, got {x}")
    if v.funding is not None and v.funding.interval_hours <= 0:
        raise err("funding.interval_hours must be > 0")
    if v.contract_size <= 0:
        raise err("contract_size must be > 0")
    if v.kind == "perp" and v.funding is None:
        raise err("perp venue requires a funding model")
    if v.kind == "spot" and (v.max_leverage != 1 or v.allows_short):
        raise err("spot venue requires max_leverage 1 and allows_short false")
    return v


def load_venue(name_or_path: str) -> Venue:
    """`"kraken_us_perp"` -> venues/kraken_us_perp.yaml, or a path to a YAML file."""
    if name_or_path.endswith((".yaml", ".yml")) or os.sep in name_or_path:
        path = name_or_path
    else:
        path = os.path.join(VENUE_DIR, name_or_path + ".yaml")
    if not os.path.isfile(path):
        raise ValueError(f"unknown venue {name_or_path!r}; available: {list_venues()}")
    with open(path) as f:
        d = yaml.safe_load(f)
    try:
        return _build(d)
    except (TypeError, KeyError, ValueError) as e:
        if isinstance(e, ValueError) and str(e).startswith("venue '"):
            raise
        raise ValueError(f"venue file {path}: {type(e).__name__}: {e}") from e


def _build(d) -> Venue:
    fund = d.get("funding")
    return _validate(Venue(
        name=d["name"], kind=d["kind"], description=d.get("description", ""),
        contract_size=float(d["contract_size"]), fees=FeeModel(**d["fees"]),
        funding=FundingModel(**fund) if fund else None,
        max_leverage=float(d["max_leverage"]),
        maintenance_margin_rate=float(d["maintenance_margin_rate"]),
        liquidation_fee_usd=float(d["liquidation_fee_usd"]),
        allows_short=bool(d["allows_short"]), as_of=str(d["as_of"]),
        sources=list(d.get("sources") or []), unverified=list(d.get("unverified") or [])))
