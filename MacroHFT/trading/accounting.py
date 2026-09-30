"""[TradeMaster] Account: the single place where position, fees, funding, liquidation and equity are computed."""
import math
from dataclasses import dataclass, fields
from typing import Dict, List

import numpy as np

from MacroHFT.trading.config import TradingConfig
from MacroHFT.trading.penalty import turnover_penalty


@dataclass
class StepResult:
    direction: float
    contracts: float        # signed contracts held during the bar
    traded: float           # |contracts change| at the bar's start
    pnl: float              # USD
    fee: float              # USD
    funding: float          # USD, > 0 = paid
    penalty: float          # USD-equivalent turnover penalty (training shaping, NOT in equity)
    liquidated: bool
    bankrupt: bool
    equity: float
    notional: float


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class Account:
    def __init__(self, cfg: TradingConfig, bar_hours: float):
        self.cfg, self.bar_hours = cfg, bar_hours
        self.venue = cfg.venue
        self.reset()

    def reset(self, direction: float = 0.0, price: float = None) -> None:
        self.equity = self.cfg.capital
        self.contracts = self.size(direction, price) if direction != 0 else 0.0
        self._log: List[StepResult] = []

    def size(self, direction: float, price: float) -> float:
        cs = self.venue.contract_size
        if self.cfg.sizing == "fixed":
            return direction * self.cfg.fixed_units / cs
        if self.equity <= 0:
            return 0.0
        return direction * math.floor(self.cfg.leverage * self.equity / (price * cs))

    def step(self, direction: float, price_prev: float, price_now: float,
             bar_low: float = None, bar_high: float = None) -> StepResult:
        v, cs = self.venue, self.venue.contract_size
        # 1. target: no rebalancing while the direction is held
        if self.contracts != 0 and _sign(direction) == _sign(self.contracts):
            target = self.contracts
        else:
            target = self.size(direction, price_prev) if direction != 0 else 0.0
        # 2. fee and penalty
        traded = abs(target - self.contracts)
        fee = v.fees.cost(traded, price_prev, cs)
        penalty = turnover_penalty(traded * cs * price_prev, self.cfg.turnover_penalty_bps)
        self.contracts = target
        # 3. pnl
        pnl = self.contracts * cs * (price_now - price_prev)
        # 4. funding (perp only)
        funding = 0.0
        if v.funding is not None:
            funding = self.contracts * cs * price_prev * v.funding.rate_per_hour * self.bar_hours
        # 5. equity
        start_equity = self.equity
        self.equity = start_equity + pnl - fee - funding
        # 6. liquidation against the bar's adverse extreme
        liquidated = False
        if v.kind == "perp" and self.contracts != 0:
            long = self.contracts > 0
            extreme = bar_low if long else bar_high
            adverse = price_now if extreme is None else extreme
            eq_adv = start_equity + self.contracts * cs * (adverse - price_prev) - fee - funding
            if eq_adv < v.maintenance_margin_rate * abs(self.contracts) * cs * adverse:
                pnl = self.contracts * cs * (adverse - price_prev)
                fee += v.fees.cost(abs(self.contracts), adverse, cs) + v.liquidation_fee_usd
                self.equity = start_equity + pnl - fee - funding
                self.contracts = 0.0
                liquidated = True
        # 7. bankruptcy
        res = StepResult(direction=direction, contracts=self.contracts, traded=traded, pnl=pnl, fee=fee,
                         funding=funding, penalty=penalty, liquidated=liquidated, bankrupt=self.equity <= 0,
                         equity=self.equity, notional=abs(self.contracts) * cs * price_now)
        self._log.append(res)
        return res

    def log(self) -> Dict[str, np.ndarray]:
        return {f.name: np.array([getattr(r, f.name) for r in self._log]) for f in fields(StepResult)}
