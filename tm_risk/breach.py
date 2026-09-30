"""[TradeMaster] Per-bar marking (stop / liquidation / pnl / funding, worst-case precedence) and deleveraging."""
import math
from dataclasses import dataclass



@dataclass
class BarMark:
    pnl: float = 0.0            # USD price pnl booked this bar (up to the exit price if the position closed)
    fee: float = 0.0            # USD exchange fees (+ liquidation fee)
    slippage: float = 0.0       # USD execution-cost stress
    funding: float = 0.0        # USD, > 0 = paid
    event: str = ""             # "" | "stop" | "liquidation" | "breach"
    side: int = 0               # side of the position that was closed / reduced
    price: float = 0.0          # fill price of the event
    realised: float = 0.0       # trade-level realised pnl of the closed contracts (price pnl vs entry minus exit costs)
    closed: float = 0.0         # |contracts| closed by the event


def mark_bar(state, stream, k, venue, policy) -> BarMark:
    """Mark bar k (k >= 1) of a position opened or held from close[k-1]; mutates `state`.

    Precedence with unknown intrabar order, worst case assumed: stop (checks only), then liquidation (perp); if both
    trigger, the one at the worse price. Funding is charged in every case."""
    m = BarMark()
    c = state.contracts
    if k < 1 or c == 0:
        return m
    cs, prev = venue.contract_size, float(stream.close[k - 1])
    long = c > 0
    side = 1 if long else -1
    m.side = side
    if venue.funding is not None:
        m.funding = c * cs * prev * venue.funding.rate_per_hour * stream.bar_hours
    e0 = state.equity
    n = abs(c)
    # (a) stop
    stop_fill = None
    if policy.checks and state.stop_price > 0:
        hit = stream.low[k] <= state.stop_price if long else stream.high[k] >= state.stop_price
        if hit:
            o = stream.open[k]
            stop_fill = min(o, state.stop_price) if long else max(o, state.stop_price)
    # (b) liquidation against the adverse extreme
    liq_price = None
    if venue.kind == "perp":
        adv = float(stream.low[k] if long else stream.high[k])
        eq_adv = e0 + c * cs * (adv - prev) - m.funding
        if eq_adv < venue.maintenance_margin_rate * n * cs * adv:
            liq_price = adv
    if liq_price is not None and (stop_fill is None or (liq_price <= stop_fill if long else liq_price >= stop_fill)):
        m.event, m.price = "liquidation", liq_price
        m.pnl = c * cs * (liq_price - prev)
        m.fee = venue.fees.cost(n, liq_price, cs) + venue.liquidation_fee_usd
    elif stop_fill is not None:
        m.event, m.price = "stop", float(stop_fill)
        m.pnl = c * cs * (stop_fill - prev)
        m.fee = venue.fees.cost(n, stop_fill, cs)
        m.slippage = policy.slippage_bps * 1e-4 * n * cs * stop_fill
    else:
        m.pnl = c * cs * (float(stream.close[k]) - prev)
    state.equity = e0 + m.pnl - m.fee - m.funding - m.slippage
    if m.event:
        m.closed = n
        m.realised = c * cs * (m.price - state.entry_price) - m.fee - m.slippage
        state.contracts, state.stop_price, state.open_risk = 0.0, 0.0, 0.0
        if policy.checks:                       # after a stop or a liquidation, do not re-enter that side until the signal changes
            state.blocked_direction = side
    return m


def deleverage(state, stream, k, venue, policy) -> BarMark:
    """Step 4: if exposure N/E > exposure_cap*(1+breach_band), reduce to floor(cap*E/(cs*close)) at close[k] with fees
    and slippage; if E <= 0, go flat and set bankrupt. Mutates `state`; returns an empty mark if no breach."""
    m = BarMark()
    c = state.contracts
    if c == 0:
        return m
    cs, p = venue.contract_size, float(stream.close[k])
    n = abs(c)
    side = 1 if c > 0 else -1
    if state.equity <= 0:
        state.contracts, state.stop_price, state.open_risk, state.bankrupt = 0.0, 0.0, 0.0, True
        m.event, m.side, m.price, m.closed = "breach", side, p, n
        return m
    if n * cs * p / state.equity <= policy.exposure_cap * (1 + policy.breach_band):
        return m
    target = math.floor(policy.exposure_cap * state.equity / (cs * p))
    cut = n - target
    m.event, m.side, m.price, m.closed = "breach", side, p, cut
    m.fee = venue.fees.cost(cut, p, cs)
    m.slippage = policy.slippage_bps * 1e-4 * cut * cs * p
    m.realised = side * cut * cs * (p - state.entry_price) - m.fee - m.slippage
    state.equity = state.equity - m.fee - m.slippage
    state.open_risk *= target / n
    state.contracts = side * float(target)
    if target == 0:
        state.stop_price = 0.0
    return m
