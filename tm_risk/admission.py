"""[TradeMaster] admit(): cap a requested new notional by loss budgets, drawdown headroom, collateral, liquidity and episode loss."""
import math

from tm_risk.types import Decision

NAMES = ("request", "trade_loss", "aggregate_loss", "drawdown_headroom", "exposure_cap", "collateral", "liq_distance",
         "participation", "episode_loss")
EPISODE = ("episode_loss",)
INF = float("inf")


def admit(k, side, f, state, stream, policy, venue, episode) -> Decision:
    """Notional capacities per spec section 8 (est_fee = (fee rate + slippage) * f*E). permitted = min over all;
    binding = first minimal name in table order; contracts rounded down to whole contracts.
    aggregate_loss, collateral and liq_distance are defensive caps: with one net position admit runs only when flat
    (open_risk 0) and exposure_cap <= L, so they cannot bind under the shipped policies."""
    from tm_risk import stops
    E, p, cs = state.equity, float(stream.close[k]), venue.contract_size
    u = stops.stressed_loss_per_unit(stream, k, venue, policy)
    per = lambda usd: usd / u if u > 0 else INF
    wanted = max(0.0, f) * E
    est_fee = (venue.fees.rate(p, cs) + policy.slippage_bps * 1e-4) * wanted
    episode.open(E)
    cap = {"request": wanted,
           "trade_loss": per(policy.trade_loss_budget * E),
           "aggregate_loss": per(max(0.0, policy.aggregate_loss_budget * E - state.open_risk)),
           "drawdown_headroom": per(max(0.0, E - (1 - policy.dd_halt) * state.peak_ref)),
           "exposure_cap": policy.exposure_cap * E,
           "collateral": max(0.0, E - est_fee) * policy.margin_leverage}
    if venue.kind == "perp":
        sd = stops.stop_distance(stream, k, policy) if policy.stop_atr_mult > 0 else 0.0
        cap["liq_distance"] = E / (sd / policy.liq_distance_factor + venue.maintenance_margin_rate)
    qv = float(stream.quote_volume[k])
    if policy.participation > 0 and qv > 0:
        cap["participation"] = policy.participation * qv
    cap["episode_loss"] = per(episode.remaining)
    for n in cap:
        cap[n] = max(0.0, cap[n])
    names = [n for n in NAMES if n in cap]
    binding = min(names, key=lambda n: cap[n])          # min() keeps the first of equal values
    permitted = cap[binding]
    preferred = min([wanted] + [cap[n] for n in EPISODE if n in cap])
    vetoes = []
    if state.halted or E < (1 - policy.dd_halt) * state.peak_ref:
        vetoes.append("halted")
    if state.bankrupt or E <= 0:
        vetoes.append("bankrupt")
    if state.blocked_direction == side:
        vetoes.append("blocked_direction")
    if vetoes:
        permitted, binding = 0.0, vetoes[0]
    contracts = float(math.floor(permitted / (cs * p) + 1e-9)) if permitted > 0 else 0.0
    d = Decision(k=k, timestamp=stream.timestamp[k], signal=side, action="enter" if contracts > 0 else "abstain",
                 wanted=wanted, preferred=preferred, permitted=permitted, contracts=contracts, binding=binding,
                 vetoes=tuple(vetoes), exposure_req=f)
    if contracts <= 0:
        d.reason = "veto: " + ",".join(vetoes) if vetoes else f"permitted {permitted:.6g} below one contract (binding {binding})"
    return d
