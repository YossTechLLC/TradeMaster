"""[TradeMaster] RiskEpisode: one flat-to-flat run of non-zero signals with a realised-loss allowance that profits never refill."""


class RiskEpisode:
    """Starts at the first non-zero signal after flat; flips stay inside; the engine ends it on a 0 signal or
    `policy.episode_max_bars`. Rejected proposals never touch it."""

    def __init__(self, policy, start_k=0, equity=None):
        self.policy = policy
        self.start_k = int(start_k)
        self.start_equity = equity
        self.loss_allowance = None      # set at the first admission: episode_loss_budget * E
        self.loss_spent = 0.0

    def open(self, equity):
        """Fix the allowance at the first admission (later calls are no-ops)."""
        if self.loss_allowance is None:
            self.loss_allowance = self.policy.episode_loss_budget * equity
        return self.loss_allowance

    def record_close(self, realised_pnl):
        """Add max(0, -pnl) to loss_spent; profits do not reduce it."""
        self.loss_spent += max(0.0, -float(realised_pnl))

    @property
    def remaining(self):
        """Loss (USD) still allowed; the full budget of `start_equity` if nothing has been admitted yet."""
        allow = self.loss_allowance
        if allow is None:
            allow = self.policy.episode_loss_budget * (self.start_equity or 0.0)
        return max(0.0, allow - self.loss_spent)

    def expired(self, k):
        return k - self.start_k >= self.policy.episode_max_bars
