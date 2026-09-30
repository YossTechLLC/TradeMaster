"""[TradeMaster] Trading layer for MacroHFT: long/short positions, venue costs (fees, funding, leverage,
liquidation) and a turnover penalty, selectable next to the upstream long-only environment.
Specification: docs/trading_mvp_spec.md. See README.md. (report.py is a CLI and is not imported here.)"""
from MacroHFT.trading.venue import load_venue
from MacroHFT.trading.config import TradingConfig, add_trading_args, config_from_args
from MacroHFT.trading.accounting import Account
from MacroHFT.trading.envs import make_env
from MacroHFT.trading.teacher import make_q_table_trading
from MacroHFT.trading.penalty import turnover_penalty

__all__ = ["load_venue", "TradingConfig", "add_trading_args", "config_from_args", "Account", "make_env",
           "make_q_table_trading", "turnover_penalty"]
