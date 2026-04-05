"""
Event-Driven Market Making Backtest Engine
==========================================
Simulates the Avellaneda-Stoikov market maker against historical/synthetic ticks.

At each tick:
  1. AS model computes optimal bid/ask quotes
  2. Engine checks whether a market order would fill our posted quote
  3. Update cash, inventory, and P&L
  4. Record all state variables

P&L Decomposition:
  Total P&L = Spread P&L + Inventory P&L + Adverse Selection Cost
    - Spread P&L:    half-spread × filled_volume
    - Inventory P&L: mid_price_change × inventory
    - Adverse P&L:   -|adverse_move_after_fill| × filled_volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from src.models.avellaneda_stoikov import AvellanedaStoikovModel, Quote


# ──────────────────────────────────────────────────────────────────────────────
#  State and results containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestState:
    cash: float = 1_000_000.0
    inventory: int = 0
    realized_pnl: float = 0.0
    total_fills: int = 0
    total_volume: float = 0.0


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    initial_inventory: int = 0
    lot_size: int = 1
    transaction_cost: float = 0.0001   # per unit (fraction of price)
    max_inventory: int = 50
    inventory_penalty: float = 0.001   # quadratic inventory penalty weight


class BacktestResults:
    """Container for all backtest outputs with analysis methods."""

    def __init__(self, ledger: pd.DataFrame, config: BacktestConfig):
        self.ledger = ledger
        self.config = config
        self._compute_metrics()

    def _compute_metrics(self):
        df = self.ledger
        if df.empty:
            return

        self.total_pnl       = df["total_pnl"].iloc[-1]
        self.realized_pnl    = df["realized_pnl"].iloc[-1]
        self.unrealized_pnl  = df["unrealized_pnl"].iloc[-1]
        self.spread_pnl      = df["spread_pnl"].cumsum().iloc[-1] if "spread_pnl" in df else 0
        self.inventory_pnl   = df["inventory_pnl"].cumsum().iloc[-1] if "inventory_pnl" in df else 0
        self.adverse_pnl     = df["adverse_pnl"].cumsum().iloc[-1] if "adverse_pnl" in df else 0
        self.total_fills     = int(df["fill_event"].sum())
        self.total_volume    = df["fill_qty"].sum()
        self.max_inventory   = int(df["inventory"].abs().max())
        self.inventory_turns = int(df["fill_event"].sum())

        # Daily P&L for Sharpe
        pnl_series = df["total_pnl"].diff().fillna(0)
        if pnl_series.std() > 0:
            self.sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252 * 390)
        else:
            self.sharpe = 0.0

        # Max drawdown
        cummax = df["total_pnl"].cummax()
        drawdown = df["total_pnl"] - cummax
        self.max_drawdown = float(drawdown.min())

        # Inventory turnover (cycles)
        inv_abs = df["inventory"].abs()
        self.avg_inventory = float(inv_abs.mean())
        self.pct_time_flat = float((inv_abs == 0).mean())

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Backtest Results — Avellaneda-Stoikov Engine",
            "=" * 60,
            f"  Total P&L         : ${self.total_pnl:>12,.2f}",
            f"  Realized P&L      : ${self.realized_pnl:>12,.2f}",
            f"  Unrealized P&L    : ${self.unrealized_pnl:>12,.2f}",
            "",
            "  ── P&L Decomposition ──────────────────────────────",
            f"  Spread capture    : ${self.spread_pnl:>12,.2f}",
            f"  Inventory P&L     : ${self.inventory_pnl:>12,.2f}",
            f"  Adverse selection : ${self.adverse_pnl:>12,.2f}",
            "",
            "  ── Risk Metrics ───────────────────────────────────",
            f"  Sharpe ratio      : {self.sharpe:>12.3f}",
            f"  Max drawdown      : ${self.max_drawdown:>12,.2f}",
            f"  Avg |inventory|   : {self.avg_inventory:>12.2f}",
            f"  Max |inventory|   : {self.max_inventory:>12}",
            f"  % time flat       : {self.pct_time_flat:>12.1%}",
            "",
            "  ── Activity ───────────────────────────────────────",
            f"  Total fills       : {self.total_fills:>12,}",
            f"  Total volume      : {self.total_volume:>12,.0f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  Main backtest engine
# ──────────────────────────────────────────────────────────────────────────────

class MarketMakingBacktest:
    """
    Event-driven backtest for the Avellaneda-Stoikov market maker.

    Parameters
    ----------
    model  : AvellanedaStoikovModel
    config : BacktestConfig
    """

    def __init__(
        self,
        model: AvellanedaStoikovModel,
        config: Optional[BacktestConfig] = None,
    ):
        self.model  = model
        self.config = config or BacktestConfig()

    def run(
        self,
        quotes_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        verbose: bool = True,
    ) -> BacktestResults:
        """
        Run the full backtest.

        Parameters
        ----------
        quotes_df : DataFrame [timestamp, mid_price, best_bid, best_ask]
        trades_df : DataFrame [timestamp, trade_side, trade_price, trade_qty]
        """
        cfg   = self.config
        model = self.model

        # Initialise state
        cash      = cfg.initial_cash
        inventory = cfg.initial_inventory
        realized_pnl  = 0.0
        spread_pnl    = 0.0
        inventory_pnl = 0.0
        adverse_pnl   = 0.0

        # Assign each trade to the nearest quote tick using searchsorted
        quote_ts  = quotes_df.timestamp.values
        trade_ts  = trades_df.timestamp.values
        # For each trade, find the quote tick it falls within
        bucket_idx = np.searchsorted(quote_ts, trade_ts, side="right") - 1
        bucket_idx = np.clip(bucket_idx, 0, len(quote_ts) - 1)
        trades_df  = trades_df.copy()
        trades_df["bucket"] = bucket_idx

        # Group trades by quote bucket index
        trades_by_bucket = trades_df.groupby("bucket")

        timestamps   = sorted(quotes_df.timestamp.unique())
        T_horizon    = model.T
        t_start      = timestamps[0]
        t_end        = timestamps[-1]
        t_range      = max(t_end - t_start, 1.0)

        prev_mid = quotes_df.iloc[0].mid_price
        ledger_rows = []

        iterator = tqdm(range(len(timestamps)), desc="Backtesting", disable=not verbose)

        for i in iterator:
            ts    = timestamps[i]
            row_q = quotes_df.iloc[i]
            mid   = row_q.mid_price
            t_norm = (ts - t_start) / t_range * T_horizon   # normalised time

            # ── 1. Compute optimal quotes ──────────────────────────────────
            quote = model.quote(mid, inventory, t_norm)

            # ── 2. Process incoming market orders ─────────────────────────
            fill_event = False
            fill_qty   = 0
            fill_side  = None
            fill_price = 0.0

            if i in trades_by_bucket.groups:
                trades_now = trades_by_bucket.get_group(i)
                for _, trade in trades_now.iterrows():
                    side  = trade.trade_side
                    price = trade.trade_price
                    qty   = int(trade.trade_qty) * cfg.lot_size

                    # Fill logic: market sell hits our bid, market buy hits our ask
                    if side == "sell":
                        # Market sell order — we can fill at our posted bid
                        # Fill if market sell price is at or below our bid (we offer to buy at bid)
                        fill_price_cand = quote.bid
                        if abs(inventory + qty) <= cfg.max_inventory:
                            cost  = fill_price_cand * qty
                            tc    = cfg.transaction_cost * cost
                            cash      -= (cost + tc)
                            inventory += qty
                            realized_pnl -= tc

                            # Spread P&L: we buy at our bid, fair value is mid
                            sp    = (mid - fill_price_cand) * qty
                            spread_pnl += sp
                            realized_pnl += sp

                            fill_event = True
                            fill_qty   += qty
                            fill_side  = "buy"
                            fill_price  = fill_price_cand

                    elif side == "buy":
                        # Market buy order — we can fill at our posted ask
                        fill_price_cand = quote.ask
                        if abs(inventory - qty) <= cfg.max_inventory:
                            revenue = fill_price_cand * qty
                            tc      = cfg.transaction_cost * revenue
                            cash      += (revenue - tc)
                            inventory -= qty
                            realized_pnl -= tc

                            # Spread P&L: we sell at our ask, fair value is mid
                            sp    = (fill_price_cand - mid) * qty
                            spread_pnl += sp
                            realized_pnl += sp

                            fill_event = True
                            fill_qty   += qty
                            fill_side  = "sell"
                            fill_price  = fill_price_cand

            # ── 3. Adverse selection: measure price move after fill ────────
            adv = 0.0
            if fill_event and fill_side is not None:
                # Adverse move: mid moves against us after fill
                mid_move = mid - prev_mid
                if fill_side == "buy" and mid_move < 0:
                    adv = mid_move * fill_qty   # negative = cost
                elif fill_side == "sell" and mid_move > 0:
                    adv = -mid_move * fill_qty  # negative = cost
                adverse_pnl += adv

            # ── 4. Mark-to-market ─────────────────────────────────────────
            unrealized   = inventory * mid
            inv_pnl_tick = inventory * (mid - prev_mid)
            inventory_pnl += inv_pnl_tick

            # Inventory penalty (quadratic cost)
            inv_penalty = cfg.inventory_penalty * inventory**2
            total_pnl   = cash + unrealized - cfg.initial_cash - inv_penalty

            # ── 5. Record row ─────────────────────────────────────────────
            ledger_rows.append({
                "timestamp":     ts,
                "mid_price":     mid,
                "bid_quote":     quote.bid,
                "ask_quote":     quote.ask,
                "reservation":   quote.reservation,
                "half_spread":   quote.half_spread,
                "inventory":     inventory,
                "cash":          cash,
                "unrealized_pnl": unrealized - inventory * row_q.mid_price
                                  if inventory != 0 else 0.0,
                "realized_pnl":  realized_pnl,
                "total_pnl":     total_pnl,
                "spread_pnl":    spread_pnl,
                "inventory_pnl": inventory_pnl,
                "adverse_pnl":   adverse_pnl,
                "fill_event":    int(fill_event),
                "fill_qty":      fill_qty,
                "fill_side":     fill_side or "",
                "fill_price":    fill_price,
            })

            prev_mid = mid

        ledger = pd.DataFrame(ledger_rows)
        # Recompute incremental columns for decomposition
        ledger["spread_pnl_incr"]    = ledger["spread_pnl"].diff().fillna(0)
        ledger["inventory_pnl_incr"] = ledger["inventory_pnl"].diff().fillna(0)
        ledger["adverse_pnl_incr"]   = ledger["adverse_pnl"].diff().fillna(0)
        ledger["unrealized_pnl"]     = ledger["inventory"] * ledger["mid_price"]

        self.results_ = BacktestResults(ledger, cfg)
        if verbose:
            print(self.results_.summary())

        return self.results_
