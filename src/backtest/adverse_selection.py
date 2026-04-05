"""
Adverse Selection Analysis
==========================
Decomposes P&L into:
  1. Spread capture        — profit from bid-ask spread
  2. Inventory holding     — P&L from mid-price drift while holding inventory
  3. Adverse selection     — cost of trading against informed flow

Also computes:
  - Fill Toxicity Score    — fraction of fills followed by adverse moves
  - VPIN proxy             — volume-synchronized probability of informed trading
  - Roll implied spread    — from price autocovariance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class AdverseSelectionReport:
    """Summary of adverse selection analysis."""

    # P&L breakdown
    total_pnl:        float = 0.0
    spread_pnl:       float = 0.0
    inventory_pnl:    float = 0.0
    adverse_pnl:      float = 0.0

    # Toxicity metrics
    fill_toxicity:    float = 0.0   # fraction of fills with adverse subsequent move
    avg_adverse_move: float = 0.0   # average mid-price move after fill (adverse side)
    vpin_proxy:       float = 0.0   # volume imbalance proxy
    roll_spread:      float = 0.0   # Roll (1984) implied spread estimate

    # Fill breakdown
    n_buy_fills:   int = 0
    n_sell_fills:  int = 0
    buy_toxicity:  float = 0.0
    sell_toxicity: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  Adverse Selection Decomposition Report",
            "=" * 62,
            "",
            "  ── P&L Attribution ────────────────────────────────────",
            f"  Total P&L         : ${self.total_pnl:>12,.2f}",
            f"  Spread capture    : ${self.spread_pnl:>12,.2f}  ({self._pct(self.spread_pnl):.1%})",
            f"  Inventory P&L     : ${self.inventory_pnl:>12,.2f}  ({self._pct(self.inventory_pnl):.1%})",
            f"  Adverse selection : ${self.adverse_pnl:>12,.2f}  ({self._pct(self.adverse_pnl):.1%})",
            "",
            "  ── Toxicity Metrics ───────────────────────────────────",
            f"  Fill toxicity     : {self.fill_toxicity:>12.1%}",
            f"  Buy  toxicity     : {self.buy_toxicity:>12.1%}  ({self.n_buy_fills:,} fills)",
            f"  Sell toxicity     : {self.sell_toxicity:>12.1%}  ({self.n_sell_fills:,} fills)",
            f"  Avg adverse move  : {self.avg_adverse_move:>12.6f}",
            "",
            "  ── Market Microstructure ──────────────────────────────",
            f"  VPIN proxy        : {self.vpin_proxy:>12.4f}",
            f"  Roll spread est.  : {self.roll_spread:>12.6f}",
            "=" * 62,
        ]
        return "\n".join(lines)

    def _pct(self, val: float) -> float:
        base = abs(self.total_pnl) + 1e-8
        return val / base


def decompose_adverse_selection(
    ledger: pd.DataFrame,
    lookahead_ticks: int = 5,
) -> AdverseSelectionReport:
    """
    Compute full adverse selection breakdown from backtest ledger.

    Parameters
    ----------
    ledger         : backtest output DataFrame
    lookahead_ticks: number of ticks forward to measure mid-price move
    """
    report = AdverseSelectionReport()

    if ledger.empty:
        return report

    report.total_pnl     = float(ledger["total_pnl"].iloc[-1])
    report.spread_pnl    = float(ledger["spread_pnl"].iloc[-1]) if "spread_pnl" in ledger else 0.0
    report.inventory_pnl = float(ledger["inventory_pnl"].iloc[-1]) if "inventory_pnl" in ledger else 0.0
    report.adverse_pnl   = float(ledger["adverse_pnl"].iloc[-1]) if "adverse_pnl" in ledger else 0.0

    # ── Fill toxicity ──────────────────────────────────────────────────────
    fills = ledger[ledger["fill_event"] == 1].copy()

    if len(fills) == 0:
        return report

    mid_prices = ledger["mid_price"].values
    fill_idxs  = fills.index.tolist()

    buy_toxic  = []
    sell_toxic = []
    adverse_moves = []

    for idx in fill_idxs:
        side = ledger.loc[idx, "fill_side"]
        pos  = ledger.index.get_loc(idx)
        fut_pos = min(pos + lookahead_ticks, len(mid_prices) - 1)
        mid_now = mid_prices[pos]
        mid_fut = mid_prices[fut_pos]
        move = mid_fut - mid_now

        if side == "buy":
            # We bought; adverse if mid moves down
            toxic = move < 0
            buy_toxic.append(toxic)
            if toxic:
                adverse_moves.append(abs(move))
        elif side == "sell":
            # We sold; adverse if mid moves up
            toxic = move > 0
            sell_toxic.append(toxic)
            if toxic:
                adverse_moves.append(abs(move))

    report.n_buy_fills   = len(buy_toxic)
    report.n_sell_fills  = len(sell_toxic)
    report.buy_toxicity  = float(np.mean(buy_toxic)) if buy_toxic else 0.0
    report.sell_toxicity = float(np.mean(sell_toxic)) if sell_toxic else 0.0
    report.fill_toxicity = float(np.mean(buy_toxic + sell_toxic)) if (buy_toxic + sell_toxic) else 0.0
    report.avg_adverse_move = float(np.mean(adverse_moves)) if adverse_moves else 0.0

    # ── VPIN proxy: volume imbalance ──────────────────────────────────────
    if "fill_side" in ledger and "fill_qty" in ledger:
        buy_vol  = ledger[ledger.fill_side == "buy"].fill_qty.sum()
        sell_vol = ledger[ledger.fill_side == "sell"].fill_qty.sum()
        total_vol = buy_vol + sell_vol + 1e-8
        report.vpin_proxy = float(abs(buy_vol - sell_vol) / total_vol)

    # ── Roll (1984) spread estimate ───────────────────────────────────────
    prices = ledger["mid_price"].values
    if len(prices) > 2:
        dp = np.diff(prices)
        cov = np.cov(dp[:-1], dp[1:])[0, 1]
        if cov < 0:
            report.roll_spread = float(2 * np.sqrt(-cov))
        else:
            report.roll_spread = 0.0

    return report


def compute_vpin(
    trades_df: pd.DataFrame,
    bucket_size: float = 1000.0,
    n_buckets: int = 50,
) -> pd.Series:
    """
    Compute a VPIN time series from trade flow.

    Classic VPIN = |V_buy - V_sell| / V_total  over rolling volume buckets.
    """
    trades = trades_df.copy()
    trades["signed_vol"] = np.where(
        trades.trade_side == "buy",
        trades.trade_qty,
        -trades.trade_qty,
    )
    trades["cum_vol"] = trades.trade_qty.cumsum()
    trades["bucket"]  = (trades.cum_vol / bucket_size).astype(int)

    buckets = trades.groupby("bucket").agg(
        buy_vol  = ("trade_qty",   lambda x: x[trades.loc[x.index, "trade_side"] == "buy"].sum()),
        sell_vol = ("trade_qty",   lambda x: x[trades.loc[x.index, "trade_side"] == "sell"].sum()),
        ts_end   = ("timestamp",  "max"),
    )
    buckets["vpin"] = (
        (buckets["buy_vol"] - buckets["sell_vol"]).abs()
        / (buckets["buy_vol"] + buckets["sell_vol"] + 1e-8)
    )
    return buckets[["ts_end", "vpin"]].set_index("ts_end")["vpin"]
