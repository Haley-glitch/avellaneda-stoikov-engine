"""
Synthetic Limit Order Book Data Generator
==========================================
Generates realistic tick data for backtesting the AS market making engine.

Mid-price follows arithmetic Brownian motion.
Order arrivals follow a Poisson process with intensity λ.
Fill probabilities follow the exponential model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticDataParams:
    """Parameters for synthetic LOB data generation."""
    mid_price_start: float = 100.0
    sigma: float = 0.02          # mid-price volatility (per tick)
    lambda_b: float = 5.0        # buy arrival intensity (per second)
    lambda_a: float = 5.0        # sell arrival intensity (per second)
    kappa: float = 1.5           # fill-rate decay parameter
    dt: float = 1.0              # seconds per tick
    n_ticks: int = 50_000
    seed: Optional[int] = 42


def generate_tick_data(params: SyntheticDataParams) -> pd.DataFrame:
    """
    Generate synthetic tick data with Poisson order arrivals.

    Returns a DataFrame with columns:
        timestamp, mid_price, best_bid, best_ask,
        trade_side, trade_price, trade_qty
    """
    rng = np.random.default_rng(params.seed)
    n = params.n_ticks

    # ── Mid-price: arithmetic Brownian motion ──────────────────────────────
    increments = rng.normal(0, params.sigma * np.sqrt(params.dt), size=n)
    mid_prices = params.mid_price_start + np.cumsum(increments)
    mid_prices = np.maximum(mid_prices, 1.0)   # price floor

    # ── Spread: realistic quoted spread ~= 0.02 * mid ──────────────────────
    base_spread = 0.02
    spreads = rng.gamma(shape=2.0, scale=base_spread / 2.0, size=n) * mid_prices
    best_bid = mid_prices - spreads / 2
    best_ask = mid_prices + spreads / 2

    # ── Order arrivals: Poisson ────────────────────────────────────────────
    buy_arrivals  = rng.poisson(params.lambda_b * params.dt, size=n)
    sell_arrivals = rng.poisson(params.lambda_a * params.dt, size=n)

    # Build event list
    timestamps, sides, prices, qtys, mids = [], [], [], [], []

    for t in range(n):
        ts_base = t * params.dt
        # Buy market orders — spread sub-tick arrivals uniformly within interval
        n_buy = buy_arrivals[t]
        if n_buy > 0:
            buy_offsets = np.sort(rng.uniform(0, params.dt * 0.999, size=n_buy))
            for off in buy_offsets:
                timestamps.append(ts_base + off)
                sides.append("buy")
                prices.append(best_ask[t])
                qtys.append(int(rng.integers(1, 5)))
                mids.append(mid_prices[t])
        # Sell market orders
        n_sell = sell_arrivals[t]
        if n_sell > 0:
            sell_offsets = np.sort(rng.uniform(0, params.dt * 0.999, size=n_sell))
            for off in sell_offsets:
                timestamps.append(ts_base + off)
                sides.append("sell")
                prices.append(best_bid[t])
                qtys.append(int(rng.integers(1, 5)))
                mids.append(mid_prices[t])

    trades_df = pd.DataFrame({
        "timestamp":   timestamps,
        "mid_price":   mids,
        "trade_side":  sides,
        "trade_price": prices,
        "trade_qty":   qtys,
    })

    # ── Quote snapshots (every tick) ──────────────────────────────────────
    quotes_df = pd.DataFrame({
        "timestamp": np.arange(n) * params.dt,
        "mid_price": mid_prices,
        "best_bid":  best_bid,
        "best_ask":  best_ask,
        "spread":    spreads,
    })

    return quotes_df, trades_df


def compute_inter_arrival_times(trades_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract inter-arrival times for each side."""
    result = {}
    for side in ("buy", "sell"):
        ts = trades_df[trades_df.trade_side == side].timestamp.values
        if len(ts) > 1:
            result[side] = np.diff(np.sort(ts))
        else:
            result[side] = np.array([])
    return result
