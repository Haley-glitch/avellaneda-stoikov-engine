"""
Avellaneda-Stoikov Market Making Model (2008)
=============================================
Closed-form solution to the HJB equation for optimal bid/ask quoting.

The market maker maximises expected terminal utility:
    max E[U(W_T)]  =  max E[-exp(-γ · (W_T + q_T · S_T))]

subject to:
    dS_t = σ dW_t                       (arithmetic Brownian motion)
    dN^a_t ~ Poisson(λ · exp(-κ · δ^a)) (ask side fills)
    dN^b_t ~ Poisson(λ · exp(-κ · δ^b)) (bid side fills)

Closed-form (symmetric, risk-neutral spread):
    r(s,q,t) = s - q·γ·σ²·(T-t)              [reservation price]
    δ*(t)    = γ·σ²·(T-t) + (2/γ)·ln(1+γ/κ)  [half-spread]

References:
    Avellaneda & Stoikov (2008), Quantitative Finance 8(3), 217-224.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import NamedTuple


class Quote(NamedTuple):
    """A pair of bid/ask quotes produced by the model."""
    bid: float
    ask: float
    reservation: float
    half_spread: float
    mid: float
    inventory: int
    time_remaining: float


@dataclass
class AvellanedaStoikovModel:
    """
    Avellaneda-Stoikov closed-form market making model.

    Parameters
    ----------
    gamma  : float  Risk aversion parameter γ > 0
    sigma  : float  Mid-price volatility (calibrated)
    kappa  : float  Fill-rate decay parameter κ > 0
    T      : float  Normalised trading horizon (e.g. 1.0)
    min_spread : float  Minimum half-spread (tick protection)
    max_inventory : int  Hard inventory limit
    """

    gamma: float = 0.1
    sigma: float = 0.02
    kappa: float = 1.5
    T:     float = 1.0
    min_spread: float = 0.001
    max_inventory: int = 50

    # ── Core equations ────────────────────────────────────────────────────────

    def reservation_price(self, mid: float, q: int, t: float) -> float:
        """
        r(s, q, t) = s - q · γ · σ² · (T - t)

        The reservation price adjusts the mid toward reducing inventory risk:
        positive inventory → lower reservation price (incentivise sells)
        negative inventory → higher reservation price (incentivise buys)
        """
        tau = max(self.T - t, 0.0)
        return mid - q * self.gamma * self.sigma**2 * tau

    def optimal_half_spread(self, t: float) -> float:
        """
        δ*(t) = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/κ)

        Two components:
            1. Inventory risk term:   γ·σ²·(T-t)        → widens with time
            2. Market making profit:  (2/γ)·ln(1+γ/κ)   → fixed premium
        """
        tau = max(self.T - t, 0.0)
        inventory_term = self.gamma * self.sigma**2 * tau
        mm_premium     = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return max(inventory_term + mm_premium, self.min_spread)

    def quote(self, mid: float, q: int, t: float) -> Quote:
        """
        Compute optimal bid and ask quotes.

        Parameters
        ----------
        mid : float  Current mid-price
        q   : int    Current inventory
        t   : float  Current time (0 ≤ t ≤ T)
        """
        r     = self.reservation_price(mid, q, t)
        delta = self.optimal_half_spread(t)

        # Inventory skewing: tighten spread on side we want filled
        # This is the natural consequence of the reservation price adjustment
        bid = r - delta
        ask = r + delta

        return Quote(
            bid=round(bid, 8),
            ask=round(ask, 8),
            reservation=round(r, 8),
            half_spread=round(delta, 8),
            mid=round(mid, 8),
            inventory=q,
            time_remaining=self.T - t,
        )

    # ── Fill probability ──────────────────────────────────────────────────────

    def fill_probability(self, delta: float) -> float:
        """P(fill | δ) = exp(-κ · δ)"""
        return np.exp(-self.kappa * max(delta, 0.0))

    def expected_fill_rate(self, lam: float, delta: float) -> float:
        """λ · exp(-κ · δ)  — the effective order arrival rate at depth δ"""
        return lam * self.fill_probability(delta)

    # ── Inventory-adjusted spread (asymmetric) ────────────────────────────────

    def asymmetric_spread(
        self,
        mid: float,
        q: int,
        t: float,
        inventory_skew_factor: float = 1.0,
    ) -> tuple[float, float]:
        """
        Returns (bid_depth, ask_depth) from mid.
        When inventory is positive, ask is tightened to encourage selling.
        """
        r     = self.reservation_price(mid, q, t)
        delta = self.optimal_half_spread(t)

        # Clip inventory contribution to avoid extreme skew
        skew = np.clip(q / max(self.max_inventory, 1), -1.0, 1.0)
        bid_depth = delta * (1.0 + skew * inventory_skew_factor)
        ask_depth = delta * (1.0 - skew * inventory_skew_factor)

        bid_depth = max(bid_depth, self.min_spread)
        ask_depth = max(ask_depth, self.min_spread)

        return r - bid_depth, r + ask_depth

    # ── Model diagnostics ─────────────────────────────────────────────────────

    def spread_term_structure(self, n_points: int = 100) -> dict:
        """Return spread as a function of time for diagnostics."""
        times = np.linspace(0, self.T, n_points)
        spreads = np.array([self.optimal_half_spread(t) * 2 for t in times])
        return {"time": times, "spread": spreads}

    def inventory_impact_curve(
        self,
        mid: float,
        t: float,
        q_range: int = 20,
    ) -> dict:
        """Return reservation price as a function of inventory."""
        qs = np.arange(-q_range, q_range + 1)
        rs = np.array([self.reservation_price(mid, q, t) for q in qs])
        return {"inventory": qs, "reservation_price": rs}

    def __repr__(self) -> str:
        return (
            f"AvellanedaStoikovModel("
            f"γ={self.gamma}, σ={self.sigma:.4f}, κ={self.kappa:.4f}, T={self.T})"
        )
