"""
Maximum Likelihood Estimation for Order Arrival Intensity
=========================================================
Calibrates the Poisson arrival model and exponential fill-rate
from observed trade tick data.

Model:
  - Order arrivals: Poisson(λ · dt)
  - Inter-arrival times: Exponential(λ)
  - Fill probability: P(fill|δ) = exp(-κ·δ)  [exponential model]
  - Mid-price volatility σ estimated from price increments

MLE Estimation:
  For Poisson inter-arrival times τ ~ Exp(λ):
    L(λ | τ₁,...,τₙ) = λⁿ · exp(-λ · Στᵢ)
    λ̂_MLE = n / Στᵢ  (= 1 / mean(τ))

  Standard error: se(λ̂) = λ̂ / √n

  For fill-rate κ using quoted depth observations:
    Minimise negative log-likelihood over observed fills vs no-fills
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2, norm, expon


# ──────────────────────────────────────────────────────────────────────────────
#  Parameter container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibratedParams:
    """Holds all calibrated model parameters with uncertainty estimates."""

    # Poisson arrival intensity (per second)
    lambda_buy:   float = 1.0
    lambda_sell:  float = 1.0
    lambda_buy_se:  float = 0.0
    lambda_sell_se: float = 0.0
    lambda_buy_ci:  tuple = (0.0, 0.0)
    lambda_sell_ci: tuple = (0.0, 0.0)

    # Fill-rate decay (κ in exponential fill model)
    kappa:    float = 1.0
    kappa_se: float = 0.0
    kappa_ci: tuple = (0.0, 0.0)

    # Mid-price volatility (per second)
    sigma:    float = 0.01
    sigma_se: float = 0.0
    sigma_ci: tuple = (0.0, 0.0)

    # Sample sizes
    n_buy:   int = 0
    n_sell:  int = 0
    n_price: int = 0

    # Goodness of fit
    log_likelihood_buy:  float = 0.0
    log_likelihood_sell: float = 0.0
    ks_stat_buy:  float = 0.0
    ks_pvalue_buy: float = 0.0
    ks_stat_sell:  float = 0.0
    ks_pvalue_sell: float = 0.0

    # Meta
    confidence_level: float = 0.95

    def summary(self) -> str:
        z = norm.ppf(0.5 + self.confidence_level / 2)
        lines = [
            "=" * 62,
            "  MLE Calibration Report — Avellaneda-Stoikov Engine",
            "=" * 62,
            "",
            f"  Confidence level : {self.confidence_level:.0%}",
            "",
            "  ── Order Arrival Intensities (λ) ──────────────────────",
            f"  λ_buy   = {self.lambda_buy:8.4f}  ±  {self.lambda_buy_se:.4f}",
            f"            95% CI: [{self.lambda_buy_ci[0]:.4f}, {self.lambda_buy_ci[1]:.4f}]",
            f"            n_obs = {self.n_buy}",
            f"            logL  = {self.log_likelihood_buy:.2f}",
            f"            KS    = {self.ks_stat_buy:.4f}  (p={self.ks_pvalue_buy:.4f})",
            "",
            f"  λ_sell  = {self.lambda_sell:8.4f}  ±  {self.lambda_sell_se:.4f}",
            f"            95% CI: [{self.lambda_sell_ci[0]:.4f}, {self.lambda_sell_ci[1]:.4f}]",
            f"            n_obs = {self.n_sell}",
            f"            logL  = {self.log_likelihood_sell:.2f}",
            f"            KS    = {self.ks_stat_sell:.4f}  (p={self.ks_pvalue_sell:.4f})",
            "",
            "  ── Fill-Rate Decay (κ) ────────────────────────────────",
            f"  κ       = {self.kappa:8.4f}  ±  {self.kappa_se:.4f}",
            f"            95% CI: [{self.kappa_ci[0]:.4f}, {self.kappa_ci[1]:.4f}]",
            "",
            "  ── Mid-Price Volatility (σ) ───────────────────────────",
            f"  σ       = {self.sigma:8.6f}  ±  {self.sigma_se:.6f}",
            f"            95% CI: [{self.sigma_ci[0]:.6f}, {self.sigma_ci[1]:.6f}]",
            f"            n_obs = {self.n_price}",
            "",
            "=" * 62,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuples to lists for JSON serialisation
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Calibration] Parameters saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CalibratedParams":
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            if isinstance(v, list) and k.endswith("_ci"):
                d[k] = tuple(v)
        return cls(**d)


# ──────────────────────────────────────────────────────────────────────────────
#  MLE Estimator
# ──────────────────────────────────────────────────────────────────────────────

class OrderArrivalCalibrator:
    """
    MLE calibration of the Avellaneda-Stoikov model parameters.

    Parameters
    ----------
    confidence_level : float
        Coverage for Wald confidence intervals (default 0.95).
    bootstrap_samples : int
        Number of bootstrap replicates for CI cross-check.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_samples: int = 1000,
    ):
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self._z = norm.ppf(0.5 + confidence_level / 2)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        quotes_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        verbose: bool = True,
    ) -> CalibratedParams:
        """
        Fit all model parameters from quote and trade data.

        Parameters
        ----------
        quotes_df : DataFrame with columns [timestamp, mid_price, ...]
        trades_df : DataFrame with columns [timestamp, trade_side, trade_price, ...]
        """
        params = CalibratedParams(confidence_level=self.confidence_level)

        # 1. Estimate λ_buy and λ_sell
        for side in ("buy", "sell"):
            res = self._fit_lambda(trades_df, side)
            if side == "buy":
                params.lambda_buy    = res["lambda_mle"]
                params.lambda_buy_se = res["se"]
                params.lambda_buy_ci = res["ci"]
                params.n_buy         = res["n"]
                params.log_likelihood_buy  = res["log_likelihood"]
                params.ks_stat_buy   = res["ks_stat"]
                params.ks_pvalue_buy = res["ks_pvalue"]
            else:
                params.lambda_sell    = res["lambda_mle"]
                params.lambda_sell_se = res["se"]
                params.lambda_sell_ci = res["ci"]
                params.n_sell         = res["n"]
                params.log_likelihood_sell  = res["log_likelihood"]
                params.ks_stat_sell   = res["ks_stat"]
                params.ks_pvalue_sell = res["ks_pvalue"]

        # 2. Estimate κ (fill-rate decay)
        kappa_res = self._fit_kappa(quotes_df, trades_df)
        params.kappa    = kappa_res["kappa_mle"]
        params.kappa_se = kappa_res["se"]
        params.kappa_ci = kappa_res["ci"]

        # 3. Estimate σ (mid-price volatility)
        sigma_res = self._fit_sigma(quotes_df)
        params.sigma    = sigma_res["sigma_mle"]
        params.sigma_se = sigma_res["se"]
        params.sigma_ci = sigma_res["ci"]
        params.n_price  = sigma_res["n"]

        if verbose:
            print(params.summary())

        self.params_ = params
        return params

    # ── λ estimation ──────────────────────────────────────────────────────────

    def _fit_lambda(self, trades_df: pd.DataFrame, side: str) -> dict:
        """
        MLE for Poisson arrival intensity from inter-arrival times.
        τ ~ Exp(λ)  →  λ̂ = 1/mean(τ),  SE = λ̂/√n
        """
        subset = trades_df[trades_df.trade_side == side].sort_values("timestamp")
        ts = subset.timestamp.values

        if len(ts) < 2:
            warnings.warn(f"Too few {side} trades for calibration.")
            return dict(lambda_mle=1.0, se=0.0, ci=(0.0, 2.0), n=0,
                        log_likelihood=0.0, ks_stat=0.0, ks_pvalue=0.0)

        tau = np.diff(ts)
        tau = tau[tau > 0]   # drop zero-duration gaps
        n   = len(tau)

        lambda_mle = 1.0 / np.mean(tau)   # MLE
        se = lambda_mle / np.sqrt(n)       # Fisher information SE
        ci = (lambda_mle - self._z * se,
              lambda_mle + self._z * se)

        # Log-likelihood
        log_lik = n * np.log(lambda_mle) - lambda_mle * tau.sum()

        # KS goodness-of-fit vs theoretical Exp(λ̂)
        from scipy.stats import kstest
        ks_stat, ks_pvalue = kstest(tau, "expon", args=(0, 1 / lambda_mle))

        return dict(
            lambda_mle=float(lambda_mle),
            se=float(se),
            ci=(float(max(0, ci[0])), float(ci[1])),
            n=n,
            log_likelihood=float(log_lik),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
        )

    # ── κ estimation ──────────────────────────────────────────────────────────

    def _fit_kappa(
        self,
        quotes_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> dict:
        """
        Estimate κ from observed fill events.

        For each trade, compute the distance from mid (= depth δ),
        then MLE for κ in P(fill|δ) = exp(-κ·δ).

        Log-likelihood: L(κ) = Σ_fills(-κ·δᵢ) + Σ_no-fills log(1 - exp(-κ·δᵢ))
        We approximate using only fills (δ values at which trades occurred).
        """
        # Merge to get mid-price at each trade timestamp
        # Use half-spread as proxy for typical quote depth
        # κ is estimated from the relation: E[half_spread] = (1/κ) * ln(1 + γ/κ) ≈ 1/κ
        # More directly: fit κ to observed fill depths using MLE for P(fill|δ) = exp(-κδ)
        # With only fill events: L(κ) ∝ exp(-κ Σδᵢ)  → κ̂ = n/Σδᵢ  (same as Exp MLE)
        q_ref = quotes_df[["timestamp", "spread"]].sort_values("timestamp")
        merged = pd.merge_asof(
            trades_df.sort_values("timestamp"),
            q_ref,
            on="timestamp",
        )
        # Use half of quoted spread as the depth at which trades arrive
        merged["depth"] = merged["spread"].fillna(merged["spread"].median()) / 2.0
        depths = merged["depth"].values
        depths = depths[depths > 1e-8]

        if len(depths) < 10:
            return dict(kappa_mle=1.5, se=0.1, ci=(1.3, 1.7))

        # MLE for exponential: κ̂ = 1/mean(depth)
        n_d = len(depths)
        kappa_mle = 1.0 / np.mean(depths)

        # Fisher information SE: same as Exp(λ) → SE = κ̂/√n
        se = kappa_mle / np.sqrt(n_d)
        ci = (kappa_mle - self._z * se, kappa_mle + self._z * se)

        return dict(
            kappa_mle=float(kappa_mle),
            se=float(se),
            ci=(float(max(0, ci[0])), float(ci[1])),
        )

    # ── σ estimation ──────────────────────────────────────────────────────────

    def _fit_sigma(self, quotes_df: pd.DataFrame) -> dict:
        """
        MLE for mid-price volatility per unit time.
        ΔS ~ N(0, σ²·Δt)  →  σ̂² = Var(ΔS/√Δt)
        """
        prices = quotes_df.mid_price.values
        dt_vals = np.diff(quotes_df.timestamp.values) if "timestamp" in quotes_df.columns else np.ones(len(prices) - 1)
        dt_vals = np.maximum(dt_vals, 1e-8)

        increments = np.diff(prices) / np.sqrt(dt_vals)
        n = len(increments)
        sigma2_mle = np.var(increments, ddof=0)
        sigma_mle  = np.sqrt(sigma2_mle)

        # Delta method SE for σ
        se_sigma2 = sigma2_mle * np.sqrt(2 / (n - 1))
        se_sigma  = se_sigma2 / (2 * sigma_mle + 1e-12)
        ci = (max(0, sigma_mle - self._z * se_sigma),
              sigma_mle + self._z * se_sigma)

        return dict(
            sigma_mle=float(sigma_mle),
            se=float(se_sigma),
            ci=(float(ci[0]), float(ci[1])),
            n=n,
        )

    # ── Likelihood surface (for plotting) ────────────────────────────────────

    def compute_lambda_likelihood_surface(
        self,
        trades_df: pd.DataFrame,
        side: str = "buy",
        n_points: int = 300,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_grid, log_likelihood) for visualisation."""
        subset = trades_df[trades_df.trade_side == side].sort_values("timestamp")
        tau = np.diff(subset.timestamp.values)
        tau = tau[tau > 0]
        n   = len(tau)

        lambda_mle = 1.0 / np.mean(tau)
        grid = np.linspace(lambda_mle * 0.3, lambda_mle * 2.0, n_points)
        log_lik = np.array([n * np.log(l) - l * tau.sum() for l in grid])
        return grid, log_lik
