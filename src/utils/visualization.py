"""
Visualization Module
====================
Produces all publication-quality figures for the AS engine pipeline.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Optional

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "border":     "#30363d",
    "text":       "#e6edf3",
    "muted":      "#8b949e",
    "green":      "#3fb950",
    "red":        "#f85149",
    "blue":       "#58a6ff",
    "yellow":     "#d29922",
    "purple":     "#bc8cff",
    "teal":       "#39d353",
}

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
        "axes.titlecolor":   PALETTE["text"],
    })

_apply_dark_style()


# ──────────────────────────────────────────────────────────────────────────────
#  Calibration plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_mle_calibration(
    calibrator,
    trades_df: pd.DataFrame,
    params,
    output_dir: str = "results/figures",
) -> list[str]:
    """Plot likelihood surface and inter-arrival diagnostics."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for side in ("buy", "sell"):
        grid, log_lik = calibrator.compute_lambda_likelihood_surface(trades_df, side=side)
        lam_hat = params.lambda_buy if side == "buy" else params.lambda_sell
        lam_se  = params.lambda_buy_se if side == "buy" else params.lambda_sell_se

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(PALETTE["bg"])

        # ── Log-likelihood surface
        ax = axes[0]
        ax.plot(grid, log_lik, color=PALETTE["blue"], linewidth=2)
        ax.axvline(lam_hat, color=PALETTE["green"], linestyle="--", linewidth=1.5,
                   label=f"λ̂ = {lam_hat:.4f}")
        ax.axvline(lam_hat - 1.96*lam_se, color=PALETTE["yellow"], linestyle=":", linewidth=1,
                   label=f"95% CI: [{lam_hat-1.96*lam_se:.3f}, {lam_hat+1.96*lam_se:.3f}]")
        ax.axvline(lam_hat + 1.96*lam_se, color=PALETTE["yellow"], linestyle=":", linewidth=1)
        ax.set_xlabel("λ (arrival intensity)")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"MLE Likelihood Surface — {side.upper()} orders")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Inter-arrival time histogram vs fitted Exp(λ̂)
        ax = axes[1]
        subset = trades_df[trades_df.trade_side == side].sort_values("timestamp")
        tau = np.diff(subset.timestamp.values)
        tau = tau[tau > 0]
        ax.hist(tau, bins=50, density=True, color=PALETTE["blue"], alpha=0.6,
                label="Observed inter-arrivals")
        x = np.linspace(0, np.percentile(tau, 99), 300)
        ax.plot(x, lam_hat * np.exp(-lam_hat * x), color=PALETTE["green"], linewidth=2,
                label=f"Exp(λ={lam_hat:.3f})")
        ax.set_xlabel("Inter-arrival time (s)")
        ax.set_ylabel("Density")
        ax.set_title(f"Inter-arrival Fit — {side.upper()} orders")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"mle_calibration_{side}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig)
        paths.append(path)
        print(f"[Plot] Saved → {path}")

    return paths


# ──────────────────────────────────────────────────────────────────────────────
#  Backtest dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_backtest_dashboard(
    ledger: pd.DataFrame,
    output_dir: str = "results/figures",
) -> str:
    """Full 4-panel backtest dashboard."""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ts = ledger["timestamp"].values

    # ── Panel 1: Mid-price + quotes ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ts, ledger["mid_price"],    color=PALETTE["text"],  linewidth=0.8, alpha=0.7, label="Mid price")
    ax1.plot(ts, ledger["bid_quote"],    color=PALETTE["green"], linewidth=0.6, alpha=0.5, label="Bid quote")
    ax1.plot(ts, ledger["ask_quote"],    color=PALETTE["red"],   linewidth=0.6, alpha=0.5, label="Ask quote")
    ax1.plot(ts, ledger["reservation"],  color=PALETTE["purple"],linewidth=0.8, linestyle="--", alpha=0.7, label="Reservation price")
    fills = ledger[ledger["fill_event"] == 1]
    buy_fills  = fills[fills["fill_side"] == "buy"]
    sell_fills = fills[fills["fill_side"] == "sell"]
    ax1.scatter(buy_fills["timestamp"],  buy_fills["fill_price"],
                marker="^", color=PALETTE["green"], s=12, zorder=5, alpha=0.7, label="Buy fill")
    ax1.scatter(sell_fills["timestamp"], sell_fills["fill_price"],
                marker="v", color=PALETTE["red"],   s=12, zorder=5, alpha=0.7, label="Sell fill")
    ax1.set_title("Mid-price vs AS Quotes (Bid/Ask/Reservation)", fontsize=11)
    ax1.legend(fontsize=7, ncol=6, loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.set_ylabel("Price")

    # ── Panel 2: P&L ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ts, ledger["total_pnl"],    color=PALETTE["blue"],   linewidth=1.5, label="Total P&L")
    ax2.plot(ts, ledger["realized_pnl"], color=PALETTE["green"],  linewidth=1,   linestyle="--", label="Realized P&L", alpha=0.8)
    ax2.axhline(0, color=PALETTE["muted"], linewidth=0.5, linestyle=":")
    ax2.fill_between(ts, ledger["total_pnl"], 0,
                     where=ledger["total_pnl"] > 0, alpha=0.15, color=PALETTE["green"])
    ax2.fill_between(ts, ledger["total_pnl"], 0,
                     where=ledger["total_pnl"] < 0, alpha=0.15, color=PALETTE["red"])
    ax2.set_title("Cumulative P&L", fontsize=10)
    ax2.set_ylabel("P&L ($)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 3: Inventory ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    inv = ledger["inventory"].values
    colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in inv]
    ax3.fill_between(ts, inv, 0, alpha=0.4,
                     where=np.array(inv) >= 0, color=PALETTE["green"])
    ax3.fill_between(ts, inv, 0, alpha=0.4,
                     where=np.array(inv) < 0,  color=PALETTE["red"])
    ax3.plot(ts, inv, color=PALETTE["text"], linewidth=0.8, alpha=0.7)
    ax3.axhline(0, color=PALETTE["muted"], linewidth=0.5, linestyle=":")
    ax3.set_title("Inventory Path", fontsize=10)
    ax3.set_ylabel("Inventory (units)")
    ax3.grid(True, alpha=0.2)

    # ── Panel 4: P&L decomposition ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(ts, ledger["spread_pnl"],    color=PALETTE["blue"],   linewidth=1.2, label="Spread capture")
    ax4.plot(ts, ledger["inventory_pnl"], color=PALETTE["yellow"], linewidth=1.2, label="Inventory P&L")
    ax4.plot(ts, ledger["adverse_pnl"],   color=PALETTE["red"],    linewidth=1.2, label="Adverse selection")
    ax4.axhline(0, color=PALETTE["muted"], linewidth=0.5, linestyle=":")
    ax4.set_title("P&L Decomposition", fontsize=10)
    ax4.set_ylabel("Cumulative P&L ($)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Panel 5: Spread distribution ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    spreads = (ledger["ask_quote"] - ledger["bid_quote"]).values
    ax5.hist(spreads, bins=60, color=PALETTE["purple"], alpha=0.7, density=True)
    ax5.axvline(np.mean(spreads), color=PALETTE["yellow"], linestyle="--",
                linewidth=1.5, label=f"Mean: {np.mean(spreads):.4f}")
    ax5.axvline(np.median(spreads), color=PALETTE["teal"], linestyle=":",
                linewidth=1.5, label=f"Median: {np.median(spreads):.4f}")
    ax5.set_title("Quoted Spread Distribution", fontsize=10)
    ax5.set_xlabel("Spread")
    ax5.set_ylabel("Density")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)

    path = os.path.join(output_dir, "backtest_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"[Plot] Saved → {path}")
    return path


def plot_adverse_selection(
    ledger: pd.DataFrame,
    adverse_report,
    output_dir: str = "results/figures",
) -> str:
    """Adverse selection breakdown figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    # ── P&L attribution bar ───────────────────────────────────────────────
    ax = axes[0]
    components = {
        "Spread\ncapture": adverse_report.spread_pnl,
        "Inventory\nP&L":  adverse_report.inventory_pnl,
        "Adverse\nsel.":   adverse_report.adverse_pnl,
    }
    colors = [PALETTE["green"] if v >= 0 else PALETTE["red"] for v in components.values()]
    bars = ax.bar(list(components.keys()), list(components.values()), color=colors, alpha=0.8, width=0.5)
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.5)
    ax.set_title("P&L Attribution", fontsize=10)
    ax.set_ylabel("P&L ($)")
    for bar, val in zip(bars, components.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"${val:,.0f}", ha="center", va="bottom", fontsize=8, color=PALETTE["text"])
    ax.grid(True, alpha=0.2, axis="y")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Toxicity by side ──────────────────────────────────────────────────
    ax = axes[1]
    sides  = ["Buy fills", "Sell fills", "Overall"]
    toxics = [adverse_report.buy_toxicity, adverse_report.sell_toxicity, adverse_report.fill_toxicity]
    bar_colors = [PALETTE["green"], PALETTE["red"], PALETTE["yellow"]]
    ax.bar(sides, toxics, color=bar_colors, alpha=0.8, width=0.5)
    ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", linewidth=1, label="50% baseline")
    ax.set_ylim(0, 1)
    ax.set_title("Fill Toxicity", fontsize=10)
    ax.set_ylabel("Fraction of toxic fills")
    for i, (side, tox) in enumerate(zip(sides, toxics)):
        ax.text(i, tox + 0.02, f"{tox:.1%}", ha="center", fontsize=9, color=PALETTE["text"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Cumulative adverse P&L ────────────────────────────────────────────
    ax = axes[2]
    ts = ledger["timestamp"].values
    ax.plot(ts, ledger["adverse_pnl"], color=PALETTE["red"], linewidth=1.5, label="Cumulative adverse P&L")
    ax.fill_between(ts, ledger["adverse_pnl"], 0, alpha=0.2, color=PALETTE["red"])
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.5, linestyle=":")
    ax.set_title("Adverse Selection Cost (cumulative)", fontsize=10)
    ax.set_ylabel("P&L ($)")
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    path = os.path.join(output_dir, "adverse_selection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"[Plot] Saved → {path}")
    return path
