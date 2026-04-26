# Avellaneda-Stoikov Market Making Engine

> A production-grade quantitative market making pipeline built on stochastic optimal control theory.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 What Is This?

This project implements the **Avellaneda-Stoikov (2008)** market making model — a mathematically rigorous framework used by quantitative trading desks to optimally quote bid/ask prices in limit order books.

A **market maker** profits by continuously posting both buy and sell quotes, earning the spread while managing the risk of holding too much inventory. The AS model solves this as a **stochastic optimal control problem**: at every moment, the market maker balances:

- 📉 **Inventory risk** — holding too much of an asset exposes you to adverse price moves
- 📊 **Fill probability** — quoting too wide means fewer trades; quoting too tight means less profit per trade
- ⏱️ **Time horizon** — risk aversion grows as a session end approaches

The result is a closed-form strategy that adapts quotes in real time based on current inventory and market conditions.

---

## 🔢 The Math

The model derives optimal quotes from a **Hamilton-Jacobi-Bellman (HJB)** equation. The two key outputs are:

**Reservation Price** — the market maker's adjusted fair value given inventory $q$:

$$r(s, q, t) = s - q \cdot \gamma \cdot \sigma^2 \cdot (T - t)$$

**Optimal Spread** — how wide to quote around the reservation price:

$$\delta^* = \gamma \cdot \sigma^2 \cdot (T - t) + \frac{2}{\gamma} \ln\!\left(1 + \frac{\gamma}{\kappa}\right)$$

**Fill Probability** — likelihood an order at distance $\delta$ gets hit:

$$P(\text{fill} \mid \delta) = e^{-\kappa \delta}$$

| Symbol | Meaning |
|--------|---------|
| $s$ | Current mid-price |
| $q$ | Current inventory |
| $\gamma$ | Risk-aversion parameter |
| $\sigma$ | Mid-price volatility |
| $\kappa$ | Order book depth / fill rate |
| $T - t$ | Time remaining in session |

---

## 🗺️ Pipeline Overview

```
Raw Tick Data
     │
     ▼
┌─────────────────────┐
│  MLE Calibration    │  ← Estimates λ (arrival intensity),
│  calibration/       │    κ (fill rate), σ (volatility)
└────────┬────────────┘
         │  calibrated params
         ▼
┌─────────────────────┐
│  AS Engine          │  ← Computes reservation price
│  models/            │    and optimal spread at each tick
└────────┬────────────┘
         │  bid/ask quotes
         ▼
┌─────────────────────┐
│  Event-Driven       │  ← Simulates order fills, tracks
│  Backtest           │    P&L, inventory, adverse selection
│  backtest/          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  HTML Reports +     │  ← P&L decomposition, MLE surfaces,
│  Visualizations     │    inventory paths, toxicity analysis
└─────────────────────┘
```

---

## ✨ Key Features

| Module | What It Does |
|--------|-------------|
| `calibration/` | MLE estimation of Poisson order arrival intensity λ, fill rate κ, mid-price volatility σ — with confidence intervals |
| `models/` | Closed-form AS reservation price and optimal spread, updated tick-by-tick |
| `backtest/` | Event-driven simulation with full P&L, inventory tracking, and adverse selection decomposition |
| `utils/` | Synthetic limit order book data generator, metrics library, and HTML report builder |

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/avellaneda-stoikov-engine.git
cd avellaneda-stoikov-engine
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, NumPy, SciPy, pandas, matplotlib

---

## ⚡ Quick Start

```python
from src.calibration import OrderArrivalCalibrator
from src.models import AvellanedaStoikovModel
from src.backtest import MarketMakingBacktest

# 1. Calibrate parameters from tick data
calibrator = OrderArrivalCalibrator()
params = calibrator.fit(tick_data)
print(params.summary())
# Output: λ=4.32/s, κ=1.87, σ=0.0023, log-likelihood=-1204.3

# 2. Build the AS model
model = AvellanedaStoikovModel(
    gamma=0.1,           # risk aversion
    sigma=params.sigma,
    kappa=params.kappa,
    T=1.0                # 1-day session
)

# 3. Backtest and visualize
bt = MarketMakingBacktest(model=model, initial_cash=1_000_000)
results = bt.run(tick_data)
results.plot_dashboard()
```

### Run the Full Pipeline

```bash
# One command: generate data → calibrate → backtest → reports
python run_pipeline.py --config config.yaml

# Or step by step:
python -m src.calibration.run_mle --data data/ticks.csv --output results/
python -m src.backtest.run --params results/calibrated_params.json
```

---

## 📁 Project Structure

```
avellaneda-stoikov-engine/
├── src/
│   ├── calibration/
│   │   ├── mle_estimator.py        # Core MLE engine
│   │   ├── arrival_intensity.py    # Poisson λ estimation
│   │   └── run_mle.py              # CLI entry point
│   ├── models/
│   │   ├── avellaneda_stoikov.py   # Closed-form AS model
│   │   └── fill_probability.py     # Exponential fill model
│   ├── backtest/
│   │   ├── engine.py               # Event-driven backtest loop
│   │   ├── metrics.py              # Sharpe, drawdown, P&L
│   │   └── adverse_selection.py    # Order toxicity decomposition
│   └── utils/
│       ├── data_generator.py       # Synthetic LOB generator
│       └── visualization.py        # Dashboard + HTML reports
├── tests/
├── notebooks/
│   └── 01_full_pipeline_walkthrough.ipynb
├── run_pipeline.py
├── config.yaml
└── README.md
```

---

## 📊 Sample Output

After running the backtest, the pipeline generates:

- **`calibration_report.html`** — MLE likelihood surface, parameter estimates with 95% confidence intervals
- **`backtest_report.html`** — Cumulative P&L, inventory path, spread over time
- **`pnl_decomposition.png`** — Spread income vs. inventory cost vs. adverse selection loss
- **`adverse_selection_breakdown.png`** — Toxicity analysis of filled orders

---

## 🧠 What I Learned / Reflection

This project pushed me to connect graduate-level stochastic control theory to practical implementation details. The most challenging part was the **MLE calibration** — fitting Poisson arrival intensities to real (noisy) tick data required careful handling of time bucketing and numerical stability in the log-likelihood computation. Implementing the **adverse selection decomposition** in the backtest was also non-trivial: separating P&L into spread income, inventory cost, and toxicity required careful bookkeeping at every fill event.

The project deepened my understanding of how market microstructure models translate into live trading logic and reinforced the importance of rigorous backtesting to validate theoretical models against realistic data.

---

## 📚 References

- Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance, 8(3), 217–224.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
