# Avellaneda-Stoikov Market Making Engine

A production-grade, end-to-end quantitative market making pipeline based on the **Avellaneda-Stoikov (2008)** stochastic optimal control framework.

```
┌─────────────────────────────────────────────────────────────┐
│              AS Market Making Pipeline                       │
│                                                             │
│  Raw Tick Data  →  MLE Calibration  →  AS Engine  →  Backtest│
│       ↓                  ↓               ↓            ↓     │
│  Synthetic Gen     λ, μ, σ, κ      Bid/Ask Quotes   Reports  │
└─────────────────────────────────────────────────────────────┘
```

## Features

| Module | Description |
|--------|-------------|
| `calibration/` | MLE estimation of Poisson order arrival intensity λ, fill rate κ, mid-price volatility σ |
| `models/` | Avellaneda-Stoikov closed-form reservation price & spread engine |
| `backtest/` | Tick-level event-driven backtest with P&L, inventory, adverse selection decomposition |
| `utils/` | Synthetic LOB data generator, metrics, and visualization |

## Theoretical Background

The AS model solves the HJB equation for a market maker who:
- Posts symmetric bid/ask quotes around a **reservation price**
- Manages **inventory risk** via a risk-aversion parameter γ
- Faces **Poisson order arrivals** with intensity λ and exponential fill probability

### Key Equations

**Reservation Price:**
```
r(s, q, t) = s - q · γ · σ² · (T - t)
```

**Optimal Spread:**
```
δ* = γ · σ² · (T - t) + (2/γ) · ln(1 + γ/κ)
```

**Fill Probability (exponential model):**
```
P(fill | δ) = exp(-κ · δ)
```

## Installation

```bash
git clone https://github.com/yourusername/avellaneda-stoikov-engine.git
cd avellaneda-stoikov-engine
pip install -e ".[dev]"
```

## Quick Start

```python
from src.calibration import OrderArrivalCalibrator
from src.models import AvellanedaStoikovModel
from src.backtest import MarketMakingBacktest

# 1. Calibrate from tick data
calibrator = OrderArrivalCalibrator()
params = calibrator.fit(tick_data)
print(params.summary())

# 2. Instantiate AS model
model = AvellanedaStoikovModel(
    gamma=0.1,      # risk aversion
    sigma=params.sigma,
    kappa=params.kappa,
    T=1.0
)

# 3. Run backtest
bt = MarketMakingBacktest(model=model, initial_cash=1_000_000)
results = bt.run(tick_data)
results.plot_dashboard()
```

## Running the Full Pipeline

```bash
# Generate synthetic data, calibrate, backtest, and produce reports
python run_pipeline.py --config config.yaml

# Or step by step:
python -m src.calibration.run_mle --data data/ticks.csv --output results/
python -m src.backtest.run --params results/calibrated_params.json
```

## Results Structure

```
results/
├── calibration_report.html     # MLE estimation with confidence intervals
├── backtest_report.html        # Full P&L, inventory, adverse selection
├── calibrated_params.json      # Serialized model parameters
└── figures/
    ├── mle_likelihood_surface.png
    ├── pnl_decomposition.png
    ├── inventory_path.png
    └── adverse_selection_breakdown.png
```

## Project Structure

```
avellaneda-stoikov-engine/
├── src/
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── mle_estimator.py        # Core MLE engine
│   │   ├── arrival_intensity.py    # Poisson λ model
│   │   └── run_mle.py              # CLI entry point
│   ├── models/
│   │   ├── __init__.py
│   │   ├── avellaneda_stoikov.py   # AS closed-form
│   │   └── fill_probability.py     # Exponential fill model
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py               # Event-driven backtest
│   │   ├── metrics.py              # P&L, Sharpe, drawdown
│   │   └── adverse_selection.py    # Toxicity decomposition
│   └── utils/
│       ├── data_generator.py       # Synthetic LOB generator
│       ├── visualization.py        # Dashboard plotting
│       └── report.py               # HTML report generation
├── tests/
│   ├── test_calibration.py
│   ├── test_model.py
│   └── test_backtest.py
├── notebooks/
│   └── 01_full_pipeline_walkthrough.ipynb
├── run_pipeline.py
├── config.yaml
├── setup.py
└── README.md
```

## References

- Avellaneda, M., & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3), 217-224.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

## License

MIT
