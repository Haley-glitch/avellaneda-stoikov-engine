"""
Test Suite for Avellaneda-Stoikov Engine
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.utils.data_generator   import generate_tick_data, SyntheticDataParams
from src.calibration             import OrderArrivalCalibrator, CalibratedParams
from src.models                  import AvellanedaStoikovModel
from src.backtest                import MarketMakingBacktest, BacktestConfig
from src.backtest.adverse_selection import decompose_adverse_selection


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tick_data():
    params = SyntheticDataParams(n_ticks=2000, seed=42)
    return generate_tick_data(params)


@pytest.fixture(scope="module")
def calibrated_params(tick_data):
    quotes_df, trades_df = tick_data
    cal = OrderArrivalCalibrator(confidence_level=0.95)
    return cal.fit(quotes_df, trades_df, verbose=False), cal


@pytest.fixture(scope="module")
def as_model(calibrated_params):
    params, _ = calibrated_params
    return AvellanedaStoikovModel(
        gamma=0.1, sigma=params.sigma, kappa=params.kappa, T=1.0
    )


# ── Data generation tests ─────────────────────────────────────────────────────

class TestDataGenerator:
    def test_quotes_shape(self, tick_data):
        quotes_df, trades_df = tick_data
        assert len(quotes_df) == 2000
        assert "mid_price" in quotes_df.columns
        assert "timestamp" in quotes_df.columns

    def test_prices_positive(self, tick_data):
        quotes_df, _ = tick_data
        assert (quotes_df.mid_price > 0).all()

    def test_bid_ask_spread(self, tick_data):
        quotes_df, _ = tick_data
        assert (quotes_df.best_ask > quotes_df.best_bid).all()

    def test_trades_sides(self, tick_data):
        _, trades_df = tick_data
        assert set(trades_df.trade_side.unique()).issubset({"buy", "sell"})

    def test_trades_count(self, tick_data):
        _, trades_df = tick_data
        assert len(trades_df) > 0, "Should have some trades"


# ── Calibration tests ─────────────────────────────────────────────────────────

class TestCalibration:
    def test_lambda_positive(self, calibrated_params):
        params, _ = calibrated_params
        assert params.lambda_buy > 0
        assert params.lambda_sell > 0

    def test_sigma_positive(self, calibrated_params):
        params, _ = calibrated_params
        assert params.sigma > 0

    def test_kappa_positive(self, calibrated_params):
        params, _ = calibrated_params
        assert params.kappa > 0

    def test_confidence_intervals_ordered(self, calibrated_params):
        params, _ = calibrated_params
        assert params.lambda_buy_ci[0] <= params.lambda_buy
        assert params.lambda_buy <= params.lambda_buy_ci[1]
        assert params.lambda_sell_ci[0] <= params.lambda_sell
        assert params.lambda_sell <= params.lambda_sell_ci[1]

    def test_lambda_close_to_truth(self, calibrated_params):
        """λ̂ should be within 20% of true λ=5.0 for n=2000 ticks."""
        params, _ = calibrated_params
        assert abs(params.lambda_buy - 5.0) / 5.0 < 0.25, \
            f"λ_buy={params.lambda_buy:.3f} too far from truth=5.0"

    def test_params_serialization(self, calibrated_params, tmp_path):
        params, _ = calibrated_params
        path = str(tmp_path / "params.json")
        params.save(path)
        loaded = CalibratedParams.load(path)
        assert abs(loaded.lambda_buy - params.lambda_buy) < 1e-6
        assert abs(loaded.sigma - params.sigma) < 1e-8

    def test_summary_is_string(self, calibrated_params):
        params, _ = calibrated_params
        s = params.summary()
        assert isinstance(s, str)
        assert "λ_buy" in s

    def test_likelihood_surface_shape(self, calibrated_params, tick_data):
        params, cal = calibrated_params
        _, trades_df = tick_data
        grid, loglik = cal.compute_lambda_likelihood_surface(trades_df, side="buy")
        assert len(grid) == len(loglik)
        assert len(grid) > 0
        # MLE should be near the maximum
        mle_idx = np.argmax(loglik)
        assert abs(grid[mle_idx] - params.lambda_buy) / params.lambda_buy < 0.05


# ── AS Model tests ────────────────────────────────────────────────────────────

class TestASModel:
    def test_reservation_price_zero_inventory(self, as_model):
        r = as_model.reservation_price(mid=100.0, q=0, t=0.0)
        assert r == pytest.approx(100.0)

    def test_reservation_price_positive_inventory(self, as_model):
        r = as_model.reservation_price(mid=100.0, q=5, t=0.0)
        assert r < 100.0, "Positive inventory should lower reservation price"

    def test_reservation_price_negative_inventory(self, as_model):
        r = as_model.reservation_price(mid=100.0, q=-5, t=0.0)
        assert r > 100.0, "Negative inventory should raise reservation price"

    def test_spread_decreases_with_time(self, as_model):
        delta_early = as_model.optimal_half_spread(t=0.0)
        delta_late  = as_model.optimal_half_spread(t=0.9)
        assert delta_early > delta_late, "Spread should shrink as horizon approaches"

    def test_spread_positive(self, as_model):
        for t in np.linspace(0, 0.99, 20):
            assert as_model.optimal_half_spread(t) > 0

    def test_quote_bid_lt_ask(self, as_model):
        for q in [-10, 0, 10]:
            quote = as_model.quote(mid=100.0, q=q, t=0.5)
            assert quote.bid < quote.ask

    def test_fill_probability_unit_interval(self, as_model):
        for delta in [0.0, 0.01, 0.1, 1.0, 10.0]:
            p = as_model.fill_probability(delta)
            assert 0.0 <= p <= 1.0

    def test_fill_probability_decreases_with_depth(self, as_model):
        p1 = as_model.fill_probability(0.01)
        p2 = as_model.fill_probability(0.10)
        assert p1 > p2

    def test_quote_namedtuple_fields(self, as_model):
        quote = as_model.quote(100.0, 0, 0.5)
        assert hasattr(quote, "bid")
        assert hasattr(quote, "ask")
        assert hasattr(quote, "reservation")
        assert hasattr(quote, "half_spread")


# ── Backtest tests ────────────────────────────────────────────────────────────

class TestBacktest:
    def _run(self, as_model, tick_data):
        quotes_df, trades_df = tick_data
        cfg = BacktestConfig(initial_cash=1_000_000, max_inventory=50)
        bt  = MarketMakingBacktest(model=as_model, config=cfg)
        return bt.run(quotes_df, trades_df, verbose=False)

    def test_backtest_runs(self, as_model, tick_data):
        results = self._run(as_model, tick_data)
        assert results is not None

    def test_ledger_columns(self, as_model, tick_data):
        results = self._run(as_model, tick_data)
        for col in ["timestamp", "mid_price", "inventory", "cash", "total_pnl", "fill_event"]:
            assert col in results.ledger.columns, f"Missing column: {col}"

    def test_no_nan_in_ledger(self, as_model, tick_data):
        results = self._run(as_model, tick_data)
        nans = results.ledger[["inventory", "cash", "total_pnl"]].isna().sum().sum()
        assert nans == 0, f"NaN values found in ledger: {nans}"

    def test_inventory_bounded(self, as_model, tick_data):
        results = self._run(as_model, tick_data)
        max_inv = results.ledger["inventory"].abs().max()
        assert max_inv <= 55, f"Inventory {max_inv} exceeds hard limit"

    def test_sharpe_finite(self, as_model, tick_data):
        results = self._run(as_model, tick_data)
        assert np.isfinite(results.sharpe)

    def test_adverse_selection_report(self, as_model, tick_data):
        results = self._run(as_model, tick_data)
        adv = decompose_adverse_selection(results.ledger)
        assert 0.0 <= adv.fill_toxicity <= 1.0
        assert isinstance(adv.summary(), str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
