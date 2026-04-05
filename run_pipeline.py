#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end Avellaneda-Stoikov Market Making Pipeline

Usage:
    python run_pipeline.py                    # use defaults from config.yaml
    python run_pipeline.py --config my.yaml   # custom config
    python run_pipeline.py --gamma 0.05 --sigma 0.02
"""

import argparse
import os
import sys
import time
import yaml

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.data_generator   import generate_tick_data, SyntheticDataParams
from src.calibration             import OrderArrivalCalibrator
from src.models                  import AvellanedaStoikovModel
from src.backtest                import MarketMakingBacktest, BacktestConfig
from src.backtest.adverse_selection import decompose_adverse_selection
from src.utils.visualization     import (
    plot_mle_calibration,
    plot_backtest_dashboard,
    plot_adverse_selection,
)
from src.utils.report import generate_calibration_report, generate_backtest_report


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       Avellaneda-Stoikov Market Making Engine                ║
║       End-to-End Pipeline                                    ║
╚══════════════════════════════════════════════════════════════╝
"""


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="AS Market Making Pipeline")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--gamma",    type=float, default=None)
    parser.add_argument("--n-ticks",  type=int,   default=None)
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.gamma:
        cfg["model"]["gamma"] = args.gamma
    if args.n_ticks:
        cfg["data"]["n_ticks"] = args.n_ticks
    if args.seed:
        cfg["data"]["seed"] = args.seed

    os.makedirs(cfg["backtest"]["output_path"], exist_ok=True)
    os.makedirs(cfg["reporting"]["figures_path"], exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Generate / load data
    # ═══════════════════════════════════════════════════════════
    print("\n[Step 1/4] Generating synthetic tick data …")
    t0 = time.time()

    syn = cfg["synthetic_params"]
    data_params = SyntheticDataParams(
        mid_price_start = syn.get("mid_price_start", 100.0),
        sigma           = syn.get("sigma_true", 0.02),
        lambda_b        = syn.get("lambda_true", 5.0),
        lambda_a        = syn.get("lambda_true", 5.0),
        kappa           = syn.get("kappa_true", 1.5),
        dt              = syn.get("dt", 1.0),
        n_ticks         = cfg["data"]["n_ticks"],
        seed            = cfg["data"]["seed"],
    )
    quotes_df, trades_df = generate_tick_data(data_params)
    print(f"   → {len(quotes_df):,} quote ticks | {len(trades_df):,} trade events  [{time.time()-t0:.1f}s]")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: MLE Calibration
    # ═══════════════════════════════════════════════════════════
    print("\n[Step 2/4] Running MLE calibration …")
    t0 = time.time()

    cal_cfg = cfg["calibration"]
    calibrator = OrderArrivalCalibrator(
        confidence_level = cal_cfg.get("confidence_level", 0.95),
        bootstrap_samples = cal_cfg.get("bootstrap_samples", 1000),
    )
    params = calibrator.fit(quotes_df, trades_df, verbose=True)
    params.save(cal_cfg["output_path"])
    print(f"   → Calibration complete  [{time.time()-t0:.1f}s]")

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Build AS Model and run backtest
    # ═══════════════════════════════════════════════════════════
    print("\n[Step 3/4] Running market making backtest …")
    t0 = time.time()

    model_cfg = cfg["model"]
    model = AvellanedaStoikovModel(
        gamma         = model_cfg["gamma"],
        sigma         = params.sigma,
        kappa         = params.kappa,
        T             = model_cfg["T"],
        min_spread    = model_cfg["min_spread"],
        max_inventory = model_cfg["max_inventory"],
    )
    print(f"   Model: {model}")

    bt_cfg = cfg["backtest"]
    bt_config = BacktestConfig(
        initial_cash     = bt_cfg["initial_cash"],
        transaction_cost = bt_cfg["transaction_cost"],
        max_inventory    = model_cfg["max_inventory"],
        inventory_penalty = bt_cfg["inventory_penalty"],
    )

    bt = MarketMakingBacktest(model=model, config=bt_config)
    results = bt.run(quotes_df, trades_df, verbose=True)

    adverse = decompose_adverse_selection(results.ledger)
    print(adverse.summary())
    print(f"   → Backtest complete  [{time.time()-t0:.1f}s]")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Reports and figures
    # ═══════════════════════════════════════════════════════════
    print("\n[Step 4/4] Generating reports and figures …")
    t0 = time.time()

    fig_dir = cfg["reporting"]["figures_path"]
    figure_paths = []

    if not args.no_plots and cfg["reporting"]["produce_plots"]:
        cal_figs = plot_mle_calibration(calibrator, trades_df, params, output_dir=fig_dir)
        figure_paths.extend(cal_figs)

        bt_fig  = plot_backtest_dashboard(results.ledger, output_dir=fig_dir)
        adv_fig = plot_adverse_selection(results.ledger, adverse, output_dir=fig_dir)
        figure_paths.extend([bt_fig, adv_fig])

    if cfg["reporting"]["produce_html"]:
        cal_report = generate_calibration_report(
            params,
            figure_paths=[p for p in figure_paths if "mle" in p],
            output_path=os.path.join(bt_cfg["output_path"], "calibration_report.html"),
        )
        bt_report  = generate_backtest_report(
            results,
            adverse,
            figure_paths=[p for p in figure_paths if "backtest" in p or "adverse" in p],
            output_path=os.path.join(bt_cfg["output_path"], "backtest_report.html"),
        )

    print(f"   → Reports complete  [{time.time()-t0:.1f}s]")
    print("\n" + "═" * 62)
    print("  Pipeline complete. Results in:", bt_cfg["output_path"])
    print("═" * 62)


if __name__ == "__main__":
    main()
