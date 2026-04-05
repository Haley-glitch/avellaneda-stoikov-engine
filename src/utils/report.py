"""
HTML Report Generator
=====================
Produces standalone self-contained HTML reports for:
  - MLE calibration summary
  - Backtest results with P&L, inventory, adverse selection
"""

from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path


def _img_to_b64(path: str) -> str:
    """Embed image as base64 for self-contained HTML."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_calibration_report(
    params,
    figure_paths: list[str],
    output_path: str = "results/calibration_report.html",
) -> str:
    """Build MLE calibration HTML report."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    imgs_html = ""
    for p in figure_paths:
        if os.path.exists(p):
            b64 = _img_to_b64(p)
            imgs_html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:10px 0;">\n'

    ci_buy  = params.lambda_buy_ci
    ci_sell = params.lambda_sell_ci

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AS Engine — MLE Calibration Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --green: #3fb950;
    --red: #f85149; --blue: #58a6ff; --yellow: #d29922; --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Courier New', monospace; padding: 32px; }}
  h1 {{ font-size: 1.6rem; color: var(--blue); margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; color: var(--purple); margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 32px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ background: #21262d; color: var(--muted); text-align: left; padding: 8px 12px; }}
  td {{ padding: 8px 12px; border-top: 1px solid var(--border); }}
  .val {{ color: var(--green); font-weight: bold; }}
  .ci  {{ color: var(--yellow); font-size: 0.82rem; }}
  .bad {{ color: var(--red); }}
  .formula {{ background: #21262d; border-left: 3px solid var(--blue); padding: 12px 16px;
              font-family: monospace; font-size: 0.9rem; margin: 10px 0; color: var(--text); border-radius: 0 6px 6px 0; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }}
  .metric-box {{ background: #21262d; border: 1px solid var(--border); border-radius: 6px; padding: 14px; text-align: center; }}
  .metric-label {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }}
  .metric-value {{ font-size: 1.3rem; color: var(--blue); font-weight: bold; }}
  img {{ border-radius: 8px; border: 1px solid var(--border); }}
</style>
</head>
<body>
  <h1>Avellaneda-Stoikov Engine</h1>
  <div class="subtitle">MLE Calibration Report · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

  <div class="card">
    <h2>Model Overview</h2>
    <div class="formula">P(fill | δ) = exp(−κ · δ)</div>
    <div class="formula">r(s,q,t)   = s − q · γ · σ² · (T−t)          [reservation price]</div>
    <div class="formula">δ*(t)      = γ·σ²·(T−t) + (2/γ)·ln(1+γ/κ)   [optimal half-spread]</div>
  </div>

  <div class="card">
    <h2>Calibrated Parameters</h2>
    <div class="metric-grid">
      <div class="metric-box">
        <div class="metric-label">λ_buy (arrivals/s)</div>
        <div class="metric-value">{params.lambda_buy:.4f}</div>
        <div class="ci">±{params.lambda_buy_se:.4f} | CI [{ci_buy[0]:.3f}, {ci_buy[1]:.3f}]</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">λ_sell (arrivals/s)</div>
        <div class="metric-value">{params.lambda_sell:.4f}</div>
        <div class="ci">±{params.lambda_sell_se:.4f} | CI [{ci_sell[0]:.3f}, {ci_sell[1]:.3f}]</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">κ (fill-rate decay)</div>
        <div class="metric-value">{params.kappa:.4f}</div>
        <div class="ci">±{params.kappa_se:.4f} | CI [{params.kappa_ci[0]:.3f}, {params.kappa_ci[1]:.3f}]</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">σ (mid-price vol)</div>
        <div class="metric-value">{params.sigma:.5f}</div>
        <div class="ci">±{params.sigma_se:.5f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">n_buy observations</div>
        <div class="metric-value">{params.n_buy:,}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">n_sell observations</div>
        <div class="metric-value">{params.n_sell:,}</div>
      </div>
    </div>
    <h2>Goodness of Fit (KS Test)</h2>
    <table>
      <tr><th>Side</th><th>KS Statistic</th><th>p-value</th><th>Result</th></tr>
      <tr>
        <td>BUY</td>
        <td class="val">{params.ks_stat_buy:.4f}</td>
        <td class="val">{params.ks_pvalue_buy:.4f}</td>
        <td class="{'val' if params.ks_pvalue_buy > 0.05 else 'bad'}">
          {'✓ Cannot reject Exp(λ)' if params.ks_pvalue_buy > 0.05 else '✗ Reject Exp(λ)'}
        </td>
      </tr>
      <tr>
        <td>SELL</td>
        <td class="val">{params.ks_stat_sell:.4f}</td>
        <td class="val">{params.ks_pvalue_sell:.4f}</td>
        <td class="{'val' if params.ks_pvalue_sell > 0.05 else 'bad'}">
          {'✓ Cannot reject Exp(λ)' if params.ks_pvalue_sell > 0.05 else '✗ Reject Exp(λ)'}
        </td>
      </tr>
    </table>
  </div>

  <div class="card">
    <h2>Likelihood Surfaces & Inter-arrival Distributions</h2>
    {imgs_html}
  </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"[Report] Calibration report → {output_path}")
    return output_path


def generate_backtest_report(
    bt_results,
    adverse_report,
    figure_paths: list[str],
    output_path: str = "results/backtest_report.html",
) -> str:
    """Build backtest HTML report."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    r   = bt_results
    adv = adverse_report

    imgs_html = ""
    for p in figure_paths:
        if os.path.exists(p):
            b64 = _img_to_b64(p)
            imgs_html += f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:10px 0;">\n'

    pnl_color  = "var(--green)" if r.total_pnl >= 0 else "var(--red)"
    sharpe_color = "var(--green)" if r.sharpe >= 1.0 else "var(--yellow)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AS Engine — Backtest Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --green: #3fb950;
    --red: #f85149; --blue: #58a6ff; --yellow: #d29922; --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Courier New', monospace; padding: 32px; }}
  h1 {{ font-size: 1.6rem; color: var(--blue); margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; color: var(--purple); margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 32px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th {{ background: #21262d; color: var(--muted); text-align: left; padding: 8px 12px; }}
  td {{ padding: 8px 12px; border-top: 1px solid var(--border); }}
  .val  {{ color: var(--green); font-weight: bold; }}
  .neg  {{ color: var(--red);   font-weight: bold; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
  .metric-box  {{ background: #21262d; border: 1px solid var(--border); border-radius: 6px; padding: 14px; text-align: center; }}
  .metric-label {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; }}
  .metric-value {{ font-size: 1.25rem; font-weight: bold; }}
  img {{ border-radius: 8px; border: 1px solid var(--border); }}
  .pnl-bar {{ display: flex; gap: 4px; height: 18px; border-radius: 4px; overflow: hidden; margin: 8px 0; }}
  .pnl-segment {{ height: 100%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; }}
</style>
</head>
<body>
  <h1>Avellaneda-Stoikov Engine</h1>
  <div class="subtitle">Backtest Report · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

  <div class="card">
    <h2>Performance Summary</h2>
    <div class="metric-grid">
      <div class="metric-box">
        <div class="metric-label">Total P&L</div>
        <div class="metric-value" style="color:{pnl_color}">${r.total_pnl:,.2f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Sharpe Ratio</div>
        <div class="metric-value" style="color:{sharpe_color}">{r.sharpe:.3f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-value" style="color:var(--red)">${r.max_drawdown:,.2f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Total Fills</div>
        <div class="metric-value" style="color:var(--blue)">{r.total_fills:,}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Total Volume</div>
        <div class="metric-value" style="color:var(--text)">{r.total_volume:,.0f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Max |Inventory|</div>
        <div class="metric-value" style="color:var(--yellow)">{r.max_inventory}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Avg |Inventory|</div>
        <div class="metric-value" style="color:var(--yellow)">{r.avg_inventory:.1f}</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">% Time Flat</div>
        <div class="metric-value" style="color:var(--teal, #39d353)">{r.pct_time_flat:.1%}</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>P&L Decomposition</h2>
    <table>
      <tr><th>Component</th><th>Cumulative P&L</th><th>% of |Total|</th><th>Interpretation</th></tr>
      <tr>
        <td>Spread Capture</td>
        <td class="val">${adv.spread_pnl:,.2f}</td>
        <td class="val">{abs(adv.spread_pnl) / (abs(adv.total_pnl)+1e-8):.1%}</td>
        <td>Profit from bid-ask spread on fills</td>
      </tr>
      <tr>
        <td>Inventory P&L</td>
        <td class="{'val' if adv.inventory_pnl >= 0 else 'neg'}">${adv.inventory_pnl:,.2f}</td>
        <td>{abs(adv.inventory_pnl) / (abs(adv.total_pnl)+1e-8):.1%}</td>
        <td>Mid-price drift while holding position</td>
      </tr>
      <tr>
        <td>Adverse Selection</td>
        <td class="{'val' if adv.adverse_pnl >= 0 else 'neg'}">${adv.adverse_pnl:,.2f}</td>
        <td>{abs(adv.adverse_pnl) / (abs(adv.total_pnl)+1e-8):.1%}</td>
        <td>Cost of trading against informed flow</td>
      </tr>
    </table>
  </div>

  <div class="card">
    <h2>Adverse Selection Metrics</h2>
    <table>
      <tr><th>Metric</th><th>Value</th><th>Notes</th></tr>
      <tr><td>Fill Toxicity (overall)</td><td class="val">{adv.fill_toxicity:.1%}</td>
          <td>Fraction of fills followed by adverse mid move</td></tr>
      <tr><td>Buy Fill Toxicity</td><td class="val">{adv.buy_toxicity:.1%}</td>
          <td>n = {adv.n_buy_fills:,} fills</td></tr>
      <tr><td>Sell Fill Toxicity</td><td class="val">{adv.sell_toxicity:.1%}</td>
          <td>n = {adv.n_sell_fills:,} fills</td></tr>
      <tr><td>Avg Adverse Move</td><td class="val">{adv.avg_adverse_move:.6f}</td>
          <td>Mean mid-price move after toxic fill</td></tr>
      <tr><td>VPIN Proxy</td><td class="val">{adv.vpin_proxy:.4f}</td>
          <td>Volume imbalance |V_buy - V_sell| / V_total</td></tr>
      <tr><td>Roll Implied Spread</td><td class="val">{adv.roll_spread:.6f}</td>
          <td>2√(−Cov(ΔP_t, ΔP_{{t+1}}))</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Charts</h2>
    {imgs_html}
  </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"[Report] Backtest report → {output_path}")
    return output_path
