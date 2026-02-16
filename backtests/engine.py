"""
Shared backtest engine for all ONDS research strategies.

Every strategy generates a daily signal in {-1, 0, +1} (or continuous [-1, +1]).
This engine converts signals into PnL, computes performance metrics, and plots results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKTEST, FIGURES_DIR, REPORTS_DIR


def compute_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns."""
    return prices.pct_change().dropna()


def signal_to_positions(signal: pd.Series, lag: int = 1) -> pd.Series:
    """
    Convert raw signal to tradeable positions, lagged by `lag` days
    to avoid look-ahead bias.  Signal on day t → position on day t+lag.
    """
    return signal.shift(lag).fillna(0)


def backtest(
    prices: pd.Series,
    signal: pd.Series,
    name: str = "Strategy",
    lag: int = 1,
    commission_bps: float = None,
    risk_free_rate: float = None,
    plot: bool = True,
    save_fig: bool = True,
) -> dict:
    """
    Run a simple long/short backtest.

    Parameters
    ----------
    prices : pd.Series
        Daily close prices (DatetimeIndex).
    signal : pd.Series
        Daily signal, same index as prices. Values in [-1, +1].
        +1 = long, -1 = short, 0 = flat.
    name : str
        Strategy name for labels.
    lag : int
        Number of days to lag signal (default 1 = next-day execution).
    commission_bps : float
        Round-trip transaction cost in basis points.
    risk_free_rate : float
        Annualized risk-free rate for Sharpe computation.

    Returns
    -------
    dict with keys: sharpe, annual_return, annual_vol, max_drawdown,
                    hit_rate, n_trades, cum_return, daily_pnl (Series)
    """
    if commission_bps is None:
        commission_bps = BACKTEST["commission_bps"]
    if risk_free_rate is None:
        risk_free_rate = BACKTEST["risk_free_rate"]

    # Align
    common = prices.index.intersection(signal.index)
    prices = prices.loc[common].sort_index()
    signal = signal.loc[common].sort_index()

    ret = compute_returns(prices)
    pos = signal_to_positions(signal, lag=lag)

    # Align after shift
    common2 = ret.index.intersection(pos.index)
    ret = ret.loc[common2]
    pos = pos.loc[common2]

    # Strategy returns (before costs)
    strat_ret = pos * ret

    # Transaction costs: charge on position changes
    turnover = pos.diff().abs()
    cost = turnover * (commission_bps / 10_000)
    strat_ret_net = strat_ret - cost

    # Cumulative
    cum_ret = (1 + strat_ret_net).cumprod()
    cum_bh = (1 + ret).cumprod()

    # Metrics
    n_days = len(strat_ret_net)
    annual_factor = 252
    ann_ret = (cum_ret.iloc[-1]) ** (annual_factor / n_days) - 1
    ann_vol = strat_ret_net.std() * np.sqrt(annual_factor)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()

    # Hit rate (% of days with positive return when positioned)
    active = strat_ret_net[pos != 0]
    hit_rate = (active > 0).mean() if len(active) > 0 else 0

    # Number of trades (position changes)
    n_trades = (pos.diff().abs() > 0).sum()

    results = {
        "name":          name,
        "sharpe":        round(sharpe, 4),
        "annual_return": round(ann_ret, 4),
        "annual_vol":    round(ann_vol, 4),
        "max_drawdown":  round(max_dd, 4),
        "hit_rate":      round(hit_rate, 4),
        "n_trades":      int(n_trades),
        "total_return":  round(cum_ret.iloc[-1] - 1, 4),
        "n_days":        n_days,
        "daily_pnl":     strat_ret_net,
        "cum_return":    cum_ret,
        "cum_benchmark": cum_bh,
    }

    if plot:
        _plot_backtest(results, save=save_fig)

    return results


def _plot_backtest(results: dict, save: bool = True):
    """Plot cumulative return chart with drawdown subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    height_ratios=[3, 1], sharex=True)
    name = results["name"]

    # Cumulative returns
    ax1.plot(results["cum_return"], label=f"{name} (Sharpe={results['sharpe']:.2f})",
             linewidth=1.5, color="#2196F3")
    ax1.plot(results["cum_benchmark"], label="Buy & Hold",
             linewidth=1, color="#9E9E9E", alpha=0.7)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title(f"{name} — Backtest Results")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    peak = results["cum_return"].cummax()
    dd = (results["cum_return"] - peak) / peak
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.4, color="#F44336")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()

    if save:
        slug = name.lower().replace(" ", "_").replace("/", "_")
        path = FIGURES_DIR / f"bt_{slug}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def print_results(results: dict):
    """Pretty-print backtest results."""
    r = results
    print(f"\n{'='*60}")
    print(f"  {r['name']} — Backtest Summary")
    print(f"{'='*60}")
    print(f"  Sharpe Ratio:    {r['sharpe']:+.4f}")
    print(f"  Annual Return:   {r['annual_return']:+.2%}")
    print(f"  Annual Vol:      {r['annual_vol']:.2%}")
    print(f"  Max Drawdown:    {r['max_drawdown']:.2%}")
    print(f"  Hit Rate:        {r['hit_rate']:.2%}")
    print(f"  Total Return:    {r['total_return']:+.2%}")
    print(f"  # Trades:        {r['n_trades']}")
    print(f"  # Trading Days:  {r['n_days']}")
    print(f"{'='*60}\n")


def compare_strategies(results_list: list, save: bool = True) -> pd.DataFrame:
    """Compare multiple strategies side-by-side."""
    rows = []
    for r in results_list:
        rows.append({
            "Strategy":       r["name"],
            "Sharpe":         r["sharpe"],
            "Annual Return":  f"{r['annual_return']:+.2%}",
            "Annual Vol":     f"{r['annual_vol']:.2%}",
            "Max Drawdown":   f"{r['max_drawdown']:.2%}",
            "Hit Rate":       f"{r['hit_rate']:.2%}",
            "Total Return":   f"{r['total_return']:+.2%}",
            "# Trades":       r["n_trades"],
        })
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    if save:
        path = REPORTS_DIR / "strategy_comparison.csv"
        df.to_csv(path, index=False)
        print(f"\n  Saved: {path}")

    # Overlay plot
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    for r, c in zip(results_list, colors):
        ax.plot(r["cum_return"], label=f"{r['name']} (S={r['sharpe']:.2f})", color=c)
    ax.set_title("Strategy Comparison — Cumulative Returns")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df
