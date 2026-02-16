"""
Plan 8: Analyst Reports & Target Price Analysis

Analyzes analyst recommendations (buy/hold/sell) and target prices.
Tests whether changes in analyst consensus predict ONDS returns.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, DATA_RAW, DATA_PROC, FIGURES_DIR


def load_recommendations() -> pd.DataFrame:
    """Load analyst recommendation data."""
    path = DATA_RAW / f"finnhub_recommendations_{TARGET_TICKER}.csv"
    if not path.exists():
        print("  No analyst data. Run collectors/news.py first (uses Finnhub).")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["period"])


def analyze_recommendations(rec_df: pd.DataFrame, onds_prices: pd.DataFrame) -> dict:
    """
    Analyze analyst recommendation changes and their impact on ONDS.

    Tests:
    1. Does a shift toward 'buy' predict positive returns?
    2. How accurate are consensus ratings?
    3. Event study: returns around recommendation changes
    """
    if rec_df.empty:
        print("  No recommendation data available.")
        return {}

    results = {}
    print(f"\n  Analyst Recommendation Analysis ({len(rec_df)} periods):")
    print("  " + "-" * 60)

    # Compute consensus score: (strongBuy*2 + buy*1 + hold*0 + sell*-1 + strongSell*-2) / total
    for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
        if col not in rec_df.columns:
            rec_df[col] = 0

    rec_df["total"] = rec_df[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum(axis=1)
    rec_df["consensus_score"] = (
        rec_df["strongBuy"] * 2 + rec_df["buy"] * 1 + rec_df["hold"] * 0
        + rec_df["sell"] * (-1) + rec_df["strongSell"] * (-2)
    ) / rec_df["total"].replace(0, np.nan)

    # Change in consensus
    rec_df["consensus_change"] = rec_df["consensus_score"].diff()

    print(f"    Latest consensus: {rec_df['consensus_score'].iloc[-1]:.2f}")
    print(f"    Buy: {rec_df['buy'].iloc[-1]}, Hold: {rec_df['hold'].iloc[-1]}, "
          f"Sell: {rec_df['sell'].iloc[-1]}")

    # Merge with returns
    rec_df = rec_df.set_index("period")
    onds_ret = onds_prices["Close"].pct_change()

    # Forward returns after each recommendation period
    for days in [5, 10, 20, 60]:
        fwd = onds_prices["Close"].pct_change(days).shift(-days)
        merged = pd.DataFrame({
            "consensus": rec_df["consensus_score"],
            f"fwd_{days}d": fwd,
        }).dropna()

        if len(merged) > 5:
            corr, pval = stats.spearmanr(merged["consensus"], merged[f"fwd_{days}d"])
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            results[f"consensus_vs_fwd_{days}d"] = {"corr": corr, "pval": pval}
            print(f"    Consensus → {days}d fwd return: r={corr:+.4f}, p={pval:.4f} {sig}")

    return results


def generate_analyst_signal(rec_df: pd.DataFrame) -> pd.Series:
    """Generate trading signal from analyst consensus changes."""
    if rec_df.empty:
        return pd.Series(dtype=float)

    for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
        if col not in rec_df.columns:
            rec_df[col] = 0

    rec_df["total"] = rec_df[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum(axis=1)
    rec_df["consensus_score"] = (
        rec_df["strongBuy"] * 2 + rec_df["buy"] * 1
        + rec_df["sell"] * (-1) + rec_df["strongSell"] * (-2)
    ) / rec_df["total"].replace(0, np.nan)

    if "period" in rec_df.columns:
        rec_df = rec_df.set_index("period")

    # Signal: consensus > 1 → long, < 0 → short
    sig = pd.Series(0.0, index=rec_df.index)
    sig[rec_df["consensus_score"] > 1] = 1.0
    sig[rec_df["consensus_score"] < 0] = -1.0

    return sig


def plot_analyst_history(rec_df: pd.DataFrame, onds_prices: pd.DataFrame, save: bool = True):
    """Plot analyst recommendation history alongside price."""
    if rec_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Price
    ax1.plot(onds_prices.index, onds_prices["Close"], color="#1976D2", linewidth=1)
    ax1.set_ylabel("ONDS Price ($)")
    ax1.set_title("ONDS Price & Analyst Recommendations")
    ax1.grid(True, alpha=0.3)

    # Stacked bar of recommendations
    idx = rec_df["period"] if "period" in rec_df.columns else rec_df.index
    width = 20  # days

    colors = {"strongBuy": "#1B5E20", "buy": "#4CAF50", "hold": "#FFC107",
              "sell": "#FF5722", "strongSell": "#B71C1C"}

    bottom = np.zeros(len(rec_df))
    for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
        if col in rec_df.columns:
            vals = rec_df[col].values
            ax2.bar(idx, vals, bottom=bottom, color=colors[col], label=col, width=width)
            bottom += vals

    ax2.set_ylabel("# Analysts")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "analyst_recommendations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
