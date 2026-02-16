"""
Plan 4: Dark Pool Analysis

Tests whether DIX/GEX and per-ticker dark pool indicators
predict ONDS returns.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROC, FIGURES_DIR


def load_dix_data() -> pd.DataFrame:
    """Load SqueezeMetrics DIX/GEX data."""
    path = DATA_RAW / "squeezemetrics_dix_gex.csv"
    if not path.exists():
        print("  No DIX/GEX data found. Run collectors/darkpool.py first.")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_stockgrid_data(ticker: str = "ONDS") -> pd.DataFrame:
    """Load Stockgrid dark pool data."""
    path = DATA_RAW / f"stockgrid_{ticker}_darkpool.csv"
    if not path.exists():
        print(f"  No Stockgrid data for {ticker}. Run collectors/darkpool.py first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_chartexchange_data() -> pd.DataFrame:
    """Load ChartExchange ONDS-specific short volume data (FINRA)."""
    path = DATA_RAW / "chartexchange_onds_short_volume.csv"
    if not path.exists():
        print("  No ChartExchange short volume data found.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Parse date column (format: YYYYMMDD)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").sort_index()
    # Compute short ratio
    df["short_ratio"] = df["total_short"] / df["total_reported"]
    df["off_exchange_short"] = df["FINRA"] / df["total_reported"]
    print(f"    Loaded {len(df)} rows of ONDS short volume from ChartExchange")
    return df


def analyze_short_volume(sv_df: pd.DataFrame, onds_prices: pd.DataFrame) -> dict:
    """
    Analyze ONDS-specific short volume data from FINRA/ChartExchange.

    Key hypothesis: Extreme short volume ratios may indicate directional bias.
    Consistently high short volume (>40%) in small-caps like ONDS often reflects
    market-maker hedging rather than bearish sentiment ("Short is Long" thesis).
    """
    if sv_df.empty:
        print("  No short volume data available.")
        return {}

    onds_ret = onds_prices["Close"].pct_change()

    merged = pd.DataFrame({
        "short_ratio": sv_df["short_ratio"],
        "off_exchange_short": sv_df["off_exchange_short"],
        "total_volume": sv_df["total_reported"],
        "ONDS_ret": onds_ret,
        "ONDS_fwd_1d": onds_ret.shift(-1),
        "ONDS_fwd_5d": onds_prices["Close"].pct_change(5).shift(-5),
        "ONDS_fwd_20d": onds_prices["Close"].pct_change(20).shift(-20),
    }).dropna()

    if len(merged) < 30:
        print(f"  Only {len(merged)} overlapping days for short volume. Insufficient.")
        return {}

    results = {}
    print(f"\n  ONDS Short Volume Predictive Power ({len(merged)} days):")
    print("  " + "-" * 60)
    print(f"    Mean short ratio:  {merged['short_ratio'].mean():.2%}")
    print(f"    Std short ratio:   {merged['short_ratio'].std():.2%}")
    print(f"    Range: [{merged['short_ratio'].min():.2%}, {merged['short_ratio'].max():.2%}]")

    for sig_col in ["short_ratio", "off_exchange_short"]:
        for fwd in ["ONDS_fwd_1d", "ONDS_fwd_5d", "ONDS_fwd_20d"]:
            valid = merged[[sig_col, fwd]].dropna()
            if len(valid) < 30:
                continue
            corr, pval = stats.spearmanr(valid[sig_col], valid[fwd])
            key = f"{sig_col}_vs_{fwd}"
            results[key] = {"corr": corr, "pval": pval, "n": len(valid)}
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"    {sig_col:20s} → {fwd:15s}  r={corr:+.4f}  p={pval:.4f} {sig}")

    # Quintile analysis
    for sig_col in ["short_ratio"]:
        for fwd in ["ONDS_fwd_5d", "ONDS_fwd_20d"]:
            valid = merged[[sig_col, fwd]].dropna()
            if len(valid) < 50:
                continue
            valid["quintile"] = pd.qcut(valid[sig_col], 5,
                                        labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"],
                                        duplicates="drop")
            quintile_means = valid.groupby("quintile", observed=True)[fwd].mean()
            print(f"\n  Short Ratio Quintile Analysis → {fwd}:")
            for q, m in quintile_means.items():
                print(f"    {q}: {m:+.4f} ({m*100:+.2f}%)")

    return results


def generate_short_volume_signal(sv_df: pd.DataFrame) -> pd.Series:
    """
    Generate signal from ONDS-specific short volume.
    High short ratio → potential bullish (market maker hedging)
    Low short ratio → potential bearish (less institutional activity)
    """
    if sv_df.empty:
        return pd.Series(dtype=float)

    sr = sv_df["short_ratio"]
    sr_z = (sr - sr.rolling(60, min_periods=20).mean()) / sr.rolling(60, min_periods=20).std()
    sig = pd.Series(0.0, index=sv_df.index)
    sig[sr_z > 1] = 0.5       # high short ratio → bullish ("short is long")
    sig[sr_z < -1] = -0.5     # low short ratio → bearish
    return sig.clip(-1, 1)


def plot_short_volume(sv_df: pd.DataFrame, onds_prices: pd.DataFrame, save: bool = True):
    """Plot ONDS short volume analysis."""
    if sv_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ONDS price
    common = onds_prices.index.intersection(sv_df.index)
    ax = axes[0]
    ax.plot(onds_prices.loc[common].index, onds_prices.loc[common]["Close"],
            color="#1976D2", linewidth=1)
    ax.set_ylabel("ONDS Price ($)")
    ax.set_title("ONDS Short Volume Analysis (FINRA/ChartExchange)")
    ax.grid(True, alpha=0.3)

    # Short ratio
    ax = axes[1]
    ax.bar(sv_df.index, sv_df["short_ratio"] * 100, color="#FF9800", alpha=0.6, width=1)
    ax.axhline(sv_df["short_ratio"].mean() * 100, color="red", linestyle="--",
               alpha=0.7, label=f"Mean: {sv_df['short_ratio'].mean():.1%}")
    ax.set_ylabel("Short Volume %")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total volume
    ax = axes[2]
    ax.bar(sv_df.index, sv_df["total_reported"] / 1e6, color="#2196F3", alpha=0.5, width=1)
    ax.set_ylabel("Total Volume (M)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "onds_short_volume.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'onds_short_volume.png'}")
    plt.close(fig)


def analyze_dix_predictive_power(dix_df: pd.DataFrame, onds_prices: pd.DataFrame) -> dict:
    """
    Test if DIX/GEX levels predict future ONDS returns.

    Key hypothesis (from SqueezeMetrics "Short is Long" paper):
    High DIX (>= 45%) → institutional buying → positive forward returns
    Low GEX → volatility amplification → larger moves
    """
    if dix_df.empty:
        print("  No DIX data available for analysis.")
        return {}

    onds_ret = onds_prices["Close"].pct_change()

    # Align dates
    merged = pd.DataFrame({
        "ONDS_ret_1d": onds_ret,
        "ONDS_fwd_1d": onds_ret.shift(-1),
        "ONDS_fwd_5d": onds_prices["Close"].pct_change(5).shift(-5),
        "ONDS_fwd_20d": onds_prices["Close"].pct_change(20).shift(-20),
    })

    # Try to merge DIX columns
    dix_cols = [c for c in dix_df.columns if c.lower() in ["dix", "gex"]]
    if not dix_cols:
        dix_cols = list(dix_df.columns[:4])  # take first few columns

    for col in dix_cols:
        merged[col] = dix_df[col]

    merged = merged.dropna()

    if len(merged) < 30:
        print(f"  Only {len(merged)} overlapping days. Insufficient data.")
        return {}

    results = {}
    print("\n  DIX/GEX Predictive Power for ONDS:")
    print("  " + "-" * 60)

    for dix_col in dix_cols:
        for fwd in ["ONDS_fwd_1d", "ONDS_fwd_5d", "ONDS_fwd_20d"]:
            valid = merged[[dix_col, fwd]].dropna()
            if len(valid) < 30:
                continue
            corr, pval = stats.spearmanr(valid[dix_col], valid[fwd])
            key = f"{dix_col}_vs_{fwd}"
            results[key] = {"corr": corr, "pval": pval, "n": len(valid)}
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            print(f"    {dix_col:8s} → {fwd:15s}  r={corr:+.4f}  p={pval:.4f} {sig}")

    # Quintile analysis: sort days by DIX, compute mean forward return per quintile
    for dix_col in dix_cols:
        for fwd in ["ONDS_fwd_5d", "ONDS_fwd_20d"]:
            valid = merged[[dix_col, fwd]].dropna()
            if len(valid) < 50:
                continue
            valid["quintile"] = pd.qcut(valid[dix_col], 5, labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"])
            quintile_means = valid.groupby("quintile")[fwd].mean()
            print(f"\n  {dix_col} Quintile Analysis → {fwd}:")
            for q, m in quintile_means.items():
                print(f"    {q}: {m:+.4f} ({m*100:+.2f}%)")

    return results


def generate_darkpool_signal(dix_df: pd.DataFrame) -> pd.Series:
    """
    Generate dark pool signal.
    High DIX → bullish (institutional buying)
    Low GEX → expect vol expansion (but direction uncertain)
    """
    if dix_df.empty:
        return pd.Series(dtype=float)

    sig = pd.Series(0.0, index=dix_df.index)

    # DIX signal: z-score relative to rolling 60-day window
    dix_cols = [c for c in dix_df.columns if "dix" in c.lower()]
    if dix_cols:
        dix = dix_df[dix_cols[0]]
        dix_z = (dix - dix.rolling(60).mean()) / dix.rolling(60).std()
        sig += np.where(dix_z > 1, 0.5, np.where(dix_z < -1, -0.5, 0))

    # GEX signal: very negative GEX → expect larger moves
    gex_cols = [c for c in dix_df.columns if "gex" in c.lower()]
    if gex_cols:
        gex = dix_df[gex_cols[0]]
        gex_z = (gex - gex.rolling(60).mean()) / gex.rolling(60).std()
        # Negative GEX amplifies moves; combine with DIX direction
        sig *= np.where(gex_z < -1, 1.5, 1.0)

    return sig.clip(-1, 1)


def plot_dix_analysis(dix_df: pd.DataFrame, onds_prices: pd.DataFrame, save: bool = True):
    """Plot DIX/GEX alongside ONDS price."""
    if dix_df.empty:
        print("  No DIX data to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ONDS price
    ax = axes[0]
    ax.plot(onds_prices.index, onds_prices["Close"], color="#1976D2", linewidth=1)
    ax.set_ylabel("ONDS Price ($)")
    ax.set_title("Dark Pool Indicators vs ONDS Price")
    ax.grid(True, alpha=0.3)

    # DIX
    dix_cols = [c for c in dix_df.columns if "dix" in c.lower()]
    if dix_cols:
        ax = axes[1]
        ax.plot(dix_df.index, dix_df[dix_cols[0]], color="#4CAF50", linewidth=0.8)
        ax.axhline(0.45, color="red", linestyle="--", alpha=0.5, label="DIX=45%")
        ax.set_ylabel("DIX")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # GEX
    gex_cols = [c for c in dix_df.columns if "gex" in c.lower()]
    if gex_cols:
        ax = axes[2]
        gex = dix_df[gex_cols[0]]
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in gex]
        ax.bar(dix_df.index, gex, color=colors, alpha=0.6, width=1)
        ax.set_ylabel("GEX")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "darkpool_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
