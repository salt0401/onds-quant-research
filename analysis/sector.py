"""
Plan 2: Related Stocks & Sector Analysis

Analyzes ONDS vs drone/defense peers (RCAT, AVAV, KTOS, JOBY)
and sector ETFs (ITA, XAR). Tests lead-lag relationships and
sector momentum signals.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, PEER_TICKERS, SECTOR_ETFS, DATA_PROC, FIGURES_DIR


def compute_sector_features(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute peer and sector features relative to ONDS.

    Features:
    - Lead-lag correlations (does peer move before ONDS?)
    - Sector momentum (ITA 5d/20d return)
    - Relative strength vs peers
    - Peer average return
    """
    target = TARGET_TICKER
    features = pd.DataFrame(index=closes.index)

    onds_ret = closes[target].pct_change()
    features["ONDS_ret"] = onds_ret
    features["ONDS_fwd_1d"] = onds_ret.shift(-1)

    # Peer returns
    peer_rets = pd.DataFrame()
    for peer in PEER_TICKERS:
        if peer in closes.columns:
            ret = closes[peer].pct_change()
            features[f"{peer}_ret_1d"] = ret
            features[f"{peer}_ret_5d"] = closes[peer].pct_change(5)
            peer_rets[peer] = ret

    # Peer average return (sector breadth)
    if not peer_rets.empty:
        features["peer_avg_ret"] = peer_rets.mean(axis=1)
        features["peer_breadth"] = (peer_rets > 0).mean(axis=1)  # % of peers positive

    # Sector ETF momentum
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            features[f"{etf}_ret_5d"]  = closes[etf].pct_change(5)
            features[f"{etf}_ret_20d"] = closes[etf].pct_change(20)

    # ONDS relative strength vs sector
    if "ITA" in closes.columns:
        features["ONDS_vs_ITA_20d"] = (
            closes[target].pct_change(20) - closes["ITA"].pct_change(20)
        )

    return features


def test_lead_lag(closes: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Test lead-lag relationships: does peer return on day t
    predict ONDS return on day t+lag?
    """
    target = TARGET_TICKER
    onds_ret = closes[target].pct_change()
    peers = [t for t in PEER_TICKERS + SECTOR_ETFS if t in closes.columns]

    results = []
    for peer in peers:
        peer_ret = closes[peer].pct_change()
        for lag in range(1, max_lag + 1):
            # Does peer on day t predict ONDS on day t+lag?
            valid = pd.DataFrame({
                "peer": peer_ret,
                "onds_fwd": onds_ret.shift(-lag),
            }).dropna()
            if len(valid) < 50:
                continue
            corr, pval = stats.spearmanr(valid["peer"], valid["onds_fwd"])
            results.append({
                "Peer": peer,
                "Lag": lag,
                "Spearman": round(corr, 4),
                "p_value": round(pval, 4),
                "sig": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "",
            })

    df = pd.DataFrame(results)
    df.to_csv(DATA_PROC / "sector_lead_lag.csv", index=False)

    print("\n  Sector Lead-Lag Analysis (peer day-t → ONDS day-t+lag):")
    print("  " + "-" * 60)
    sig_rows = df[df["p_value"] < 0.1].sort_values("p_value")
    for _, row in sig_rows.head(15).iterrows():
        print(f"    {row['Peer']:8s} lag={row['Lag']}  r={row['Spearman']:+.4f}  "
              f"p={row['p_value']:.4f} {row['sig']}")
    if sig_rows.empty:
        print("    No significant lead-lag relationships found.")

    return df


def generate_sector_signal(features: pd.DataFrame) -> pd.Series:
    """
    Generate sector-based trading signal for ONDS.
    Logic: when sector momentum is positive and peer breadth is high, go long.
    """
    sig = pd.Series(0.0, index=features.index)

    # Sector momentum
    if "ITA_ret_5d" in features:
        sig += np.where(features["ITA_ret_5d"] > 0.02, 0.3,
               np.where(features["ITA_ret_5d"] < -0.02, -0.3, 0))

    # Peer breadth (>60% of peers positive → bullish)
    if "peer_breadth" in features:
        sig += np.where(features["peer_breadth"] > 0.6, 0.3,
               np.where(features["peer_breadth"] < 0.4, -0.3, 0))

    # Peer average momentum
    if "peer_avg_ret" in features:
        peer_z = (features["peer_avg_ret"] - features["peer_avg_ret"].rolling(60).mean()) / \
                 features["peer_avg_ret"].rolling(60).std()
        sig += np.where(peer_z > 1, 0.3, np.where(peer_z < -1, -0.3, 0))

    return sig.clip(-1, 1)


def plot_sector_comparison(closes: pd.DataFrame, save: bool = True):
    """Plot normalized price comparison of ONDS vs peers."""
    target = TARGET_TICKER
    tickers = [target] + [t for t in PEER_TICKERS + SECTOR_ETFS if t in closes.columns]

    # Normalize to 100
    subset = closes[tickers].dropna(how="all")
    normalized = subset / subset.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
    for ticker, color in zip(tickers, colors):
        lw = 2.5 if ticker == target else 1
        alpha = 1.0 if ticker == target else 0.6
        ax.plot(normalized.index, normalized[ticker], label=ticker,
                linewidth=lw, alpha=alpha, color=color)

    ax.set_title("ONDS vs Drone/Defense Peers (Normalized to 100)")
    ax.set_ylabel("Normalized Price")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "sector_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'sector_comparison.png'}")
    plt.close(fig)
