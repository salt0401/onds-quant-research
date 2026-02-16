"""
Plan 3: Cross-Asset Signal Analysis

Tests whether gold, silver, bitcoin, VIX, bonds, and USD movements
have predictive power for ONDS returns.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_TICKER, CROSS_ASSET, DATA_PROC, FIGURES_DIR


def compute_cross_asset_features(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-asset features relative to ONDS.

    Features per cross-asset:
    - 1d, 5d, 20d return of the cross-asset
    - Rolling 20d correlation with ONDS
    - Rolling 60d correlation with ONDS
    - Relative strength (ONDS return - asset return, 20d)
    """
    target = TARGET_TICKER
    if target not in closes.columns:
        raise ValueError(f"{target} not found in close prices")

    onds_ret = closes[target].pct_change()
    features = pd.DataFrame(index=closes.index)
    features["ONDS_ret_1d"] = onds_ret

    for name, ticker in CROSS_ASSET.items():
        col = ticker
        if col not in closes.columns:
            print(f"  WARNING: {col} not in data, skipping {name}")
            continue

        ret = closes[col].pct_change()

        features[f"{name}_ret_1d"]  = ret
        features[f"{name}_ret_5d"]  = closes[col].pct_change(5)
        features[f"{name}_ret_20d"] = closes[col].pct_change(20)

        # Rolling correlations
        features[f"{name}_corr_20d"] = onds_ret.rolling(20).corr(ret)
        features[f"{name}_corr_60d"] = onds_ret.rolling(60).corr(ret)

        # Relative strength
        features[f"{name}_relstr_20d"] = (
            closes[target].pct_change(20) - closes[col].pct_change(20)
        )

    return features


def test_predictive_power(features: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Test if cross-asset returns predict ONDS next-day returns (Spearman).
    Also tests Granger-like lead-lag: does asset return on day t predict ONDS on day t+1?
    """
    onds_fwd = features["ONDS_ret_1d"].shift(-1)  # next-day ONDS return

    results = []
    for col in features.columns:
        if col == "ONDS_ret_1d":
            continue
        valid = pd.DataFrame({"feature": features[col], "target": onds_fwd}).dropna()
        if len(valid) < 50:
            continue
        corr, pval = stats.spearmanr(valid["feature"], valid["target"])
        results.append({
            "Feature": col,
            "Spearman": round(corr, 4),
            "p_value": round(pval, 4),
            "sig": "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "",
            "n": len(valid),
        })

    df = pd.DataFrame(results).sort_values("p_value")

    if save:
        df.to_csv(DATA_PROC / "crossasset_predictive_power.csv", index=False)

    print("\n  Cross-Asset Predictive Power (vs ONDS next-day return):")
    print("  " + "-" * 70)
    for _, row in df.head(20).iterrows():
        print(f"    {row['Feature']:30s}  r={row['Spearman']:+.4f}  p={row['p_value']:.4f} {row['sig']}")

    return df


def generate_cross_asset_signal(features: pd.DataFrame) -> pd.Series:
    """
    Generate a composite cross-asset signal.
    Logic: when safe-haven assets (gold, bonds) surge and risk assets (BTC, SPY) drop,
    expect defensive names like ONDS to benefit. When VIX spikes, go defensive.
    """
    sig = pd.Series(0.0, index=features.index)

    # VIX spike → defensive stocks benefit in short term
    if "vix_ret_1d" in features:
        vix_z = (features["vix_ret_1d"] - features["vix_ret_1d"].rolling(60).mean()) / \
                features["vix_ret_1d"].rolling(60).std()
        sig += np.where(vix_z > 1, 0.3, np.where(vix_z < -1, -0.3, 0))

    # Gold momentum → defense sector correlation
    if "gold_ret_5d" in features:
        gold_z = (features["gold_ret_5d"] - features["gold_ret_5d"].rolling(60).mean()) / \
                 features["gold_ret_5d"].rolling(60).std()
        sig += np.where(gold_z > 1, 0.2, np.where(gold_z < -1, -0.2, 0))

    # Bitcoin risk-on/off
    if "bitcoin_ret_5d" in features:
        btc_z = (features["bitcoin_ret_5d"] - features["bitcoin_ret_5d"].rolling(60).mean()) / \
                features["bitcoin_ret_5d"].rolling(60).std()
        # For defense stock: BTC crash → risk-off → could go either way
        sig += np.where(btc_z < -2, 0.2, 0)  # extreme risk-off → flight to defense

    # Clip to [-1, 1]
    sig = sig.clip(-1, 1)
    return sig


def plot_correlation_heatmap(closes: pd.DataFrame, window: int = 60, save: bool = True):
    """Plot rolling correlation heatmap between ONDS and cross-assets."""
    rets = closes.pct_change().dropna()
    corr = rets.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Annotate
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Full-Period Return Correlation Matrix")
    plt.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "crossasset_correlation_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIGURES_DIR / 'crossasset_correlation_heatmap.png'}")
    plt.close(fig)
