"""
Plan 11: Regime Detection & Switching Models

Applies multiple regime detection methods to ONDS:
1. Hidden Markov Model (HMM) - Gaussian emissions
2. Gaussian Mixture Model (GMM) - static clustering
3. Rolling volatility regimes (simple rule-based)
4. Markov Switching (statsmodels)

Then tests whether other signals behave differently across regimes.
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REGIME, DATA_PROC, FIGURES_DIR

warnings.filterwarnings("ignore")


def detect_regimes_hmm(returns: pd.Series, n_regimes: int = None) -> pd.DataFrame:
    """
    Fit a Gaussian HMM to daily returns.
    Returns DataFrame with regime labels and probabilities.
    """
    if n_regimes is None:
        n_regimes = REGIME["n_regimes"]

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  hmmlearn not installed. pip install hmmlearn")
        return pd.DataFrame()

    # Prepare data
    X = returns.dropna().values.reshape(-1, 1)
    idx = returns.dropna().index

    # Fit HMM
    print(f"  Fitting Gaussian HMM with {n_regimes} regimes...")
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type=REGIME["hmm_covariance"],
        n_iter=200,
        random_state=42,
    )
    model.fit(X)

    # Predict regimes
    regime_labels = model.predict(X)
    regime_probs = model.predict_proba(X)

    # Sort regimes by mean return (0=bearish, 1=neutral, 2=bullish)
    regime_means = {}
    for r in range(n_regimes):
        mask = regime_labels == r
        regime_means[r] = X[mask].mean()

    sorted_regimes = sorted(regime_means.keys(), key=lambda r: regime_means[r])
    label_map = {old: new for new, old in enumerate(sorted_regimes)}
    regime_labels = np.array([label_map[r] for r in regime_labels])

    result = pd.DataFrame(index=idx)
    result["regime"] = regime_labels
    for r in range(n_regimes):
        result[f"prob_regime_{r}"] = regime_probs[:, sorted_regimes[r]]
    result["return"] = returns.loc[idx]

    # Print regime statistics
    regime_names = {0: "Bearish", 1: "Neutral", 2: "Bullish"} if n_regimes == 3 else {}
    print(f"\n  HMM Regime Statistics:")
    print("  " + "-" * 50)
    for r in range(n_regimes):
        mask = result["regime"] == r
        name = regime_names.get(r, f"Regime {r}")
        rets = result.loc[mask, "return"]
        print(f"    {name}: n={mask.sum()} days ({mask.mean():.1%}), "
              f"mean={rets.mean():.4f}, vol={rets.std():.4f}")

    # Transition matrix
    print(f"\n  Transition Matrix:")
    trans = model.transmat_
    # Apply same sorting
    sorted_trans = trans[sorted_regimes][:, sorted_regimes]
    for i in range(n_regimes):
        name = regime_names.get(i, f"R{i}")
        probs = "  ".join(f"{p:.3f}" for p in sorted_trans[i])
        print(f"    {name}: [{probs}]")

    return result


def detect_regimes_gmm(returns: pd.Series, n_regimes: int = None) -> pd.DataFrame:
    """
    Fit a Gaussian Mixture Model (static, no temporal dynamics).
    """
    if n_regimes is None:
        n_regimes = REGIME["n_regimes"]

    from sklearn.mixture import GaussianMixture

    X = returns.dropna().values.reshape(-1, 1)
    idx = returns.dropna().index

    print(f"  Fitting GMM with {n_regimes} components...")
    gmm = GaussianMixture(n_components=n_regimes, random_state=42)
    gmm.fit(X)

    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)

    # Sort by mean
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    label_map = {old: new for new, old in enumerate(sorted_idx)}
    labels = np.array([label_map[l] for l in labels])

    result = pd.DataFrame(index=idx)
    result["regime"] = labels
    result["return"] = returns.loc[idx]

    return result


def detect_regimes_volatility(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Simple rule-based regime detection using rolling volatility.
    Low vol → calm/bullish, High vol → stressed/bearish.
    """
    print(f"  Computing volatility-based regimes (window={window})...")

    vol = returns.rolling(window).std() * np.sqrt(252)
    vol_median = vol.median()
    vol_75 = vol.quantile(0.75)

    result = pd.DataFrame(index=returns.index)
    result["return"] = returns
    result["rolling_vol"] = vol

    # 3 regimes based on vol quantiles
    result["regime"] = 1  # normal
    result.loc[vol <= vol_median, "regime"] = 0       # low vol (calm)
    result.loc[vol >= vol_75, "regime"] = 2           # high vol (stressed)

    regime_names = {0: "Low Vol", 1: "Normal", 2: "High Vol"}
    print(f"\n  Volatility Regime Statistics:")
    for r in range(3):
        mask = result["regime"] == r
        name = regime_names[r]
        rets = result.loc[mask, "return"]
        print(f"    {name}: n={mask.sum()} ({mask.mean():.1%}), "
              f"mean_ret={rets.mean():.4f}, vol={rets.std():.4f}")

    return result


def detect_regimes_markov_switching(returns: pd.Series, n_regimes: int = 2) -> pd.DataFrame:
    """
    Hamilton (1989) Markov Switching model via statsmodels.
    """
    import statsmodels.api as sm

    print(f"  Fitting Markov Switching model with {n_regimes} regimes...")

    clean = returns.dropna()

    try:
        model = sm.tsa.MarkovRegression(
            clean,
            k_regimes=n_regimes,
            trend="c",
            switching_variance=True,
        )
        result = model.fit(disp=False)

        regime_df = pd.DataFrame(index=clean.index)
        regime_df["return"] = clean

        # Smoothed probabilities
        for r in range(n_regimes):
            regime_df[f"prob_regime_{r}"] = result.smoothed_marginal_probabilities[r]

        regime_df["regime"] = regime_df[[f"prob_regime_{r}" for r in range(n_regimes)]].values.argmax(axis=1)

        # Sort by regime mean
        regime_means = {}
        for r in range(n_regimes):
            mask = regime_df["regime"] == r
            regime_means[r] = regime_df.loc[mask, "return"].mean()

        sorted_r = sorted(regime_means, key=lambda r: regime_means[r])
        label_map = {old: new for new, old in enumerate(sorted_r)}
        regime_df["regime"] = regime_df["regime"].map(label_map)

        print(f"  Markov Switching Results:")
        for r in range(n_regimes):
            mask = regime_df["regime"] == r
            rets = regime_df.loc[mask, "return"]
            print(f"    Regime {r}: n={mask.sum()} ({mask.mean():.1%}), "
                  f"mean={rets.mean():.4f}, vol={rets.std():.4f}")

        return regime_df

    except Exception as e:
        print(f"  Markov Switching failed: {e}")
        return pd.DataFrame()


def analyze_regime_dependent_signals(
    regime_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    signal_name: str = "signal",
) -> pd.DataFrame:
    """
    Test whether a signal's predictive power varies by regime.
    """
    merged = regime_df.join(signal_df, how="inner").dropna()
    if len(merged) < 50:
        print("  Insufficient data for regime-dependent analysis.")
        return pd.DataFrame()

    signal_col = [c for c in signal_df.columns if "signal" in c.lower() or "sentiment" in c.lower()]
    if not signal_col:
        signal_col = signal_df.columns[:1]
    signal_col = signal_col[0]

    results = []
    print(f"\n  Regime-Dependent Signal Analysis ({signal_name}):")
    print("  " + "-" * 50)

    for regime in sorted(merged["regime"].unique()):
        subset = merged[merged["regime"] == regime]
        if len(subset) < 20:
            continue

        fwd = subset["return"].shift(-1).dropna()
        sig = subset.loc[fwd.index, signal_col]

        corr, pval = stats.spearmanr(sig, fwd)
        results.append({
            "Regime": regime,
            "N": len(subset),
            "Signal_mean": sig.mean(),
            "Return_mean": subset["return"].mean(),
            "Corr_sig_fwd_ret": corr,
            "p_value": pval,
        })
        sig_marker = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"    Regime {regime}: n={len(subset)}, corr={corr:+.4f}, p={pval:.4f} {sig_marker}")

    return pd.DataFrame(results)


def plot_regimes(regime_df: pd.DataFrame, prices: pd.Series, method: str = "HMM",
                 save: bool = True):
    """Plot regimes colored over price chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                    height_ratios=[3, 1])

    # Color map
    cmap = {0: "#F44336", 1: "#FFC107", 2: "#4CAF50"}  # red, yellow, green
    regime_names = {0: "Bearish/Low", 1: "Neutral", 2: "Bullish/High"}

    # Price with regime background
    aligned_prices = prices.loc[prices.index.intersection(regime_df.index)]
    ax1.plot(aligned_prices.index, aligned_prices, color="black", linewidth=1)

    # Color background by regime
    for regime in sorted(regime_df["regime"].unique()):
        mask = regime_df["regime"] == regime
        dates = regime_df.index[mask]
        for d in dates:
            if d in aligned_prices.index:
                ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.15,
                           color=cmap.get(regime, "gray"))

    ax1.set_ylabel("ONDS Price ($)")
    ax1.set_title(f"ONDS Regime Detection ({method})")
    ax1.grid(True, alpha=0.3)

    # Regime probabilities or labels
    ax2.plot(regime_df.index, regime_df["regime"], ".", markersize=2)
    ax2.set_ylabel("Regime")
    ax2.set_yticks(sorted(regime_df["regime"].unique()))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"regime_{method.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
