"""
Cross-Sectional Validation + Bayesian Analysis
================================================
Alternative statistical evidence for ONDS strategy performance.

Part 1: Cross-sectional panel test
    Do strategies generalize to peer stocks? If a strategy works not just on
    ONDS but also on RCAT, JOBY, AVAV, KTOS, LMT, RTX with the SAME params,
    that is strong evidence the pattern is real, not overfitting.

Part 2: Bayesian posterior analysis
    Instead of "reject H0 at 5%" (which needs 300+ days), compute:
    "Given 84 days of data, what is the probability that alpha > 0?"
"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from scipy.stats import wilcoxon, ttest_1samp, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FIGURES_DIR, REPORTS_DIR, BACKTEST, PEER_TICKERS
from backtests.engine import backtest
from analysis.advanced_research import (
    load_all_data, build_advanced_features, select_features,
    vol_adjusted_signal,
)
from analysis.param_optimization import (
    make_split, period_backtest,
    momentum_stop_params, market_contrarian_params,
    multi_tf_momentum_params, mean_reversion_params,
    ml_direction_params, combined_best_params,
)

# Binomtest: scipy >= 1.7 has binomtest, older has binom_test
try:
    from scipy.stats import binomtest
    def _sign_test_pval(k, n):
        return binomtest(k, n, 0.5, alternative="greater").pvalue
except ImportError:
    from scipy.stats import binom_test
    def _sign_test_pval(k, n):
        return binom_test(k, n, 0.5, alternative="greater")


# =====================================================================
# OPTIMIZED PARAMETERS (from grid search on ONDS)
# =====================================================================

OPTIMIZED_PARAMS = {
    "Momentum+Stop": {
        "sma_short": 15, "sma_long": 50,
        "trailing_stop": 0.15, "max_loss": 0.1,
    },
    "Market Contrarian": {"z_threshold": 0.5, "lookback": 30},
    "ML Direction": {
        "n_features": 20, "prob_threshold": 0.58, "max_depth": 5,
    },
    "Multi-TF Momentum": {"periods": (10, 20, 50), "strength_div": 0.2},
    "Mean Reversion": {"z_threshold": 1.5, "lookback": 10},
    "Combined Best": {
        "w_peer": 0.2, "w_contra": 0.4, "w_mtf": 0.3, "w_mr": 0.3,
    },
}

# Cross-section universe
CROSS_TICKERS = ["ONDS"] + list(PEER_TICKERS)  # 7 stocks

# Strategies that generalize to any stock (only need OHLCV + QQQ/SPY)
GENERALIZABLE = [
    "Momentum+Stop", "Multi-TF Momentum", "Mean Reversion", "Market Contrarian",
]


# =====================================================================
# GENERIC STRATEGY FUNCTIONS (work on any ticker's close price)
# =====================================================================

def momentum_stop_generic(close, params):
    """Momentum + trailing stop -- generalized for any ticker."""
    sma_short = params.get("sma_short", 10)
    sma_long = params.get("sma_long", 50)
    trailing = params.get("trailing_stop", 0.10)
    max_loss = params.get("max_loss", 0.15)

    sma_s = close.rolling(sma_short).mean()
    sma_l = close.rolling(sma_long).mean()
    sig = pd.Series(0.0, index=close.index)

    position = 0
    entry_price = 0.0
    peak_price = 0.0
    warmup = max(sma_short, sma_long)

    for i in range(warmup, len(sig)):
        price = close.iloc[i]
        if position == 0:
            if price > sma_s.iloc[i] and sma_s.iloc[i] > sma_l.iloc[i]:
                position = 1
                entry_price = price
                peak_price = price
        elif position == 1:
            peak_price = max(peak_price, price)
            if price < sma_s.iloc[i]:
                position = 0
            elif price < peak_price * (1 - trailing):
                position = 0
            elif price < entry_price * (1 - max_loss):
                position = 0
        sig.iloc[i] = position
    return sig


def multi_tf_momentum_generic(close, params):
    """Multi-timeframe momentum -- generalized for any ticker."""
    periods = params.get("periods", (5, 10, 20))
    strength_div = params.get("strength_div", 0.15)
    sig = pd.Series(0.0, index=close.index)

    p1, p2, p3 = periods
    m1 = close.pct_change(p1)
    m2 = close.pct_change(p2)
    m3 = close.pct_change(p3)

    warmup = max(periods)
    for i in range(warmup, len(sig)):
        v1, v2, v3 = m1.iloc[i], m2.iloc[i], m3.iloc[i]
        if pd.isna(v1) or pd.isna(v2) or pd.isna(v3):
            continue
        if v1 > 0 and v2 > 0 and v3 > 0:
            sig.iloc[i] = min(1.0, (v1 + v2 + v3) / strength_div)
        elif v1 < 0 and v2 < 0 and v3 < 0:
            sig.iloc[i] = -min(1.0, abs(v1 + v2 + v3) / strength_div)
    return sig.clip(-1, 1)


def mean_reversion_generic(close, params):
    """Mean reversion (z-score of daily returns) -- generalized for any ticker."""
    ret = close.pct_change()
    z_thresh = params.get("z_threshold", 2.0)
    lookback = params.get("lookback", 20)
    sig = pd.Series(0.0, index=close.index)

    ret_z = (ret - ret.rolling(lookback).mean()) / ret.rolling(lookback).std()

    for i in range(lookback, len(sig)):
        z = ret_z.iloc[i]
        if pd.isna(z):
            continue
        if z > z_thresh:
            sig.iloc[i] = -1.0
        elif z > z_thresh / 2:
            sig.iloc[i] = -0.5
        elif z < -z_thresh:
            sig.iloc[i] = 1.0
        elif z < -z_thresh / 2:
            sig.iloc[i] = 0.5
    return sig


def market_contrarian_generic(closes_all, target_index, params):
    """Market contrarian -- signal from QQQ/SPY z-scores, applied to any target."""
    z_thresh = params.get("z_threshold", 0.75)
    lookback = params.get("lookback", 20)
    closes = closes_all.reindex(target_index)
    sig = pd.Series(0.0, index=target_index)

    if "QQQ" in closes.columns:
        nr = closes["QQQ"].pct_change()
        nz = (nr - nr.rolling(lookback).mean()) / nr.rolling(lookback).std()
        sig += np.where(nz > z_thresh, -0.5, np.where(nz < -z_thresh, 0.5, 0))

    if "SPY" in closes.columns:
        sr = closes["SPY"].pct_change()
        sz = (sr - sr.rolling(lookback).mean()) / sr.rolling(lookback).std()
        sig += np.where(sz > z_thresh, -0.3, np.where(sz < -z_thresh, 0.3, 0))

    return pd.Series(sig, index=target_index).clip(-1, 1)


# Map strategy name -> generic function
_GENERIC_MAP = {
    "Momentum+Stop":      momentum_stop_generic,
    "Multi-TF Momentum":  multi_tf_momentum_generic,
    "Mean Reversion":     mean_reversion_generic,
    "Market Contrarian":  market_contrarian_generic,
}


# =====================================================================
# PART 1: CROSS-SECTIONAL VALIDATION
# =====================================================================

def cross_sectional_one_strategy(closes_all, strategy_name, params,
                                  test_start, test_end):
    """
    Apply ONE strategy with the SAME ONDS-optimized params to multiple stocks.
    Returns (per_stock_results, panel_stats, pooled_daily_excess).
    """
    results = []
    all_daily_excess = []
    rf_daily = BACKTEST["risk_free_rate"] / 252

    for ticker in CROSS_TICKERS:
        if ticker not in closes_all.columns:
            continue
        close = closes_all[ticker].dropna()
        if len(close) < 60:
            continue

        # Generate signal using generic function
        if strategy_name == "Market Contrarian":
            sig = market_contrarian_generic(closes_all, close.index, params)
        else:
            fn = _GENERIC_MAP[strategy_name]
            sig = fn(close, params)

        # Backtest on the test period
        bt = period_backtest(close, sig, test_start, test_end,
                             name=f"{strategy_name}_{ticker}")
        if bt is None:
            continue

        daily_excess = bt["daily_pnl"] - rf_daily
        all_daily_excess.append(daily_excess)

        results.append({
            "ticker": ticker,
            "strategy": strategy_name,
            "sharpe": bt["sharpe"],
            "total_return": bt["total_return"],
            "annual_return": bt["annual_return"],
            "max_drawdown": bt["max_drawdown"],
            "hit_rate": bt["hit_rate"],
            "n_trades": bt["n_trades"],
            "n_days": bt["n_days"],
        })

    # ---------- Panel-level statistical tests ----------
    if len(results) < 3:
        return results, {}, all_daily_excess

    sharpes = np.array([r["sharpe"] for r in results])
    n_pos = int((sharpes > 0).sum())
    n_tot = len(sharpes)

    # 1. Sign test (binomial): H0: P(Sharpe > 0) = 0.5
    sign_pval = _sign_test_pval(n_pos, n_tot)

    # 2. Wilcoxon signed-rank: H0: median Sharpe = 0
    try:
        w_stat, w_pval = wilcoxon(sharpes, alternative="greater")
    except Exception:
        w_stat, w_pval = np.nan, np.nan

    # 3. Pooled daily excess returns t-test
    if all_daily_excess:
        pooled = pd.concat(all_daily_excess).dropna()
        t_stat, t_pval_2 = ttest_1samp(pooled, 0)
        pooled_pval = t_pval_2 / 2 if t_stat > 0 else 1 - t_pval_2 / 2
        pooled_n = len(pooled)
        pooled_mean = pooled.mean()
    else:
        pooled_pval, pooled_n, pooled_mean = np.nan, 0, np.nan

    panel = {
        "strategy": strategy_name,
        "n_stocks": n_tot,
        "n_positive": n_pos,
        "mean_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "sign_test_p": float(sign_pval),
        "wilcoxon_p": float(w_pval),
        "pooled_n": pooled_n,
        "pooled_mean_daily": float(pooled_mean) if not np.isnan(pooled_mean) else 0,
        "pooled_t_p": float(pooled_pval),
    }
    return results, panel, all_daily_excess


def run_cross_sectional(data, test_start, test_end):
    """Run cross-sectional validation for all generalizable strategies."""
    closes_all = data["closes"]
    all_per_stock = []
    all_panel = []

    for strat in GENERALIZABLE:
        params = OPTIMIZED_PARAMS[strat]
        print(f"\n  {strat}  (params: {params})")

        per_stock, panel, _ = cross_sectional_one_strategy(
            closes_all, strat, params, test_start, test_end,
        )

        for r in per_stock:
            all_per_stock.append(r)
            sign = "+" if r["sharpe"] > 0 else ""
            print(f"    {r['ticker']:<6s}  Sharpe {sign}{r['sharpe']:.3f}  "
                  f"Ret {r['total_return']:+.2%}  MaxDD {r['max_drawdown']:.2%}")

        if panel:
            all_panel.append(panel)
            print(f"    --- Panel: {panel['n_positive']}/{panel['n_stocks']} positive"
                  f" | Sign p={panel['sign_test_p']:.4f}"
                  f" | Wilcoxon p={panel['wilcoxon_p']:.4f}"
                  f" | Pooled p={panel['pooled_t_p']:.4f}"
                  f" (n={panel['pooled_n']})")

    return all_per_stock, all_panel


# =====================================================================
# PART 2: BAYESIAN ANALYSIS
# =====================================================================

def bayesian_sharpe(daily_excess, prior_mean=0.0, prior_var=None):
    """
    Conjugate normal-normal posterior for daily alpha.

    Model:
        Prior:      mu ~ N(prior_mean, prior_var)
        Likelihood: x_i ~ N(mu, sigma^2)    (sigma^2 estimated from data)
        Posterior:   mu | data ~ N(mu_post, sigma_post^2)

    Returns dict with posterior parameters, P(alpha > 0), credible interval,
    and Bayes Factor (Savage-Dickey density ratio: evidence for mu != 0).
    """
    daily_excess = daily_excess.dropna()
    n = len(daily_excess)
    if n < 5:
        return None

    x_bar = float(daily_excess.mean())
    sigma2 = float(daily_excess.var(ddof=1))  # unbiased sample variance
    if sigma2 <= 0:
        return None

    if prior_var is None:
        prior_var = (0.001) ** 2  # skeptical: sd = 0.1% daily (~25% annual)

    # Conjugate posterior
    post_precision = 1.0 / prior_var + n / sigma2
    post_var = 1.0 / post_precision
    post_mean = post_var * (prior_mean / prior_var + n * x_bar / sigma2)
    post_std = np.sqrt(post_var)

    # P(alpha > 0 | data)
    prob_positive = float(1.0 - norm.cdf(0, loc=post_mean, scale=post_std))

    # 95% credible interval
    ci_lo = float(norm.ppf(0.025, loc=post_mean, scale=post_std))
    ci_hi = float(norm.ppf(0.975, loc=post_mean, scale=post_std))

    # Bayes Factor: Savage-Dickey density ratio
    #   BF_10 = prior(mu=0) / posterior(mu=0)
    #   BF > 1 means data moves posterior away from 0 -> evidence for alpha != 0
    prior_at_0 = norm.pdf(0, loc=prior_mean, scale=np.sqrt(prior_var))
    post_at_0 = norm.pdf(0, loc=post_mean, scale=post_std)
    bf_10 = float(prior_at_0 / post_at_0) if post_at_0 > 0 else np.inf

    # Annualized
    ann_alpha = post_mean * 252
    ann_vol = np.sqrt(sigma2 * 252)

    return {
        "n_days": n,
        "daily_mean": x_bar,
        "daily_std": np.sqrt(sigma2),
        "posterior_mean": post_mean,
        "posterior_std": post_std,
        "prob_positive": prob_positive,
        "ci_95_lower": ci_lo,
        "ci_95_upper": ci_hi,
        "bayes_factor": bf_10,
        "ann_alpha_pct": ann_alpha * 100,
        "ann_vol_pct": ann_vol * 100,
    }


def _bf_label(bf):
    """Jeffreys scale interpretation for Bayes Factor."""
    if bf > 100:
        return "decisive"
    if bf > 30:
        return "very strong"
    if bf > 10:
        return "strong"
    if bf > 3:
        return "substantial"
    if bf > 1:
        return "anecdotal"
    return "supports H0"


def run_bayesian_analysis(data, df, feature_cols, test_start, test_end):
    """
    Bayesian posterior analysis for all optimized ONDS strategies.
    Tests: P(daily alpha > 0 | data) under skeptical and uninformative priors.
    """
    onds_close = data["onds"]["Close"]
    rf_daily = BACKTEST["risk_free_rate"] / 252

    # Strategy signal generators for ONDS
    strat_generators = {
        "Momentum+Stop": lambda: momentum_stop_params(
            data, OPTIMIZED_PARAMS["Momentum+Stop"]),
        "Market Contrarian": lambda: market_contrarian_params(
            data, OPTIMIZED_PARAMS["Market Contrarian"]),
        "Multi-TF Momentum": lambda: multi_tf_momentum_params(
            data, OPTIMIZED_PARAMS["Multi-TF Momentum"]),
        "Mean Reversion": lambda: mean_reversion_params(
            data, OPTIMIZED_PARAMS["Mean Reversion"]),
        "ML Direction": lambda: ml_direction_params(
            df, feature_cols, OPTIMIZED_PARAMS["ML Direction"]),
        "Combined Best": lambda: combined_best_params(
            data, OPTIMIZED_PARAMS["Combined Best"]),
    }

    results = []
    for strat_name, gen_fn in strat_generators.items():
        print(f"    {strat_name}...", end=" ", flush=True)
        try:
            sig = gen_fn()
        except Exception as e:
            print(f"signal failed: {e}")
            continue

        bt = period_backtest(onds_close, sig, test_start, test_end,
                             name=strat_name)
        if bt is None:
            print("backtest failed (too few days)")
            continue

        daily_excess = bt["daily_pnl"] - rf_daily

        # Skeptical prior: sd = 0.1% daily (~25% annual)
        b_skep = bayesian_sharpe(daily_excess, prior_var=(0.001) ** 2)
        # Uninformative prior: sd = 1% daily (~158% annual)
        b_unin = bayesian_sharpe(daily_excess, prior_var=(0.01) ** 2)

        if b_skep is None or b_unin is None:
            print("too few data points")
            continue

        results.append({
            "strategy": strat_name,
            "test_sharpe": bt["sharpe"],
            "test_return": bt["total_return"],
            "n_days": b_skep["n_days"],
            # Skeptical prior
            "P(a>0)|skep": b_skep["prob_positive"],
            "BF_skep": b_skep["bayes_factor"],
            "CI95_skep_lo": b_skep["ci_95_lower"],
            "CI95_skep_hi": b_skep["ci_95_upper"],
            "ann_alpha_skep": b_skep["ann_alpha_pct"],
            # Uninformative prior
            "P(a>0)|unin": b_unin["prob_positive"],
            "BF_unin": b_unin["bayes_factor"],
            "CI95_unin_lo": b_unin["ci_95_lower"],
            "CI95_unin_hi": b_unin["ci_95_upper"],
            "ann_alpha_unin": b_unin["ann_alpha_pct"],
            # Internal
            "_skep": b_skep,
            "_unin": b_unin,
            "_daily_excess": daily_excess,
        })
        print(f"P(a>0)={b_skep['prob_positive']:.1%} [skeptical]  "
              f"BF={b_skep['bayes_factor']:.2f} ({_bf_label(b_skep['bayes_factor'])})")

    return results


# =====================================================================
# PLOTTING
# =====================================================================

def plot_cross_sectional(all_per_stock, save=True):
    """Bar chart: Sharpe ratio by stock, grouped by strategy."""
    if not all_per_stock:
        return
    df = pd.DataFrame(all_per_stock)
    strategies = df["strategy"].unique()
    n_strat = len(strategies)

    fig, axes = plt.subplots(1, n_strat, figsize=(4.5 * n_strat, 5.5), sharey=True)
    if n_strat == 1:
        axes = [axes]

    for ax, strat in zip(axes, strategies):
        sub = df[df["strategy"] == strat].set_index("ticker")
        colors = ["#2196F3" if s > 0 else "#F44336" for s in sub["sharpe"]]
        bars = ax.bar(range(len(sub)), sub["sharpe"], color=colors, edgecolor="none")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(strat, fontsize=9, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Test-Period Sharpe Ratio")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate values
        for bar, val in zip(bars, sub["sharpe"]):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:+.2f}", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=7)

    fig.suptitle("Cross-Sectional Validation: ONDS-Optimized Params on Peer Stocks",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        path = FIGURES_DIR / "cross_sectional_panel.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_bayesian_posteriors(bayes_results, save=True):
    """Posterior distributions of daily alpha with credible intervals."""
    n = len(bayes_results)
    if n == 0:
        return

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for idx, br in enumerate(bayes_results):
        ax = axes[idx]
        strat = br["strategy"]

        for label, key, color, ls in [
            ("Skeptical", "_skep", "#2196F3", "-"),
            ("Uninformative", "_unin", "#FF9800", "--"),
        ]:
            info = br[key]
            mu = info["posterior_mean"]
            sigma = info["posterior_std"]

            # Plot in annualized % units
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            y = norm.pdf(x, loc=mu, scale=sigma)
            x_pct = x * 252 * 100  # daily -> annual %
            ax.plot(x_pct, y, label=f"{label} prior", color=color,
                    linewidth=1.5, linestyle=ls)
            ax.fill_between(x_pct, y, alpha=0.1, color=color)

            # 95% CI markers
            ci_l = info["ci_95_lower"] * 252 * 100
            ci_h = info["ci_95_upper"] * 252 * 100
            ax.axvline(ci_l, color=color, linestyle=":", alpha=0.5, linewidth=0.7)
            ax.axvline(ci_h, color=color, linestyle=":", alpha=0.5, linewidth=0.7)

        ax.axvline(0, color="red", linewidth=1, alpha=0.6)
        p_skep = br["P(a>0)|skep"]
        p_unin = br["P(a>0)|unin"]
        ax.set_title(f"{strat}\n"
                     f"P(a>0): {p_skep:.1%} (skep) / {p_unin:.1%} (unin)",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Annualized Alpha (%)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Bayesian Posterior: P(Strategy Alpha > 0 | 84 Days of Data)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        path = FIGURES_DIR / "bayesian_posteriors.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def run_statistical_validation():
    print("=" * 70)
    print("  STATISTICAL VALIDATION")
    print("  Cross-Sectional Panel Test + Bayesian Posterior Analysis")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1/6] Loading data...")
    data = load_all_data()
    onds = data["onds"]
    onds_close = onds["Close"]
    idx = onds_close.index
    train_end, val_end = make_split(idx)
    test_start = idx[idx > val_end][0] if (idx > val_end).any() else val_end
    test_end = idx[-1]
    n_test = (idx > val_end).sum()
    print(f"  Test period: {test_start.date()} to {test_end.date()} ({n_test} days)")
    print(f"  Cross-section tickers: {CROSS_TICKERS}")

    # ---- Build features (for ML Direction) ----
    print("\n[2/6] Building features (for ML Direction Bayesian analysis)...")
    df = build_advanced_features(data)
    leakage = {"ret_open_close", "ret_high_low", "ret_close_high", "ret_close_low",
               "gap", "volume_change", "volume_ma_ratio", "dollar_volume",
               "price_range_pct", "ret_1d", "dow", "is_monday", "is_friday"}
    all_feat = [c for c in df.columns
                if not c.startswith("target_") and c not in leakage]
    feature_cols = select_features(df, "target_direction", n_top=20,
                                   allowed_cols=all_feat)

    # ---- Cross-sectional validation ----
    print("\n[3/6] Cross-Sectional Validation...")
    print("  Testing 4 generalizable strategies across 7 stocks")
    print("  (using ONDS-optimized params transferred to peer stocks)")
    cs_per_stock, cs_panel = run_cross_sectional(data, test_start, test_end)

    # ---- Bayesian analysis ----
    print("\n[4/6] Bayesian Posterior Analysis (ONDS test period)...")
    bayes_results = run_bayesian_analysis(
        data, df, feature_cols, test_start, test_end,
    )

    # ---- Save CSVs ----
    print("\n[5/6] Saving results...")

    # Cross-sectional per-stock results
    cs_df = pd.DataFrame(cs_per_stock)
    cs_path = REPORTS_DIR / "cross_sectional_results.csv"
    cs_df.to_csv(cs_path, index=False)
    print(f"  Saved: {cs_path}  ({len(cs_df)} rows)")

    # Cross-sectional panel stats
    if cs_panel:
        panel_df = pd.DataFrame(cs_panel)
        panel_path = REPORTS_DIR / "cross_sectional_panel_stats.csv"
        panel_df.to_csv(panel_path, index=False)
        print(f"  Saved: {panel_path}")

    # Bayesian analysis
    bayes_clean = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in bayes_results
    ]
    bayes_df = pd.DataFrame(bayes_clean)
    bayes_path = REPORTS_DIR / "bayesian_analysis.csv"
    bayes_df.to_csv(bayes_path, index=False)
    print(f"  Saved: {bayes_path}  ({len(bayes_df)} rows)")

    # ---- Plot ----
    print("\n[6/6] Plotting...")
    plot_cross_sectional(cs_per_stock)
    plot_bayesian_posteriors(bayes_results)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  CROSS-SECTIONAL PANEL RESULTS")
    print("=" * 70)
    if cs_panel:
        hdr = (f"  {'Strategy':<20s} {'Pos/Tot':>7s} {'Mean S':>7s} "
               f"{'Sign p':>8s} {'Wilcox p':>9s} {'Pool p':>8s} {'Pool N':>7s}")
        print(hdr)
        print("  " + "-" * (len(hdr.strip())))
        for p in cs_panel:
            print(f"  {p['strategy']:<20s} "
                  f"{p['n_positive']}/{p['n_stocks']:>4d} "
                  f"{p['mean_sharpe']:>+7.3f} "
                  f"{p['sign_test_p']:>8.4f} "
                  f"{p['wilcoxon_p']:>9.4f} "
                  f"{p['pooled_t_p']:>8.4f} "
                  f"{p['pooled_n']:>7d}")

    print("\n" + "=" * 70)
    print("  BAYESIAN POSTERIOR ANALYSIS")
    print("=" * 70)
    if bayes_results:
        hdr2 = (f"  {'Strategy':<20s} {'Test S':>7s} {'P(a>0) sk':>10s} "
                f"{'BF sk':>7s} {'BF interp':>12s} "
                f"{'P(a>0) un':>10s} {'Ann Alpha':>10s}")
        print(hdr2)
        print("  " + "-" * (len(hdr2.strip())))
        for br in sorted(bayes_results,
                         key=lambda x: x["P(a>0)|skep"], reverse=True):
            bf_s = br["BF_skep"]
            print(f"  {br['strategy']:<20s} "
                  f"{br['test_sharpe']:>+7.3f} "
                  f"{br['P(a>0)|skep']:>10.1%} "
                  f"{bf_s:>7.2f} "
                  f"{_bf_label(bf_s):>12s} "
                  f"{br['P(a>0)|unin']:>10.1%} "
                  f"{br['ann_alpha_skep']:>+9.1f}%")

    # ---- Interpretation ----
    print("\n" + "-" * 70)
    print("  INTERPRETATION GUIDE")
    print("-" * 70)
    print("  Cross-Sectional:")
    print("    Sign test p < 0.05  -> strategies work on significantly many stocks")
    print("    Wilcoxon p < 0.05   -> Sharpe ratios are systematically positive")
    print("    Pooled p < 0.05     -> daily returns are significantly positive")
    print("    (Pooled test has ~588 obs but correlated -- interpret cautiously)")
    print()
    print("  Bayesian:")
    print("    P(alpha > 0) > 90%  -> strong evidence of positive alpha")
    print("    P(alpha > 0) > 75%  -> moderate evidence")
    print("    BF > 3: substantial | BF > 10: strong | BF > 30: very strong")
    print("    Skeptical prior (sd=0.1%/day) is more conservative")
    print("    Uninformative prior (sd=1%/day) lets data speak")

    return cs_per_stock, cs_panel, bayes_results


if __name__ == "__main__":
    run_statistical_validation()
