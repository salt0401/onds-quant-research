"""
Meta-Ensemble: Stacked Model Combination for ONDS
===================================================
Plan 16 -- Combines ALL models/strategies into one meta-ensemble:

A. Enhanced Distribution Features
   - Log-return moments (mean, std, skew, kurtosis) for 5d/10d/20d
   - Jarque-Bera normality test statistic (20d)
   - Tail ratio (20d)
   - Hurst exponent (20d, R/S method)
   - SSVI static features (rho, eta, atm_iv)

B. Signal Stacking (Level 1)
   - 12 rule-based signals from advanced_research.py
   - 3 IV-based signals from iv_surface.py
   - 3 ML walk-forward signals (RF, XGBoost, LightGBM)
   = 18 total out-of-sample signals

C. Meta-Learner (Level 2)
   - ML Meta-Ensemble: LogReg(ElasticNet) stacker on ~47 meta-features
   - Rank-Weighted Ensemble: trailing Sharpe-weighted (non-ML benchmark)

D. Full Validation (60/10/30 + bootstrap + permutation + Bayesian)

E. Output: CSVs, correlation heatmap, equity curves, feature importance
"""
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sys
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FIGURES_DIR, REPORTS_DIR, BACKTEST
from backtests.engine import backtest, print_results
from analysis.advanced_research import (
    load_all_data, build_advanced_features, vol_adjusted_signal,
    strategy_peer_leadlag, strategy_market_contrarian,
    strategy_regime_conditional, strategy_volatility_breakout,
    strategy_dix_enhanced, strategy_mean_reversion,
    strategy_momentum_with_stop, strategy_gap_fade,
    strategy_volume_spike, strategy_rsi_divergence,
    strategy_multi_timeframe_momentum, strategy_bollinger_mean_reversion,
)
from analysis.iv_surface import (
    build_iv_proxy_features,
    strategy_vrp_mean_reversion,
    strategy_iv_regime_conditional,
    strategy_vol_surface_composite,
    fit_ssvi_surface,
)
from analysis.ml_models import (
    walk_forward_predict, WalkForwardConfig, get_safe_features,
    rf_factory, xgb_factory, lgbm_factory, full_validation,
)
from analysis.param_optimization import make_split, period_backtest
from analysis.robustness import bootstrap_sharpe, permutation_test


# =====================================================================
# SECTION A: ENHANCED DISTRIBUTION FEATURES
# =====================================================================

def _hurst_rs(series: np.ndarray) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) method.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending

    Uses sub-series of lengths [10, 15, 20] to fit log(R/S) vs log(n).
    """
    n = len(series)
    if n < 20:
        return 0.5

    sub_lengths = [s for s in [10, 15, 20] if s <= n]
    if len(sub_lengths) < 2:
        return 0.5

    rs_values = []
    for length in sub_lengths:
        n_subs = n // length
        if n_subs < 1:
            continue
        rs_list = []
        for i in range(n_subs):
            sub = series[i * length:(i + 1) * length]
            mean_sub = np.mean(sub)
            deviations = np.cumsum(sub - mean_sub)
            r = np.max(deviations) - np.min(deviations)
            s = np.std(sub, ddof=1)
            if s > 1e-12:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append((length, np.mean(rs_list)))

    if len(rs_values) < 2:
        return 0.5

    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    # Linear regression: log(R/S) = H * log(n) + c
    slope, _, _, _, _ = stats.linregress(log_n, log_rs)
    return np.clip(slope, 0.0, 1.0)


def build_distribution_features(data: dict) -> pd.DataFrame:
    """
    Build ~19 new distribution features from log returns + normality tests.

    Features (all shifted by 1 day for causality):
    - log_ret_mean_Xd, log_ret_std_Xd, log_ret_skew_Xd, log_ret_kurt_Xd
      for X in {5, 10, 20}  -> 12 features
    - jb_stat_20d: Jarque-Bera test statistic (departure from normality)
    - tail_ratio_20d: |5th percentile| / |95th percentile| of returns
    - hurst_20d: Hurst exponent (R/S method)
    - ssvi_rho: SSVI skew parameter (static)
    - ssvi_eta: SSVI vol-of-vol (static)
    - ssvi_atm_iv: ATM IV from nearest expiration (static)
    = 15 rolling + 3 static + 1 normality = 19 total
    """
    onds = data["onds"]
    close = onds["Close"]
    idx = onds.index

    # Log returns
    log_ret = np.log(close / close.shift(1))

    features = pd.DataFrame(index=idx)

    # --- Rolling log-return moments (4 stats x 3 windows = 12 features) ---
    for window in [5, 10, 20]:
        features[f"log_ret_mean_{window}d"] = (
            log_ret.rolling(window).mean().shift(1)
        )
        features[f"log_ret_std_{window}d"] = (
            log_ret.rolling(window).std().shift(1)
        )
        features[f"log_ret_skew_{window}d"] = (
            log_ret.rolling(window).skew().shift(1)
        )
        features[f"log_ret_kurt_{window}d"] = (
            log_ret.rolling(window).kurt().shift(1)
        )

    # --- Jarque-Bera test statistic (20d) ---
    def _jb_stat(x):
        if len(x) < 8:
            return 0.0
        jb, _ = stats.jarque_bera(x)
        return jb

    features["jb_stat_20d"] = (
        log_ret.rolling(20).apply(_jb_stat, raw=True).shift(1)
    )

    # --- Tail ratio (20d): |5th percentile| / |95th percentile| ---
    def _tail_ratio(x):
        p5 = np.percentile(x, 5)
        p95 = np.percentile(x, 95)
        if abs(p95) < 1e-12:
            return 1.0
        return abs(p5) / abs(p95)

    features["tail_ratio_20d"] = (
        log_ret.rolling(20).apply(_tail_ratio, raw=True).shift(1)
    )

    # --- Hurst exponent (20d) ---
    features["hurst_20d"] = (
        log_ret.rolling(20).apply(
            lambda x: _hurst_rs(x.values) if len(x) >= 20 else 0.5,
            raw=False
        ).shift(1)
    )

    # --- SSVI static features ---
    ssvi_rho, ssvi_eta, ssvi_atm_iv = _get_ssvi_features()
    features["ssvi_rho"] = ssvi_rho
    features["ssvi_eta"] = ssvi_eta
    features["ssvi_atm_iv"] = ssvi_atm_iv

    features = features.replace([np.inf, -np.inf], np.nan)

    n_feat = len(features.columns)
    n_valid = features.dropna(how="all").shape[0]
    print(f"  Distribution features: {n_feat} columns, "
          f"{n_valid}/{len(features)} valid rows")

    return features


def _get_ssvi_features() -> Tuple[float, float, float]:
    """
    Load SSVI parameters from options data (static, one snapshot).

    Returns (rho, eta, atm_iv) or (0, 0, 0) if unavailable.
    """
    try:
        from config import TARGET_TICKER
        from analysis.options_iv import load_options_data
        from collectors.prices import load_prices

        options_df = load_options_data(TARGET_TICKER)
        if options_df.empty:
            print("  SSVI: no options data, using zeros")
            return 0.0, 0.0, 0.0

        onds = load_prices(TARGET_TICKER)
        current_price = float(onds["Close"].iloc[-1])
        result = fit_ssvi_surface(options_df, current_price)

        if "error" in result:
            print(f"  SSVI: fit failed ({result['error']}), using zeros")
            return 0.0, 0.0, 0.0

        rho = result["params"]["rho"]
        eta = result["params"]["eta"]
        # ATM IV from nearest expiration theta
        thetas = list(result["theta_by_tau"].values())
        atm_iv = np.sqrt(np.mean(thetas)) if thetas else 0.0

        print(f"  SSVI static: rho={rho:+.3f}, eta={eta:.3f}, "
              f"atm_iv={atm_iv:.3f}")
        return rho, eta, atm_iv

    except Exception as e:
        print(f"  SSVI: could not load ({e}), using zeros")
        return 0.0, 0.0, 0.0


# =====================================================================
# SECTION B: SIGNAL STACKING (LEVEL 1)
# =====================================================================

def generate_level1_signals(data: dict, df: pd.DataFrame,
                             iv_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 18 out-of-sample signals from 3 categories.

    Category 1: Rule-based (12 signals, fast)
    Category 2: IV-based (3 signals)
    Category 3: ML walk-forward (3 models, slow)

    All signals are in [-1, +1] and causally aligned.
    """
    onds = data["onds"]
    idx = onds.index
    signals = pd.DataFrame(index=idx)

    # --- Category 1: Rule-based signals (12) ---
    print("  [L1] Rule-based signals (12)...")
    t0 = time.time()

    rule_strategies = {
        "sig_peer_leadlag":     strategy_peer_leadlag(data),
        "sig_market_contrarian": strategy_market_contrarian(data),
        "sig_regime_cond":      strategy_regime_conditional(data),
        "sig_vol_breakout":     strategy_volatility_breakout(data),
        "sig_dix_enhanced":     strategy_dix_enhanced(data),
        "sig_mean_reversion":   strategy_mean_reversion(data),
        "sig_momentum_stop":    strategy_momentum_with_stop(data),
        "sig_gap_fade":         strategy_gap_fade(data),
        "sig_volume_spike":     strategy_volume_spike(data),
        "sig_rsi_divergence":   strategy_rsi_divergence(data),
        "sig_mtf_momentum":     strategy_multi_timeframe_momentum(data),
        "sig_bb_mean_rev":      strategy_bollinger_mean_reversion(data),
    }

    for name, sig in rule_strategies.items():
        signals[name] = sig.reindex(idx).fillna(0)

    print(f"    Done in {time.time() - t0:.1f}s")

    # --- Category 2: IV-based signals (3) ---
    print("  [L1] IV-based signals (3)...")
    t0 = time.time()

    iv_strategies = {
        "sig_vrp_mean_rev":     strategy_vrp_mean_reversion(data, iv_features),
        "sig_iv_regime_cond":   strategy_iv_regime_conditional(data, iv_features),
        "sig_vol_surf_comp":    strategy_vol_surface_composite(data, iv_features),
    }

    for name, sig in iv_strategies.items():
        signals[name] = sig.reindex(idx).fillna(0)

    print(f"    Done in {time.time() - t0:.1f}s")

    # --- Category 3: ML walk-forward signals (3 models) ---
    print("  [L1] ML walk-forward signals (3 models)...")
    feature_cols = get_safe_features(df)
    target = "target_direction"
    config = WalkForwardConfig()

    ml_factories = {
        "sig_ml_rf":     rf_factory,
        "sig_ml_xgb":    xgb_factory,
        "sig_ml_lgbm":   lgbm_factory,
    }

    for name, factory in ml_factories.items():
        t0 = time.time()
        sig = walk_forward_predict(
            df, feature_cols, target, factory, config, verbose=False
        )
        signals[name] = sig.reindex(idx).fillna(0)
        n_active = (signals[name] != 0).sum()
        print(f"    {name}: {time.time() - t0:.0f}s, "
              f"{n_active} active days")

    # Summary
    n_signals = len(signals.columns)
    nonzero_pct = [(signals[c] != 0).mean() * 100 for c in signals.columns]
    print(f"  Total L1 signals: {n_signals}")
    print(f"  Activity range: {min(nonzero_pct):.0f}% - {max(nonzero_pct):.0f}%")

    return signals


# =====================================================================
# SECTION C: META-LEARNER (LEVEL 2)
# =====================================================================

def build_meta_features(l1_signals: pd.DataFrame,
                         dist_features: pd.DataFrame,
                         iv_features: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Level 1 signals + distribution features + IV features
    into a single meta-feature matrix.

    ~47 features:
    - 18 L1 signal values
    - ~19 distribution features (log moments, JB, tail ratio, Hurst, SSVI)
    - 7 IV proxy features
    - 3 SSVI static (already in dist_features)
    """
    meta = pd.concat([l1_signals, dist_features, iv_features], axis=1)
    # Drop duplicate columns (SSVI statics might appear in both)
    meta = meta.loc[:, ~meta.columns.duplicated()]
    meta = meta.replace([np.inf, -np.inf], np.nan)

    print(f"  Meta-features: {meta.shape[1]} columns, "
          f"{meta.dropna().shape[0]} clean rows")
    return meta


def meta_ensemble_ml(meta_features: pd.DataFrame,
                      onds_ret: pd.Series) -> pd.Series:
    """
    ML Meta-Ensemble: LogReg(ElasticNet) stacker.

    Walk-forward with:
      train = [i-80 : i-3]  (shorter window, signals sparse early)
      purge = [i-3 : i]     (3-day purge)
      test = [i]

    Target: next-day direction (1 if ret > 0, else 0).
    """
    # Build target: next-day direction
    target = (onds_ret.shift(-1) > 0).astype(int)
    target.name = "target_dir"

    # Merge meta features + target
    df = meta_features.copy()
    df["_target"] = target

    # Drop rows with all NaN in features
    feat_cols = [c for c in df.columns if c != "_target"]
    df = df.dropna(subset=["_target"])

    signal = pd.Series(0.0, index=df.index)

    train_window = 80
    purge_gap = 3
    min_train = 40
    start_idx = train_window + purge_gap

    if start_idx >= len(df):
        print("    WARNING: Not enough data for meta walk-forward")
        return signal

    scaler = StandardScaler()
    n_long, n_short, n_flat = 0, 0, 0

    for i in range(start_idx, len(df)):
        train_start = max(0, i - train_window - purge_gap)
        train_end = i - purge_gap

        X_train = df[feat_cols].iloc[train_start:train_end]
        y_train = df["_target"].iloc[train_start:train_end]
        X_test = df[feat_cols].iloc[i:i + 1]

        # Drop columns with all NaN in training window
        valid_cols = X_train.columns[X_train.notna().any()]
        X_train = X_train[valid_cols].fillna(0)
        X_test = X_test[valid_cols].fillna(0)

        if len(X_train) < min_train or len(valid_cols) < 5:
            continue

        # Scale
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        if np.any(np.isnan(X_tr_scaled)) or np.any(np.isnan(X_te_scaled)):
            continue

        # LogReg ElasticNet meta-learner
        model = LogisticRegression(
            C=0.1, penalty="elasticnet", solver="saga",
            l1_ratio=0.5, max_iter=2000, random_state=42,
        )

        try:
            model.fit(X_tr_scaled, y_train)
            prob = model.predict_proba(X_te_scaled)[0]
        except Exception:
            continue

        if len(prob) >= 2:
            p_up = prob[1] if len(prob) == 2 else prob[-1]
            if p_up > 0.58:
                signal.iloc[i] = 1.0
                n_long += 1
            elif p_up < 0.42:
                signal.iloc[i] = -1.0
                n_short += 1
            else:
                n_flat += 1

    print(f"    ML Meta: Long={n_long} Short={n_short} Flat={n_flat}")
    return signal


def meta_ensemble_rank_weighted(l1_signals: pd.DataFrame,
                                 onds_ret: pd.Series) -> pd.Series:
    """
    Rank-Weighted Ensemble (non-ML benchmark).

    Trailing 40-day holdout-based Sharpe per signal.
    Weight = max(0, sharpe) normalized.
    Rebalance every 20 days.
    """
    idx = l1_signals.index
    signal = pd.Series(0.0, index=idx)
    sig_cols = l1_signals.columns.tolist()

    eval_window = 40
    rebalance_freq = 20

    # Initialize equal weights
    weights = {c: 1.0 / len(sig_cols) for c in sig_cols}

    for i in range(eval_window + 1, len(idx)):
        if (i - eval_window - 1) % rebalance_freq == 0:
            # Compute trailing Sharpe for each signal on holdout
            trail_start = max(0, i - eval_window)
            # Use last 30% as holdout
            holdout_start = trail_start + int(eval_window * 0.7)

            new_weights = {}
            total_w = 0

            for col in sig_cols:
                pos = l1_signals[col].iloc[holdout_start:i].shift(1)
                ret = onds_ret.iloc[holdout_start:i]
                strat_ret = (pos * ret).dropna()

                if len(strat_ret) > 3 and strat_ret.std() > 0:
                    sharpe = strat_ret.mean() / strat_ret.std()
                else:
                    sharpe = 0

                w = max(0, sharpe)
                new_weights[col] = w
                total_w += w

            # Normalize
            if total_w > 0:
                for c in new_weights:
                    new_weights[c] /= total_w
            else:
                for c in new_weights:
                    new_weights[c] = 1.0 / len(sig_cols)

            weights = new_weights

        # Apply weights
        weighted = 0.0
        for col in sig_cols:
            val = l1_signals[col].iloc[i]
            if not pd.isna(val):
                weighted += val * weights[col]
        signal.iloc[i] = weighted

    return signal.clip(-1, 1)


def _get_logreg_coefficients(meta_features: pd.DataFrame,
                              onds_ret: pd.Series) -> pd.Series:
    """
    Fit a single LogReg on the full training set (60%) to extract
    coefficient importance for visualization.

    This is NOT used for trading -- only for plotting/analysis.
    """
    target = (onds_ret.shift(-1) > 0).astype(int)
    df = meta_features.copy()
    df["_target"] = target
    feat_cols = [c for c in df.columns if c != "_target"]

    # Use first 60% only
    n = len(df)
    train_end = int(n * 0.60)
    train = df.iloc[:train_end].dropna()

    if len(train) < 30:
        return pd.Series(dtype=float)

    X = train[feat_cols].fillna(0)
    y = train["_target"]

    # Drop zero-variance columns
    valid = X.columns[X.std() > 1e-8]
    X = X[valid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=0.1, penalty="elasticnet", solver="saga",
        l1_ratio=0.5, max_iter=2000, random_state=42,
    )
    model.fit(X_scaled, y)

    coefs = pd.Series(model.coef_[0], index=X.columns)
    return coefs.sort_values(key=abs, ascending=False)


# =====================================================================
# SECTION D: VALIDATION
# =====================================================================

def validate_meta_strategies(prices: pd.Series,
                              signals: Dict[str, pd.Series],
                              n_perm: int = 500) -> List[dict]:
    """
    Run full_validation on each meta-strategy signal.

    Includes: 60/10/30 split, bootstrap CI, permutation test, Bayesian.
    """
    results = []

    for name, sig in signals.items():
        n_active = (sig != 0).sum()
        if n_active < 10:
            print(f"  SKIPPED {name}: only {n_active} active days")
            continue

        print(f"\n  Validating: {name} ({n_active} active days)")
        res = full_validation(
            prices, sig, name,
            run_permutation=True, n_perm=n_perm,
        )
        results.append(res)

        print(f"    Sharpe: Full={res['Full Sharpe']:+.2f} | "
              f"Train={res['Train Sharpe']:+.2f} "
              f"Val={res['Val Sharpe']:+.2f} "
              f"Test={res['Test Sharpe']:+.2f} | "
              f"Gap={res['Train-Test Gap']:.2f}")
        print(f"    Bootstrap 95% CI: "
              f"[{res['CI 2.5%']:+.2f}, {res['CI 97.5%']:+.2f}] | "
              f"P(Sharpe>0)={res['P(Sharpe>0)']:.1%}")
        if not np.isnan(res["Perm p-value"]):
            print(f"    Permutation p={res['Perm p-value']:.4f}")

    return results


# =====================================================================
# SECTION E: OUTPUT & PLOTTING
# =====================================================================

def plot_signal_correlation(l1_signals: pd.DataFrame) -> None:
    """Plot 18x18 signal correlation heatmap."""
    corr = l1_signals.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Labels
    short_names = [c.replace("sig_", "") for c in corr.columns]
    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    # Values
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Level 1 Signal Correlation Matrix (18 signals)")
    plt.tight_layout()

    path = FIGURES_DIR / "signal_correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save CSV
    csv_path = REPORTS_DIR / "signal_correlation_matrix.csv"
    corr.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Summary: diversity check
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    mean_corr = upper.stack().mean()
    max_corr = upper.stack().abs().max()
    high_corr_pairs = (upper.stack().abs() > 0.7).sum()
    print(f"  Mean pairwise |correlation|: {mean_corr:.3f}")
    print(f"  Max absolute correlation: {max_corr:.3f}")
    print(f"  Pairs with |corr| > 0.7: {high_corr_pairs}")


def plot_equity_curves(results: List[dict]) -> None:
    """Plot equity curves for meta-ensemble strategies."""
    if not results:
        return

    # Filter to non-VolAdj for cleaner plot
    base = [r for r in results if "(VolAdj)" not in r["Model"]]
    if not base:
        base = results

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(base)))

    for r, c in zip(base, colors):
        cum = r["_cum_return"]
        label = f"{r['Model']} (S={r['Test Sharpe']:+.2f})"
        ax.plot(cum.index, cum.values, label=label, color=c, linewidth=1.3)

    # Benchmark
    if base:
        bm = base[0]["_cum_benchmark"]
        ax.plot(bm.index, bm.values, label="Buy & Hold",
                color="gray", linewidth=1, linestyle="--", alpha=0.6)

    # Mark train/val/test boundaries
    if base:
        idx = base[0]["_cum_return"].index
        train_end, val_end = make_split(idx)
        ymin, ymax = ax.get_ylim()
        ax.axvline(x=train_end, color="orange", linestyle=":", alpha=0.7)
        ax.axvline(x=val_end, color="red", linestyle=":", alpha=0.7)
        ax.text(train_end, ymax * 0.95, " Train|Val", fontsize=8,
                color="orange")
        ax.text(val_end, ymax * 0.95, " Val|Test", fontsize=8, color="red")

    ax.set_title("Meta-Ensemble Strategies -- Equity Curves")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()

    path = FIGURES_DIR / "meta_ensemble_equity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(coefs: pd.Series) -> None:
    """Plot top LogReg coefficients (meta-feature importance)."""
    if coefs.empty:
        print("  No coefficients to plot")
        return

    # Top 25 by absolute value
    top = coefs.head(25)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in top.values]
    short_names = [n.replace("sig_", "").replace("log_ret_", "lr_")
                   for n in top.index]
    ax.barh(range(len(top)), top.values, color=colors, alpha=0.8,
            edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("LogReg Coefficient (ElasticNet)")
    ax.set_title("Meta-Feature Importance (Top 25 by |coef|)")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()

    path = FIGURES_DIR / "meta_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# SECTION F: MAIN RUNNER
# =====================================================================

def run_meta_ensemble(n_perm: int = 500):
    """
    Run the complete meta-ensemble pipeline.

    1. Load data + build features
    2. Build distribution features
    3. Generate Level 1 signals
    4. Build meta-feature matrix
    5. Run meta-learners (ML + Rank-Weighted)
    6. Full validation
    7. Plots + output
    """
    print("=" * 70)
    print("  META-ENSEMBLE (Plan 16)")
    print("  Stacked combination of ALL models/strategies")
    print("=" * 70)

    # --- 1. Load data ---
    print("\n[1/7] Loading data and building base features...")
    data = load_all_data()
    df = build_advanced_features(data)
    onds_close = data["onds"]["Close"]
    onds_ret = onds_close.pct_change()

    # --- 2. Build distribution features ---
    print("\n[2/7] Building distribution features (log moments, JB, Hurst)...")
    dist_features = build_distribution_features(data)

    # Check non-zero variance
    for col in dist_features.columns:
        valid = dist_features[col].dropna()
        if len(valid) > 0 and valid.std() > 0:
            status = "OK"
        else:
            status = "ZERO-VARIANCE"
        if col.startswith("ssvi_"):
            # Static features: just print value
            val = valid.iloc[0] if len(valid) > 0 else 0
            print(f"    {col:25s}: value={val:.4f} [{status}]")
        else:
            if len(valid) > 0:
                print(f"    {col:25s}: mean={valid.mean():+.4f}  "
                      f"std={valid.std():.4f} [{status}]")

    # --- 3. Build IV proxy features ---
    print("\n[3/7] Building IV proxy features...")
    iv_features = build_iv_proxy_features(data)
    for col in iv_features.columns:
        valid = iv_features[col].dropna()
        print(f"    {col:25s}: {len(valid)} valid days")

    # --- 4. Generate Level 1 signals ---
    print("\n[4/7] Generating Level 1 signals (18 total)...")
    l1_signals = generate_level1_signals(data, df, iv_features)

    # Signal correlation analysis
    print("\n  Signal correlation analysis:")
    plot_signal_correlation(l1_signals)

    # --- 5. Build meta-feature matrix ---
    print("\n[5/7] Building meta-feature matrix...")
    meta_features = build_meta_features(l1_signals, dist_features, iv_features)

    # --- 6. Run meta-learners ---
    print("\n[6/7] Running meta-learners...")
    print("-" * 70)

    # ML Meta-Ensemble
    print("\n  >> ML Meta-Ensemble (LogReg ElasticNet stacker)...")
    t0 = time.time()
    sig_ml_meta = meta_ensemble_ml(meta_features, onds_ret)
    print(f"    Done in {time.time() - t0:.1f}s")

    # Rank-Weighted Ensemble
    print("\n  >> Rank-Weighted Ensemble (non-ML benchmark)...")
    t0 = time.time()
    sig_rank_wt = meta_ensemble_rank_weighted(l1_signals, onds_ret)
    print(f"    Done in {time.time() - t0:.1f}s")

    # Collect all signals for validation
    all_signals = {
        "ML Meta-Ensemble": sig_ml_meta,
        "Rank-Weighted Ensemble": sig_rank_wt,
    }

    # Vol-adjusted variants
    for name, sig in list(all_signals.items()):
        va_sig = vol_adjusted_signal(sig, onds_close, target_vol=0.30)
        all_signals[f"{name} (VolAdj)"] = va_sig

    # Print activity stats
    for name, sig in all_signals.items():
        n_active = (sig != 0).sum()
        n_long = (sig > 0).sum()
        n_short = (sig < 0).sum()
        print(f"    {name:40s}: {n_active} active "
              f"(L={n_long}, S={n_short})")

    # --- 7. Full validation ---
    print(f"\n[7/7] Validating meta-strategies "
          f"({n_perm} permutations each)...")
    print("-" * 70)

    results = validate_meta_strategies(
        onds_close, all_signals, n_perm=n_perm
    )

    if not results:
        print("\n  ERROR: No meta-strategies produced valid results!")
        return

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  META-ENSEMBLE VALIDATION SUMMARY")
    print("=" * 70)

    display_cols = [c for c in results[0].keys() if not c.startswith("_")]
    summary_df = pd.DataFrame([{k: r[k] for k in display_cols}
                                for r in results])
    summary_df = summary_df.sort_values("Test Sharpe", ascending=False)
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # Leakage check
    print("\n  LEAKAGE CHECK:")
    for r in results:
        gap = r["Train-Test Gap"]
        status = "OK" if gap < 2.0 else "WARNING: possible leakage"
        print(f"    {r['Model']:<40s} Gap={gap:.2f}  [{status}]")

    # --- Save results ---
    csv_path = REPORTS_DIR / "meta_ensemble_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # --- Plots ---
    print("\n  Generating plots...")
    plot_equity_curves(results)

    # Feature importance from LogReg coefficients
    coefs = _get_logreg_coefficients(meta_features, onds_ret)
    plot_feature_importance(coefs)

    # --- Final ranking ---
    print("\n" + "=" * 70)
    print("  FINAL META-ENSEMBLE RANKING (by Test Sharpe)")
    print("=" * 70)
    ranked = sorted(results, key=lambda r: r["Test Sharpe"], reverse=True)
    for i, r in enumerate(ranked):
        marker = " << BEST" if i == 0 else ""
        print(f"  {i + 1:2d}. {r['Model']:<40s} "
              f"TestSharpe={r['Test Sharpe']:+.4f}  "
              f"FullSharpe={r['Full Sharpe']:+.4f}  "
              f"Gap={r['Train-Test Gap']:.2f}  "
              f"Trades={r['N Trades']}{marker}")

    print("\n" + "=" * 70)
    print("  META-ENSEMBLE COMPLETE")
    print("=" * 70)

    return {
        "results": results,
        "l1_signals": l1_signals,
        "meta_features": meta_features,
        "dist_features": dist_features,
        "coefs": coefs,
    }


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    output = run_meta_ensemble(n_perm=500)
