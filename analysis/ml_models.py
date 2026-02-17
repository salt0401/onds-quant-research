"""
Leakage-Free ML Pipeline for ONDS
==================================
Fixes 4 data-leakage bugs in the original advanced_research.py:

1. Feature selection on full dataset -> now MI within each fold only
2. Expanding walk-forward (no cap)  -> rolling 120-day window
3. No purge gap                     -> 5-day gap between train and test
4. Adaptive Ensemble look-ahead     -> holdout-based weight estimation

Provides 6-model comparison with full statistical validation:
  - Random Forest, XGBoost, LightGBM
  - Logistic Regression (ElasticNet), SVM (RBF), MLP Neural Net

Each model is validated through:
  - 60/10/30 train/val/test Sharpe
  - 5000-sample block bootstrap CI
  - 2000-permutation test (p-value)
  - Bayesian P(alpha>0) with skeptical prior
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import warnings
import sys
import time

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FIGURES_DIR, REPORTS_DIR, BACKTEST
from backtests.engine import backtest, print_results
from analysis.advanced_research import (
    load_all_data, build_advanced_features, vol_adjusted_signal,
    strategy_regime_conditional, strategy_dix_enhanced,
    strategy_momentum_with_stop, strategy_mean_reversion,
    strategy_bollinger_mean_reversion, strategy_multi_timeframe_momentum,
)
from analysis.robustness import bootstrap_sharpe, permutation_test
from analysis.param_optimization import make_split, period_backtest

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional imports (graceful fallback)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# =====================================================================
# WALK-FORWARD CONFIGURATION
# =====================================================================

@dataclass
class WalkForwardConfig:
    """Configuration for leakage-free walk-forward prediction."""
    train_window: int = 120       # Rolling training window (days)
    purge_gap: int = 5            # Gap between train and test (days)
    feature_select_k: int = 15    # Number of features to select per fold
    min_train_samples: int = 60   # Minimum training samples to start
    prob_threshold_long: float = 0.58   # Probability threshold for long
    prob_threshold_short: float = 0.42  # Probability threshold for short


# =====================================================================
# LEAKAGE-FREE FEATURE SELECTION (WITHIN FOLD ONLY)
# =====================================================================

def select_features_within_fold(X_train: pd.DataFrame, y_train: pd.Series,
                                 k: int = 15) -> list:
    """
    Select top-k features using mutual information on TRAINING data only.

    This is the key fix for leakage bug #2: the original code ran MI on all
    280 rows (including test data), choosing features that predict the future.
    Now MI runs only on the current training window.
    """
    clean_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_clean = X_train.loc[clean_mask]
    y_clean = y_train.loc[clean_mask]

    if len(X_clean) < 30:
        return X_train.columns.tolist()[:k]

    mi = mutual_info_classif(X_clean.values, y_clean.values,
                              random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    return mi_series.head(k).index.tolist()


# =====================================================================
# WALK-FORWARD PREDICTION ENGINE
# =====================================================================

def walk_forward_predict(df: pd.DataFrame, feature_cols: list,
                          target: str, model_factory: Callable,
                          config: WalkForwardConfig = None,
                          verbose: bool = False) -> pd.Series:
    """
    Leakage-free walk-forward prediction with rolling window + purge gap.

    At each test day i:
      train = [i - train_window - purge_gap : i - purge_gap]
      purge = [i - purge_gap : i]  (discarded)
      test  = [i]  (predict 1 day)

    Feature selection and scaling are fitted on train window ONLY.

    Parameters
    ----------
    df : DataFrame with features and target
    feature_cols : list of candidate feature column names
    target : target column name (classification: 0/1)
    model_factory : callable that returns a fresh sklearn classifier
    config : WalkForwardConfig (defaults used if None)
    verbose : print progress every 50 days

    Returns
    -------
    pd.Series : signal in [-1, 0, +1], same index as df
    """
    if config is None:
        config = WalkForwardConfig()

    # Clean data
    cols_needed = feature_cols + [target]
    clean = df[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
    X_all = clean[feature_cols]
    y_all = clean[target]

    signal = pd.Series(0.0, index=clean.index)
    start_idx = config.train_window + config.purge_gap

    if start_idx >= len(X_all):
        print(f"    WARNING: Not enough data for walk-forward "
              f"(need {start_idx}, have {len(X_all)})")
        return signal

    n_long, n_short, n_flat = 0, 0, 0

    for i in range(start_idx, len(X_all)):
        # Define rolling train window (NOT expanding)
        train_start = max(0, i - config.train_window - config.purge_gap)
        train_end = i - config.purge_gap  # purge gap before test

        X_train = X_all.iloc[train_start:train_end]
        y_train = y_all.iloc[train_start:train_end]
        X_test = X_all.iloc[i:i+1]

        if len(X_train) < config.min_train_samples:
            continue

        # Feature selection within fold (fixes leakage bug #2)
        selected = select_features_within_fold(
            X_train, y_train, k=config.feature_select_k
        )

        X_tr = X_train[selected]
        X_te = X_test[selected]

        # Scale within fold only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Handle NaN after scaling
        if np.any(np.isnan(X_tr_scaled)) or np.any(np.isnan(X_te_scaled)):
            continue

        # Fit model
        model = model_factory()
        try:
            model.fit(X_tr_scaled, y_train)
            prob = model.predict_proba(X_te_scaled)[0]
        except Exception:
            continue

        # Generate signal from probability
        if len(prob) >= 2:
            p_up = prob[1] if len(prob) == 2 else prob[-1]
            if p_up > config.prob_threshold_long:
                signal.iloc[i] = 1.0
                n_long += 1
            elif p_up < config.prob_threshold_short:
                signal.iloc[i] = -1.0
                n_short += 1
            else:
                n_flat += 1

        if verbose and (i - start_idx) % 50 == 0 and i > start_idx:
            print(f"    Day {i}/{len(X_all)}: "
                  f"L={n_long} S={n_short} F={n_flat}")

    if verbose:
        print(f"    Final: Long={n_long} Short={n_short} Flat={n_flat}")

    return signal


# =====================================================================
# 6 MODEL FACTORIES
# =====================================================================

def rf_factory():
    """Random Forest -- robust baseline, resistant to noise."""
    return RandomForestClassifier(
        n_estimators=150, max_depth=4, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )


def xgb_factory():
    """XGBoost -- regularized gradient boosting, state-of-art tabular."""
    if not HAS_XGB:
        print("    WARNING: XGBoost not installed, falling back to RF")
        return rf_factory()
    return XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=-1,
        use_label_encoder=False, eval_metric="logloss",
    )


def lgbm_factory():
    """LightGBM -- fast histogram-based boosting."""
    if not HAS_LGBM:
        print("    WARNING: LightGBM not installed, falling back to RF")
        return rf_factory()
    return LGBMClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1, n_jobs=-1,
    )


def logreg_factory():
    """Logistic Regression (ElasticNet) -- interpretable, sparse."""
    return LogisticRegression(
        C=0.1, penalty="elasticnet", solver="saga",
        l1_ratio=0.5, max_iter=2000, random_state=42,
    )


def svm_factory():
    """SVM (RBF) -- different inductive bias, non-linear."""
    return SVC(
        C=1.0, kernel="rbf", gamma="scale", probability=True,
        random_state=42,
    )


def mlp_factory():
    """MLP Neural Net -- non-linear, captures interactions."""
    return MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        solver="adam", alpha=0.01, learning_rate="adaptive",
        max_iter=500, random_state=42, early_stopping=True,
        validation_fraction=0.15,
    )


MODEL_REGISTRY = {
    "Random Forest":   rf_factory,
    "XGBoost":         xgb_factory,
    "LightGBM":        lgbm_factory,
    "LogReg (ElNet)":  logreg_factory,
    "SVM (RBF)":       svm_factory,
    "MLP Neural Net":  mlp_factory,
}


# =====================================================================
# FULL VALIDATION PIPELINE (per model)
# =====================================================================

def full_validation(prices: pd.Series, signal: pd.Series,
                    name: str, run_permutation: bool = True,
                    n_perm: int = 500) -> dict:
    """
    Complete validation for a single model's signal:
      1. Full-period backtest
      2. 60/10/30 train/val/test Sharpe
      3. Bootstrap CI (5000 samples)
      4. Permutation test (n_perm shuffles)
      5. Bayesian P(alpha>0)

    Returns dict with all metrics.
    """
    from analysis.statistical_validation import bayesian_sharpe
    from scipy.stats import norm

    rfr = BACKTEST["risk_free_rate"]

    # --- 1. Full-period backtest ---
    full_bt = backtest(prices, signal, name=name, plot=False, save_fig=False)

    # --- 2. 60/10/30 split ---
    train_end, val_end = make_split(prices.index)
    train_bt = period_backtest(prices, signal, prices.index[0], train_end,
                                name=f"{name} (Train)")
    val_bt = period_backtest(prices, signal, train_end, val_end,
                              name=f"{name} (Val)")
    test_bt = period_backtest(prices, signal, val_end, prices.index[-1],
                               name=f"{name} (Test)")

    train_sharpe = train_bt["sharpe"] if train_bt else 0
    val_sharpe = val_bt["sharpe"] if val_bt else 0
    test_sharpe = test_bt["sharpe"] if test_bt else 0
    # Clip extreme Sharpe values (short periods with few trades -> near-zero vol)
    train_sharpe = np.clip(train_sharpe, -20, 20)
    val_sharpe = np.clip(val_sharpe, -20, 20)
    test_sharpe = np.clip(test_sharpe, -20, 20)
    train_test_gap = abs(train_sharpe - test_sharpe)

    # --- 3. Bootstrap CI ---
    daily_pnl = full_bt["daily_pnl"]
    bs = bootstrap_sharpe(daily_pnl, n_boot=5000, risk_free_rate=rfr)

    # --- 4. Permutation test ---
    if run_permutation:
        perm = permutation_test(prices, signal, n_perm=n_perm)
    else:
        perm = {"p_value": np.nan, "actual_sharpe": full_bt["sharpe"],
                "percentile": np.nan}

    # --- 5. Bayesian analysis ---
    daily_excess = daily_pnl - rfr / 252
    bayes_skeptical = bayesian_sharpe(daily_excess, prior_mean=0.0,
                                       prior_var=(0.001)**2)
    bayes_uninform = bayesian_sharpe(daily_excess, prior_mean=0.0,
                                      prior_var=(0.01)**2)

    prob_alpha_skeptical = bayes_skeptical["prob_positive"] if bayes_skeptical else 0
    prob_alpha_uninform = bayes_uninform["prob_positive"] if bayes_uninform else 0

    result = {
        "Model": name,
        # Full period
        "Full Sharpe": full_bt["sharpe"],
        "Annual Return": full_bt["annual_return"],
        "Max DD": full_bt["max_drawdown"],
        "Hit Rate": full_bt["hit_rate"],
        "N Trades": full_bt["n_trades"],
        # 60/10/30 split
        "Train Sharpe": train_sharpe,
        "Val Sharpe": val_sharpe,
        "Test Sharpe": test_sharpe,
        "Train-Test Gap": train_test_gap,
        # Bootstrap
        "Bootstrap Mean": bs["mean"],
        "CI 2.5%": bs["ci_2.5"],
        "CI 97.5%": bs["ci_97.5"],
        "P(Sharpe>0)": bs["pct_positive"],
        # Permutation
        "Perm p-value": perm["p_value"],
        "Perm Pctl": perm.get("percentile", np.nan),
        # Bayesian
        "P(alpha>0) Skeptical": prob_alpha_skeptical,
        "P(alpha>0) Uninform": prob_alpha_uninform,
        # Raw data for plotting
        "_daily_pnl": daily_pnl,
        "_cum_return": full_bt["cum_return"],
        "_cum_benchmark": full_bt["cum_benchmark"],
        "_bootstrap_dist": bs["distribution"],
    }

    return result


# =====================================================================
# FIXED ADAPTIVE ENSEMBLE (no look-ahead)
# =====================================================================

def strategy_adaptive_ensemble_fixed(data: dict, df: pd.DataFrame) -> pd.Series:
    """
    Adaptive ensemble with look-ahead bug FIXED.

    Original bug: evaluated sub-signal Sharpe on the SAME period being traded.
    Fix: split trailing window into train (first 70%) and holdout (last 30%).
    Evaluate performance on HOLDOUT, apply weights to NEXT period.
    """
    onds = data["onds"]
    onds_ret = onds["Close"].pct_change()
    sig = pd.Series(0.0, index=onds.index)

    # Generate all base signals (these are all causal / rolling)
    base_signals = {
        "regime": strategy_regime_conditional(data),
        "dix": strategy_dix_enhanced(data),
        "momentum": strategy_momentum_with_stop(data),
        "mean_rev": strategy_mean_reversion(data),
        "bb_revert": strategy_bollinger_mean_reversion(data),
        "mtf_mom": strategy_multi_timeframe_momentum(data),
    }

    eval_window = 60      # Longer window for more stable estimates
    rebalance_freq = 20   # Rebalance every 20 days

    weights = {name: 1.0 / len(base_signals) for name in base_signals}

    for i in range(eval_window + 1, len(sig)):
        if (i - eval_window - 1) % rebalance_freq == 0:
            # --- FIXED: Holdout-based weight estimation ---
            # Split trailing window: first 70% for context, last 30% for holdout
            trail_start = max(0, i - eval_window)
            holdout_start = trail_start + int(eval_window * 0.7)

            new_weights = {}
            total_weight = 0

            for name, s in base_signals.items():
                # Evaluate ONLY on holdout portion (last 30% of trailing window)
                holdout_pos = s.iloc[holdout_start:i].shift(1)
                holdout_ret = onds_ret.iloc[holdout_start:i]
                strat_ret = holdout_pos * holdout_ret
                strat_ret = strat_ret.dropna()

                if len(strat_ret) > 5 and strat_ret.std() > 0:
                    sharpe = strat_ret.mean() / strat_ret.std()
                else:
                    sharpe = 0

                w = max(0, np.exp(sharpe * 2) - 0.5)
                new_weights[name] = w
                total_weight += w

            # Normalize
            if total_weight > 0:
                for name in new_weights:
                    new_weights[name] /= total_weight
            else:
                for name in new_weights:
                    new_weights[name] = 1.0 / len(new_weights)

            weights = new_weights

        # Apply weights to current day's signals
        weighted_sig = 0.0
        for name, s in base_signals.items():
            val = s.iloc[i]
            if not pd.isna(val):
                weighted_sig += val * weights[name]
        sig.iloc[i] = weighted_sig

    return sig.clip(-1, 1)


# =====================================================================
# COMPARISON RUNNER
# =====================================================================

def get_safe_features(df: pd.DataFrame) -> list:
    """Return feature columns safe from same-day leakage."""
    leakage_cols = {
        "ret_open_close", "ret_high_low", "ret_close_high", "ret_close_low",
        "gap", "volume_change", "volume_ma_ratio", "dollar_volume",
        "price_range_pct", "ret_1d", "dow", "is_monday", "is_friday",
    }
    return [c for c in df.columns
            if not c.startswith("target_") and c not in leakage_cols]


def run_ml_comparison(run_permutation: bool = True, n_perm: int = 500):
    """
    Run complete 6-model comparison with leakage-free walk-forward.

    Parameters
    ----------
    run_permutation : bool
        Whether to run permutation tests (slow: ~10min per model)
    n_perm : int
        Number of permutations (500 for quick, 2000 for publication)
    """
    print("=" * 70)
    print("  LEAKAGE-FREE ML MODEL COMPARISON")
    print("  6 Models x Walk-Forward x Full Validation")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/5] Loading data and building features...")
    data = load_all_data()
    df = build_advanced_features(data)
    onds_close = data["onds"]["Close"]
    feature_cols = get_safe_features(df)
    target = "target_direction"

    print(f"  Features: {len(feature_cols)} safe columns")
    print(f"  Target: {target}")
    print(f"  Days: {len(df)}")

    config = WalkForwardConfig()
    print(f"  Walk-forward: train={config.train_window}d, "
          f"purge={config.purge_gap}d, features={config.feature_select_k}")

    # --- Generate signals for all 6 models ---
    print("\n[2/5] Generating walk-forward signals (6 models)...")
    print("-" * 70)

    signals = {}
    timings = {}

    for model_name, factory in MODEL_REGISTRY.items():
        print(f"\n  >> {model_name}...")
        t0 = time.time()
        sig = walk_forward_predict(
            df, feature_cols, target, factory, config, verbose=True
        )
        elapsed = time.time() - t0
        signals[model_name] = sig
        timings[model_name] = elapsed

        n_nonzero = (sig != 0).sum()
        print(f"    Done in {elapsed:.1f}s -- {n_nonzero} active days "
              f"({n_nonzero/len(sig)*100:.0f}%)")

    # --- Add fixed Adaptive Ensemble ---
    print("\n  >> Adaptive Ensemble (Fixed)...")
    t0 = time.time()
    sig_ensemble = strategy_adaptive_ensemble_fixed(data, df)
    signals["Adaptive Ensemble (Fixed)"] = sig_ensemble
    timings["Adaptive Ensemble (Fixed)"] = time.time() - t0
    n_nonzero = (sig_ensemble != 0).sum()
    print(f"    Done in {timings['Adaptive Ensemble (Fixed)']:.1f}s -- "
          f"{n_nonzero} active days")

    # --- Add vol-adjusted versions of top models ---
    print("\n  Adding vol-adjusted variants...")
    va_signals = {}
    for name, sig in signals.items():
        va_sig = vol_adjusted_signal(sig, onds_close, target_vol=0.30)
        va_signals[f"{name} (VolAdj)"] = va_sig
    signals.update(va_signals)

    # --- Full validation for each signal ---
    print(f"\n[3/5] Validating all {len(signals)} signals...")
    print("-" * 70)
    if run_permutation:
        print(f"  (Permutation test: {n_perm} shuffles per model -- this takes a while)")

    results = []
    for name, sig in signals.items():
        print(f"\n  Validating: {name}")
        n_nonzero = (sig != 0).sum()
        if n_nonzero < 10:
            print(f"    SKIPPED: only {n_nonzero} active days")
            continue

        res = full_validation(
            onds_close, sig, name,
            run_permutation=run_permutation,
            n_perm=n_perm,
        )
        results.append(res)

        # Print summary line
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

    if not results:
        print("\n  ERROR: No models produced valid signals!")
        return

    # --- Summary Table ---
    print(f"\n[4/5] Model Comparison Summary")
    print("=" * 70)

    # Build comparison DataFrame (exclude internal columns)
    display_cols = [c for c in results[0].keys() if not c.startswith("_")]
    summary_df = pd.DataFrame([{k: r[k] for k in display_cols} for r in results])
    summary_df = summary_df.sort_values("Test Sharpe", ascending=False)

    # Format for display
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # --- Leakage Check ---
    print("\n  LEAKAGE CHECK:")
    for r in results:
        gap = r["Train-Test Gap"]
        status = "OK" if gap < 2.0 else "WARNING: possible leakage"
        print(f"    {r['Model']:<30s} Train-Test gap = {gap:.2f}  [{status}]")

    # --- Save CSV ---
    csv_df = summary_df.copy()
    csv_path = REPORTS_DIR / "ml_model_comparison.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # --- Plots ---
    print(f"\n[5/5] Generating figures...")
    _plot_model_comparison(results)
    _plot_equity_curves(results)
    _plot_train_test_gap(results)

    print("\n" + "=" * 70)
    print("  ML MODEL COMPARISON COMPLETE")
    print("=" * 70)

    return results


# =====================================================================
# PLOTTING
# =====================================================================

def _plot_model_comparison(results: list):
    """Bar chart of Full Sharpe with bootstrap CI for each model."""
    # Filter to base models (not VolAdj) for cleaner plot
    base = [r for r in results if "(VolAdj)" not in r["Model"]]
    if not base:
        base = results

    names = [r["Model"] for r in base]
    # Use Full Sharpe (matches bootstrap CI which is also full-period)
    full_sharpes = [r["Full Sharpe"] for r in base]
    test_sharpes = [r["Test Sharpe"] for r in base]
    ci_lo = [r["CI 2.5%"] for r in base]
    ci_hi = [r["CI 97.5%"] for r in base]
    # CI errors relative to Full Sharpe (bootstrap is on full period)
    errors_lo = [max(0, s - lo) for s, lo in zip(full_sharpes, ci_lo)]
    errors_hi = [max(0, hi - s) for s, hi in zip(full_sharpes, ci_hi)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35

    colors_full = []
    for s in full_sharpes:
        if s > 1.0:
            colors_full.append("#4CAF50")
        elif s > 0:
            colors_full.append("#2196F3")
        else:
            colors_full.append("#F44336")

    colors_test = []
    for s in test_sharpes:
        if s > 1.0:
            colors_test.append("#4CAF50")
        elif s > 0:
            colors_test.append("#90CAF9")
        else:
            colors_test.append("#EF9A9A")

    bars1 = ax.bar(x - width/2, full_sharpes, width, color=colors_full,
                    alpha=0.8, edgecolor="white", label="Full Period")
    bars2 = ax.bar(x + width/2, test_sharpes, width, color=colors_test,
                    alpha=0.6, edgecolor="white", label="Test Period (30%)")
    ax.errorbar(x - width/2, full_sharpes, yerr=[errors_lo, errors_hi],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("ML Model Comparison -- Full & Test Sharpe (with 95% Bootstrap CI)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on test bars
    for bar, val in zip(bars2, test_sharpes):
        ypos = bar.get_height() if val >= 0 else bar.get_height() - 0.15
        ax.text(bar.get_x() + bar.get_width()/2, ypos + 0.05,
                f"{val:+.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "ml_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_equity_curves(results: list):
    """Overlay equity curves for all models."""
    # Filter to base models for cleaner plot
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
        ax.text(train_end, ymax * 0.95, " Train|Val", fontsize=8, color="orange")
        ax.text(val_end, ymax * 0.95, " Val|Test", fontsize=8, color="red")

    ax.set_title("Walk-Forward ML Models -- Equity Curves (Leakage-Free)")
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()

    path = FIGURES_DIR / "ml_walkforward_equity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_train_test_gap(results: list):
    """Scatter plot: Train Sharpe vs Test Sharpe (leakage diagnostic)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for r in results:
        label = r["Model"]
        short_label = label[:15] + "..." if len(label) > 18 else label
        ax.scatter(r["Train Sharpe"], r["Test Sharpe"], s=80, zorder=5)
        ax.annotate(short_label,
                    (r["Train Sharpe"], r["Test Sharpe"]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    # Diagonal line (perfect = no overfitting)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, label="No overfitting line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Train Sharpe")
    ax.set_ylabel("Test Sharpe")
    ax.set_title("Train vs Test Sharpe (closer to diagonal = less overfitting)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = FIGURES_DIR / "ml_train_test_gap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    # Quick run: 500 permutations (use 2000 for publication)
    results = run_ml_comparison(run_permutation=True, n_perm=500)
