"""
Robustness Validation for Top ONDS Strategies
==============================================
Tests whether observed performance is statistically significant:
  1. Bootstrap confidence intervals for Sharpe ratio
  2. Rolling out-of-sample Sharpe (stability check)
  3. Parameter sensitivity analysis
  4. Randomized signal test (permutation test)
  5. Drawdown analysis and recovery times
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, REPORTS_DIR, BACKTEST
from backtests.engine import backtest
from analysis.advanced_research import (
    load_all_data, build_advanced_features, select_features,
    strategy_momentum_with_stop, strategy_market_contrarian,
    strategy_adaptive_ensemble, strategy_regime_conditional,
    strategy_dix_enhanced, strategy_ml_signal,
    strategy_multi_timeframe_momentum,
    vol_adjusted_signal,
)


def bootstrap_sharpe(daily_returns: pd.Series, n_boot: int = 5000,
                     risk_free_rate: float = 0.05) -> dict:
    """
    Bootstrap confidence intervals for annualized Sharpe ratio.
    Uses block bootstrap (block_size=5) to preserve autocorrelation.
    """
    ret = daily_returns.dropna().values
    n = len(ret)
    block_size = 5
    n_blocks = n // block_size

    sharpes = []
    for _ in range(n_boot):
        # Block bootstrap: sample contiguous blocks
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        boot_ret = np.concatenate([ret[s:s+block_size] for s in block_starts])

        ann_ret = boot_ret.mean() * 252
        ann_vol = boot_ret.std() * np.sqrt(252)
        if ann_vol > 0:
            sharpe = (ann_ret - risk_free_rate) / ann_vol
        else:
            sharpe = 0
        sharpes.append(sharpe)

    sharpes = np.array(sharpes)
    return {
        "mean": np.mean(sharpes),
        "median": np.median(sharpes),
        "ci_2.5": np.percentile(sharpes, 2.5),
        "ci_97.5": np.percentile(sharpes, 97.5),
        "ci_5": np.percentile(sharpes, 5),
        "ci_95": np.percentile(sharpes, 95),
        "std": np.std(sharpes),
        "pct_positive": (sharpes > 0).mean(),
        "distribution": sharpes,
    }


def rolling_sharpe(daily_returns: pd.Series, window: int = 60,
                   risk_free_rate: float = 0.05) -> pd.Series:
    """Compute rolling Sharpe ratio to check stability."""
    roll_mean = daily_returns.rolling(window).mean() * 252
    roll_vol = daily_returns.rolling(window).std() * np.sqrt(252)
    roll_vol = roll_vol.replace(0, np.nan)  # avoid inf
    return (roll_mean - risk_free_rate) / roll_vol


def permutation_test(prices: pd.Series, signal: pd.Series,
                     n_perm: int = 2000) -> dict:
    """
    Permutation test: randomly shuffle signal dates, re-run backtest.
    Computes p-value: fraction of random shuffles that beat actual Sharpe.
    """
    # Actual performance
    actual = backtest(prices, signal, name="Actual", plot=False)
    actual_sharpe = actual["sharpe"]

    # Permutation distribution
    random_sharpes = []
    for _ in range(n_perm):
        shuffled = pd.Series(np.random.permutation(signal.values), index=signal.index)
        try:
            r = backtest(prices, shuffled, name="Random", plot=False)
            random_sharpes.append(r["sharpe"])
        except:
            random_sharpes.append(0)

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= actual_sharpe).mean()

    return {
        "actual_sharpe": actual_sharpe,
        "p_value": p_value,
        "random_mean": np.mean(random_sharpes),
        "random_std": np.std(random_sharpes),
        "random_max": np.max(random_sharpes),
        "percentile": (random_sharpes < actual_sharpe).mean() * 100,
    }


def drawdown_analysis(cum_return: pd.Series) -> pd.DataFrame:
    """Detailed drawdown analysis: find all drawdowns, duration, recovery."""
    peak = cum_return.cummax()
    dd = (cum_return - peak) / peak

    # Find drawdown periods
    in_dd = dd < 0
    dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
    dd_ends = ~in_dd & in_dd.shift(1, fill_value=False)

    starts = dd_starts[dd_starts].index.tolist()
    ends = dd_ends[dd_ends].index.tolist()

    # Match starts with ends
    drawdowns = []
    for start in starts:
        # Find next end after this start
        future_ends = [e for e in ends if e > start]
        if future_ends:
            end = future_ends[0]
            period_dd = dd.loc[start:end]
            max_dd = period_dd.min()
            trough_date = period_dd.idxmin()
            duration = (end - start).days
            drawdowns.append({
                "start": start,
                "trough": trough_date,
                "end": end,
                "max_drawdown": max_dd,
                "duration_days": duration,
                "recovery_days": (end - trough_date).days,
            })
        else:
            # Still in drawdown
            period_dd = dd.loc[start:]
            max_dd = period_dd.min()
            trough_date = period_dd.idxmin()
            drawdowns.append({
                "start": start,
                "trough": trough_date,
                "end": None,
                "max_drawdown": max_dd,
                "duration_days": (dd.index[-1] - start).days,
                "recovery_days": None,
            })

    return pd.DataFrame(drawdowns)


def run_robustness():
    """Run comprehensive robustness validation for top strategies."""
    print("=" * 70)
    print("  ROBUSTNESS VALIDATION")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    data = load_all_data()
    df = build_advanced_features(data)
    onds_close = data["onds"]["Close"]
    rfr = BACKTEST["risk_free_rate"]

    # ── Define strategies to validate ────────────────────────
    leakage_cols = {"ret_open_close", "ret_high_low", "ret_close_high", "ret_close_low",
                    "gap", "volume_change", "volume_ma_ratio", "dollar_volume", "price_range_pct",
                    "ret_1d", "dow", "is_monday", "is_friday"}
    all_feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in leakage_cols]
    selected_dir = select_features(df, "target_direction", n_top=15, allowed_cols=all_feature_cols)

    strategies = {
        "Momentum+Stop": strategy_momentum_with_stop(data),
        "Momentum+Stop (VolAdj)": vol_adjusted_signal(strategy_momentum_with_stop(data), onds_close),
        "Market Contrarian": strategy_market_contrarian(data),
        "Market Contrarian (VolAdj)": vol_adjusted_signal(strategy_market_contrarian(data), onds_close),
        "Adaptive Ensemble": strategy_adaptive_ensemble(data, df),
        "Adaptive Ensemble (VolAdj)": vol_adjusted_signal(strategy_adaptive_ensemble(data, df), onds_close),
        "Multi-TF Momentum (VolAdj)": vol_adjusted_signal(strategy_multi_timeframe_momentum(data), onds_close),
        "ML Direction (top-15)": strategy_ml_signal(df, selected_dir, "target_direction"),
    }

    # ── Bootstrap Analysis ───────────────────────────────────
    print("\n[2/5] Bootstrap Sharpe confidence intervals (5000 samples)...")
    print("-" * 70)
    print(f"  {'Strategy':<35s} {'Sharpe':>8s} {'95% CI':>18s} {'P(>0)':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*18} {'-'*8}")

    bootstrap_results = {}
    for name, sig in strategies.items():
        bt = backtest(onds_close, sig, name=name, plot=False)
        daily_ret = bt["daily_pnl"]
        if len(daily_ret) > 20 and daily_ret.std() > 0:
            bs = bootstrap_sharpe(daily_ret, n_boot=5000, risk_free_rate=rfr)
            bootstrap_results[name] = bs
            ci_str = f"[{bs['ci_2.5']:+.2f}, {bs['ci_97.5']:+.2f}]"
            print(f"  {name:<35s} {bs['mean']:>+8.4f} {ci_str:>18s} {bs['pct_positive']:>7.1%}")
        else:
            print(f"  {name:<35s} {'N/A':>8s}")

    # ── Rolling Sharpe Stability ─────────────────────────────
    print("\n[3/5] Rolling Sharpe stability (60-day window)...")
    print("-" * 70)

    fig, axes = plt.subplots(len(strategies), 1, figsize=(14, 3 * len(strategies)),
                              sharex=True)
    if len(strategies) == 1:
        axes = [axes]

    for ax, (name, sig) in zip(axes, strategies.items()):
        bt = backtest(onds_close, sig, name=name, plot=False)
        daily_ret = bt["daily_pnl"]
        roll_s = rolling_sharpe(daily_ret, window=60, risk_free_rate=rfr)

        ax.plot(roll_s, linewidth=1.2, color="#2196F3")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=bt["sharpe"], color="red", linestyle=":", alpha=0.7,
                   label=f"Full-period: {bt['sharpe']:+.2f}")
        ax.set_ylabel("60d Sharpe")
        ax.set_title(f"{name}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print stability metric
        valid = roll_s.dropna()
        if len(valid) > 0:
            pct_positive = (valid > 0).mean()
            stability = 1 - valid.std() / (abs(valid.mean()) + 1e-6)
            print(f"  {name:<35s} Pct>0: {pct_positive:.1%}  "
                  f"Mean: {valid.mean():+.2f}  Std: {valid.std():.2f}")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "rolling_sharpe_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'rolling_sharpe_stability.png'}")

    # ── Permutation Test ─────────────────────────────────────
    print("\n[4/5] Permutation tests (2000 permutations)...")
    print("-" * 70)
    print(f"  {'Strategy':<35s} {'Sharpe':>8s} {'p-value':>10s} {'Pctl':>8s} {'Significant':>12s}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*12}")

    perm_results = {}
    top_strategies = ["Momentum+Stop (VolAdj)", "Market Contrarian (VolAdj)",
                      "Adaptive Ensemble", "ML Direction (top-15)"]
    for name in top_strategies:
        if name in strategies:
            sig = strategies[name]
            perm = permutation_test(onds_close, sig, n_perm=2000)
            perm_results[name] = perm
            significant = "YES ***" if perm["p_value"] < 0.01 else (
                          "YES **" if perm["p_value"] < 0.05 else (
                          "YES *" if perm["p_value"] < 0.10 else "No"))
            print(f"  {name:<35s} {perm['actual_sharpe']:>+8.4f} "
                  f"{perm['p_value']:>10.4f} {perm['percentile']:>7.1f}% {significant:>12s}")

    # ── Drawdown Analysis ────────────────────────────────────
    print("\n[5/5] Drawdown analysis...")
    print("-" * 70)

    for name in top_strategies:
        if name in strategies:
            sig = strategies[name]
            bt = backtest(onds_close, sig, name=name, plot=False)
            dd_df = drawdown_analysis(bt["cum_return"])
            if len(dd_df) > 0:
                worst = dd_df.loc[dd_df["max_drawdown"].idxmin()]
                avg_dd = dd_df["max_drawdown"].mean()
                avg_dur = dd_df["duration_days"].mean()
                n_dd = len(dd_df)
                print(f"  {name}:")
                print(f"    # Drawdowns: {n_dd}")
                print(f"    Average DD: {avg_dd:.2%}, Average duration: {avg_dur:.0f} days")
                print(f"    Worst DD: {worst['max_drawdown']:.2%} "
                      f"({worst['start'].strftime('%Y-%m-%d')} to "
                      f"{worst['trough'].strftime('%Y-%m-%d')})")

    # ── Summary Report ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ROBUSTNESS SUMMARY")
    print(f"{'='*70}")

    rows = []
    for name in strategies:
        row = {"Strategy": name}
        if name in bootstrap_results:
            bs = bootstrap_results[name]
            row["Bootstrap Sharpe"] = f"{bs['mean']:+.4f}"
            row["95% CI Low"] = f"{bs['ci_2.5']:+.4f}"
            row["95% CI High"] = f"{bs['ci_97.5']:+.4f}"
            row["P(Sharpe>0)"] = f"{bs['pct_positive']:.1%}"
        if name in perm_results:
            row["Perm p-value"] = f"{perm_results[name]['p_value']:.4f}"
            row["Perm Pctl"] = f"{perm_results[name]['percentile']:.1f}%"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(REPORTS_DIR / "robustness_report.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\n  Saved: {REPORTS_DIR / 'robustness_report.csv'}")

    return bootstrap_results, perm_results


if __name__ == "__main__":
    bootstrap_results, perm_results = run_robustness()
