"""
Train/Test Split Validation
============================
Proper out-of-sample test: develop on first 70% of data, evaluate on last 30%.
This answers: "Does the strategy work on data it's never seen?"

Also compares every strategy against the buy-and-hold benchmark to separate
genuine alpha from simply riding the stock's natural growth.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIGURES_DIR, REPORTS_DIR, BACKTEST
from backtests.engine import backtest, compare_strategies
from analysis.advanced_research import (
    load_all_data, build_advanced_features, select_features,
    strategy_momentum_with_stop, strategy_market_contrarian,
    strategy_regime_conditional, strategy_dix_enhanced,
    strategy_mean_reversion, strategy_gap_fade,
    strategy_multi_timeframe_momentum, strategy_rsi_divergence,
    strategy_bollinger_mean_reversion, strategy_adaptive_ensemble,
    strategy_peer_leadlag, strategy_volatility_breakout,
    strategy_volume_spike, strategy_ml_signal,
    strategy_regression_signal, strategy_combined_best,
    vol_adjusted_signal,
)


def buy_and_hold_metrics(prices: pd.Series, rfr: float = 0.05) -> dict:
    """Compute buy-and-hold benchmark metrics."""
    ret = prices.pct_change().dropna()
    n = len(ret)
    cum = (1 + ret).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / n) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ann_ret - rfr) / ann_vol if ann_vol > 0 else 0
    peak = cum.cummax()
    max_dd = ((cum - peak) / peak).min()
    hit = (ret > 0).mean()
    return {
        "name": "Buy & Hold",
        "sharpe": round(sharpe, 4),
        "annual_return": round(ann_ret, 4),
        "annual_vol": round(ann_vol, 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate": round(hit, 4),
        "total_return": round(total_ret, 4),
        "n_trades": 1,
        "n_days": n,
    }


def run_train_test_validation():
    """
    Split data into train (first 70%) and test (last 30%).
    Evaluate all strategies on BOTH periods, compare vs buy-and-hold.
    """
    print("=" * 70)
    print("  TRAIN/TEST SPLIT VALIDATION")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    data = load_all_data()
    df = build_advanced_features(data)
    onds = data["onds"]
    onds_close = onds["Close"]
    rfr = BACKTEST["risk_free_rate"]

    # ── Split point ──────────────────────────────────────────
    n = len(onds_close)
    split_idx = int(n * 0.70)
    split_date = onds_close.index[split_idx]

    train_close = onds_close.iloc[:split_idx]
    test_close = onds_close.iloc[split_idx:]

    train_bh = buy_and_hold_metrics(train_close, rfr)
    test_bh = buy_and_hold_metrics(test_close, rfr)

    print(f"\n  Total: {n} trading days ({onds_close.index[0].strftime('%Y-%m-%d')} to {onds_close.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Train: {split_idx} days ({onds_close.index[0].strftime('%Y-%m-%d')} to {split_date.strftime('%Y-%m-%d')})")
    print(f"  Test:  {n - split_idx} days ({split_date.strftime('%Y-%m-%d')} to {onds_close.index[-1].strftime('%Y-%m-%d')})")
    print(f"\n  Train B&H: {train_bh['total_return']:+.2%} (Sharpe {train_bh['sharpe']:+.4f})")
    print(f"  Test  B&H: {test_bh['total_return']:+.2%} (Sharpe {test_bh['sharpe']:+.4f})")
    print(f"  Full  B&H: {(onds_close.iloc[-1]/onds_close.iloc[0]-1):+.2%}")

    # ── Generate all signals on FULL data ────────────────────
    # (strategies look at the full history for technical indicators,
    #  but we evaluate ONLY on the test period)
    print("\n[2/4] Generating strategy signals...")

    leakage_cols = {"ret_open_close", "ret_high_low", "ret_close_high", "ret_close_low",
                    "gap", "volume_change", "volume_ma_ratio", "dollar_volume", "price_range_pct",
                    "ret_1d", "dow", "is_monday", "is_friday"}
    all_feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in leakage_cols]
    selected_dir = select_features(df, "target_direction", n_top=15, allowed_cols=all_feature_cols)

    strategies = {
        "Peer Lead-Lag": strategy_peer_leadlag(data),
        "Market Contrarian": strategy_market_contrarian(data),
        "Regime Conditional": strategy_regime_conditional(data),
        "Vol Breakout": strategy_volatility_breakout(data),
        "DIX Enhanced": strategy_dix_enhanced(data),
        "Mean Reversion": strategy_mean_reversion(data),
        "Momentum+Stop": strategy_momentum_with_stop(data),
        "Gap Fade": strategy_gap_fade(data),
        "Volume Spike": strategy_volume_spike(data),
        "RSI Divergence": strategy_rsi_divergence(data),
        "Multi-TF Momentum": strategy_multi_timeframe_momentum(data),
        "BB Mean Reversion": strategy_bollinger_mean_reversion(data),
        "Combined Best": strategy_combined_best(data, df),
        "Adaptive Ensemble": strategy_adaptive_ensemble(data, df),
        "ML Direction": strategy_ml_signal(df, selected_dir, "target_direction"),
    }

    # Add vol-adjusted versions
    va_strategies = {}
    for name, sig in strategies.items():
        va_strategies[f"{name} (VA)"] = vol_adjusted_signal(sig, onds_close)
    strategies.update(va_strategies)

    # ── Evaluate on train and test periods ───────────────────
    print("\n[3/4] Evaluating on train and test periods...")
    print("-" * 70)

    results_rows = []

    for name, sig in strategies.items():
        # Split signal into train and test
        train_sig = sig.loc[sig.index < split_date]
        test_sig = sig.loc[sig.index >= split_date]

        # Backtest on train
        try:
            train_bt = backtest(train_close, train_sig, name=f"{name}", plot=False)
            train_sharpe = train_bt["sharpe"]
            train_return = train_bt["total_return"]
            train_dd = train_bt["max_drawdown"]
            train_trades = train_bt["n_trades"]
        except:
            train_sharpe = train_return = train_dd = 0
            train_trades = 0

        # Backtest on test (the actual out-of-sample evaluation)
        try:
            test_bt = backtest(test_close, test_sig, name=f"{name}", plot=False)
            test_sharpe = test_bt["sharpe"]
            test_return = test_bt["total_return"]
            test_dd = test_bt["max_drawdown"]
            test_trades = test_bt["n_trades"]
            test_hit = test_bt["hit_rate"]
        except:
            test_sharpe = test_return = test_dd = 0
            test_trades = 0
            test_hit = 0

        # Alpha vs B&H in test period
        test_alpha = test_return - test_bh["total_return"]

        results_rows.append({
            "Strategy": name,
            "Train Sharpe": train_sharpe,
            "Train Return": train_return,
            "Train MaxDD": train_dd,
            "Test Sharpe": test_sharpe,
            "Test Return": test_return,
            "Test MaxDD": test_dd,
            "Test Alpha": test_alpha,
            "Test Hit": test_hit,
            "Test Trades": test_trades,
            "Consistent": "YES" if (train_sharpe > 0 and test_sharpe > 0) else "no",
        })

    # Sort by test Sharpe
    results_df = pd.DataFrame(results_rows).sort_values("Test Sharpe", ascending=False)

    # ── Print results ────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  TRAIN vs TEST COMPARISON (sorted by Test Sharpe)")
    print(f"{'='*110}")
    print(f"  {'Strategy':<30s} | {'TRAIN':^27s} | {'TEST (out-of-sample)':^42s} | {'OK?':>4s}")
    print(f"  {'':30s} | {'Sharpe':>8s} {'Return':>9s} {'MaxDD':>8s} | {'Sharpe':>8s} {'Return':>9s} {'MaxDD':>8s} {'Alpha':>8s} {'Hit':>6s} | {'':>4s}")
    print(f"  {'-'*30} + {'-'*27} + {'-'*42} + {'-'*4}")

    # Print B&H first
    print(f"  {'** Buy & Hold **':<30s} | {train_bh['sharpe']:>+8.4f} {train_bh['total_return']:>+8.2%} {train_bh['max_drawdown']:>8.2%} | "
          f"{test_bh['sharpe']:>+8.4f} {test_bh['total_return']:>+8.2%} {test_bh['max_drawdown']:>8.2%} {'0.00%':>8s} {test_bh['hit_rate']:>5.1%} | {'BASE':>4s}")

    for _, row in results_df.iterrows():
        consistent = row["Consistent"]
        print(f"  {row['Strategy']:<30s} | {row['Train Sharpe']:>+8.4f} {row['Train Return']:>+8.2%} {row['Train MaxDD']:>8.2%} | "
              f"{row['Test Sharpe']:>+8.4f} {row['Test Return']:>+8.2%} {row['Test MaxDD']:>8.2%} {row['Test Alpha']:>+7.2%} {row['Test Hit']:>5.1%} | {consistent:>4s}")

    # ── Highlight consistent winners ─────────────────────────
    print(f"\n{'='*70}")
    print("  STRATEGIES THAT WORK IN BOTH TRAIN AND TEST")
    print(f"{'='*70}")

    consistent = results_df[results_df["Consistent"] == "YES"].copy()
    beats_bh = consistent[consistent["Test Sharpe"] > test_bh["sharpe"]]

    if len(consistent) > 0:
        print(f"\n  Strategies with positive Sharpe in BOTH periods: {len(consistent)}")
        for _, row in consistent.iterrows():
            beat = " >> BEATS B&H" if row["Test Sharpe"] > test_bh["sharpe"] else ""
            print(f"    {row['Strategy']:<30s} Train={row['Train Sharpe']:+.4f}  Test={row['Test Sharpe']:+.4f}  "
                  f"TestReturn={row['Test Return']:+.2%}{beat}")
    else:
        print("  No strategies have positive Sharpe in both periods.")

    if len(beats_bh) > 0:
        print(f"\n  Strategies that BEAT buy-and-hold in test period: {len(beats_bh)}")
        for _, row in beats_bh.iterrows():
            print(f"    {row['Strategy']:<30s} Test Sharpe={row['Test Sharpe']:+.4f} vs B&H={test_bh['sharpe']:+.4f}")
    else:
        print(f"\n  No strategies beat buy-and-hold (Sharpe {test_bh['sharpe']:+.4f}) in test period.")

    # ── Save results ─────────────────────────────────────────
    results_df.to_csv(REPORTS_DIR / "train_test_validation.csv", index=False)
    print(f"\n  Saved: {REPORTS_DIR / 'train_test_validation.csv'}")

    # ── Plot: Train vs Test Sharpe scatter ───────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in results_df.iterrows():
        color = "#4CAF50" if row["Consistent"] == "YES" else "#F44336"
        marker = "D" if row["Test Sharpe"] > test_bh["sharpe"] else "o"
        ax.scatter(row["Train Sharpe"], row["Test Sharpe"], c=color, marker=marker, s=60, alpha=0.7)
        # Label only interesting ones
        if abs(row["Train Sharpe"]) > 1.5 or abs(row["Test Sharpe"]) > 1.5 or row["Consistent"] == "YES":
            short_name = row["Strategy"].replace(" (VA)", "*")
            if len(short_name) > 20:
                short_name = short_name[:18] + ".."
            ax.annotate(short_name, (row["Train Sharpe"], row["Test Sharpe"]),
                       fontsize=6, alpha=0.8, ha="left", va="bottom")

    # Add B&H reference point
    ax.scatter(train_bh["sharpe"], test_bh["sharpe"], c="gold", marker="*", s=200, zorder=10,
               edgecolors="black", linewidths=0.5)
    ax.annotate("Buy & Hold", (train_bh["sharpe"], test_bh["sharpe"]),
               fontsize=9, fontweight="bold", ha="left", va="bottom")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=test_bh["sharpe"], color="gold", linestyle=":", alpha=0.5, label=f"B&H Test Sharpe={test_bh['sharpe']:+.2f}")
    ax.axvline(x=train_bh["sharpe"], color="gold", linestyle=":", alpha=0.5)
    ax.set_xlabel("Train Period Sharpe")
    ax.set_ylabel("Test Period Sharpe (Out-of-Sample)")
    ax.set_title("Strategy Consistency: Train vs Test Sharpe\n(green=both positive, red=inconsistent, diamond=beats B&H)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "train_test_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'train_test_scatter.png'}")

    # ── Plot: Equity curves on test period only ──────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    # B&H equity curve on test
    test_ret = test_close.pct_change().dropna()
    bh_equity = (1 + test_ret).cumprod()
    ax.plot(bh_equity, label=f"Buy & Hold (S={test_bh['sharpe']:+.2f})", color="gray",
            linewidth=2, linestyle="--")

    # Plot top strategies on test period
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    top_test = results_df.head(8)
    for i, (_, row) in enumerate(top_test.iterrows()):
        sig = strategies[row["Strategy"]]
        test_sig = sig.loc[sig.index >= split_date]
        try:
            bt = backtest(test_close, test_sig, name=row["Strategy"], plot=False)
            ax.plot(bt["cum_return"], label=f"{row['Strategy']} (S={row['Test Sharpe']:+.2f})",
                    color=colors[i % len(colors)], linewidth=1)
        except:
            pass

    ax.axvline(x=split_date, color="red", linestyle="-", alpha=0.3, linewidth=2)
    ax.set_title(f"Out-of-Sample Test Period: {split_date.strftime('%Y-%m-%d')} onwards")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "test_period_equity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'test_period_equity.png'}")

    return results_df


if __name__ == "__main__":
    results_df = run_train_test_validation()
